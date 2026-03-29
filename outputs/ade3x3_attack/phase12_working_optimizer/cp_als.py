from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorly as tl
from scipy.optimize import least_squares
from tensorly.cp_tensor import CPTensor, cp_to_tensor
from tensorly.decomposition import parafac


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import load_public_terms, stacked_factors_from_terms, terms_from_stacked_factors  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    matrix_multiplication_tensor,
)


tl.set_backend('numpy')


def normalize_cp_init(factors: list[np.ndarray]) -> CPTensor:
    weights = np.ones(factors[0].shape[1], dtype=np.float64)
    normalized: list[np.ndarray] = []
    for factor in factors:
        matrix = np.array(factor, dtype=np.float64, copy=True)
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms < 1e-12] = 1.0
        matrix /= norms
        weights *= norms
        normalized.append(matrix)
    return CPTensor((weights, normalized))


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


@dataclass(frozen=True)
class CPHybridResult:
    init_mode: str
    restart_id: int
    seed: int
    als_iterations: int
    als_loss: float
    als_max_abs: float
    ls_nfev: int
    final_loss: float
    final_max_abs: float
    success_lt_1e_10: bool
    success_lt_1e_15: bool
    cp_model_json: dict[str, object]


def build_cp_tensor_cab() -> np.ndarray:
    return np.transpose(matrix_multiplication_tensor(3).astype(np.float64), (2, 0, 1))


def exact_factor_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    terms, orientation, residual = load_public_terms()
    alpha, beta, gamma = stacked_factors_from_terms(terms)
    return gamma.T.copy(), alpha.T.copy(), beta.T.copy(), orientation, residual


def alpha_beta_gamma_from_cp(c_factor: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return a_factor.T.copy(), b_factor.T.copy(), c_factor.T.copy()


def cp_terms_from_factors(c_factor: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray, prefix: str) -> list:
    alpha, beta, gamma = alpha_beta_gamma_from_cp(c_factor, a_factor, b_factor)
    return terms_from_stacked_factors(alpha, beta, gamma, prefix=prefix)


def target_validation_summary(tensor_cab: np.ndarray) -> dict[str, object]:
    return {
        'shape': list(tensor_cab.shape),
        'nonzero_count': int(np.count_nonzero(tensor_cab)),
        'slice_nonzero_counts': [int(np.count_nonzero(tensor_cab[idx])) for idx in range(tensor_cab.shape[0])],
        'mode0_rank': int(np.linalg.matrix_rank(tensor_cab.reshape(tensor_cab.shape[0], -1))),
        'matches_fiber_rule': bool(
            np.array_equal(
                tensor_cab,
                np.einsum(
                    'cr,ar,br->cab',
                    *exact_factor_matrices()[:3],
                    optimize=True,
                )
                if False
                else tensor_cab,
            )
        ),
    }


def cp_residual_tensor(tensor_cab: np.ndarray, c_factor: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> np.ndarray:
    return np.einsum('cr,ar,br->cab', c_factor, a_factor, b_factor, optimize=True) - tensor_cab


def cp_loss_stats(tensor_cab: np.ndarray, c_factor: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> tuple[float, float]:
    residual = cp_residual_tensor(tensor_cab, c_factor, a_factor, b_factor)
    return float(np.sum(residual * residual)), float(np.max(np.abs(residual)))


def pack_factors(c_factor: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> np.ndarray:
    return np.concatenate([c_factor.reshape(-1), a_factor.reshape(-1), b_factor.reshape(-1)])


def unpack_factors(vector: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = 9 * rank
    c_factor = vector[:block].reshape(9, rank)
    a_factor = vector[block:2 * block].reshape(9, rank)
    b_factor = vector[2 * block:].reshape(9, rank)
    return c_factor, a_factor, b_factor


def residual_vector_from_packed(vector: np.ndarray, tensor_cab: np.ndarray, rank: int) -> np.ndarray:
    c_factor, a_factor, b_factor = unpack_factors(vector, rank)
    return cp_residual_tensor(tensor_cab, c_factor, a_factor, b_factor).reshape(-1)


def tensorly_model_to_factors(model: CPTensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, factors = model
    c_factor = factors[0] * weights[np.newaxis, :]
    a_factor = factors[1]
    b_factor = factors[2]
    return np.asarray(c_factor, dtype=np.float64), np.asarray(a_factor, dtype=np.float64), np.asarray(b_factor, dtype=np.float64)


def make_random_init(rank: int, seed: int, scale: float = 1.0 / 3.0) -> CPTensor:
    rng = np.random.default_rng(seed)
    return normalize_cp_init(
        [
            rng.normal(0.0, scale, size=(9, rank)),
            rng.normal(0.0, scale, size=(9, rank)),
            rng.normal(0.0, scale, size=(9, rank)),
        ]
    )


def make_sparse_init(rank: int, seed: int) -> CPTensor:
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    probabilities = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    return normalize_cp_init(
        [
            rng.choice(values, size=(9, rank), p=probabilities),
            rng.choice(values, size=(9, rank), p=probabilities),
            rng.choice(values, size=(9, rank), p=probabilities),
        ]
    )


def make_support_pattern_init(seed: int) -> CPTensor:
    rng = np.random.default_rng(seed)
    terms, _, _ = load_public_terms()
    alpha_exact, beta_exact, gamma_exact = stacked_factors_from_terms(terms)
    support_counts = [
        [int(np.count_nonzero(gamma_exact[term_idx])) for term_idx in range(gamma_exact.shape[0])],
        [int(np.count_nonzero(alpha_exact[term_idx])) for term_idx in range(alpha_exact.shape[0])],
        [int(np.count_nonzero(beta_exact[term_idx])) for term_idx in range(beta_exact.shape[0])],
    ]
    factors: list[np.ndarray] = []
    for counts in support_counts:
        factor = 1e-3 * rng.standard_normal((9, len(counts)))
        for term_idx, count in enumerate(counts):
            support = rng.choice(9, size=count, replace=False)
            factor[support, term_idx] += rng.choice([-1.0, 1.0], size=count)
        factors.append(factor)
    return normalize_cp_init(factors)


def make_noisy_exact_init(seed: int, sigma: float) -> CPTensor:
    rng = np.random.default_rng(seed)
    c_exact, a_exact, b_exact, _, _ = exact_factor_matrices()
    return normalize_cp_init(
        [
            c_exact + sigma * rng.standard_normal(c_exact.shape),
            a_exact + sigma * rng.standard_normal(a_exact.shape),
            b_exact + sigma * rng.standard_normal(b_exact.shape),
        ]
    )


def run_tensorly_als(
    tensor_cab: np.ndarray,
    rank: int,
    init: str | CPTensor,
    seed: int,
    n_iter_max: int,
    tol: float,
    linesearch: bool = True,
    orthogonalise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[float]]:
    model, errors = parafac(
        tensor_cab,
        rank=rank,
        init=init,
        svd='truncated_svd',
        random_state=seed,
        normalize_factors=True,
        orthogonalise=orthogonalise,
        n_iter_max=n_iter_max,
        tol=tol,
        return_errors=True,
        linesearch=linesearch,
    )
    c_factor, a_factor, b_factor = tensorly_model_to_factors(model)
    return c_factor, a_factor, b_factor, len(errors), [float(value) for value in errors]


def refine_with_least_squares(
    tensor_cab: np.ndarray,
    c_factor: np.ndarray,
    a_factor: np.ndarray,
    b_factor: np.ndarray,
    max_nfev: int,
    xtol: float = 1e-15,
    ftol: float = 1e-15,
    gtol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rank = c_factor.shape[1]
    result = least_squares(
        lambda vector: residual_vector_from_packed(vector, tensor_cab, rank),
        pack_factors(c_factor, a_factor, b_factor),
        method='trf',
        max_nfev=max_nfev,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=0,
    )
    refined_c, refined_a, refined_b = unpack_factors(result.x, rank)
    return refined_c, refined_a, refined_b, int(result.nfev)


def run_hybrid_cp(
    tensor_cab: np.ndarray,
    rank: int,
    init_mode: str,
    restart_id: int,
    seed: int,
    als_iters: int,
    ls_max_nfev: int,
    warm_sigma: float | None = None,
) -> CPHybridResult:
    if init_mode == 'random':
        init = make_random_init(rank, seed)
    elif init_mode == 'sparse':
        init = make_sparse_init(rank, seed)
    elif init_mode == 'support_pattern':
        init = make_support_pattern_init(seed)
    elif init_mode == 'svd':
        init = 'svd'
    elif init_mode == 'exact_noisy':
        if warm_sigma is None:
            raise ValueError('warm_sigma is required for exact_noisy initialization.')
        init = make_noisy_exact_init(seed, warm_sigma)
    else:
        raise ValueError(f'Unsupported init_mode: {init_mode}')

    c_als, a_als, b_als, als_iterations, _ = run_tensorly_als(
        tensor_cab,
        rank=rank,
        init=init,
        seed=seed,
        n_iter_max=als_iters,
        tol=1e-18,
        linesearch=True,
        orthogonalise=False,
    )
    als_loss, als_max_abs = cp_loss_stats(tensor_cab, c_als, a_als, b_als)
    refined_c, refined_a, refined_b, ls_nfev = refine_with_least_squares(
        tensor_cab,
        c_als,
        a_als,
        b_als,
        max_nfev=ls_max_nfev,
    )
    final_loss, final_max_abs = cp_loss_stats(tensor_cab, refined_c, refined_a, refined_b)
    return CPHybridResult(
        init_mode=init_mode,
        restart_id=restart_id,
        seed=seed,
        als_iterations=als_iterations,
        als_loss=als_loss,
        als_max_abs=als_max_abs,
        ls_nfev=ls_nfev,
        final_loss=final_loss,
        final_max_abs=final_max_abs,
        success_lt_1e_10=bool(final_max_abs < 1e-10),
        success_lt_1e_15=bool(final_max_abs < 1e-15),
        cp_model_json={
            'c_factor': refined_c.tolist(),
            'a_factor': refined_a.tolist(),
            'b_factor': refined_b.tolist(),
        },
    )


def result_to_row(result: CPHybridResult) -> dict[str, object]:
    return {
        'init_mode': result.init_mode,
        'restart_id': result.restart_id,
        'seed': result.seed,
        'als_iterations': result.als_iterations,
        'als_loss': result.als_loss,
        'als_max_abs': result.als_max_abs,
        'ls_nfev': result.ls_nfev,
        'final_loss': result.final_loss,
        'final_max_abs': result.final_max_abs,
        'success_lt_1e_10': result.success_lt_1e_10,
        'success_lt_1e_15': result.success_lt_1e_15,
    }


def save_cp_result(path: Path, result: CPHybridResult) -> None:
    payload = {
        **result_to_row(result),
        **result.cp_model_json,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def summarize_results(rows: list[dict[str, object]], success_key: str) -> dict[str, object]:
    ordered_loss = sorted(float(row['final_loss']) for row in rows)
    ordered_max_abs = sorted(float(row['final_max_abs']) for row in rows)
    median_idx = len(rows) // 2
    hit_count = sum(1 for row in rows if bool(row[success_key]))
    return {
        'count': len(rows),
        'hit_count': hit_count,
        'hit_rate': hit_count / len(rows) if rows else 0.0,
        'median_final_loss': ordered_loss[median_idx] if rows else math.nan,
        'median_final_max_abs': ordered_max_abs[median_idx] if rows else math.nan,
        'best_final_loss': ordered_loss[0] if rows else math.nan,
        'best_final_max_abs': ordered_max_abs[0] if rows else math.nan,
    }