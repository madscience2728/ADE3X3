from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    build_mode_matrices,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
    reconstruct_tensor,
)


S3 = list(permutations(range(3)))
ACTIONS = [(pi_rA, pi_shared, pi_cB) for pi_rA in S3 for pi_shared in S3 for pi_cB in S3]
TARGET_TENSOR = matrix_multiplication_tensor(3).astype(np.float64)
TARGET_TENSOR_TORCH = None
NULL_VECTOR_PATH = ATTACK_ROOT / 'phase5_diagnostics' / 'null_vectors.csv'
SHARED_ETA_MODE_MATRICES: dict[tuple[int, int, int], sp.Matrix] = {
    (0, 1, 2): sp.Matrix([[1, 0], [0, 1]]),
    (0, 2, 1): sp.Matrix([[1, 1], [0, -1]]),
    (1, 0, 2): sp.Matrix([[-1, 0], [1, 1]]),
    (1, 2, 0): sp.Matrix([[0, 1], [-1, -1]]),
    (2, 0, 1): sp.Matrix([[-1, -1], [1, 0]]),
    (2, 1, 0): sp.Matrix([[0, -1], [-1, 0]]),
}


@dataclass(frozen=True)
class NumericProfile:
    rank_h: int
    rank_delta: int
    rank_joint: int
    delta_in_eta: bool
    projection_max_residual: float
    projection_nonzero_count: int
    projection_unique_nonzero_coefficients: list[str]
    projection_allclose_to_integer: bool


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':')))
            handle.write('\n')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eta_coordinate_labels() -> list[str]:
    labels: list[str] = []
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'eta1[{row_idx},{col_idx}]')
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'eta2[{row_idx},{col_idx}]')
    return labels


def delta_coordinate_labels() -> list[str]:
    labels: list[str] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    labels.append(f'delta[{row_idx},{sum_left},{sum_right},{col_idx}]')
    return labels


def factor_coordinate_labels(prefix: str) -> list[str]:
    return [f'{prefix}[{row_idx},{col_idx}]' for row_idx in range(3) for col_idx in range(3)]


def read_phase5_null_vectors(path: Path | None = None) -> list[dict[str, object]]:
    source = NULL_VECTOR_PATH if path is None else path
    rows: list[dict[str, object]] = []
    with open(source, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            payload: dict[str, object] = {
                'basis_id': int(row['basis_id']),
                'support_size': int(row['support_size']),
                'formula': row['formula'],
            }
            for label in eta_coordinate_labels():
                payload[label] = int(row[label])
            rows.append(payload)
    return rows


def phase5_identity_basis(path: Path | None = None) -> tuple[list[str], list[sp.Matrix]]:
    labels = eta_coordinate_labels()
    vectors: list[sp.Matrix] = []
    for row in read_phase5_null_vectors(path=path):
        vectors.append(sp.Matrix([int(row[label]) for label in labels]))
    return labels, vectors


def eta_identity_to_q_matrix(identity_vector: sp.Matrix) -> sp.Matrix:
    if identity_vector.rows != 18 or identity_vector.cols != 1:
        raise ValueError('Identity vector must be an 18x1 eta-coordinate column.')

    q_matrix = sp.zeros(9, 9)
    for row_idx in range(3):
        for col_idx in range(3):
            eta1_coeff = sp.nsimplify(identity_vector[3 * row_idx + col_idx, 0])
            eta2_coeff = sp.nsimplify(identity_vector[9 + 3 * row_idx + col_idx, 0])
            if eta1_coeff != 0:
                q_matrix[3 * row_idx + 0, 3 * 0 + col_idx] += eta1_coeff
                q_matrix[3 * row_idx + 1, 3 * 1 + col_idx] -= eta1_coeff
            if eta2_coeff != 0:
                q_matrix[3 * row_idx + 1, 3 * 1 + col_idx] += eta2_coeff
                q_matrix[3 * row_idx + 2, 3 * 2 + col_idx] -= eta2_coeff
    return q_matrix


def eta_action_matrix(
    action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> sp.Matrix:
    pi_rA, pi_shared, pi_cB = action
    mode_mix = SHARED_ETA_MODE_MATRICES[pi_shared]
    matrix = sp.zeros(18, 18)
    inverse_r = {dst_idx: src_idx for src_idx, dst_idx in enumerate(pi_rA)}
    inverse_c = {dst_idx: src_idx for src_idx, dst_idx in enumerate(pi_cB)}

    for row_idx in range(3):
        for col_idx in range(3):
            source_row = inverse_r[row_idx]
            source_col = inverse_c[col_idx]
            out_eta1 = 3 * row_idx + col_idx
            out_eta2 = 9 + 3 * row_idx + col_idx
            in_eta1 = 3 * source_row + source_col
            in_eta2 = 9 + 3 * source_row + source_col
            matrix[out_eta1, in_eta1] = mode_mix[0, 0]
            matrix[out_eta1, in_eta2] = mode_mix[0, 1]
            matrix[out_eta2, in_eta1] = mode_mix[1, 0]
            matrix[out_eta2, in_eta2] = mode_mix[1, 1]
    return matrix


def permutation_matrix(perm: tuple[int, int, int]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for src_idx, dst_idx in enumerate(perm):
        matrix[dst_idx, src_idx] = 1
    return matrix


def action_label(action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> str:
    pi_rA, pi_shared, pi_cB = action
    return f'pi_rA={pi_rA};pi_shared={pi_shared};pi_cB={pi_cB}'


def apply_action_to_term(term: Term, action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> Term:
    pi_rA, pi_shared, pi_cB = action
    p_r = permutation_matrix(pi_rA)
    p_s = permutation_matrix(pi_shared)
    p_c = permutation_matrix(pi_cB)
    alpha = p_r @ np.asarray(term.alpha) @ p_s.T
    beta = p_s @ np.asarray(term.beta) @ p_c.T
    gamma = p_r @ np.asarray(term.gamma) @ p_c.T
    return Term(term.term_id, term.source_label, alpha, beta, gamma)


def apply_action_to_terms(
    terms: list[Term],
    action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> list[Term]:
    return [apply_action_to_term(term, action) for term in terms]


def serialize_term(term: Term) -> dict:
    return {
        'term_id': term.term_id,
        'source_label': term.source_label,
        'alpha': np.asarray(term.alpha, dtype=np.float64).tolist(),
        'beta': np.asarray(term.beta, dtype=np.float64).tolist(),
        'gamma': np.asarray(term.gamma, dtype=np.float64).tolist(),
    }


def deserialize_term(payload: dict) -> Term:
    return Term(
        term_id=str(payload['term_id']),
        source_label=str(payload.get('source_label', payload['term_id'])),
        alpha=np.array(payload['alpha'], dtype=np.float64),
        beta=np.array(payload['beta'], dtype=np.float64),
        gamma=np.array(payload['gamma'], dtype=np.float64),
    )


def serialize_decomposition(
    decomposition_id: str,
    source: str,
    source_detail: str,
    terms: list[Term],
    max_abs_residual: float,
    loss_value: float,
    metadata: dict | None = None,
) -> dict:
    return {
        'decomposition_id': decomposition_id,
        'source': source,
        'source_detail': source_detail,
        'term_count': len(terms),
        'max_abs_residual': float(max_abs_residual),
        'loss_value': float(loss_value),
        'terms': [serialize_term(term) for term in terms],
        'metadata': metadata or {},
    }


def load_serialized_decompositions(path: Path) -> list[dict]:
    return read_jsonl(path)


def terms_from_decomposition(payload: dict) -> list[Term]:
    return [deserialize_term(term_payload) for term_payload in payload['terms']]


def stacked_factors_from_terms(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta, gamma


def terms_from_stacked_factors(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    prefix: str,
) -> list[Term]:
    terms: list[Term] = []
    for term_idx in range(alpha.shape[0]):
        terms.append(
            Term(
                term_id=f'{prefix}{term_idx + 1:02d}',
                source_label=f'{prefix}{term_idx + 1:02d}',
                alpha=np.array(alpha[term_idx], dtype=np.float64).reshape(3, 3),
                beta=np.array(beta[term_idx], dtype=np.float64).reshape(3, 3),
                gamma=np.array(gamma[term_idx], dtype=np.float64).reshape(3, 3),
            )
        )
    return terms


def tensor_residual_stats(terms: list[Term]) -> tuple[float, float]:
    total = np.zeros((9, 9, 9), dtype=np.float64)
    for term in terms:
        total += np.einsum(
            'a,b,c->abc',
            np.asarray(term.alpha, dtype=np.float64).reshape(-1),
            np.asarray(term.beta, dtype=np.float64).reshape(-1),
            np.asarray(term.gamma, dtype=np.float64).reshape(-1),
            optimize=True,
        )
    residual = total - TARGET_TENSOR
    return float(np.max(np.abs(residual))), float(np.sum(residual * residual))


def output_unfolding_9x81(tensor: np.ndarray | None = None) -> np.ndarray:
    source = TARGET_TENSOR if tensor is None else np.asarray(tensor, dtype=np.float64)
    return np.transpose(source, (2, 0, 1)).reshape(9, 81)


def target_output_unfolding_9x81() -> np.ndarray:
    return output_unfolding_9x81(TARGET_TENSOR)


def solve_gamma_least_squares(alpha: np.ndarray, beta: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    target_tensor = TARGET_TENSOR if target is None else np.asarray(target, dtype=np.float64)
    features = np.einsum('ra,rb->abr', alpha, beta, optimize=True).reshape(81, alpha.shape[0])
    target_flat = target_tensor.reshape(81, 9)
    gamma, *_ = np.linalg.lstsq(features, target_flat, rcond=None)
    return np.asarray(gamma, dtype=np.float64)


def build_mode_matrices_numeric(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha, beta, _ = stacked_factors_from_terms(terms)
    rank = alpha.shape[0]
    sigma = np.zeros((rank, 9), dtype=np.float64)
    eta1 = np.zeros((rank, 9), dtype=np.float64)
    eta2 = np.zeros((rank, 9), dtype=np.float64)
    delta = np.zeros((rank, 54), dtype=np.float64)

    sigma_col = 0
    for row_idx in range(3):
        for col_idx in range(3):
            lambda0 = alpha[:, 3 * row_idx + 0] * beta[:, 0 * 3 + col_idx]
            lambda1 = alpha[:, 3 * row_idx + 1] * beta[:, 1 * 3 + col_idx]
            lambda2 = alpha[:, 3 * row_idx + 2] * beta[:, 2 * 3 + col_idx]
            sigma[:, sigma_col] = lambda0 + lambda1 + lambda2
            eta1[:, sigma_col] = lambda0 - lambda1
            eta2[:, sigma_col] = lambda1 - lambda2
            sigma_col += 1

    delta_col = 0
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    delta[:, delta_col] = alpha[:, 3 * row_idx + sum_left] * beta[:, 3 * sum_right + col_idx]
                    delta_col += 1

    h_matrix = np.hstack([eta1, eta2])
    nuisance = np.hstack([h_matrix, delta])
    return sigma, eta1, eta2, delta, h_matrix, nuisance


def numeric_rank_report(matrix: np.ndarray, tol: float = 1e-10) -> dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > tol))
    boundary_sv = float(singular_values[rank - 1]) if rank else 0.0
    next_sv = float(singular_values[rank]) if rank < singular_values.size else 0.0
    gap_ratio = float('inf') if next_sv == 0.0 else boundary_sv / next_sv
    return {
        'rank': rank,
        'threshold': tol,
        'boundary_singular_value': boundary_sv,
        'next_singular_value': next_sv,
        'gap_ratio': gap_ratio,
        'singular_values': [float(value) for value in singular_values],
    }


def delta_in_eta_numeric(
    h_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    rank_tol: float = 1e-10,
    projection_tol: float = 1e-8,
    nonzero_tol: float = 1e-9,
    integrality_tol: float = 1e-8,
) -> NumericProfile:
    rank_h = numeric_rank_report(h_matrix, tol=rank_tol)['rank']
    rank_delta = numeric_rank_report(delta_matrix, tol=rank_tol)['rank']
    joint = np.hstack([h_matrix, delta_matrix])
    rank_joint = numeric_rank_report(joint, tol=rank_tol)['rank']
    projection = np.linalg.lstsq(h_matrix, delta_matrix, rcond=rank_tol)[0]
    projection_residual = h_matrix @ projection - delta_matrix
    projection_max_residual = float(np.max(np.abs(projection_residual)))
    delta_in_eta = bool(rank_joint == rank_h and projection_max_residual <= projection_tol)
    nz_mask = np.abs(projection) > nonzero_tol
    nz_values = projection[nz_mask]
    rounded = np.rint(nz_values)
    unique_coeffs = sorted({f'{value:.8g}' for value in rounded}) if nz_values.size else []
    allclose_to_integer = bool(nz_values.size == 0 or np.all(np.abs(nz_values - rounded) <= integrality_tol))
    return NumericProfile(
        rank_h=rank_h,
        rank_delta=rank_delta,
        rank_joint=rank_joint,
        delta_in_eta=delta_in_eta,
        projection_max_residual=projection_max_residual,
        projection_nonzero_count=int(nz_mask.sum()),
        projection_unique_nonzero_coefficients=unique_coeffs,
        projection_allclose_to_integer=allclose_to_integer,
    )


def exact_projection_stats(h_matrix: sp.Matrix, delta_matrix: sp.Matrix) -> dict[str, object]:
    h_rank = int(h_matrix.rank())
    joint_rank = int(sp.Matrix.hstack(h_matrix, delta_matrix).rank())
    if joint_rank != h_rank:
        return {
            'delta_in_eta_exact': False,
            'projection_nonzero_count': 0,
            'projection_unique_nonzero_coefficients': [],
            'projection_integral': False,
            'projection_support_min': 0,
            'projection_support_max': 0,
        }

    _, pivots = h_matrix.rref()
    h_basis = h_matrix[:, pivots]
    projection_columns: list[list[sp.Expr]] = []
    support_sizes: list[int] = []
    coeff_values: list[sp.Expr] = []
    for delta_idx in range(delta_matrix.cols):
        basis_solution = list(h_basis.gauss_jordan_solve(delta_matrix[:, delta_idx])[0])
        full_solution = [sp.Integer(0)] * h_matrix.cols
        for basis_idx, pivot in enumerate(pivots):
            full_solution[pivot] = sp.simplify(basis_solution[basis_idx])
        projection_columns.append(full_solution)
        support_size = sum(1 for value in full_solution if value != 0)
        support_sizes.append(support_size)
        coeff_values.extend(value for value in full_solution if value != 0)

    projection_matrix = sp.Matrix(
        h_matrix.cols,
        delta_matrix.cols,
        lambda row_idx, col_idx: projection_columns[col_idx][row_idx],
    )
    if h_matrix * projection_matrix != delta_matrix:
        raise RuntimeError('Exact projection verification failed.')

    unique_coefficients = sorted({sp.sstr(value) for value in coeff_values})
    return {
        'delta_in_eta_exact': True,
        'projection_nonzero_count': len(coeff_values),
        'projection_unique_nonzero_coefficients': unique_coefficients,
        'projection_integral': all(bool(value.is_integer) for value in coeff_values),
        'projection_support_min': min(support_sizes) if support_sizes else 0,
        'projection_support_max': max(support_sizes) if support_sizes else 0,
    }


def profile_terms(
    terms: list[Term],
    rank_tol: float = 1e-10,
    projection_tol: float = 1e-8,
) -> dict[str, object]:
    sigma_num, eta1_num, eta2_num, delta_num, h_num, nuisance_num = build_mode_matrices_numeric(terms)
    numeric_profile = delta_in_eta_numeric(h_num, delta_num, rank_tol=rank_tol, projection_tol=projection_tol)
    max_abs_residual, loss_value = tensor_residual_stats(terms)
    rows = {
        'rank_H_numeric': numeric_profile.rank_h,
        'rank_Delta_numeric': numeric_profile.rank_delta,
        'rank_HDelta_numeric': numeric_profile.rank_joint,
        'delta_in_eta_numeric': numeric_profile.delta_in_eta,
        'projection_max_residual_numeric': numeric_profile.projection_max_residual,
        'projection_nonzero_count_numeric': numeric_profile.projection_nonzero_count,
        'projection_unique_nonzero_coefficients_numeric': ';'.join(numeric_profile.projection_unique_nonzero_coefficients),
        'projection_allclose_to_integer_numeric': numeric_profile.projection_allclose_to_integer,
        'rank_Nuisance_numeric': numeric_rank_report(nuisance_num, tol=rank_tol)['rank'],
        'rank_Sigma_numeric': numeric_rank_report(sigma_num, tol=rank_tol)['rank'],
        'quotient_gain_numeric': numeric_rank_report(np.hstack([sigma_num, nuisance_num]), tol=rank_tol)['rank'] - numeric_rank_report(nuisance_num, tol=rank_tol)['rank'],
        'max_abs_residual': max_abs_residual,
        'loss_value': loss_value,
    }

    if all(np.allclose(np.rint(np.asarray(term.alpha)), np.asarray(term.alpha)) and np.allclose(np.rint(np.asarray(term.beta)), np.asarray(term.beta)) for term in terms):
        sigma_exact, eta1_exact, eta2_exact, delta_exact, nuisance_exact = build_mode_matrices(terms, 3)
        h_exact = sp.Matrix.hstack(eta1_exact, eta2_exact)
        exact_stats = exact_projection_stats(h_exact, delta_exact)
        rows.update(
            {
                'rank_H_exact': int(h_exact.rank()),
                'rank_Delta_exact': int(delta_exact.rank()),
                'rank_HDelta_exact': int(sp.Matrix.hstack(h_exact, delta_exact).rank()),
                'rank_Nuisance_exact': int(nuisance_exact.rank()),
                'rank_Sigma_exact': int(sigma_exact.rank()),
                'quotient_gain_exact': int(sp.Matrix.hstack(sigma_exact, nuisance_exact).rank()) - int(nuisance_exact.rank()),
                'eta_nullity_exact': h_exact.cols - int(h_exact.rank()),
                **exact_stats,
            }
        )

    return rows


def primitive_integer_vector(vector: sp.Matrix) -> list[int]:
    entries = [sp.nsimplify(value) for value in vector]
    denominators = [int(sp.denom(value)) for value in entries if value != 0]
    lcm = 1
    for denominator in denominators:
        lcm = math.lcm(lcm, denominator)
    scaled = [int(sp.expand(value * lcm)) for value in entries]
    gcd = 0
    for value in scaled:
        gcd = math.gcd(gcd, abs(value))
    gcd = gcd or 1
    primitive = [value // gcd for value in scaled]
    for value in primitive:
        if value < 0:
            primitive = [-entry for entry in primitive]
            break
        if value > 0:
            break
    return primitive


def load_public_terms() -> tuple[list[Term], str, int]:
    return load_public_rank23_terms()