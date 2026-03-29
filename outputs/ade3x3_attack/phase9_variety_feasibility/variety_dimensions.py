from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import write_csv, write_json  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def random_q_matrices(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((count, 9, 9))


def matrix_rank(matrix: np.ndarray, tol: float = 1e-8) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def sigma_vector(a_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    return (a_vec.reshape(3, 3) @ b_vec.reshape(3, 3)).reshape(9)


def sample_point_on_variety(q_matrices: np.ndarray, rng: np.random.Generator, max_attempts: int = 512) -> tuple[np.ndarray, np.ndarray] | None:
    count = q_matrices.shape[0]
    for _ in range(max_attempts):
        a_vec = rng.standard_normal(9)
        if np.linalg.norm(a_vec) < 1e-12:
            continue
        constraint = np.stack([q_matrix.T @ a_vec for q_matrix in q_matrices], axis=0)
        rank = matrix_rank(constraint)
        if rank >= 9:
            continue
        _, _, vh = np.linalg.svd(constraint, full_matrices=True)
        nullspace = vh[rank:].T
        if nullspace.size == 0:
            continue
        coeffs = rng.standard_normal(nullspace.shape[1])
        b_vec = nullspace @ coeffs
        if np.linalg.norm(b_vec) < 1e-12:
            continue
        a_vec = a_vec / np.linalg.norm(a_vec)
        b_vec = b_vec / np.linalg.norm(b_vec)
        return a_vec, b_vec
    return None


def jacobian_rank(q_matrices: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray) -> int:
    rows = []
    for q_matrix in q_matrices:
        rows.append(np.concatenate([q_matrix @ b_vec, q_matrix.T @ a_vec]))
    return matrix_rank(np.stack(rows, axis=0))


def find_singular_point_for_k9(q_matrices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    best_a: np.ndarray | None = None
    best_vh: np.ndarray | None = None
    best_sigma_min = float('inf')
    for _ in range(2048):
        a_vec = rng.standard_normal(9)
        norm = np.linalg.norm(a_vec)
        if norm < 1e-12:
            continue
        a_vec = a_vec / norm
        constraint = np.stack([q_matrix.T @ a_vec for q_matrix in q_matrices], axis=0)
        _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
        sigma_min = float(singular_values[-1])
        if sigma_min < best_sigma_min:
            best_sigma_min = sigma_min
            best_a = a_vec.copy()
            best_vh = vh.copy()
            if sigma_min < 1e-8:
                break
    if best_a is None or best_vh is None or best_sigma_min > 1e-6:
        return None
    b_vec = best_vh[-1]
    if np.linalg.norm(b_vec) < 1e-12:
        return None
    return best_a, b_vec / np.linalg.norm(b_vec)


def main() -> None:
    trials_per_k = env_int('ADE3X3_PHASE9_DIMENSION_TRIALS', 16)
    sigma_samples_per_trial = env_int('ADE3X3_PHASE9_DIMENSION_SIGMA_SAMPLES', 48)
    rng = np.random.default_rng(20260329)

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for k in range(1, 10):
        observed_dimensions: list[int] = []
        observed_jacobian_ranks: list[int] = []
        observed_sigma_ranks: list[int] = []
        successful_trials = 0
        for trial_idx in range(trials_per_k):
            q_matrices = random_q_matrices(k, rng)
            point = find_singular_point_for_k9(q_matrices, rng) if k == 9 else sample_point_on_variety(q_matrices, rng)
            if point is None:
                rows.append(
                    {
                        'k': k,
                        'trial': trial_idx,
                        'sample_found': False,
                        'jacobian_rank': '',
                        'observed_dimension': '',
                        'sigma_sample_rank': '',
                    }
                )
                continue
            successful_trials += 1
            a_vec, b_vec = point
            jac_rank = jacobian_rank(q_matrices, a_vec, b_vec)
            sigma_vectors: list[np.ndarray] = []
            for _ in range(sigma_samples_per_trial):
                sample = find_singular_point_for_k9(q_matrices, rng) if k == 9 else sample_point_on_variety(q_matrices, rng)
                if sample is None:
                    continue
                sigma_vectors.append(sigma_vector(*sample))
            sigma_rank = matrix_rank(np.stack(sigma_vectors, axis=0), tol=1e-7) if sigma_vectors else 0
            observed_jacobian_ranks.append(jac_rank)
            observed_dimensions.append(18 - jac_rank)
            observed_sigma_ranks.append(sigma_rank)
            rows.append(
                {
                    'k': k,
                    'trial': trial_idx,
                    'sample_found': True,
                    'jacobian_rank': jac_rank,
                    'observed_dimension': 18 - jac_rank,
                    'sigma_sample_rank': sigma_rank,
                }
            )

        summary.append(
            {
                'k': k,
                'expected_dimension': 18 - k,
                'successful_trials': successful_trials,
                'trial_count': trials_per_k,
                'median_observed_dimension': float(np.median(observed_dimensions)) if observed_dimensions else None,
                'min_observed_dimension': min(observed_dimensions) if observed_dimensions else None,
                'max_observed_dimension': max(observed_dimensions) if observed_dimensions else None,
                'median_sigma_sample_rank': float(np.median(observed_sigma_ranks)) if observed_sigma_ranks else None,
            }
        )

    write_csv(
        OUTPUT_DIR / 'variety_dimension_trials.csv',
        rows,
        ['k', 'trial', 'sample_found', 'jacobian_rank', 'observed_dimension', 'sigma_sample_rank'],
    )
    write_json(
        OUTPUT_DIR / 'variety_dimension_summary.json',
        {
            'trial_count_per_k': trials_per_k,
            'sigma_samples_per_trial': sigma_samples_per_trial,
            'summary': summary,
            'interpretation': (
                'For generic random bilinear constraints, the observed Jacobian codimension tracks k across V_k. '
                'For k <= 8 the sampled sigma-image remains generically full 9-dimensional; k=9 requires singular-a sampling '
                'because nonzero points lie on a determinantal hypersurface.'
            ),
        },
    )


if __name__ == '__main__':
    main()