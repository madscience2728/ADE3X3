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


def matrix_rank(matrix: np.ndarray, tol: float = 1e-8) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def sample_point(q_matrices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(512):
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
        return a_vec / np.linalg.norm(a_vec), b_vec / np.linalg.norm(b_vec)
    return None


def sigma_rank_for_trial(q_matrices: np.ndarray, point_count: int, rng: np.random.Generator) -> tuple[int, int]:
    sigma_vectors: list[np.ndarray] = []
    for _ in range(point_count):
        point = sample_point(q_matrices, rng)
        if point is None:
            continue
        a_vec, b_vec = point
        sigma_vectors.append((a_vec.reshape(3, 3) @ b_vec.reshape(3, 3)).reshape(9))
    if not sigma_vectors:
        return 0, 0
    sigma_matrix = np.stack(sigma_vectors, axis=0)
    return matrix_rank(sigma_matrix, tol=1e-7), len(sigma_vectors)


def main() -> None:
    trial_count = env_int('ADE3X3_PHASE9_SIGMA_TRIALS', 100)
    point_count = env_int('ADE3X3_PHASE9_SIGMA_POINTS', 22)
    rng = np.random.default_rng(20260330)
    rows: list[dict[str, object]] = []
    success_count = 0

    for trial_idx in range(trial_count):
        q_matrices = rng.standard_normal((5, 9, 9))
        sigma_rank, sampled_points = sigma_rank_for_trial(q_matrices, point_count, rng)
        if sigma_rank == 9:
            success_count += 1
        rows.append(
            {
                'trial': trial_idx,
                'sigma_rank': sigma_rank,
                'sampled_points': sampled_points,
                'success_rank9': sigma_rank == 9,
            }
        )

    write_csv(
        OUTPUT_DIR / 'sigma_span_trials.csv',
        rows,
        ['trial', 'sigma_rank', 'sampled_points', 'success_rank9'],
    )
    write_json(
        OUTPUT_DIR / 'sigma_span_summary.json',
        {
            'trial_count': trial_count,
            'point_count_per_trial': point_count,
            'rank9_success_count': success_count,
            'rank9_success_rate': success_count / trial_count if trial_count else 0.0,
            'rank_histogram': {
                str(rank): sum(1 for row in rows if int(row['sigma_rank']) == rank) for rank in sorted({int(row['sigma_rank']) for row in rows})
            },
        },
    )


if __name__ == '__main__':
    main()