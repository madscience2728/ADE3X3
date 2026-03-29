from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import target_output_unfolding_9x81, write_csv, write_json  # noqa: E402


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
        norm_a = np.linalg.norm(a_vec)
        if norm_a < 1e-12:
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
        norm_b = np.linalg.norm(b_vec)
        if norm_b < 1e-12:
            continue
        return a_vec / norm_a, b_vec / norm_b
    return None


def build_w_matrix(q_matrices: np.ndarray, point_count: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    rows: list[np.ndarray] = []
    for _ in range(point_count):
        point = sample_point(q_matrices, rng)
        if point is None:
            continue
        a_vec, b_vec = point
        rows.append(np.kron(a_vec, b_vec))
    if not rows:
        return np.zeros((0, 81), dtype=np.float64), 0
    return np.stack(rows, axis=0), len(rows)


def projection_residual(target_unfold: np.ndarray, w_matrix: np.ndarray) -> tuple[float, float, int]:
    if w_matrix.size == 0:
        return float('inf'), float('inf'), 0
    gamma = np.linalg.lstsq(w_matrix.T, target_unfold.T, rcond=None)[0]
    residual = w_matrix.T @ gamma - target_unfold.T
    return float(np.max(np.abs(residual))), float(np.sum(residual * residual)), matrix_rank(w_matrix, tol=1e-7)


def main() -> None:
    trial_count = env_int('ADE3X3_PHASE9_TENSOR_TRIALS', 100)
    point_count = env_int('ADE3X3_PHASE9_TENSOR_POINTS', 22)
    residual_tol = float(os.environ.get('ADE3X3_PHASE9_TENSOR_TOL', '1e-8'))
    rng = np.random.default_rng(20260331)
    target_unfold = target_output_unfolding_9x81()
    rows: list[dict[str, object]] = []
    exactish_count = 0

    for trial_idx in range(trial_count):
        q_matrices = rng.standard_normal((5, 9, 9))
        w_matrix, sampled_points = build_w_matrix(q_matrices, point_count, rng)
        max_abs, loss_value, w_rank = projection_residual(target_unfold, w_matrix)
        if max_abs <= residual_tol:
            exactish_count += 1
        rows.append(
            {
                'trial': trial_idx,
                'sampled_points': sampled_points,
                'rank_W': w_rank,
                'projection_max_abs': max_abs,
                'projection_loss': loss_value,
                'tensor_feasible': max_abs <= residual_tol,
            }
        )

    rows.sort(key=lambda row: float(row['projection_loss']))
    write_csv(
        OUTPUT_DIR / 'tensor_feasibility_trials.csv',
        rows,
        ['trial', 'sampled_points', 'rank_W', 'projection_max_abs', 'projection_loss', 'tensor_feasible'],
    )
    write_json(
        OUTPUT_DIR / 'tensor_feasibility_summary.json',
        {
            'trial_count': trial_count,
            'point_count_per_trial': point_count,
            'residual_tolerance': residual_tol,
            'feasible_count': exactish_count,
            'feasible_rate': exactish_count / trial_count if trial_count else 0.0,
            'best_trial': rows[0] if rows else None,
            'median_projection_loss': float(np.median([float(row['projection_loss']) for row in rows])) if rows else None,
        },
    )


if __name__ == '__main__':
    main()