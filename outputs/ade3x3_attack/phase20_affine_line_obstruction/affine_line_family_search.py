from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (  # noqa: E402
    build_mode_matrices_numeric,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)


def build_affine_line_family(rank_value: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    u = rng.standard_normal((rank_value, 3))
    v = rng.standard_normal((rank_value, 3))
    a = rng.standard_normal(3)
    b = rng.standard_normal(3)

    alpha = np.zeros((rank_value, 9), dtype=np.float64)
    beta = np.zeros((rank_value, 9), dtype=np.float64)
    for shared_idx in range(3):
        alpha[:, shared_idx + 0] = u[:, 0] * a[shared_idx]
        alpha[:, shared_idx + 3] = u[:, 1] * a[shared_idx]
        alpha[:, shared_idx + 6] = u[:, 2] * a[shared_idx]

        beta[:, 3 * shared_idx + 0] = b[shared_idx] * v[:, 0]
        beta[:, 3 * shared_idx + 1] = b[shared_idx] * v[:, 1]
        beta[:, 3 * shared_idx + 2] = b[shared_idx] * v[:, 2]

    metadata = {
        'a': [float(value) for value in a],
        'b': [float(value) for value in b],
    }
    return alpha, beta, metadata


def matrix_rank_numeric(matrix: np.ndarray, tol: float = 1e-10) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def max_abs_numeric(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def evaluate_trial(rank_value: int, trial_index: int, alpha: np.ndarray, beta: np.ndarray, metadata: dict[str, object]) -> dict[str, object]:
    gamma = solve_gamma_least_squares(alpha, beta)
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix=f'affine_{rank_value}_{trial_index:03d}_')
    sigma, eta1, eta2, delta, h_matrix, _ = build_mode_matrices_numeric(terms)
    max_abs_residual, loss_value = tensor_residual_stats(terms)

    p0 = np.zeros((rank_value, 9), dtype=np.float64)
    p1 = np.zeros((rank_value, 9), dtype=np.float64)
    p2 = np.zeros((rank_value, 9), dtype=np.float64)
    sigma_col = 0
    for row_idx in range(3):
        for col_idx in range(3):
            p0[:, sigma_col] = alpha[:, 3 * row_idx + 0] * beta[:, 0 * 3 + col_idx]
            p1[:, sigma_col] = alpha[:, 3 * row_idx + 1] * beta[:, 1 * 3 + col_idx]
            p2[:, sigma_col] = alpha[:, 3 * row_idx + 2] * beta[:, 2 * 3 + col_idx]
            sigma_col += 1

    center = (p0 + p1 + p2) / 3.0
    lifts = [p0 - center, p1 - center, p2 - center]
    flat_lifts = np.column_stack([lift.reshape(-1) for lift in lifts])
    gamma_num = gamma.T
    return {
        'R': rank_value,
        'trial_index': trial_index,
        'a0': metadata['a'][0],
        'a1': metadata['a'][1],
        'a2': metadata['a'][2],
        'b0': metadata['b'][0],
        'b1': metadata['b'][1],
        'b2': metadata['b'][2],
        'rank_H_numeric': matrix_rank_numeric(h_matrix),
        'rank_HDelta_numeric': matrix_rank_numeric(np.hstack([h_matrix, delta])),
        'lift_span_rank_numeric': matrix_rank_numeric(flat_lifts),
        'channel0_identity_max_abs': max_abs_numeric(gamma_num @ p0 - np.eye(9)),
        'channel1_identity_max_abs': max_abs_numeric(gamma_num @ p1 - np.eye(9)),
        'channel2_identity_max_abs': max_abs_numeric(gamma_num @ p2 - np.eye(9)),
        'tensor_max_abs_residual': max_abs_residual,
        'tensor_loss': loss_value,
    }


def main() -> None:
    rng = np.random.default_rng(20260329)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for rank_value in (23, 27):
        local_rows: list[dict[str, object]] = []
        for trial_index in range(1, 121):
            alpha, beta, metadata = build_affine_line_family(rank_value, rng)
            local_rows.append(evaluate_trial(rank_value, trial_index, alpha, beta, metadata))
        rows.extend(local_rows)
        best_row = min(local_rows, key=lambda row: (float(row['tensor_max_abs_residual']), float(row['tensor_loss'])))
        summaries.append(
            {
                'R': rank_value,
                'trial_count': len(local_rows),
                'best_tensor_max_abs_residual': best_row['tensor_max_abs_residual'],
                'best_tensor_loss': best_row['tensor_loss'],
                'best_rank_H_numeric': best_row['rank_H_numeric'],
                'best_rank_HDelta_numeric': best_row['rank_HDelta_numeric'],
                'best_lift_span_rank_numeric': best_row['lift_span_rank_numeric'],
                'best_channel_identity_max_abs': max(
                    float(best_row['channel0_identity_max_abs']),
                    float(best_row['channel1_identity_max_abs']),
                    float(best_row['channel2_identity_max_abs']),
                ),
                'best_trial_index': best_row['trial_index'],
            }
        )

    write_csv(
        OUT_DIR / 'affine_line_family_trials.csv',
        rows,
        [
            'R', 'trial_index', 'a0', 'a1', 'a2', 'b0', 'b1', 'b2',
            'rank_H_numeric', 'rank_HDelta_numeric', 'lift_span_rank_numeric',
            'channel0_identity_max_abs', 'channel1_identity_max_abs', 'channel2_identity_max_abs',
            'tensor_max_abs_residual', 'tensor_loss',
        ],
    )
    write_json(OUT_DIR / 'affine_line_family_summary.json', {'summaries': summaries})
    print('Phase 20B complete.')
    for summary in summaries:
        print(
            f"  R={summary['R']}: best max-abs residual={summary['best_tensor_max_abs_residual']:.6g}, "
            f"best rank(H)={summary['best_rank_H_numeric']}, best lift-span rank={summary['best_lift_span_rank_numeric']}"
        )


if __name__ == '__main__':
    main()