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
    load_public_terms,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)
from outputs.ade3x3_attack.phase22_kernel_linked_annihilator.kernel_linked_annihilator_homotopy import (  # noqa: E402
    choose_lambda_candidates,
    stacked_alpha_beta,
    standard_rank27_terms,
)
from outputs.ade3x3_attack.phase24_tangent_transversality.kernel_forcing_tangent_transversality import (  # noqa: E402
    apply_channel_delta,
    build_constraint_matrix,
    build_tangent_jacobian,
    principal_angle_degrees,
    solve_affine_min_residual,
)


RANK_TOL = 1e-10
SCAN_T_VALUES = [idx / 10 for idx in range(11)]


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def vector_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 90.0
    cosine = abs(float(left @ right) / (left_norm * right_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def soft_kernel_mode(gamma_matrix: np.ndarray, h_matrix: np.ndarray) -> dict[str, object]:
    kernel_basis = nullspace_basis_numeric(gamma_matrix.T)
    restricted = kernel_basis.T @ h_matrix
    if restricted.size == 0:
        return {
            'kernel_dim': int(kernel_basis.shape[1]),
            'restricted_rank': 0,
            'kappa_ker': 0.0,
            'soft_lambda': np.zeros(gamma_matrix.shape[1], dtype=np.float64),
            'soft_lambda_h_max_abs': 0.0,
            'soft_lambda_gamma_max_abs': 0.0,
        }

    left_u, singular_values, _ = np.linalg.svd(restricted, full_matrices=False)
    soft_coords = left_u[:, -1]
    soft_lambda = kernel_basis @ soft_coords
    soft_lambda /= max(np.linalg.norm(soft_lambda), 1e-15)
    soft_lambda_h = soft_lambda @ h_matrix
    soft_lambda_gamma = gamma_matrix.T @ soft_lambda
    restricted_rank = int(np.sum(singular_values > RANK_TOL))
    return {
        'kernel_dim': int(kernel_basis.shape[1]),
        'restricted_rank': restricted_rank,
        'kappa_ker': float(singular_values[-1]),
        'soft_lambda': soft_lambda,
        'soft_lambda_h_max_abs': float(np.max(np.abs(soft_lambda_h))),
        'soft_lambda_gamma_max_abs': float(np.max(np.abs(soft_lambda_gamma))),
    }


def coupled_defect_singular_value(gamma_matrix: np.ndarray, h_matrix: np.ndarray) -> float:
    coupled = np.vstack([gamma_matrix.T, h_matrix.T])
    singular_values = np.linalg.svd(coupled, compute_uv=False)
    return float(singular_values[-1]) if singular_values.size else 0.0


def optimized_direction(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> np.ndarray:
    gamma = solve_gamma_least_squares(alpha, beta)
    jacobian, _ = build_tangent_jacobian(alpha, beta, gamma)
    a_matrix, rhs = build_constraint_matrix(alpha, beta, lam)
    affine_solution = solve_affine_min_residual(jacobian, a_matrix, rhs)
    return np.asarray(affine_solution['delta_opt'], dtype=np.float64)


def evaluate_path(base_label: str, candidate_name: str, alpha: np.ndarray, beta: np.ndarray, base_lambda: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    delta_opt = optimized_direction(alpha, beta, base_lambda)
    rows: list[dict[str, object]] = []
    base_soft_lambda: np.ndarray | None = None

    for t_value in SCAN_T_VALUES:
        alpha_t = apply_channel_delta(alpha, delta_opt, t_value)
        gamma_t = solve_gamma_least_squares(alpha_t, beta)
        terms_t = terms_from_stacked_factors(alpha_t, beta, gamma_t, prefix=f'honest_{candidate_name}_{t_value:.1f}_')
        _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms_t)
        gamma_matrix = gamma_t
        mode = soft_kernel_mode(gamma_matrix, h_matrix)
        soft_lambda = np.asarray(mode['soft_lambda'], dtype=np.float64)
        if base_soft_lambda is None:
            base_soft_lambda = soft_lambda.copy()
        max_abs_residual, loss_value = tensor_residual_stats(terms_t)
        rows.append(
            {
                'base_label': base_label,
                'candidate': candidate_name,
                't_value': t_value,
                'R': int(alpha.shape[0]),
                'tensor_max_abs_residual': max_abs_residual,
                'tensor_loss': loss_value,
                'kernel_dim': int(mode['kernel_dim']),
                'restricted_rank': int(mode['restricted_rank']),
                'kappa_ker': float(mode['kappa_ker']),
                'kappa_full': coupled_defect_singular_value(gamma_matrix, h_matrix),
                'soft_lambda_h_max_abs': float(mode['soft_lambda_h_max_abs']),
                'soft_lambda_gamma_max_abs': float(mode['soft_lambda_gamma_max_abs']),
                'base_candidate_angle_deg': principal_angle_degrees(base_lambda, nullspace_basis_numeric(gamma_matrix.T)),
                'soft_mode_rotation_deg': vector_angle_degrees(base_soft_lambda, soft_lambda),
            }
        )

    base_row = rows[0]
    min_kappa_row = min(rows, key=lambda row: (float(row['kappa_ker']), float(row['tensor_max_abs_residual'])))
    min_full_row = min(rows, key=lambda row: (float(row['kappa_full']), float(row['tensor_max_abs_residual'])))
    first_small_kappa = next((row for row in rows if float(row['kappa_ker']) <= 0.1 * float(base_row['kappa_ker'])), None)
    first_small_full = next((row for row in rows if float(row['kappa_full']) <= 0.1 * float(base_row['kappa_full'])), None)

    summary = {
        'base_label': base_label,
        'candidate': candidate_name,
        'base_kappa_ker': float(base_row['kappa_ker']),
        'base_kappa_full': float(base_row['kappa_full']),
        'min_kappa_ker': float(min_kappa_row['kappa_ker']),
        'min_kappa_ker_t': float(min_kappa_row['t_value']),
        'min_kappa_full': float(min_full_row['kappa_full']),
        'min_kappa_full_t': float(min_full_row['t_value']),
        'tensor_residual_at_min_kappa_ker': float(min_kappa_row['tensor_max_abs_residual']),
        'tensor_residual_at_min_kappa_full': float(min_full_row['tensor_max_abs_residual']),
        'first_kappa_ker_below_10pct': first_small_kappa,
        'first_kappa_full_below_10pct': first_small_full,
    }
    return rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for base_label, alpha, beta in base_cases:
        gamma = solve_gamma_least_squares(alpha, beta)
        for candidate_name, base_lambda in choose_lambda_candidates(alpha, beta, gamma, count=3):
            path_rows, summary = evaluate_path(base_label, candidate_name, alpha, beta, base_lambda)
            rows.extend(path_rows)
            summaries.append(summary)

    write_csv(
        OUT_DIR / 'honest_kernel_retention_scan.csv',
        rows,
        [
            'base_label', 'candidate', 't_value', 'R', 'tensor_max_abs_residual', 'tensor_loss',
            'kernel_dim', 'restricted_rank', 'kappa_ker', 'kappa_full',
            'soft_lambda_h_max_abs', 'soft_lambda_gamma_max_abs',
            'base_candidate_angle_deg', 'soft_mode_rotation_deg',
        ],
    )
    write_json(
        OUT_DIR / 'honest_kernel_retention_scan.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'summaries': summaries,
        },
    )

    print('Phase 25 honest-kernel-retention scan complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']} / {summary['candidate']}: base kappa_ker={summary['base_kappa_ker']:.6g}, "
            f"min kappa_ker={summary['min_kappa_ker']:.6g} at t={summary['min_kappa_ker_t']}, "
            f"base kappa_full={summary['base_kappa_full']:.6g}, min kappa_full={summary['min_kappa_full']:.6g}"
        )


if __name__ == '__main__':
    main()