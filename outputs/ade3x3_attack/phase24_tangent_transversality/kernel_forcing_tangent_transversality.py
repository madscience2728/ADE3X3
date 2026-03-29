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
    live_q_vectors,
    stacked_alpha_beta,
    standard_rank27_terms,
)


RANK_TOL = 1e-10
FINITE_DIFFERENCE_EPS = 1e-4
CURVATURE_EPSILONS = [1e-3, 1e-2, 5e-2, 1e-1]
PROGRESS_EPS = 1e-1
PATH_SCAN_EPSILONS = [1e-1, 2e-1, 5e-1, 1.0]


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def principal_angle_degrees(vector: np.ndarray, subspace_basis: np.ndarray) -> float:
    if subspace_basis.size == 0:
        return 90.0
    coeff = subspace_basis.T @ vector
    proj_norm = float(np.linalg.norm(coeff))
    vec_norm = float(np.linalg.norm(vector))
    cosine = 0.0 if vec_norm == 0.0 else max(-1.0, min(1.0, proj_norm / vec_norm))
    return float(np.degrees(np.arccos(cosine)))


def build_constraint_matrix(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank_value = alpha.shape[0]
    q0 = live_q_vectors(alpha, beta, 0)
    q1 = live_q_vectors(alpha, beta, 1)
    q2 = live_q_vectors(alpha, beta, 2)

    a_matrix = np.zeros((18, 3 * rank_value), dtype=np.float64)
    for term_idx in range(rank_value):
        a_matrix[0:9, term_idx] = lam[term_idx] * q0[term_idx, :]
        a_matrix[0:9, rank_value + term_idx] = -lam[term_idx] * q1[term_idx, :]
        a_matrix[9:18, rank_value + term_idx] = lam[term_idx] * q1[term_idx, :]
        a_matrix[9:18, 2 * rank_value + term_idx] = -lam[term_idx] * q2[term_idx, :]

    baseline = np.ones(3 * rank_value, dtype=np.float64)
    rhs = -(a_matrix @ baseline)
    return a_matrix, rhs


def alpha_channel_delta_vector(alpha: np.ndarray, term_idx: int, shared_idx: int) -> np.ndarray:
    delta = np.zeros(alpha.shape[1], dtype=np.float64)
    for row_idx in range(3):
        flat_idx = 3 * row_idx + shared_idx
        delta[flat_idx] = alpha[term_idx, flat_idx]
    return delta


def feature_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.einsum('ra,rb->abr', alpha, beta, optimize=True).reshape(81, alpha.shape[0])


def build_tangent_jacobian(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    features = feature_matrix(alpha, beta)
    projector = np.eye(features.shape[0], dtype=np.float64) - features @ np.linalg.pinv(features, rcond=RANK_TOL)
    rank_value = alpha.shape[0]
    jacobian = np.zeros((81 * 9, 3 * rank_value), dtype=np.float64)

    for shared_idx in range(3):
        for term_idx in range(rank_value):
            alpha_delta = alpha_channel_delta_vector(alpha, term_idx, shared_idx)
            d_feature_col = np.outer(alpha_delta, beta[term_idx]).reshape(81)
            projected_col = projector @ d_feature_col
            d_residual = np.outer(projected_col, gamma[term_idx, :]).reshape(-1)
            jacobian[:, shared_idx * rank_value + term_idx] = d_residual

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank_j = int(np.sum(singular_values > RANK_TOL))
    return jacobian, {
        'rank': rank_j,
        'nullity': int(jacobian.shape[1] - rank_j),
        'largest_singular_value': float(singular_values[0]) if singular_values.size else 0.0,
        'smallest_positive_singular_value': float(singular_values[rank_j - 1]) if rank_j else 0.0,
    }


def solve_affine_min_residual(jacobian: np.ndarray, a_matrix: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    delta_min_norm, *_ = np.linalg.lstsq(a_matrix, rhs, rcond=None)
    null_a = nullspace_basis_numeric(a_matrix)

    residual_min_norm = jacobian @ delta_min_norm
    if null_a.size == 0:
        delta_opt = delta_min_norm
        residual_opt = residual_min_norm
        feasible_null_dim = 0
    else:
        reduced = jacobian @ null_a
        correction, *_ = np.linalg.lstsq(reduced, -residual_min_norm, rcond=None)
        delta_opt = delta_min_norm + null_a @ correction
        residual_opt = jacobian @ delta_opt
        feasible_null_dim = int(null_a.shape[1] - matrix_rank_numeric(reduced))

    return {
        'delta_min_norm': delta_min_norm,
        'delta_opt': delta_opt,
        'residual_min_norm': residual_min_norm,
        'residual_opt': residual_opt,
        'constraint_rank': matrix_rank_numeric(a_matrix),
        'constraint_nullity': int(a_matrix.shape[1] - matrix_rank_numeric(a_matrix)),
        'feasible_exact_tangent_dim': feasible_null_dim,
        'constraint_residual_min_norm': float(np.max(np.abs(a_matrix @ delta_min_norm - rhs))),
        'constraint_residual_opt': float(np.max(np.abs(a_matrix @ delta_opt - rhs))),
    }


def apply_channel_delta(alpha: np.ndarray, delta: np.ndarray, eps: float) -> np.ndarray:
    rank_value = alpha.shape[0]
    scaled = alpha.copy().reshape(rank_value, 3, 3)
    delta_reshaped = delta.reshape(3, rank_value)
    for shared_idx in range(3):
        scales = 1.0 + eps * delta_reshaped[shared_idx]
        scaled[:, :, shared_idx] *= scales[:, None]
    return scaled.reshape(rank_value, 9)


def finite_difference_check(alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    alpha_eps = apply_channel_delta(alpha, delta, FINITE_DIFFERENCE_EPS)
    gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
    terms_eps = terms_from_stacked_factors(alpha_eps, beta, gamma_eps, prefix='fd_')
    max_abs_residual, loss_value = tensor_residual_stats(terms_eps)
    return {
        'eps': FINITE_DIFFERENCE_EPS,
        'residual_over_eps': float(max_abs_residual / FINITE_DIFFERENCE_EPS),
        'loss_over_eps_sq': float(loss_value / (FINITE_DIFFERENCE_EPS ** 2)),
    }


def curvature_scan(alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    first_over_1e_2: float | None = None
    first_over_5e_2: float | None = None
    for eps in CURVATURE_EPSILONS:
        alpha_eps = apply_channel_delta(alpha, delta, eps)
        gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
        terms_eps = terms_from_stacked_factors(alpha_eps, beta, gamma_eps, prefix='curv_')
        max_abs_residual, loss_value = tensor_residual_stats(terms_eps)
        eps_key = str(eps).replace('.', 'p')
        payload[f'curvature_residual_eps_{eps_key}'] = float(max_abs_residual)
        payload[f'curvature_loss_eps_{eps_key}'] = float(loss_value)
        payload[f'curvature_residual_over_eps_sq_{eps_key}'] = float(max_abs_residual / (eps ** 2))
        if first_over_1e_2 is None and max_abs_residual > 1e-2:
            first_over_1e_2 = eps
        if first_over_5e_2 is None and max_abs_residual > 5e-2:
            first_over_5e_2 = eps
    payload['first_curvature_residual_over_1e_2'] = first_over_1e_2
    payload['first_curvature_residual_over_5e_2'] = first_over_5e_2
    return payload


def lambda_h_max_abs(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, lam: np.ndarray) -> float:
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix='probe_')
    _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
    return float(np.max(np.abs(lam @ h_matrix)))


def progress_probe(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray, delta: np.ndarray, base_lambda_h: float) -> dict[str, float]:
    alpha_eps = apply_channel_delta(alpha, delta, PROGRESS_EPS)
    gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
    lambda_h_eps = lambda_h_max_abs(alpha_eps, beta, gamma_eps, lam)
    return {
        'progress_lambda_h_eps_0p1': lambda_h_eps,
        'progress_lambda_h_ratio_eps_0p1': float(lambda_h_eps / max(base_lambda_h, 1e-15)),
    }


def optimized_path_scan(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray, delta: np.ndarray, base_lambda_h: float) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    first_residual_over_1e_2: float | None = None
    first_residual_over_5e_2: float | None = None
    first_angle_over_5: float | None = None
    first_angle_over_15: float | None = None
    for eps in PATH_SCAN_EPSILONS:
        alpha_eps = apply_channel_delta(alpha, delta, eps)
        gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
        terms_eps = terms_from_stacked_factors(alpha_eps, beta, gamma_eps, prefix='path_')
        max_abs_residual, loss_value = tensor_residual_stats(terms_eps)
        lambda_h_eps = lambda_h_max_abs(alpha_eps, beta, gamma_eps, lam)
        angle_eps = principal_angle_degrees(lam, nullspace_basis_numeric(gamma_eps.T))
        eps_key = str(eps).replace('.', 'p')
        payload[f'path_residual_eps_{eps_key}'] = float(max_abs_residual)
        payload[f'path_loss_eps_{eps_key}'] = float(loss_value)
        payload[f'path_lambda_h_ratio_eps_{eps_key}'] = float(lambda_h_eps / max(base_lambda_h, 1e-15))
        payload[f'path_angle_deg_eps_{eps_key}'] = angle_eps
        if first_residual_over_1e_2 is None and max_abs_residual > 1e-2:
            first_residual_over_1e_2 = eps
        if first_residual_over_5e_2 is None and max_abs_residual > 5e-2:
            first_residual_over_5e_2 = eps
        if first_angle_over_5 is None and angle_eps > 5.0:
            first_angle_over_5 = eps
        if first_angle_over_15 is None and angle_eps > 15.0:
            first_angle_over_15 = eps
    payload['first_path_residual_over_1e_2'] = first_residual_over_1e_2
    payload['first_path_residual_over_5e_2'] = first_residual_over_5e_2
    payload['first_path_angle_over_5_deg'] = first_angle_over_5
    payload['first_path_angle_over_15_deg'] = first_angle_over_15
    return payload


def evaluate_case(base_label: str, alpha: np.ndarray, beta: np.ndarray, candidate_name: str, lam: np.ndarray) -> dict[str, object]:
    gamma = solve_gamma_least_squares(alpha, beta)
    jacobian, tangent_summary = build_tangent_jacobian(alpha, beta, gamma)
    a_matrix, rhs = build_constraint_matrix(alpha, beta, lam)
    affine_solution = solve_affine_min_residual(jacobian, a_matrix, rhs)

    residual_min_norm = affine_solution['residual_min_norm']
    residual_opt = affine_solution['residual_opt']
    delta_min_norm = affine_solution['delta_min_norm']
    delta_opt = affine_solution['delta_opt']
    correction = delta_opt - delta_min_norm
    fd_min_norm = finite_difference_check(alpha, beta, delta_min_norm)
    fd_opt = finite_difference_check(alpha, beta, delta_opt)
    curvature = curvature_scan(alpha, beta, delta_opt)
    base_lambda_h = lambda_h_max_abs(alpha, beta, gamma, lam)
    progress_min_norm = progress_probe(alpha, beta, lam, delta_min_norm, base_lambda_h)
    progress_opt = progress_probe(alpha, beta, lam, delta_opt, base_lambda_h)
    path_scan = optimized_path_scan(alpha, beta, lam, delta_opt, base_lambda_h)

    row = {
        'base_label': base_label,
        'candidate': candidate_name,
        'R': int(alpha.shape[0]),
        'tangent_rank': int(tangent_summary['rank']),
        'tangent_nullity': int(tangent_summary['nullity']),
        'tangent_smallest_positive_singular': float(tangent_summary['smallest_positive_singular_value']),
        'constraint_rank': int(affine_solution['constraint_rank']),
        'constraint_nullity': int(affine_solution['constraint_nullity']),
        'feasible_exact_tangent_dim': int(affine_solution['feasible_exact_tangent_dim']),
        'min_norm_direction_norm': float(np.linalg.norm(delta_min_norm)),
        'min_norm_first_order_residual_l2': float(np.linalg.norm(residual_min_norm)),
        'min_norm_first_order_residual_max_abs': float(np.max(np.abs(residual_min_norm))),
        'optimized_direction_norm': float(np.linalg.norm(delta_opt)),
        'tangent_correction_norm': float(np.linalg.norm(correction)),
        'tangent_correction_ratio': float(np.linalg.norm(correction) / max(np.linalg.norm(delta_min_norm), 1e-15)),
        'optimized_first_order_residual_l2': float(np.linalg.norm(residual_opt)),
        'optimized_first_order_residual_max_abs': float(np.max(np.abs(residual_opt))),
        'optimized_residual_improvement_factor': float(
            np.linalg.norm(residual_min_norm) / max(np.linalg.norm(residual_opt), 1e-15)
        ),
        'relative_transversality_min_norm': float(np.linalg.norm(residual_min_norm) / max(np.linalg.norm(delta_min_norm), 1e-15)),
        'relative_transversality_optimized': float(np.linalg.norm(residual_opt) / max(np.linalg.norm(delta_opt), 1e-15)),
        'constraint_residual_min_norm': float(affine_solution['constraint_residual_min_norm']),
        'constraint_residual_optimized': float(affine_solution['constraint_residual_opt']),
        'fd_min_norm_residual_over_eps': float(fd_min_norm['residual_over_eps']),
        'fd_optimized_residual_over_eps': float(fd_opt['residual_over_eps']),
        'fd_min_norm_loss_over_eps_sq': float(fd_min_norm['loss_over_eps_sq']),
        'fd_optimized_loss_over_eps_sq': float(fd_opt['loss_over_eps_sq']),
    }
    row.update(curvature)
    row.update(path_scan)
    row['base_lambda_h_max_abs'] = base_lambda_h
    row['min_norm_progress_lambda_h_eps_0p1'] = progress_min_norm['progress_lambda_h_eps_0p1']
    row['min_norm_progress_lambda_h_ratio_eps_0p1'] = progress_min_norm['progress_lambda_h_ratio_eps_0p1']
    row['optimized_progress_lambda_h_eps_0p1'] = progress_opt['progress_lambda_h_eps_0p1']
    row['optimized_progress_lambda_h_ratio_eps_0p1'] = progress_opt['progress_lambda_h_ratio_eps_0p1']
    return row


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    rows: list[dict[str, object]] = []
    for base_label, alpha, beta in base_cases:
        gamma = solve_gamma_least_squares(alpha, beta)
        for candidate_name, lam in choose_lambda_candidates(alpha, beta, gamma, count=3):
            rows.append(evaluate_case(base_label, alpha, beta, candidate_name, lam))

    write_csv(
        OUT_DIR / 'kernel_forcing_tangent_transversality.csv',
        rows,
        [
            'base_label', 'candidate', 'R', 'tangent_rank', 'tangent_nullity', 'tangent_smallest_positive_singular',
            'constraint_rank', 'constraint_nullity', 'feasible_exact_tangent_dim',
            'min_norm_direction_norm', 'min_norm_first_order_residual_l2', 'min_norm_first_order_residual_max_abs',
            'optimized_direction_norm', 'tangent_correction_norm', 'tangent_correction_ratio',
            'optimized_first_order_residual_l2', 'optimized_first_order_residual_max_abs',
            'optimized_residual_improvement_factor', 'relative_transversality_min_norm', 'relative_transversality_optimized',
            'constraint_residual_min_norm', 'constraint_residual_optimized',
            'fd_min_norm_residual_over_eps', 'fd_optimized_residual_over_eps',
            'fd_min_norm_loss_over_eps_sq', 'fd_optimized_loss_over_eps_sq',
            'curvature_residual_eps_0p001', 'curvature_loss_eps_0p001', 'curvature_residual_over_eps_sq_0p001',
            'curvature_residual_eps_0p01', 'curvature_loss_eps_0p01', 'curvature_residual_over_eps_sq_0p01',
            'curvature_residual_eps_0p05', 'curvature_loss_eps_0p05', 'curvature_residual_over_eps_sq_0p05',
            'curvature_residual_eps_0p1', 'curvature_loss_eps_0p1', 'curvature_residual_over_eps_sq_0p1',
            'first_curvature_residual_over_1e_2', 'first_curvature_residual_over_5e_2',
            'path_residual_eps_0p1', 'path_loss_eps_0p1', 'path_lambda_h_ratio_eps_0p1',
            'path_angle_deg_eps_0p1', 'path_residual_eps_0p2', 'path_loss_eps_0p2', 'path_lambda_h_ratio_eps_0p2',
            'path_angle_deg_eps_0p2', 'path_residual_eps_0p5', 'path_loss_eps_0p5', 'path_lambda_h_ratio_eps_0p5',
            'path_angle_deg_eps_0p5', 'path_residual_eps_1p0', 'path_loss_eps_1p0', 'path_lambda_h_ratio_eps_1p0',
            'path_angle_deg_eps_1p0',
            'first_path_residual_over_1e_2', 'first_path_residual_over_5e_2',
            'first_path_angle_over_5_deg', 'first_path_angle_over_15_deg',
            'base_lambda_h_max_abs', 'min_norm_progress_lambda_h_eps_0p1', 'min_norm_progress_lambda_h_ratio_eps_0p1',
            'optimized_progress_lambda_h_eps_0p1', 'optimized_progress_lambda_h_ratio_eps_0p1',
        ],
    )
    write_json(
        OUT_DIR / 'kernel_forcing_tangent_transversality.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rows': rows,
        },
    )

    print('Phase 24 tangent-transversality complete.')
    for row in rows:
        print(
            f"  {row['base_label']} / {row['candidate']}: optimized first-order residual "
            f"L2={row['optimized_first_order_residual_l2']:.6g}, max-abs={row['optimized_first_order_residual_max_abs']:.6g}, "
            f"feasible exact tangent dim={row['feasible_exact_tangent_dim']}"
        )


if __name__ == '__main__':
    main()