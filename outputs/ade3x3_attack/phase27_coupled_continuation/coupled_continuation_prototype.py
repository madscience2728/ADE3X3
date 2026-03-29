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
    build_tangent_jacobian,
)
from outputs.ade3x3_attack.phase25_honest_kernel_retention.honest_kernel_retention_scan import (  # noqa: E402
    nullspace_basis_numeric,
    soft_kernel_mode,
)


RANK_TOL = 1e-10
FD_EPS = 1e-6
PROBE_EPSILONS = [1e-3, 1e-2, 5e-2]


def build_h_and_gamma(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = solve_gamma_least_squares(alpha, beta)
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix='coupled_')
    _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
    return h_matrix, gamma


def finite_difference_h_gamma_derivatives(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0, gamma0 = build_h_and_gamma(alpha, beta)
    num_vars = alpha.shape[0] * 3
    d_h_lambda_cols: list[np.ndarray] = []
    d_gamma_cols: list[np.ndarray] = []
    d_h_mats: list[np.ndarray] = []
    for var_idx in range(num_vars):
        delta = np.zeros(num_vars, dtype=np.float64)
        delta[var_idx] = 1.0
        alpha_eps = apply_channel_delta(alpha, delta, FD_EPS)
        h_eps, gamma_eps = build_h_and_gamma(alpha_eps, beta)
        d_h_mats.append((h_eps - h0) / FD_EPS)
        d_gamma_cols.append(((gamma_eps - gamma0) / FD_EPS).reshape(-1))
    return h0, gamma0, np.stack(d_h_mats, axis=0), np.column_stack(d_gamma_cols), np.arange(num_vars)


def constraint_nullspace(constraint_matrix: np.ndarray) -> np.ndarray:
    return nullspace_basis_numeric(constraint_matrix)


def coupled_objective_matrices(
    h0: np.ndarray,
    gamma0: np.ndarray,
    d_h_mats: np.ndarray,
    d_gamma_matrix: np.ndarray,
    lam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_vars = d_h_mats.shape[0]
    rank_value = lam.shape[0]

    h_target = h0.T @ lam
    h_delta = np.column_stack([d_h_mats[var_idx].T @ lam for var_idx in range(num_vars)])
    h_lambda = h0.T
    objective_matrix = np.hstack([h_delta, h_lambda])

    gamma0_t = gamma0.T

    gamma_delta = np.column_stack([
        d_gamma_matrix[:, var_idx].reshape(gamma0.shape[0], gamma0.shape[1]).T @ lam
        for var_idx in range(num_vars)
    ])
    gamma_lambda = gamma0_t
    kernel_constraint = np.hstack([gamma_delta, gamma_lambda])

    normalization = np.zeros((1, num_vars + rank_value), dtype=np.float64)
    normalization[0, num_vars:] = lam
    return objective_matrix, h_target, kernel_constraint, normalization


def solve_coupled_step(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> dict[str, object]:
    gamma = solve_gamma_least_squares(alpha, beta)
    jacobian, _ = build_tangent_jacobian(alpha, beta, gamma)
    h0, gamma0, d_h_mats, d_gamma_matrix, _ = finite_difference_h_gamma_derivatives(alpha, beta)
    num_vars = alpha.shape[0] * 3
    rank_value = alpha.shape[0]

    exactness_constraint = np.hstack([jacobian, np.zeros((jacobian.shape[0], rank_value), dtype=np.float64)])
    objective_matrix, h_target, kernel_constraint, normalization = coupled_objective_matrices(h0, gamma0, d_h_mats, d_gamma_matrix, lam)
    full_constraints = np.vstack([exactness_constraint, kernel_constraint, normalization])
    null_basis = constraint_nullspace(full_constraints)

    if null_basis.size == 0:
        step = np.zeros(num_vars + rank_value, dtype=np.float64)
    else:
        reduced = objective_matrix @ null_basis
        coeff, *_ = np.linalg.lstsq(reduced, -h_target, rcond=None)
        step = null_basis @ coeff

    delta = step[:num_vars]
    dot_lambda = step[num_vars:]
    predicted = h_target + objective_matrix @ step
    return {
        'delta': delta,
        'dot_lambda': dot_lambda,
        'base_h_target_l2': float(np.linalg.norm(h_target)),
        'predicted_h_target_l2': float(np.linalg.norm(predicted)),
        'predicted_reduction_factor': float(np.linalg.norm(predicted) / max(np.linalg.norm(h_target), 1e-15)),
        'delta_norm': float(np.linalg.norm(delta)),
        'dot_lambda_norm': float(np.linalg.norm(dot_lambda)),
        'nullspace_dim': int(null_basis.shape[1]) if null_basis.ndim == 2 else 0,
    }


def finite_probe(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray, delta: np.ndarray, dot_lambda: np.ndarray, eps: float) -> dict[str, float]:
    alpha_eps = apply_channel_delta(alpha, delta, eps)
    gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
    terms_eps = terms_from_stacked_factors(alpha_eps, beta, gamma_eps, prefix='probe_')
    _, _, _, _, h_matrix_eps, _ = build_mode_matrices_numeric(terms_eps)
    lambda_eps = lam + eps * dot_lambda
    lambda_eps /= max(np.linalg.norm(lambda_eps), 1e-15)
    gamma_matrix_eps = gamma_eps
    h_vec = h_matrix_eps.T @ lambda_eps
    g_vec = gamma_matrix_eps.T @ lambda_eps
    max_abs_residual, loss_value = tensor_residual_stats(terms_eps)
    mode = soft_kernel_mode(gamma_matrix_eps, h_matrix_eps)
    return {
        'tensor_max_abs_residual': max_abs_residual,
        'tensor_loss': loss_value,
        'honest_defect_l2': float(np.sqrt(np.linalg.norm(h_vec) ** 2 + np.linalg.norm(g_vec) ** 2)),
        'lambda_h_l2': float(np.linalg.norm(h_vec)),
        'gamma_lambda_l2': float(np.linalg.norm(g_vec)),
        'kappa_ker_state': float(mode['kappa_ker']),
    }


def evaluate_case(base_label: str, alpha: np.ndarray, beta: np.ndarray, candidate_name: str, candidate_lambda: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    h0, gamma0 = build_h_and_gamma(alpha, beta)
    mode0 = soft_kernel_mode(gamma0, h0)
    lam = np.asarray(mode0['soft_lambda'], dtype=np.float64)
    step = solve_coupled_step(alpha, beta, lam)

    probe_rows: list[dict[str, object]] = []
    for eps in PROBE_EPSILONS:
        probe = finite_probe(alpha, beta, lam, np.asarray(step['delta']), np.asarray(step['dot_lambda']), eps)
        probe_rows.append(
            {
                'base_label': base_label,
                'candidate': candidate_name,
                'eps': eps,
                'tensor_max_abs_residual': float(probe['tensor_max_abs_residual']),
                'tensor_loss': float(probe['tensor_loss']),
                'honest_defect_l2': float(probe['honest_defect_l2']),
                'lambda_h_l2': float(probe['lambda_h_l2']),
                'gamma_lambda_l2': float(probe['gamma_lambda_l2']),
                'kappa_ker_state': float(probe['kappa_ker_state']),
            }
        )

    summary = {
        'base_label': base_label,
        'candidate': candidate_name,
        'base_kappa_ker': float(mode0['kappa_ker']),
        'base_soft_lambda_h_l2': float(np.linalg.norm(h0.T @ lam)),
        'predicted_h_target_l2': float(step['predicted_h_target_l2']),
        'predicted_reduction_factor': float(step['predicted_reduction_factor']),
        'delta_norm': float(step['delta_norm']),
        'dot_lambda_norm': float(step['dot_lambda_norm']),
        'nullspace_dim': int(step['nullspace_dim']),
        'probe_eps_1e_3_honest_defect_l2': float(probe_rows[0]['honest_defect_l2']),
        'probe_eps_1e_2_honest_defect_l2': float(probe_rows[1]['honest_defect_l2']),
        'probe_eps_5e_2_honest_defect_l2': float(probe_rows[2]['honest_defect_l2']),
        'probe_eps_1e_2_tensor_residual': float(probe_rows[1]['tensor_max_abs_residual']),
    }
    return summary, probe_rows


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    summaries: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    for base_label, alpha, beta in base_cases:
        gamma = solve_gamma_least_squares(alpha, beta)
        for candidate_name, candidate_lambda in choose_lambda_candidates(alpha, beta, gamma, count=3):
            summary, probes = evaluate_case(base_label, alpha, beta, candidate_name, candidate_lambda)
            summaries.append(summary)
            probe_rows.extend(probes)

    write_csv(
        OUT_DIR / 'coupled_continuation_probe.csv',
        probe_rows,
        [
            'base_label', 'candidate', 'eps', 'tensor_max_abs_residual', 'tensor_loss',
            'honest_defect_l2', 'lambda_h_l2', 'gamma_lambda_l2', 'kappa_ker_state',
        ],
    )
    write_json(
        OUT_DIR / 'coupled_continuation_probe.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'summaries': summaries,
        },
    )

    print('Phase 27 coupled-continuation prototype complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']} / {summary['candidate']}: predicted reduction factor="
            f"{summary['predicted_reduction_factor']:.6g}, probe eps=1e-2 honest defect="
            f"{summary['probe_eps_1e_2_honest_defect_l2']:.6g}"
        )


if __name__ == '__main__':
    main()