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
EXACTNESS_TOL = 1e-10
MAX_STEPS = 8
STEP_EPSILONS = [1e-3, 3e-3, 1e-2, 3e-2, 5e-2]
WILDCARD_SAMPLES = 48
WILDCARD_RELATIVE_SCALE = 0.75


def build_h_and_gamma(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gamma = solve_gamma_least_squares(alpha, beta)
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix='wildcard_')
    _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
    return h_matrix, gamma


def finite_difference_h_gamma_derivatives(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0, gamma0 = build_h_and_gamma(alpha, beta)
    num_vars = alpha.shape[0] * 3
    d_h_mats: list[np.ndarray] = []
    d_gamma_cols: list[np.ndarray] = []
    for var_idx in range(num_vars):
        delta = np.zeros(num_vars, dtype=np.float64)
        delta[var_idx] = 1.0
        alpha_eps = apply_channel_delta(alpha, delta, FD_EPS)
        h_eps, gamma_eps = build_h_and_gamma(alpha_eps, beta)
        d_h_mats.append((h_eps - h0) / FD_EPS)
        d_gamma_cols.append(((gamma_eps - gamma0) / FD_EPS).reshape(-1))
    return h0, gamma0, np.stack(d_h_mats, axis=0), np.column_stack(d_gamma_cols)


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

    gamma_delta = np.column_stack([
        d_gamma_matrix[:, var_idx].reshape(gamma0.shape[0], gamma0.shape[1]).T @ lam
        for var_idx in range(num_vars)
    ])
    gamma_lambda = gamma0.T
    kernel_constraint = np.hstack([gamma_delta, gamma_lambda])

    normalization = np.zeros((1, num_vars + rank_value), dtype=np.float64)
    normalization[0, num_vars:] = lam
    return objective_matrix, h_target, kernel_constraint, normalization


def solve_coupled_step_details(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> dict[str, object]:
    gamma = solve_gamma_least_squares(alpha, beta)
    jacobian, _ = build_tangent_jacobian(alpha, beta, gamma)
    h0, gamma0, d_h_mats, d_gamma_matrix = finite_difference_h_gamma_derivatives(alpha, beta)
    num_vars = alpha.shape[0] * 3
    rank_value = alpha.shape[0]
    exactness_constraint = np.hstack([jacobian, np.zeros((jacobian.shape[0], rank_value), dtype=np.float64)])
    objective_matrix, h_target, kernel_constraint, normalization = coupled_objective_matrices(
        h0,
        gamma0,
        d_h_mats,
        d_gamma_matrix,
        lam,
    )
    full_constraints = np.vstack([exactness_constraint, kernel_constraint, normalization])
    null_basis = nullspace_basis_numeric(full_constraints)

    if null_basis.size == 0:
        step = np.zeros(num_vars + rank_value, dtype=np.float64)
    else:
        reduced = objective_matrix @ null_basis
        coeff, *_ = np.linalg.lstsq(reduced, -h_target, rcond=None)
        step = null_basis @ coeff

    predicted = h_target + objective_matrix @ step
    return {
        'step': step,
        'null_basis': null_basis,
        'num_vars': num_vars,
        'rank_value': rank_value,
        'base_h_target_l2': float(np.linalg.norm(h_target)),
        'predicted_h_target_l2': float(np.linalg.norm(predicted)),
    }


def propose_candidate_steps(step: np.ndarray, null_basis: np.ndarray, rng: np.random.Generator) -> list[dict[str, object]]:
    candidates = [
        {
            'branch': 'structured',
            'sample_index': 0,
            'full_step': step.copy(),
        }
    ]
    if null_basis.size == 0:
        return candidates

    base_norm = float(np.linalg.norm(step))
    if base_norm <= 1e-15:
        base_norm = 1.0
    for sample_idx in range(1, WILDCARD_SAMPLES + 1):
        coeff = rng.standard_normal(null_basis.shape[1])
        coeff /= max(np.linalg.norm(coeff), 1e-15)
        perturb = null_basis @ coeff
        perturb /= max(np.linalg.norm(perturb), 1e-15)
        relative_scale = WILDCARD_RELATIVE_SCALE * float(rng.uniform(0.5, 1.5))
        candidates.append(
            {
                'branch': 'wildcard',
                'sample_index': sample_idx,
                'full_step': step + relative_scale * base_norm * perturb,
            }
        )
    return candidates


def finite_probe(
    alpha: np.ndarray,
    beta: np.ndarray,
    lam: np.ndarray,
    delta: np.ndarray,
    dot_lambda: np.ndarray,
    eps: float,
) -> dict[str, float | int]:
    alpha_eps = apply_channel_delta(alpha, delta, eps)
    gamma_eps = solve_gamma_least_squares(alpha_eps, beta)
    terms_eps = terms_from_stacked_factors(alpha_eps, beta, gamma_eps, prefix='wild_probe_')
    _, _, _, _, h_matrix_eps, _ = build_mode_matrices_numeric(terms_eps)
    lambda_eps = lam + eps * dot_lambda
    lambda_eps /= max(np.linalg.norm(lambda_eps), 1e-15)
    h_vec = h_matrix_eps.T @ lambda_eps
    g_vec = gamma_eps.T @ lambda_eps
    max_abs_residual, loss_value = tensor_residual_stats(terms_eps)
    mode = soft_kernel_mode(gamma_eps, h_matrix_eps)
    return {
        'tensor_max_abs_residual': float(max_abs_residual),
        'tensor_loss': float(loss_value),
        'honest_defect_l2': float(np.sqrt(np.linalg.norm(h_vec) ** 2 + np.linalg.norm(g_vec) ** 2)),
        'lambda_h_l2': float(np.linalg.norm(h_vec)),
        'gamma_lambda_l2': float(np.linalg.norm(g_vec)),
        'kappa_ker_state': float(mode['kappa_ker']),
        'restricted_rank': int(mode['restricted_rank']),
        'kernel_dim': int(mode['kernel_dim']),
    }


def choose_best_regular_probe(
    alpha: np.ndarray,
    beta: np.ndarray,
    lam: np.ndarray,
    step_details: dict[str, object],
    target_rank: int,
    rng: np.random.Generator,
    base_label: str,
    step_idx: int,
    base_kappa_ker: float,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    candidates = propose_candidate_steps(
        np.asarray(step_details['step'], dtype=np.float64),
        np.asarray(step_details['null_basis'], dtype=np.float64),
        rng,
    )
    candidate_rows: list[dict[str, object]] = []
    best_regular: dict[str, object] | None = None
    structured_best_kappa: float | None = None
    wildcard_best_kappa: float | None = None

    for candidate in candidates:
        full_step = np.asarray(candidate['full_step'], dtype=np.float64)
        num_vars = int(step_details['num_vars'])
        delta = full_step[:num_vars]
        dot_lambda = full_step[num_vars:]
        for eps in STEP_EPSILONS:
            probe = finite_probe(alpha, beta, lam, delta, dot_lambda, eps)
            is_regular = (
                float(probe['tensor_max_abs_residual']) <= EXACTNESS_TOL
                and int(probe['restricted_rank']) == target_rank
            )
            row = {
                'base_label': base_label,
                'step_idx': step_idx,
                'branch': str(candidate['branch']),
                'sample_index': int(candidate['sample_index']),
                'eps': float(eps),
                'step_norm': float(np.linalg.norm(full_step)),
                'delta_norm': float(np.linalg.norm(delta)),
                'dot_lambda_norm': float(np.linalg.norm(dot_lambda)),
                'base_kappa_ker': float(base_kappa_ker),
                'kappa_ker_state': float(probe['kappa_ker_state']),
                'honest_defect_l2': float(probe['honest_defect_l2']),
                'tensor_max_abs_residual': float(probe['tensor_max_abs_residual']),
                'tensor_loss': float(probe['tensor_loss']),
                'restricted_rank': int(probe['restricted_rank']),
                'kernel_dim': int(probe['kernel_dim']),
                'is_regular': int(is_regular),
            }
            candidate_rows.append(row)
            if row['branch'] == 'structured' and is_regular:
                structured_best_kappa = row['kappa_ker_state'] if structured_best_kappa is None else min(structured_best_kappa, row['kappa_ker_state'])
            if row['branch'] == 'wildcard' and is_regular:
                wildcard_best_kappa = row['kappa_ker_state'] if wildcard_best_kappa is None else min(wildcard_best_kappa, row['kappa_ker_state'])
            if not is_regular:
                continue
            if best_regular is None or (
                row['kappa_ker_state'],
                row['honest_defect_l2'],
                row['eps'],
            ) < (
                best_regular['row']['kappa_ker_state'],
                best_regular['row']['honest_defect_l2'],
                best_regular['row']['eps'],
            ):
                best_regular = {
                    'row': dict(row),
                    'delta': delta.copy(),
                    'dot_lambda': dot_lambda.copy(),
                }

    if best_regular is not None:
        best_regular['row']['structured_best_regular_kappa_ker'] = structured_best_kappa
        best_regular['row']['wildcard_best_regular_kappa_ker'] = wildcard_best_kappa
        best_regular['row']['wildcard_beats_structured'] = int(
            wildcard_best_kappa is not None and (
                structured_best_kappa is None or wildcard_best_kappa < structured_best_kappa
            )
        )
    return best_regular, candidate_rows


def run_case(base_label: str, alpha: np.ndarray, beta: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(20260329 + 17 * alpha.shape[0] + sum(ord(ch) for ch in base_label))
    current_alpha = alpha.copy()
    current_beta = beta.copy()
    h0, gamma0 = build_h_and_gamma(current_alpha, current_beta)
    mode0 = soft_kernel_mode(gamma0, h0)
    initial_rank = int(mode0['restricted_rank'])
    base_kappa_ker = float(mode0['kappa_ker'])

    trajectory_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    min_kappa_ker = base_kappa_ker
    final_kappa_ker = base_kappa_ker
    accepted_steps = 0
    wildcard_wins = 0

    for step_idx in range(MAX_STEPS):
        h_matrix, gamma = build_h_and_gamma(current_alpha, current_beta)
        mode = soft_kernel_mode(gamma, h_matrix)
        lam = np.asarray(mode['soft_lambda'], dtype=np.float64)
        current_kappa_ker = float(mode['kappa_ker'])
        step_details = solve_coupled_step_details(current_alpha, current_beta, lam)
        best_regular, step_candidate_rows = choose_best_regular_probe(
            current_alpha,
            current_beta,
            lam,
            step_details,
            initial_rank,
            rng,
            base_label,
            step_idx,
            current_kappa_ker,
        )
        candidate_rows.extend(step_candidate_rows)
        if best_regular is None:
            trajectory_rows.append(
                {
                    'base_label': base_label,
                    'step_idx': step_idx,
                    'accepted': 0,
                    'branch': 'none',
                    'sample_index': -1,
                    'eps': 0.0,
                    'base_kappa_ker': current_kappa_ker,
                    'next_kappa_ker': current_kappa_ker,
                    'next_honest_defect_l2': float(np.linalg.norm(h_matrix.T @ lam)),
                    'tensor_max_abs_residual': 0.0,
                    'restricted_rank': int(mode['restricted_rank']),
                    'wildcard_beats_structured': 0,
                    'structured_best_regular_kappa_ker': None,
                    'wildcard_best_regular_kappa_ker': None,
                }
            )
            final_kappa_ker = current_kappa_ker
            break

        chosen = best_regular['row']
        eps = float(chosen['eps'])
        current_alpha = apply_channel_delta(current_alpha, np.asarray(best_regular['delta']), eps)
        accepted_steps += 1
        wildcard_wins += int(chosen['branch'] == 'wildcard')
        min_kappa_ker = min(min_kappa_ker, float(chosen['kappa_ker_state']))
        final_kappa_ker = float(chosen['kappa_ker_state'])
        trajectory_rows.append(
            {
                'base_label': base_label,
                'step_idx': step_idx,
                'accepted': 1,
                'branch': chosen['branch'],
                'sample_index': int(chosen['sample_index']),
                'eps': eps,
                'base_kappa_ker': current_kappa_ker,
                'next_kappa_ker': float(chosen['kappa_ker_state']),
                'next_honest_defect_l2': float(chosen['honest_defect_l2']),
                'tensor_max_abs_residual': float(chosen['tensor_max_abs_residual']),
                'restricted_rank': int(chosen['restricted_rank']),
                'wildcard_beats_structured': int(chosen['wildcard_beats_structured']),
                'structured_best_regular_kappa_ker': chosen['structured_best_regular_kappa_ker'],
                'wildcard_best_regular_kappa_ker': chosen['wildcard_best_regular_kappa_ker'],
            }
        )

    summary = {
        'base_label': base_label,
        'initial_kappa_ker': base_kappa_ker,
        'final_kappa_ker': final_kappa_ker,
        'min_kappa_ker': min_kappa_ker,
        'accepted_steps': accepted_steps,
        'wildcard_wins': wildcard_wins,
        'initial_restricted_rank': initial_rank,
    }
    return trajectory_rows, candidate_rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    trajectory_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for base_label, alpha, beta in base_cases:
        trajectory, candidates, summary = run_case(base_label, alpha, beta)
        trajectory_rows.extend(trajectory)
        candidate_rows.extend(candidates)
        summaries.append(summary)

    write_csv(
        OUT_DIR / 'wildcard_regularized_trajectory.csv',
        trajectory_rows,
        [
            'base_label', 'step_idx', 'accepted', 'branch', 'sample_index', 'eps',
            'base_kappa_ker', 'next_kappa_ker', 'next_honest_defect_l2',
            'tensor_max_abs_residual', 'restricted_rank', 'wildcard_beats_structured',
            'structured_best_regular_kappa_ker', 'wildcard_best_regular_kappa_ker',
        ],
    )
    write_csv(
        OUT_DIR / 'wildcard_regularized_candidates.csv',
        candidate_rows,
        [
            'base_label', 'step_idx', 'branch', 'sample_index', 'eps', 'step_norm',
            'delta_norm', 'dot_lambda_norm', 'base_kappa_ker', 'kappa_ker_state',
            'honest_defect_l2', 'tensor_max_abs_residual', 'tensor_loss',
            'restricted_rank', 'kernel_dim', 'is_regular',
        ],
    )
    write_json(
        OUT_DIR / 'wildcard_regularized_continuation.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'exactness_tolerance': EXACTNESS_TOL,
            'max_steps': MAX_STEPS,
            'step_epsilons': STEP_EPSILONS,
            'wildcard_samples': WILDCARD_SAMPLES,
            'summaries': summaries,
        },
    )

    print('Phase 28 wildcard-regularized continuation complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']}: initial kappa_ker={summary['initial_kappa_ker']:.6g}, "
            f"min kappa_ker={summary['min_kappa_ker']:.6g}, final kappa_ker={summary['final_kappa_ker']:.6g}, "
            f"accepted_steps={summary['accepted_steps']}, wildcard_wins={summary['wildcard_wins']}"
        )


if __name__ == '__main__':
    main()