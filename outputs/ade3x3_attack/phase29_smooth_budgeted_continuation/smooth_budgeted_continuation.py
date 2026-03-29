from __future__ import annotations

import math
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
    terms_from_stacked_factors,
    write_csv,
    write_json,
)
from outputs.ade3x3_attack.phase22_kernel_linked_annihilator.kernel_linked_annihilator_homotopy import (  # noqa: E402
    stacked_alpha_beta,
    standard_rank27_terms,
)
from outputs.ade3x3_attack.phase25_honest_kernel_retention.honest_kernel_retention_scan import (  # noqa: E402
    soft_kernel_mode,
)
from outputs.ade3x3_attack.phase28_wildcard_regularized_continuation.wildcard_regularized_continuation import (  # noqa: E402
    EXACTNESS_TOL,
    STEP_EPSILONS,
    build_h_and_gamma,
    finite_probe,
    propose_candidate_steps,
    solve_coupled_step_details,
)
from outputs.ade3x3_attack.phase24_tangent_transversality.kernel_forcing_tangent_transversality import (  # noqa: E402
    apply_channel_delta,
)


POLICIES = [
    {
        'policy_name': 'smooth_greedy',
        'max_steps': 8,
        'total_budget': math.inf,
        'rotation_cap_deg': 12.0,
        'budget_weight': 0.0,
        'rotation_weight': 0.0,
    },
    {
        'policy_name': 'smooth_budgeted_long',
        'max_steps': 24,
        'total_budget': 1.2,
        'rotation_cap_deg': 12.0,
        'budget_weight': 0.6,
        'rotation_weight': 0.02,
    },
]


def vector_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 90.0
    cosine = abs(float(left @ right) / (left_norm * right_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def probe_next_state(
    alpha: np.ndarray,
    beta: np.ndarray,
    lam: np.ndarray,
    delta: np.ndarray,
    dot_lambda: np.ndarray,
    eps: float,
) -> dict[str, object]:
    probe = finite_probe(alpha, beta, lam, delta, dot_lambda, eps)
    alpha_next = apply_channel_delta(alpha, delta, eps)
    gamma_next = solve_gamma_least_squares(alpha_next, beta)
    terms_next = terms_from_stacked_factors(alpha_next, beta, gamma_next, prefix='smooth_budgeted_')
    _, _, _, _, h_next, _ = build_mode_matrices_numeric(terms_next)
    mode_next = soft_kernel_mode(gamma_next, h_next)
    soft_lambda_next = np.asarray(mode_next['soft_lambda'], dtype=np.float64)
    lambda_next = lam + eps * dot_lambda
    lambda_next /= max(np.linalg.norm(lambda_next), 1e-15)
    return {
        'alpha_next': alpha_next,
        'probe': probe,
        'mode_next': mode_next,
        'soft_lambda_next': soft_lambda_next,
        'transported_lambda_next': lambda_next,
    }


def step_cost(eps: float, delta: np.ndarray, dot_lambda: np.ndarray) -> float:
    return float(eps * (np.linalg.norm(delta) + 0.5 * np.linalg.norm(dot_lambda)))


def select_candidate(
    alpha: np.ndarray,
    beta: np.ndarray,
    lam: np.ndarray,
    step_details: dict[str, object],
    target_rank: int,
    rng: np.random.Generator,
    base_label: str,
    step_idx: int,
    current_kappa_ker: float,
    used_budget: float,
    policy: dict[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    candidates = propose_candidate_steps(
        np.asarray(step_details['step'], dtype=np.float64),
        np.asarray(step_details['null_basis'], dtype=np.float64),
        rng,
    )
    candidate_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for candidate in candidates:
        full_step = np.asarray(candidate['full_step'], dtype=np.float64)
        num_vars = int(step_details['num_vars'])
        delta = full_step[:num_vars]
        dot_lambda = full_step[num_vars:]
        for eps in STEP_EPSILONS:
            next_state = probe_next_state(alpha, beta, lam, delta, dot_lambda, eps)
            probe = next_state['probe']
            soft_lambda_next = np.asarray(next_state['soft_lambda_next'], dtype=np.float64)
            rotation_deg = vector_angle_degrees(lam, soft_lambda_next)
            candidate_cost = step_cost(eps, delta, dot_lambda)
            new_budget = used_budget + candidate_cost
            is_regular = (
                float(probe['tensor_max_abs_residual']) <= EXACTNESS_TOL
                and int(probe['restricted_rank']) == target_rank
            )
            within_rotation = rotation_deg <= float(policy['rotation_cap_deg'])
            within_budget = new_budget <= float(policy['total_budget']) + 1e-12
            score = (
                float(probe['kappa_ker_state'])
                + float(policy['budget_weight']) * candidate_cost
                + float(policy['rotation_weight']) * (rotation_deg / max(float(policy['rotation_cap_deg']), 1e-12))
            )
            row = {
                'policy_name': str(policy['policy_name']),
                'base_label': base_label,
                'step_idx': step_idx,
                'branch': str(candidate['branch']),
                'sample_index': int(candidate['sample_index']),
                'eps': float(eps),
                'base_kappa_ker': float(current_kappa_ker),
                'kappa_ker_state': float(probe['kappa_ker_state']),
                'honest_defect_l2': float(probe['honest_defect_l2']),
                'tensor_max_abs_residual': float(probe['tensor_max_abs_residual']),
                'restricted_rank': int(probe['restricted_rank']),
                'rotation_deg': rotation_deg,
                'step_cost': candidate_cost,
                'used_budget_before': used_budget,
                'used_budget_after': new_budget,
                'score': score,
                'is_regular': int(is_regular),
                'within_rotation': int(within_rotation),
                'within_budget': int(within_budget),
                'accepted_by_policy': int(is_regular and within_rotation and within_budget),
            }
            candidate_rows.append(row)
            if not (is_regular and within_rotation and within_budget):
                continue
            if best is None or (
                score,
                row['kappa_ker_state'],
                row['rotation_deg'],
                row['eps'],
            ) < (
                best['row']['score'],
                best['row']['kappa_ker_state'],
                best['row']['rotation_deg'],
                best['row']['eps'],
            ):
                best = {
                    'row': row,
                    'delta': delta.copy(),
                    'dot_lambda': dot_lambda.copy(),
                    'alpha_next': np.asarray(next_state['alpha_next'], dtype=np.float64),
                    'soft_lambda_next': soft_lambda_next.copy(),
                }
    return best, candidate_rows


def run_policy(
    policy: dict[str, object],
    base_label: str,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    seed = 20260329 + 101 * alpha.shape[0] + sum(ord(ch) for ch in str(policy['policy_name']) + base_label)
    rng = np.random.default_rng(seed)
    current_alpha = alpha.copy()
    current_beta = beta.copy()
    h0, gamma0 = build_h_and_gamma(current_alpha, current_beta)
    mode0 = soft_kernel_mode(gamma0, h0)
    initial_rank = int(mode0['restricted_rank'])
    initial_kappa_ker = float(mode0['kappa_ker'])
    used_budget = 0.0
    accepted_steps = 0
    wildcard_wins = 0
    min_kappa_ker = initial_kappa_ker
    final_kappa_ker = initial_kappa_ker

    trajectory_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for step_idx in range(int(policy['max_steps'])):
        h_matrix, gamma = build_h_and_gamma(current_alpha, current_beta)
        mode = soft_kernel_mode(gamma, h_matrix)
        lam = np.asarray(mode['soft_lambda'], dtype=np.float64)
        current_kappa_ker = float(mode['kappa_ker'])
        step_details = solve_coupled_step_details(current_alpha, current_beta, lam)
        best, step_candidate_rows = select_candidate(
            current_alpha,
            current_beta,
            lam,
            step_details,
            initial_rank,
            rng,
            base_label,
            step_idx,
            current_kappa_ker,
            used_budget,
            policy,
        )
        candidate_rows.extend(step_candidate_rows)
        if best is None:
            trajectory_rows.append(
                {
                    'policy_name': str(policy['policy_name']),
                    'base_label': base_label,
                    'step_idx': step_idx,
                    'accepted': 0,
                    'branch': 'none',
                    'sample_index': -1,
                    'eps': 0.0,
                    'base_kappa_ker': current_kappa_ker,
                    'next_kappa_ker': current_kappa_ker,
                    'rotation_deg': 0.0,
                    'step_cost': 0.0,
                    'used_budget_after': used_budget,
                    'score': current_kappa_ker,
                    'tensor_max_abs_residual': 0.0,
                    'restricted_rank': int(mode['restricted_rank']),
                }
            )
            final_kappa_ker = current_kappa_ker
            break

        chosen = best['row']
        current_alpha = np.asarray(best['alpha_next'], dtype=np.float64)
        used_budget = float(chosen['used_budget_after'])
        accepted_steps += 1
        wildcard_wins += int(chosen['branch'] == 'wildcard')
        min_kappa_ker = min(min_kappa_ker, float(chosen['kappa_ker_state']))
        final_kappa_ker = float(chosen['kappa_ker_state'])
        trajectory_rows.append(
            {
                'policy_name': str(policy['policy_name']),
                'base_label': base_label,
                'step_idx': step_idx,
                'accepted': 1,
                'branch': str(chosen['branch']),
                'sample_index': int(chosen['sample_index']),
                'eps': float(chosen['eps']),
                'base_kappa_ker': current_kappa_ker,
                'next_kappa_ker': float(chosen['kappa_ker_state']),
                'rotation_deg': float(chosen['rotation_deg']),
                'step_cost': float(chosen['step_cost']),
                'used_budget_after': used_budget,
                'score': float(chosen['score']),
                'tensor_max_abs_residual': float(chosen['tensor_max_abs_residual']),
                'restricted_rank': int(chosen['restricted_rank']),
            }
        )

    summary = {
        'policy_name': str(policy['policy_name']),
        'base_label': base_label,
        'initial_kappa_ker': initial_kappa_ker,
        'final_kappa_ker': final_kappa_ker,
        'min_kappa_ker': min_kappa_ker,
        'accepted_steps': accepted_steps,
        'wildcard_wins': wildcard_wins,
        'initial_restricted_rank': initial_rank,
        'rotation_cap_deg': float(policy['rotation_cap_deg']),
        'total_budget': None if math.isinf(float(policy['total_budget'])) else float(policy['total_budget']),
        'budget_used': used_budget,
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
    for policy in POLICIES:
        for base_label, alpha, beta in base_cases:
            trajectory, candidates, summary = run_policy(policy, base_label, alpha, beta)
            trajectory_rows.extend(trajectory)
            candidate_rows.extend(candidates)
            summaries.append(summary)

    write_csv(
        OUT_DIR / 'smooth_budgeted_trajectory.csv',
        trajectory_rows,
        [
            'policy_name', 'base_label', 'step_idx', 'accepted', 'branch', 'sample_index', 'eps',
            'base_kappa_ker', 'next_kappa_ker', 'rotation_deg', 'step_cost', 'used_budget_after',
            'score', 'tensor_max_abs_residual', 'restricted_rank',
        ],
    )
    write_csv(
        OUT_DIR / 'smooth_budgeted_candidates.csv',
        candidate_rows,
        [
            'policy_name', 'base_label', 'step_idx', 'branch', 'sample_index', 'eps',
            'base_kappa_ker', 'kappa_ker_state', 'honest_defect_l2', 'tensor_max_abs_residual',
            'restricted_rank', 'rotation_deg', 'step_cost', 'used_budget_before', 'used_budget_after',
            'score', 'is_regular', 'within_rotation', 'within_budget', 'accepted_by_policy',
        ],
    )
    write_json(
        OUT_DIR / 'smooth_budgeted_continuation.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'exactness_tolerance': EXACTNESS_TOL,
            'step_epsilons': STEP_EPSILONS,
            'policies': POLICIES,
            'summaries': summaries,
        },
    )

    print('Phase 29 smooth-budgeted continuation complete.')
    for summary in summaries:
        print(
            f"  {summary['policy_name']} / {summary['base_label']}: initial={summary['initial_kappa_ker']:.6g}, "
            f"min={summary['min_kappa_ker']:.6g}, final={summary['final_kappa_ker']:.6g}, "
            f"accepted_steps={summary['accepted_steps']}, budget_used={summary['budget_used']:.6g}, "
            f"wildcard_wins={summary['wildcard_wins']}"
        )


if __name__ == '__main__':
    main()