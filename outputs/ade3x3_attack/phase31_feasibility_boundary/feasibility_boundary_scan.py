from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import load_public_terms, write_csv, write_json  # noqa: E402
from outputs.ade3x3_attack.phase22_kernel_linked_annihilator.kernel_linked_annihilator_homotopy import (  # noqa: E402
    stacked_alpha_beta,
    standard_rank27_terms,
)
from outputs.ade3x3_attack.phase29_smooth_budgeted_continuation.smooth_budgeted_continuation import (  # noqa: E402
    run_policy,
)


ROTATION_CAPS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
BUDGET_LIMITS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
MAX_STEPS = 16
GAIN_TOL = 1e-9
MAX_WORKERS = max(1, min(8, max(1, (os.cpu_count() or 2) // 2)))


def build_policy(rotation_cap_deg: float, total_budget: float) -> dict[str, object]:
    rot_label = str(rotation_cap_deg).replace('.', 'p')
    budget_label = str(total_budget).replace('.', 'p')
    return {
        'policy_name': f'rot{rot_label}_budget{budget_label}',
        'max_steps': MAX_STEPS,
        'total_budget': total_budget,
        'rotation_cap_deg': rotation_cap_deg,
        'budget_weight': 0.0,
        'rotation_weight': 0.0,
    }


def summarize_cell(policy: dict[str, object], base_label: str, summary: dict[str, object]) -> dict[str, object]:
    initial = float(summary['initial_kappa_ker'])
    final = float(summary['final_kappa_ker'])
    total_gain = initial - final
    feasible = int(int(summary['accepted_steps']) > 0 and total_gain > GAIN_TOL)
    return {
        'base_label': base_label,
        'policy_name': str(policy['policy_name']),
        'rotation_cap_deg': float(policy['rotation_cap_deg']),
        'total_budget': float(policy['total_budget']),
        'accepted_steps': int(summary['accepted_steps']),
        'wildcard_wins': int(summary['wildcard_wins']),
        'budget_used': float(summary['budget_used']),
        'initial_kappa_ker': initial,
        'final_kappa_ker': final,
        'total_gain': total_gain,
        'feasible_descent': feasible,
    }


def evaluate_grid_cell(task: tuple[dict[str, object], str, object, object]) -> dict[str, object]:
    policy, base_label, alpha, beta = task
    _, _, summary = run_policy(policy, base_label, alpha, beta)
    return summarize_cell(policy, base_label, summary)


def summarize_frontier(rows: list[dict[str, object]], base_label: str) -> list[dict[str, object]]:
    base_rows = [row for row in rows if str(row['base_label']) == base_label]
    frontier: list[dict[str, object]] = []
    for budget in BUDGET_LIMITS:
        feasible_rows = [
            row for row in base_rows
            if float(row['total_budget']) == budget and int(row['feasible_descent']) == 1
        ]
        if feasible_rows:
            best = min(feasible_rows, key=lambda row: (float(row['rotation_cap_deg']), -float(row['total_gain'])))
            frontier.append(
                {
                    'base_label': base_label,
                    'total_budget': budget,
                    'min_rotation_cap_deg_for_descent': float(best['rotation_cap_deg']),
                    'accepted_steps': int(best['accepted_steps']),
                    'total_gain': float(best['total_gain']),
                    'wildcard_wins': int(best['wildcard_wins']),
                }
            )
        else:
            frontier.append(
                {
                    'base_label': base_label,
                    'total_budget': budget,
                    'min_rotation_cap_deg_for_descent': None,
                    'accepted_steps': 0,
                    'total_gain': 0.0,
                    'wildcard_wins': 0,
                }
            )
    return frontier


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    tasks: list[tuple[dict[str, object], str, object, object]] = []
    for rotation_cap in ROTATION_CAPS:
        for budget_limit in BUDGET_LIMITS:
            policy = build_policy(rotation_cap, budget_limit)
            for base_label, alpha, beta in base_cases:
                tasks.append((policy, base_label, alpha, beta))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        grid_rows = list(executor.map(evaluate_grid_cell, tasks))

    frontier_rows: list[dict[str, object]] = []
    for base_label, *_ in base_cases:
        frontier_rows.extend(summarize_frontier(grid_rows, base_label))

    delta_rows: list[dict[str, object]] = []
    for budget_limit in BUDGET_LIMITS:
        alpha_row = next(
            row for row in frontier_rows
            if str(row['base_label']).startswith('alphatensor') and float(row['total_budget']) == budget_limit
        )
        standard_row = next(
            row for row in frontier_rows
            if str(row['base_label']) == 'standard_rank27' and float(row['total_budget']) == budget_limit
        )
        alpha_cap = alpha_row['min_rotation_cap_deg_for_descent']
        standard_cap = standard_row['min_rotation_cap_deg_for_descent']
        cap_gap = None
        if alpha_cap is not None and standard_cap is not None:
            cap_gap = float(standard_cap) - float(alpha_cap)
        delta_rows.append(
            {
                'total_budget': budget_limit,
                'alpha_min_rotation_cap_deg_for_descent': alpha_cap,
                'standard_min_rotation_cap_deg_for_descent': standard_cap,
                'rotation_cap_gap_standard_minus_alpha': cap_gap,
            }
        )

    write_csv(
        OUT_DIR / 'feasibility_boundary_grid.csv',
        grid_rows,
        [
            'base_label', 'policy_name', 'rotation_cap_deg', 'total_budget', 'accepted_steps',
            'wildcard_wins', 'budget_used', 'initial_kappa_ker', 'final_kappa_ker', 'total_gain',
            'feasible_descent',
        ],
    )
    write_csv(
        OUT_DIR / 'feasibility_boundary_frontier.csv',
        frontier_rows,
        [
            'base_label', 'total_budget', 'min_rotation_cap_deg_for_descent', 'accepted_steps',
            'total_gain', 'wildcard_wins',
        ],
    )
    write_csv(
        OUT_DIR / 'feasibility_boundary_deltas.csv',
        delta_rows,
        [
            'total_budget', 'alpha_min_rotation_cap_deg_for_descent',
            'standard_min_rotation_cap_deg_for_descent', 'rotation_cap_gap_standard_minus_alpha',
        ],
    )
    write_json(
        OUT_DIR / 'feasibility_boundary_scan.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rotation_caps': ROTATION_CAPS,
            'budget_limits': BUDGET_LIMITS,
            'max_steps': MAX_STEPS,
            'max_workers': MAX_WORKERS,
            'grid_rows': grid_rows,
            'frontier_rows': frontier_rows,
            'delta_rows': delta_rows,
        },
    )

    print('Phase 31 feasibility-boundary scan complete.')
    for row in delta_rows:
        print(
            f"  budget={row['total_budget']}: alpha min cap={row['alpha_min_rotation_cap_deg_for_descent']}, "
            f"standard min cap={row['standard_min_rotation_cap_deg_for_descent']}, "
            f"gap={row['rotation_cap_gap_standard_minus_alpha']}"
        )


if __name__ == '__main__':
    main()