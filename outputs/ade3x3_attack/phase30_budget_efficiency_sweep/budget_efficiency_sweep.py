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

from outputs.ade3x3_attack.attack_common import load_public_terms, write_csv, write_json  # noqa: E402
from outputs.ade3x3_attack.phase22_kernel_linked_annihilator.kernel_linked_annihilator_homotopy import (  # noqa: E402
    stacked_alpha_beta,
    standard_rank27_terms,
)
from outputs.ade3x3_attack.phase29_smooth_budgeted_continuation.smooth_budgeted_continuation import (  # noqa: E402
    run_policy,
)


POLICY_GRID = [
    {
        'policy_name': 'sweep_rot8_bw0p0',
        'max_steps': 16,
        'total_budget': 1.2,
        'rotation_cap_deg': 8.0,
        'budget_weight': 0.0,
        'rotation_weight': 0.0,
    },
    {
        'policy_name': 'sweep_rot12_bw0p0',
        'max_steps': 16,
        'total_budget': 1.2,
        'rotation_cap_deg': 12.0,
        'budget_weight': 0.0,
        'rotation_weight': 0.0,
    },
    {
        'policy_name': 'sweep_rot12_bw0p3',
        'max_steps': 16,
        'total_budget': 1.2,
        'rotation_cap_deg': 12.0,
        'budget_weight': 0.3,
        'rotation_weight': 0.02,
    },
    {
        'policy_name': 'sweep_rot12_bw0p6',
        'max_steps': 24,
        'total_budget': 1.2,
        'rotation_cap_deg': 12.0,
        'budget_weight': 0.6,
        'rotation_weight': 0.02,
    },
    {
        'policy_name': 'sweep_rot16_bw0p6',
        'max_steps': 24,
        'total_budget': 1.2,
        'rotation_cap_deg': 16.0,
        'budget_weight': 0.6,
        'rotation_weight': 0.01,
    },
]


def safe_ratio(num: float, den: float) -> float | None:
    if abs(den) <= 1e-15:
        return None
    return float(num / den)


def summarize_efficiency(
    policy_name: str,
    base_label: str,
    trajectory: list[dict[str, object]],
    summary: dict[str, object],
) -> dict[str, object]:
    accepted = [row for row in trajectory if int(row['accepted']) == 1]
    initial = float(summary['initial_kappa_ker'])
    final = float(summary['final_kappa_ker'])
    total_gain = initial - final
    total_budget = float(summary['budget_used'])
    total_gain_per_budget = safe_ratio(total_gain, total_budget)
    total_rotation = float(sum(float(row['rotation_deg']) for row in accepted))
    total_gain_per_rotation = safe_ratio(total_gain, total_rotation)

    first_gain = None
    first_budget = None
    first_gain_per_budget = None
    if accepted:
        first_gain = initial - float(accepted[0]['next_kappa_ker'])
        first_budget = float(accepted[0]['step_cost'])
        first_gain_per_budget = safe_ratio(first_gain, first_budget)

    avg_step_gain = safe_ratio(total_gain, len(accepted)) if accepted else None
    avg_step_cost = safe_ratio(total_budget, len(accepted)) if accepted else None

    return {
        'policy_name': policy_name,
        'base_label': base_label,
        'rotation_cap_deg': float(summary['rotation_cap_deg']),
        'budget_weight': float(next(policy['budget_weight'] for policy in POLICY_GRID if policy['policy_name'] == policy_name)),
        'max_steps': int(next(policy['max_steps'] for policy in POLICY_GRID if policy['policy_name'] == policy_name)),
        'initial_kappa_ker': initial,
        'final_kappa_ker': final,
        'accepted_steps': int(summary['accepted_steps']),
        'wildcard_wins': int(summary['wildcard_wins']),
        'budget_used': total_budget,
        'total_gain': total_gain,
        'total_gain_per_budget': total_gain_per_budget,
        'total_rotation_deg': total_rotation,
        'total_gain_per_rotation_deg': total_gain_per_rotation,
        'first_step_gain': first_gain,
        'first_step_cost': first_budget,
        'first_step_gain_per_budget': first_gain_per_budget,
        'avg_step_gain': avg_step_gain,
        'avg_step_cost': avg_step_cost,
    }


def alpha_vs_standard_delta(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row['policy_name']), {})[str(row['base_label'])] = row

    deltas: list[dict[str, object]] = []
    for policy_name, mapping in grouped.items():
        alpha = next((row for label, row in mapping.items() if label.startswith('alphatensor')), None)
        standard = mapping.get('standard_rank27')
        if alpha is None or standard is None:
            continue
        alpha_eff = row_float(alpha, 'total_gain_per_budget')
        standard_eff = row_float(standard, 'total_gain_per_budget')
        alpha_rot = row_float(alpha, 'total_gain_per_rotation_deg')
        standard_rot = row_float(standard, 'total_gain_per_rotation_deg')
        deltas.append(
            {
                'policy_name': policy_name,
                'alpha_total_gain_per_budget': alpha_eff,
                'standard_total_gain_per_budget': standard_eff,
                'efficiency_gap_alpha_minus_standard': none_sub(alpha_eff, standard_eff),
                'alpha_total_gain_per_rotation_deg': alpha_rot,
                'standard_total_gain_per_rotation_deg': standard_rot,
                'rotation_efficiency_gap_alpha_minus_standard': none_sub(alpha_rot, standard_rot),
                'alpha_wildcard_wins': int(alpha['wildcard_wins']),
                'standard_wildcard_wins': int(standard['wildcard_wins']),
            }
        )
    return deltas


def row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def none_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    trajectory_rows: list[dict[str, object]] = []
    efficiency_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for policy in POLICY_GRID:
        for base_label, alpha, beta in base_cases:
            trajectory, _, summary = run_policy(policy, base_label, alpha, beta)
            trajectory_rows.extend(trajectory)
            summary_rows.append(summary)
            efficiency_rows.append(summarize_efficiency(str(policy['policy_name']), base_label, trajectory, summary))

    delta_rows = alpha_vs_standard_delta(efficiency_rows)

    write_csv(
        OUT_DIR / 'budget_efficiency_trajectory.csv',
        trajectory_rows,
        [
            'policy_name', 'base_label', 'step_idx', 'accepted', 'branch', 'sample_index', 'eps',
            'base_kappa_ker', 'next_kappa_ker', 'rotation_deg', 'step_cost', 'used_budget_after',
            'score', 'tensor_max_abs_residual', 'restricted_rank',
        ],
    )
    write_csv(
        OUT_DIR / 'budget_efficiency_summary.csv',
        efficiency_rows,
        [
            'policy_name', 'base_label', 'rotation_cap_deg', 'budget_weight', 'max_steps',
            'initial_kappa_ker', 'final_kappa_ker', 'accepted_steps', 'wildcard_wins', 'budget_used',
            'total_gain', 'total_gain_per_budget', 'total_rotation_deg', 'total_gain_per_rotation_deg',
            'first_step_gain', 'first_step_cost', 'first_step_gain_per_budget', 'avg_step_gain', 'avg_step_cost',
        ],
    )
    write_csv(
        OUT_DIR / 'budget_efficiency_deltas.csv',
        delta_rows,
        [
            'policy_name', 'alpha_total_gain_per_budget', 'standard_total_gain_per_budget',
            'efficiency_gap_alpha_minus_standard', 'alpha_total_gain_per_rotation_deg',
            'standard_total_gain_per_rotation_deg', 'rotation_efficiency_gap_alpha_minus_standard',
            'alpha_wildcard_wins', 'standard_wildcard_wins',
        ],
    )
    write_json(
        OUT_DIR / 'budget_efficiency_sweep.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'policy_grid': POLICY_GRID,
            'efficiency_rows': efficiency_rows,
            'delta_rows': delta_rows,
        },
    )

    print('Phase 30 budget-efficiency sweep complete.')
    for row in delta_rows:
        print(
            f"  {row['policy_name']}: alpha gain/budget={row['alpha_total_gain_per_budget']}, "
            f"standard gain/budget={row['standard_total_gain_per_budget']}, "
            f"gap={row['efficiency_gap_alpha_minus_standard']}"
        )


if __name__ == '__main__':
    main()