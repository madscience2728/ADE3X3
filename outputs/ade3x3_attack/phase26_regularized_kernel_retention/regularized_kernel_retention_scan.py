from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import write_csv, write_json  # noqa: E402


PHASE25_CSV = ATTACK_ROOT / 'phase25_honest_kernel_retention' / 'honest_kernel_retention_scan.csv'
EXACTNESS_TOL = 1e-10


def read_rows() -> list[dict[str, object]]:
    with open(PHASE25_CSV, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            parsed: dict[str, object] = {
                'base_label': row['base_label'],
                'candidate': row['candidate'],
                't_value': float(row['t_value']),
                'R': int(row['R']),
                'tensor_max_abs_residual': float(row['tensor_max_abs_residual']),
                'tensor_loss': float(row['tensor_loss']),
                'kernel_dim': int(row['kernel_dim']),
                'restricted_rank': int(row['restricted_rank']),
                'kappa_ker': float(row['kappa_ker']),
                'kappa_full': float(row['kappa_full']),
                'soft_lambda_h_max_abs': float(row['soft_lambda_h_max_abs']),
                'soft_lambda_gamma_max_abs': float(row['soft_lambda_gamma_max_abs']),
                'base_candidate_angle_deg': float(row['base_candidate_angle_deg']),
                'soft_mode_rotation_deg': float(row['soft_mode_rotation_deg']),
            }
            rows.append(parsed)
        return rows


def main() -> None:
    rows = read_rows()
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row['base_label']), str(row['candidate']))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (base_label, candidate), group_rows in grouped.items():
        group_rows.sort(key=lambda row: float(row['t_value']))
        base_row = group_rows[0]
        base_rank = int(base_row['restricted_rank'])
        regular_rows = [
            row for row in group_rows
            if float(row['tensor_max_abs_residual']) <= EXACTNESS_TOL and int(row['restricted_rank']) == base_rank
        ]
        min_regular = min(regular_rows, key=lambda row: (float(row['kappa_ker']), float(row['t_value']))) if regular_rows else None
        first_below_10pct = next(
            (row for row in regular_rows if float(row['kappa_ker']) <= 0.1 * float(base_row['kappa_ker'])),
            None,
        )
        summary_rows.append(
            {
                'base_label': base_label,
                'candidate': candidate,
                'base_restricted_rank': base_rank,
                'base_kappa_ker': float(base_row['kappa_ker']),
                'regular_row_count': len(regular_rows),
                'min_regular_kappa_ker': float(min_regular['kappa_ker']) if min_regular else None,
                'min_regular_kappa_ker_t': float(min_regular['t_value']) if min_regular else None,
                'tensor_residual_at_min_regular_kappa_ker': float(min_regular['tensor_max_abs_residual']) if min_regular else None,
                'first_regular_kappa_ker_below_10pct_t': float(first_below_10pct['t_value']) if first_below_10pct else None,
            }
        )

    write_csv(
        OUT_DIR / 'regularized_kernel_retention_summary.csv',
        summary_rows,
        [
            'base_label', 'candidate', 'base_restricted_rank', 'base_kappa_ker', 'regular_row_count',
            'min_regular_kappa_ker', 'min_regular_kappa_ker_t', 'tensor_residual_at_min_regular_kappa_ker',
            'first_regular_kappa_ker_below_10pct_t',
        ],
    )
    write_json(
        OUT_DIR / 'regularized_kernel_retention_summary.json',
        {
            'exactness_tolerance': EXACTNESS_TOL,
            'summaries': summary_rows,
        },
    )

    print('Phase 26 regularized-kernel-retention summary complete.')
    for row in summary_rows:
        print(
            f"  {row['base_label']} / {row['candidate']}: min regular kappa_ker="
            f"{row['min_regular_kappa_ker']} at t={row['min_regular_kappa_ker_t']}"
        )


if __name__ == '__main__':
    main()