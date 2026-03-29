from __future__ import annotations

import csv
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
PHASE22_CSV = REPO_ROOT / 'outputs' / 'ade3x3_attack' / 'phase22_kernel_linked_annihilator' / 'kernel_linked_annihilator_homotopy.csv'


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(PHASE22_CSV, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, object] = {
                'base_label': row['base_label'],
                'candidate': row['candidate'],
                't_value': float(row['t_value']),
                'R': int(row['R']),
                'annihilator_projection_residual': float(row['annihilator_projection_residual']),
                'lambda_H_max_abs': float(row['lambda_H_max_abs']),
                'lambda_to_ker_angle_deg': float(row['lambda_to_ker_angle_deg']),
                'rank_H_numeric': int(row['rank_H_numeric']),
                'restricted_rank_numeric': int(row['restricted_rank_numeric']),
                'functional_defect_dim_numeric': int(row['functional_defect_dim_numeric']),
                'tensor_max_abs_residual': float(row['tensor_max_abs_residual']),
                'tensor_loss': float(row['tensor_loss']),
            }
            rows.append(parsed)
    return rows


def threshold_row(rows: list[dict[str, object]], key: str, threshold: float, leq: bool) -> dict[str, object] | None:
    for row in rows:
        value = float(row[key])
        if (value <= threshold) if leq else (value >= threshold):
            return row
    return None


def summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    rows = sorted(rows, key=lambda row: float(row['t_value']))
    return {
        'first_residual_over_0p05': threshold_row(rows, 'tensor_max_abs_residual', 0.05, leq=False),
        'first_residual_over_0p10': threshold_row(rows, 'tensor_max_abs_residual', 0.10, leq=False),
        'first_angle_over_5deg': threshold_row(rows, 'lambda_to_ker_angle_deg', 5.0, leq=False),
        'first_angle_over_15deg': threshold_row(rows, 'lambda_to_ker_angle_deg', 15.0, leq=False),
        'first_lambdaH_below_0p10': threshold_row(rows, 'lambda_H_max_abs', 0.10, leq=True),
        'endpoint': rows[-1],
    }


def main() -> None:
    rows = load_rows()
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row['base_label']), str(row['candidate'])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (base_label, candidate), group_rows in sorted(groups.items()):
        summaries.append({
            'base_label': base_label,
            'candidate': candidate,
            **summarize_group(group_rows),
        })

    out_path = OUT_DIR / 'staged_homotopy_thresholds.json'
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump({'summaries': summaries}, handle, indent=2)

    print('Phase 23 threshold extraction complete.')
    for summary in summaries:
        first_bad = summary['first_residual_over_0p05']
        t_text = 'none'
        if first_bad:
            t_text = str(first_bad['t_value'])
        print(f"  {summary['base_label']} / {summary['candidate']}: first residual > 0.05 at t={t_text}")


if __name__ == '__main__':
    main()