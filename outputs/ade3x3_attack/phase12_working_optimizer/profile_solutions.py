from __future__ import annotations

import csv
import json
from pathlib import Path

from cp_als import cp_terms_from_factors


OUT_DIR = Path(__file__).resolve().parent
SUCCESS_DIR = OUT_DIR / 'successful_solutions'
ATTACK_ROOT = OUT_DIR.parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    import sys

    if str(ATTACK_ROOT) not in sys.path:
        sys.path.insert(0, str(ATTACK_ROOT))
    from attack_common import profile_terms  # noqa: E402

    rows: list[dict[str, object]] = []
    for path in sorted(SUCCESS_DIR.glob('*.json')):
        payload = load_json(path)
        c_factor = __import__('numpy').array(payload['c_factor'], dtype=float)
        a_factor = __import__('numpy').array(payload['a_factor'], dtype=float)
        b_factor = __import__('numpy').array(payload['b_factor'], dtype=float)
        terms = cp_terms_from_factors(c_factor, a_factor, b_factor, prefix=path.stem + '_')
        profile = profile_terms(terms, rank_tol=1e-8, projection_tol=1e-6)
        rows.append(
            {
                'solution_id': path.stem,
                'final_loss': payload['final_loss'],
                'final_max_abs': payload['final_max_abs'],
                'rank_H_numeric': profile['rank_H_numeric'],
                'rank_Delta_numeric': profile['rank_Delta_numeric'],
                'rank_HDelta_numeric': profile['rank_HDelta_numeric'],
                'rank_Nuisance_numeric': profile['rank_Nuisance_numeric'],
                'quotient_gain_numeric': profile['quotient_gain_numeric'],
                'delta_in_eta_numeric': profile['delta_in_eta_numeric'],
                'projection_max_residual_numeric': profile['projection_max_residual_numeric'],
            }
        )

    fieldnames = [
        'solution_id',
        'final_loss',
        'final_max_abs',
        'rank_H_numeric',
        'rank_Delta_numeric',
        'rank_HDelta_numeric',
        'rank_Nuisance_numeric',
        'quotient_gain_numeric',
        'delta_in_eta_numeric',
        'projection_max_residual_numeric',
    ]
    with open(OUT_DIR / 'solution_profiles.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tensor_summary = load_json(OUT_DIR / 'tensor_construction_summary.json') if (OUT_DIR / 'tensor_construction_summary.json').exists() else {}
    warm_summary = load_json(OUT_DIR / 'warm_start_summary.json') if (OUT_DIR / 'warm_start_summary.json').exists() else {'rows': []}
    cold_summary = load_json(OUT_DIR / 'cold_start_summary.json') if (OUT_DIR / 'cold_start_summary.json').exists() else {}

    warm_rows = warm_summary.get('rows', [])
    warm_pass = all(bool(row.get('success_lt_1e_15')) for row in warm_rows) if warm_rows else False
    cold_hit_rate = cold_summary.get('lt_1e_10', {}).get('hit_rate', 0.0)
    results_md = f'''# Phase 12 Results

## 12a-12b. CP Tensor Construction

- Tensor shape: {tensor_summary.get('shape')}
- Nonzero entries: {tensor_summary.get('nonzero_count')}
- Slice nonzero counts: {tensor_summary.get('slice_nonzero_counts')}
- Mode-0 unfolding rank: {tensor_summary.get('mode0_rank')}
- Exact AlphaTensor reconstruction max-abs: {tensor_summary.get('exact_reconstruction_max_abs')}

## 12c. Warm Start Validation

- Warm-start rows: {len(warm_rows)}
- All warm starts below `1e-15` max-abs: {warm_pass}

## 12d. Cold Start R=23 Benchmark

- Init mode: {cold_summary.get('init_mode')}
- Restart count: {cold_summary.get('restart_count')}
- Success rate below `1e-10`: {cold_summary.get('lt_1e_10', {}).get('hit_rate')}
- Success rate below `1e-15`: {cold_summary.get('lt_1e_15', {}).get('hit_rate')}
- Best final max-abs residual: {cold_summary.get('best_restart', {}).get('final_max_abs')}

## 12e. Successful Solution Profiles

- Successful solutions profiled: {len(rows)}
- Calibration verdict: {'PASS' if warm_pass and cold_hit_rate >= 0.05 else 'FAIL'}
'''
    (OUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')


if __name__ == '__main__':
    main()