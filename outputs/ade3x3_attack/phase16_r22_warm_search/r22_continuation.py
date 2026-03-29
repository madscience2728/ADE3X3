from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
PHASE12_DIR = ATTACK_ROOT / 'phase12_working_optimizer'
if str(PHASE12_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE12_DIR))

from cp_als import build_cp_tensor_cab, cp_loss_stats, exact_factor_matrices, run_hybrid_from_factors  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def worker_count(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    default = max(1, min(task_count, cpu_total - 1 if cpu_total > 1 else 1))
    return max(1, min(task_count, int(os.environ.get('ADE3X3_PHASE16_CONT_WORKERS', str(default)))))


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['removed_term_id', 'epsilon', 'fixed_scale', 'als_iterations', 'als_loss', 'als_max_abs', 'ls_nfev', 'final_loss', 'final_max_abs', 'success_lt_1e_10', 'success_lt_1e_15']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row['removed_term_id']), float(row['epsilon']))))


def run_continuation_path(remove_idx: int, als_iters: int, ls_max_nfev: int, epsilons: list[float]) -> list[dict[str, object]]:
    tensor_cab = build_cp_tensor_cab()
    c_exact, a_exact, b_exact, _, _ = exact_factor_matrices()
    rows: list[dict[str, object]] = []
    keep = [idx for idx in range(c_exact.shape[1]) if idx != remove_idx]
    current_c = c_exact[:, keep].copy()
    current_a = a_exact[:, keep].copy()
    current_b = b_exact[:, keep].copy()
    fixed_c = c_exact[:, [remove_idx]]
    fixed_a = a_exact[:, [remove_idx]]
    fixed_b = b_exact[:, [remove_idx]]
    fixed_tensor_base = np.einsum('cr,ar,br->cab', fixed_c, fixed_a, fixed_b, optimize=True)
    for epsilon in epsilons:
        scale = 1.0 - epsilon
        target_residual = tensor_cab - scale * fixed_tensor_base
        result = run_hybrid_from_factors(
            target_residual,
            current_c,
            current_a,
            current_b,
            restart_id=remove_idx,
            seed=20262200 + 100 * remove_idx + int(round(100 * epsilon)),
            als_iters=als_iters,
            ls_max_nfev=ls_max_nfev,
            init_mode='continuation',
        )
        current_c = np.array(result.cp_model_json['c_factor'], dtype=np.float64)
        current_a = np.array(result.cp_model_json['a_factor'], dtype=np.float64)
        current_b = np.array(result.cp_model_json['b_factor'], dtype=np.float64)
        full_c = np.column_stack([scale * fixed_c, current_c])
        full_a = np.column_stack([fixed_a, current_a])
        full_b = np.column_stack([fixed_b, current_b])
        full_loss, full_max_abs = cp_loss_stats(tensor_cab, full_c, full_a, full_b)
        rows.append(
            {
                'removed_term_id': f't{remove_idx + 1:02d}',
                'epsilon': epsilon,
                'fixed_scale': scale,
                'als_iterations': result.als_iterations,
                'als_loss': result.als_loss,
                'als_max_abs': result.als_max_abs,
                'ls_nfev': result.ls_nfev,
                'final_loss': full_loss,
                'final_max_abs': full_max_abs,
                'success_lt_1e_10': bool(full_max_abs < 1e-10),
                'success_lt_1e_15': bool(full_max_abs < 1e-15),
            }
        )
    return rows


def main() -> None:
    als_iters = int(__import__('os').environ.get('ADE3X3_PHASE16_CONT_ALS_ITERS', '300'))
    ls_max_nfev = int(__import__('os').environ.get('ADE3X3_PHASE16_CONT_LS_NFEV', '200'))
    epsilons = [0.01, 0.05, 0.1] + [round(value, 1) for value in np.arange(0.2, 1.0, 0.1)] + [0.99, 1.0]

    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count(23)) as executor:
        futures = [executor.submit(run_continuation_path, remove_idx, als_iters, ls_max_nfev, epsilons) for remove_idx in range(23)]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
            write_rows_csv(OUT_DIR / 'r22_continuation_results.csv', rows)

    best_at_full_removal = []
    for remove_idx in range(c_exact.shape[1]):
        term_rows = [row for row in rows if row['removed_term_id'] == f't{remove_idx + 1:02d}' and abs(float(row['epsilon']) - 1.0) < 1e-12]
        if term_rows:
            best_at_full_removal.append(term_rows[0])

    single_summary = load_json(OUT_DIR / 'r22_single_removal_summary.json') if (OUT_DIR / 'r22_single_removal_summary.json').exists() else {'best_row': None}
    pair_summary = load_json(OUT_DIR / 'r22_pair_addition_summary.json') if (OUT_DIR / 'r22_pair_addition_summary.json').exists() else {'best_by_config': []}

    best_cont = min(best_at_full_removal, key=lambda row: float(row['final_loss'])) if best_at_full_removal else None
    summary = {
        'best_full_removal': best_cont,
        'rows': rows,
    }
    (OUT_DIR / 'r22_continuation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    results_md = f'''# Phase 16 Results

## 16a. Single-Term Removal Re-optimization

- Best single-removal residual: {single_summary.get('best_row', {}).get('final_max_abs')}
- Best single-removal removed term: {single_summary.get('best_row', {}).get('removed_term_id')}

## 16b. Pair-Structured Addition Search

- Configurations tested: {len(pair_summary.get('best_by_config', []))}
- Best pair-addition residual: {min((row['final_max_abs'] for row in pair_summary.get('best_by_config', [])), default=None)}

## 16c. Continuation from R=23 to R=22

- Epsilon schedule: {epsilons}
- Best full-removal continuation residual: {best_cont.get('final_max_abs') if best_cont else None}
- Best full-removal continuation term: {best_cont.get('removed_term_id') if best_cont else None}
'''
    (OUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')


if __name__ == '__main__':
    main()