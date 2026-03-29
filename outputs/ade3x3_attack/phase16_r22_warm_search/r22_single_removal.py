from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
BEST_DIR = OUT_DIR / 'single_removal_best'
ATTACK_ROOT = OUT_DIR.parent
PHASE12_DIR = ATTACK_ROOT / 'phase12_working_optimizer'
if str(PHASE12_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE12_DIR))

from cp_als import build_cp_tensor_cab, exact_factor_matrices, result_to_row, run_hybrid_from_factors, save_cp_result  # noqa: E402


def worker_count(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    default = max(1, min(task_count, cpu_total - 1 if cpu_total > 1 else 1))
    return max(1, min(task_count, int(os.environ.get('ADE3X3_PHASE16_SINGLE_WORKERS', str(default)))))


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['removed_term_id', 'init_mode', 'restart_id', 'seed', 'als_iterations', 'als_loss', 'als_max_abs', 'ls_nfev', 'final_loss', 'final_max_abs', 'success_lt_1e_10', 'success_lt_1e_15']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row['final_loss'])))


def run_single_removal(remove_idx: int, als_iters: int, ls_max_nfev: int) -> tuple[dict[str, object], dict[str, object]]:
    tensor_cab = build_cp_tensor_cab()
    c_exact, a_exact, b_exact, _, _ = exact_factor_matrices()
    keep = [idx for idx in range(c_exact.shape[1]) if idx != remove_idx]
    result = run_hybrid_from_factors(
        tensor_cab,
        c_exact[:, keep],
        a_exact[:, keep],
        b_exact[:, keep],
        restart_id=remove_idx,
        seed=20262000 + remove_idx,
        als_iters=als_iters,
        ls_max_nfev=ls_max_nfev,
        init_mode='single_removal',
    )
    row = result_to_row(result)
    row['removed_term_id'] = f't{remove_idx + 1:02d}'
    payload = {
        **row,
        **result.cp_model_json,
    }
    return row, payload


def main() -> None:
    als_iters = int(__import__('os').environ.get('ADE3X3_PHASE16_SINGLE_ALS_ITERS', '1000'))
    ls_max_nfev = int(__import__('os').environ.get('ADE3X3_PHASE16_SINGLE_LS_NFEV', '500'))

    rows: list[dict[str, object]] = []
    saved_payloads: list[tuple[dict[str, object], dict[str, object]]] = []
    task_count = 23
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count(task_count)) as executor:
        futures = [executor.submit(run_single_removal, remove_idx, als_iters, ls_max_nfev) for remove_idx in range(task_count)]
        for future in concurrent.futures.as_completed(futures):
            row, payload = future.result()
            rows.append(row)
            saved_payloads.append((row, payload))
            write_rows_csv(OUT_DIR / 'r22_single_removal_results.csv', rows)

    rows.sort(key=lambda row: float(row['final_loss']))
    for save_idx, (row, payload) in enumerate(sorted(saved_payloads, key=lambda item: float(item[0]['final_loss']))[:5], start=1):
        BEST_DIR.mkdir(parents=True, exist_ok=True)
        (BEST_DIR / f'{save_idx:02d}_{row["removed_term_id"]}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    summary = {'best_row': rows[0] if rows else None, 'rows': rows}
    (OUT_DIR / 'r22_single_removal_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()