from __future__ import annotations

import csv
import json
from pathlib import Path

from cp_als import (
    build_cp_tensor_cab,
    result_to_row,
    run_hybrid_cp,
    save_cp_result,
    summarize_results,
)


OUT_DIR = Path(__file__).resolve().parent
SUCCESS_DIR = OUT_DIR / 'successful_solutions'


def main() -> None:
    tensor_cab = build_cp_tensor_cab()
    restart_count = __import__('os').environ.get('ADE3X3_PHASE12_COLD_RESTARTS', '40')
    als_iters = __import__('os').environ.get('ADE3X3_PHASE12_COLD_ALS_ITERS', '1500')
    ls_max_nfev = __import__('os').environ.get('ADE3X3_PHASE12_COLD_LS_NFEV', '120')
    init_mode = __import__('os').environ.get('ADE3X3_PHASE12_COLD_INIT', 'random')
    restart_count_int = int(restart_count)
    als_iters_int = int(als_iters)
    ls_max_nfev_int = int(ls_max_nfev)

    rows: list[dict[str, object]] = []
    for restart_id in range(restart_count_int):
        seed = 910000 + restart_id
        result = run_hybrid_cp(
            tensor_cab,
            rank=23,
            init_mode=init_mode,
            restart_id=restart_id,
            seed=seed,
            als_iters=als_iters_int,
            ls_max_nfev=ls_max_nfev_int,
        )
        row = result_to_row(result)
        rows.append(row)
        if result.success_lt_1e_10:
            save_cp_result(SUCCESS_DIR / f'{restart_id:04d}.json', result)

    fieldnames = [
        'init_mode',
        'restart_id',
        'seed',
        'als_iterations',
        'als_loss',
        'als_max_abs',
        'ls_nfev',
        'final_loss',
        'final_max_abs',
        'success_lt_1e_10',
        'success_lt_1e_15',
    ]
    with open(OUT_DIR / 'cold_start_results.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows_sorted = sorted(rows, key=lambda row: (float(row['final_max_abs']), float(row['final_loss'])))
    summary = {
        'init_mode': init_mode,
        'restart_count': restart_count_int,
        'als_iters': als_iters_int,
        'ls_max_nfev': ls_max_nfev_int,
        'lt_1e_10': summarize_results(rows, 'success_lt_1e_10'),
        'lt_1e_15': summarize_results(rows, 'success_lt_1e_15'),
        'best_restart': rows_sorted[0] if rows_sorted else None,
    }
    (OUT_DIR / 'cold_start_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()