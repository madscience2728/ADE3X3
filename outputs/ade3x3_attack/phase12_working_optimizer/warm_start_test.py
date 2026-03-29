from __future__ import annotations

import json
from pathlib import Path

from cp_als import build_cp_tensor_cab, result_to_row, run_hybrid_cp


OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    tensor_cab = build_cp_tensor_cab()
    rows: list[dict[str, object]] = []
    for restart_id, sigma in enumerate([1e-3, 1e-2, 1e-1], start=1):
        result = run_hybrid_cp(
            tensor_cab,
            rank=23,
            init_mode='exact_noisy',
            restart_id=restart_id,
            seed=20260500 + restart_id,
            als_iters=300,
            ls_max_nfev=200,
            warm_sigma=sigma,
        )
        row = result_to_row(result)
        row['noise_sigma'] = sigma
        rows.append(row)

    fieldnames = [
        'noise_sigma',
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
    import csv
    with open(OUT_DIR / 'warm_start_results.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / 'warm_start_summary.json').write_text(json.dumps({'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()