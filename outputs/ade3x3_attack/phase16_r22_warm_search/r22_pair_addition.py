from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
PAIR_GROUPS = [('t03', 't06'), ('t09', 't12'), ('t15', 't17')]
ATTACK_ROOT = OUT_DIR.parent
PHASE12_DIR = ATTACK_ROOT / 'phase12_working_optimizer'
if str(PHASE12_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE12_DIR))

from cp_als import build_cp_tensor_cab, exact_factor_matrices, result_to_row, run_hybrid_from_factors  # noqa: E402


def term_index(term_id: str) -> int:
    return int(term_id[1:]) - 1


def worker_count(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    default = max(1, min(task_count, cpu_total - 1 if cpu_total > 1 else 1))
    return max(1, min(task_count, int(os.environ.get('ADE3X3_PHASE16_PAIR_WORKERS', str(default)))))


def config_list() -> list[tuple[str, list[str]]]:
    configs: list[tuple[str, list[str]]] = []
    for mask in range(8):
        removed: list[str] = []
        for pair_idx, pair in enumerate(PAIR_GROUPS):
            removed.append(pair[(mask >> pair_idx) & 1])
        configs.append((f'mask_{mask:03b}', removed))
    return configs


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['config_label', 'removed_terms', 'trial_idx', 'init_mode', 'restart_id', 'seed', 'als_iterations', 'als_loss', 'als_max_abs', 'ls_nfev', 'final_loss', 'final_max_abs', 'success_lt_1e_10', 'success_lt_1e_15']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row['config_label']), float(row['final_loss']))))


def run_pair_trial(config_label: str, removed_terms: list[str], trial_idx: int, restart_id: int, als_iters: int, ls_max_nfev: int) -> dict[str, object]:
    tensor_cab = build_cp_tensor_cab()
    c_exact, a_exact, b_exact, _, _ = exact_factor_matrices()
    remove_indices = {term_index(term_id) for term_id in removed_terms}
    keep = [idx for idx in range(c_exact.shape[1]) if idx not in remove_indices]
    base_c = c_exact[:, keep]
    base_a = a_exact[:, keep]
    base_b = b_exact[:, keep]
    rng = np.random.default_rng(20262100 + restart_id)
    c_init = np.column_stack([base_c, rng.normal(0.0, 1.0, size=(9, 2))])
    a_init = np.column_stack([base_a, rng.normal(0.0, 1.0, size=(9, 2))])
    b_init = np.column_stack([base_b, rng.normal(0.0, 1.0, size=(9, 2))])
    result = run_hybrid_from_factors(
        tensor_cab,
        c_init,
        a_init,
        b_init,
        restart_id=restart_id,
        seed=20262100 + restart_id,
        als_iters=als_iters,
        ls_max_nfev=ls_max_nfev,
        init_mode='pair_addition',
    )
    row = result_to_row(result)
    row['config_label'] = config_label
    row['removed_terms'] = ','.join(removed_terms)
    row['trial_idx'] = trial_idx
    return row


def main() -> None:
    als_iters = int(__import__('os').environ.get('ADE3X3_PHASE16_PAIR_ALS_ITERS', '500'))
    ls_max_nfev = int(__import__('os').environ.get('ADE3X3_PHASE16_PAIR_LS_NFEV', '250'))
    trials_per_config = int(__import__('os').environ.get('ADE3X3_PHASE16_PAIR_TRIALS', '50'))

    configs = config_list()

    rows: list[dict[str, object]] = []
    jobs: list[tuple[str, list[str], int, int]] = []
    restart_id = 0
    for config_label, removed_terms in configs:
        for trial_idx in range(trials_per_config):
            jobs.append((config_label, removed_terms, trial_idx, restart_id))
            restart_id += 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count(len(jobs))) as executor:
        futures = [
            executor.submit(run_pair_trial, config_label, removed_terms, trial_idx, restart_id, als_iters, ls_max_nfev)
            for config_label, removed_terms, trial_idx, restart_id in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            write_rows_csv(OUT_DIR / 'r22_pair_addition_results.csv', rows)

    best_by_config: list[dict[str, object]] = []
    for config_label, _ in configs:
        config_rows = [row for row in rows if row['config_label'] == config_label]
        if config_rows:
            best_by_config.append(min(config_rows, key=lambda row: float(row['final_loss'])))
    (OUT_DIR / 'r22_pair_addition_summary.json').write_text(json.dumps({'best_by_config': best_by_config, 'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()