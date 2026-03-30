"""
ade3x3_step79_basis_rotated_pilot.py

Step 79: Basis-rotated pilot search.

This is a pilot for checklist item 2. It does not attempt the full 50-100 rotation
campaign yet; instead it verifies the basis-change machinery on a small structured
and random set of GL(9)^3 transforms, then runs a cold CP scan on the rotated full
tensor at ranks 19..22.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
    term_tensor,
)
from src.ade3x3.steps.ade3x3_step68_layered_correction_tiling import rank_scan_target


EXPORTS = Path('outputs/exports')
WING_SUMMARY = Path('experiments/wing it/summary.json')


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


def env_rank_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    values: list[int] = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


RANKS = env_rank_list('STEP79_RANKS', [19, 20, 21, 22])
RESTARTS = env_int('STEP79_RESTARTS', 6)
WORKERS = env_int('STEP79_WORKERS', min(4, max(1, os.cpu_count() or 1)))
RANDOM_ROTATIONS = env_int('STEP79_RANDOM_ROTATIONS', 3)
COND_BOUND = env_float('STEP79_COND_BOUND', 3.0)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def scalar_to_str(value: float) -> str:
    return f'{float(value):.16e}'


def dct_orthogonal_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    factor0 = math.sqrt(1.0 / n)
    factor = math.sqrt(2.0 / n)
    for k in range(n):
        for idx in range(n):
            base = math.cos(math.pi * (idx + 0.5) * k / n)
            matrix[k, idx] = factor0 * base if k == 0 else factor * base
    return matrix


def random_orthogonal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((9, 9))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def random_well_conditioned_gl(seed: int, cond_bound: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = random_orthogonal(seed)
    right = random_orthogonal(seed + 1)
    log_cond = math.log(cond_bound)
    exponents = rng.uniform(-log_cond, log_cond, size=9)
    scales = np.exp(exponents)
    scales = scales / np.exp(np.mean(np.log(scales)))
    return left @ np.diag(scales) @ right


def mode_transform_tensor(target: np.ndarray, a_mode: np.ndarray, b_mode: np.ndarray, c_mode: np.ndarray) -> np.ndarray:
    return np.einsum('ia,jb,kc,abc->ijk', a_mode, b_mode, c_mode, target, optimize=True)


def transform_term(term: Term, a_mode: np.ndarray, b_mode: np.ndarray, c_mode: np.ndarray, prefix: str) -> Term:
    alpha = (a_mode @ term.alpha.reshape(-1)).reshape(3, 3)
    beta = (b_mode @ term.beta.reshape(-1)).reshape(3, 3)
    gamma = (c_mode @ term.gamma.reshape(-1)).reshape(3, 3)
    return Term(f'{prefix}_{term.term_id}', prefix, alpha, beta, gamma)


def reconstruct_terms(terms: list[Term]) -> np.ndarray:
    total = np.zeros((9, 9, 9), dtype=np.float64)
    for term in terms:
        total += term_tensor(term)
    return total


def transform_spec_rows() -> list[dict]:
    rows: list[dict] = []
    dct = dct_orthogonal_matrix(9)
    rows.append({'rotation_id': 'rot_dct_shared', 'family': 'structured_dct', 'seed': '', 'a': dct, 'b': dct, 'c': dct})
    for idx in range(RANDOM_ROTATIONS):
        seed = 910000 + 100 * idx
        rows.append(
            {
                'rotation_id': f'rot_gl_{idx + 1:02d}',
                'family': 'random_well_conditioned_gl',
                'seed': str(seed),
                'a': random_well_conditioned_gl(seed + 11, COND_BOUND),
                'b': random_well_conditioned_gl(seed + 22, COND_BOUND),
                'c': random_well_conditioned_gl(seed + 33, COND_BOUND),
            }
        )
    return rows


def matrix_condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return float(singular_values[0] / singular_values[-1])


def best_known_rank19() -> float | None:
    if not WING_SUMMARY.exists():
        return None
    with open(WING_SUMMARY, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    rank19 = payload.get('best_by_rank', {}).get('19', {})
    value = rank19.get('final_max_abs')
    return None if value is None else float(value)


def markdown_report(summary_rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 79: Basis-Rotated Pilot Search')
    w('')
    w(f'[MEASURED_FROM_CODE] Pilot scan over {len(summary_rows)} rotated targets with ranks {RANKS} and {RESTARTS} cold restarts per rank.')
    w('')
    w('| rotation | family | best rank tested | best max-abs residual | exact hit found | public rank-23 calibration residual | max condition |')
    w('|----------|--------|------------------|-----------------------|----------------|-----------------------------------|---------------|')
    for row in summary_rows:
        w(
            f"| {row['rotation_id']} | {row['family']} | {row['pilot_best_rank']} | {row['pilot_best_max_abs_residual']} | "
            f"{row['pilot_any_exact_hit']} | {row['public_rank23_calibration_residual']} | {row['max_condition_number']} |"
        )
    return '\n'.join(lines) + '\n'


def main() -> None:
    start = time.time()
    target = matrix_multiplication_tensor(3).astype(np.float64)
    public_terms, _, _ = load_public_rank23_terms()
    known_rank19 = best_known_rank19()

    result_rows: list[dict] = []
    summary_rows: list[dict] = []

    print('=== Step 79: Basis-rotated pilot search ===', flush=True)
    for spec in transform_spec_rows():
        rotation_id = str(spec['rotation_id'])
        family = str(spec['family'])
        a_mode = spec['a']
        b_mode = spec['b']
        c_mode = spec['c']

        rotated_target = mode_transform_tensor(target, a_mode, b_mode, c_mode)
        transformed_terms = [transform_term(term, a_mode, b_mode, c_mode, rotation_id) for term in public_terms]
        calibration_residual = float(np.max(np.abs(reconstruct_terms(transformed_terms) - rotated_target)))
        print(f'Running pilot scan for {rotation_id}...', flush=True)
        rows, verdict = rank_scan_target(rotation_id, rotated_target, RANKS, RESTARTS, WORKERS)

        for row in rows:
            result_rows.append(
                {
                    'rotation_id': rotation_id,
                    'family': family,
                    'seed': spec['seed'],
                    'rank_tested': row['rank_tested'],
                    'restarts': row['restarts'],
                    'best_verified_loss': row['best_verified_loss'],
                    'best_max_abs_residual': row['best_max_abs_residual'],
                    'verified_exact': row['verified_exact'],
                    'provenance': row['provenance'],
                }
            )

        pilot_best = min(rows, key=lambda row: float(row['best_max_abs_residual']))
        rank19_row = next((row for row in rows if int(row['rank_tested']) == 19), None)
        rank19_beats_known = False
        if rank19_row is not None and known_rank19 is not None:
            rank19_beats_known = float(rank19_row['best_max_abs_residual']) < known_rank19

        summary_rows.append(
            {
                'rotation_id': rotation_id,
                'family': family,
                'seed': spec['seed'],
                'public_rank23_calibration_residual': scalar_to_str(calibration_residual),
                'cond_a': scalar_to_str(matrix_condition_number(a_mode)),
                'cond_b': scalar_to_str(matrix_condition_number(b_mode)),
                'cond_c': scalar_to_str(matrix_condition_number(c_mode)),
                'max_condition_number': scalar_to_str(max(matrix_condition_number(a_mode), matrix_condition_number(b_mode), matrix_condition_number(c_mode))),
                'pilot_best_rank': str(pilot_best['rank_tested']),
                'pilot_best_verified_loss': pilot_best['best_verified_loss'],
                'pilot_best_max_abs_residual': pilot_best['best_max_abs_residual'],
                'pilot_any_exact_hit': str(any(row['verified_exact'] == 'True' for row in rows)),
                'pilot_exact_rank_upper_bound': verdict.get('exact_rank_upper_bound', ''),
                'rank19_beats_known_best': str(rank19_beats_known),
                'known_best_rank19_max_abs': '' if known_rank19 is None else scalar_to_str(known_rank19),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )

    write_csv(
        EXPORTS / 'step79_basis_rotated_results.csv',
        result_rows,
        ['rotation_id', 'family', 'seed', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'verified_exact', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step79_basis_rotated_summary.csv',
        summary_rows,
        ['rotation_id', 'family', 'seed', 'public_rank23_calibration_residual', 'cond_a', 'cond_b', 'cond_c', 'max_condition_number', 'pilot_best_rank', 'pilot_best_verified_loss', 'pilot_best_max_abs_residual', 'pilot_any_exact_hit', 'pilot_exact_rank_upper_bound', 'rank19_beats_known_best', 'known_best_rank19_max_abs', 'provenance'],
    )
    write_text(EXPORTS / 'step79_basis_rotated_report.md', markdown_report(summary_rows))
    print(f'Step 79 complete in {time.time() - start:.2f}s', flush=True)


if __name__ == '__main__':
    main()