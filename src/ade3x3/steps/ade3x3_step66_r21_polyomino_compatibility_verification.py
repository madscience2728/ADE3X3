"""
ade3x3_step66_r21_polyomino_compatibility_verification.py

Step 66: R=21 Polyomino Compatibility Verification.

This step directly tests whether explicit 21-term polyomino tilings can be
assembled into valid full 3x3 matrix multiplication algorithms. The workflow is:

1. Recompute raw float64 rank-1 terms for the tetromino pieces in the candidate
   21-cost tilings using L-BFGS on the reduced piece tensor.
2. Embed those terms back into the full 3x3 output grid with gamma support only
   on the piece outputs.
3. Measure per-term leakage, per-piece masked-tensor residual, and the global
   729-equation residual of the assembled 21-term algorithm.
4. If a tiling verifies, export the full coefficient table as a candidate exact
   rank-21 algorithm and stop. Additional fallback tasks only run if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

EXPORTS = Path('outputs/exports')
EXACT_TOL = 1e-10
OPTIMIZER_TOL = 1e-18
MAXITER = 400
RESTARTS = 1000
PATIENCE = 5

GRID_CELLS = tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3))
CELL_TO_INDEX = {cell: idx for idx, cell in enumerate(GRID_CELLS)}

TILING_DEFS = {
    'A': {
        'monomino': ((0, 0),),
        's_z_tetromino': ((0, 1), (0, 2), (1, 0), (1, 1)),
        'l_tetromino': ((1, 2), (2, 0), (2, 1), (2, 2)),
    },
    'B': {
        'monomino': ((0, 0),),
        's_z_tetromino': ((0, 1), (1, 0), (1, 1), (2, 0)),
        'l_tetromino': ((0, 2), (1, 2), (2, 1), (2, 2)),
    },
}


@dataclass(frozen=True)
class ReducedTerm:
    alpha: np.ndarray
    beta: np.ndarray
    gamma_reduced: np.ndarray


@dataclass(frozen=True)
class FullTerm:
    tiling_id: str
    piece_id: str
    term_index: int
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray


def log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {message}', flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Step 66 rank-21 polyomino compatibility verification.')
    parser.add_argument('--restarts', type=int, default=RESTARTS, help='Restart count for each numerical tetromino decomposition.')
    parser.add_argument('--tilings', nargs='*', default=['A', 'B'], help='Which candidate tilings to verify.')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1), help='Worker count for parallel restart scans.')
    parser.add_argument('--fresh-piece-check', action='store_true', help='Run a direct rank scan for a single reduced output piece instead of the full tiling workflow.')
    parser.add_argument('--piece-cells', default='', help='Semicolon-separated output cells such as "0,0;0,1;1,0;2,1" for --fresh-piece-check.')
    parser.add_argument('--ranks', default='9,10,11,12', help='Comma-separated ranks to test for --fresh-piece-check.')
    parser.add_argument('--fresh-label', default='', help='Optional label used in fresh-piece-check export filenames.')
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Wrote {len(rows)} rows -> {path}')


def write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f'  Wrote text -> {path}')


def cells_to_str(cells: tuple[tuple[int, int], ...]) -> str:
    return '; '.join(f'({row_idx},{col_idx})' for row_idx, col_idx in cells)


def float_matrix_to_json(matrix: np.ndarray) -> str:
    return json.dumps(matrix.astype(float).tolist(), separators=(',', ':'))


def vector_to_json(vector: np.ndarray) -> str:
    return json.dumps(vector.astype(float).tolist(), separators=(',', ':'))


def matrix_multiplication_tensor() -> np.ndarray:
    tensor = np.zeros((9, 9, 9), dtype=np.float64)
    for row_idx in range(3):
        for sum_idx in range(3):
            for col_idx in range(3):
                a_idx = 3 * row_idx + sum_idx
                b_idx = 3 * sum_idx + col_idx
                c_idx = 3 * row_idx + col_idx
                tensor[a_idx, b_idx, c_idx] = 1.0
    return tensor


def reduced_output_tensor(selected_outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    tensor = np.zeros((9, 9, len(selected_outputs)), dtype=np.float64)
    for output_idx, (row_idx, col_idx) in enumerate(selected_outputs):
        for sum_idx in range(3):
            a_idx = 3 * row_idx + sum_idx
            b_idx = 3 * sum_idx + col_idx
            tensor[a_idx, b_idx, output_idx] = 1.0
    return tensor


def masked_output_tensor(selected_outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    full = matrix_multiplication_tensor()
    mask = np.zeros(9, dtype=np.float64)
    for cell in selected_outputs:
        mask[CELL_TO_INDEX[cell]] = 1.0
    return full * mask.reshape(1, 1, 9)


def numeric_rank(matrix: np.ndarray, tol: float | None = None) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    if tol is None:
        tol = max(matrix.shape) * np.finfo(float).eps * float(np.max(singular_values)) * 10.0
    return int(np.sum(singular_values > tol))


def flattening_ranks(tensor: np.ndarray) -> tuple[int, int, int, int]:
    rank_a = numeric_rank(tensor.reshape(tensor.shape[0], -1))
    rank_b = numeric_rank(np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1))
    rank_c = numeric_rank(np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1))
    return rank_a, rank_b, rank_c, max(rank_a, rank_b, rank_c)


def pack_factors(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([alpha.reshape(-1), beta.reshape(-1), gamma.reshape(-1)])


def unpack_factors(vector: np.ndarray, rank: int, output_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split1 = rank * 9
    split2 = 2 * rank * 9
    alpha = vector[:split1].reshape(rank, 9)
    beta = vector[split1:split2].reshape(rank, 9)
    gamma = vector[split2:].reshape(rank, output_dim)
    return alpha, beta, gamma


def cp_objective(vector: np.ndarray, target: np.ndarray, rank: int) -> tuple[float, np.ndarray]:
    output_dim = target.shape[2]
    alpha, beta, gamma = unpack_factors(vector, rank, output_dim)
    approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    residual = approx - target
    loss = 0.5 * float(np.sum(residual * residual))
    grad_alpha = np.einsum('abc,rb,rc->ra', residual, beta, gamma, optimize=True)
    grad_beta = np.einsum('abc,ra,rc->rb', residual, alpha, gamma, optimize=True)
    grad_gamma = np.einsum('abc,ra,rb->rc', residual, alpha, beta, optimize=True)
    return loss, pack_factors(grad_alpha, grad_beta, grad_gamma)


def random_initialization(rank: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    alpha = 0.25 * rng.standard_normal((rank, 9))
    beta = 0.25 * rng.standard_normal((rank, 9))
    gamma = 0.25 * rng.standard_normal((rank, output_dim))
    return pack_factors(alpha, beta, gamma)


def parse_cells_arg(text: str) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for chunk in text.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        row_text, col_text = [part.strip() for part in chunk.split(',')]
        cells.append((int(row_text), int(col_text)))
    return tuple(sorted(cells))


def parse_ranks_arg(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(',') if part.strip()]


def slugify_label(text: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text)
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'fresh_piece'


def single_restart_worker(payload: dict) -> dict:
    target = np.array(payload['target'], dtype=np.float64)
    rank = int(payload['rank'])
    seed = int(payload['seed'])
    x0 = random_initialization(rank, target.shape[2], seed=seed)
    result = minimize(
        lambda x: cp_objective(x, target, rank),
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': MAXITER, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50},
    )
    alpha, beta, gamma = unpack_factors(result.x, rank, target.shape[2])
    approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    residual = approx - target
    verified_loss = float(np.sum(residual * residual))
    max_abs = float(np.max(np.abs(residual)))
    return {
        'seed': seed,
        'verified_loss': verified_loss,
        'max_abs_residual': max_abs,
        'vector': result.x,
        'optimizer_message': str(result.message),
    }


def run_parallel_restart_batch(piece_id: str, target: np.ndarray, rank: int, restarts: int, workers: int) -> tuple[np.ndarray, list[dict], dict]:
    seeds = [100000 * rank + 701 * restart + sum(ord(ch) for ch in piece_id) for restart in range(restarts)]
    payloads = [{'target': target.tolist(), 'rank': rank, 'seed': seed} for seed in seeds]
    best_loss = float('inf')
    best_max_abs = float('inf')
    best_vector: np.ndarray | None = None
    exact_hits = 0
    restart_rows: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(single_restart_worker, payload): payload['seed'] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            verified_loss = float(result['verified_loss'])
            max_abs = float(result['max_abs_residual'])
            restart_rows.append({
                'piece_id': piece_id,
                'restart_index': completed,
                'seed': result['seed'],
                'verified_loss': f'{verified_loss:.16e}',
                'max_abs_residual': f'{max_abs:.16e}',
                'optimizer_message': result['optimizer_message'],
                'provenance': 'MEASURED_FROM_CODE',
            })
            if verified_loss < best_loss:
                best_loss = verified_loss
                best_max_abs = max_abs
                best_vector = np.array(result['vector'], dtype=np.float64)
            if verified_loss < EXACT_TOL:
                exact_hits += 1
            if completed == 1 or completed % 50 == 0 or verified_loss < EXACT_TOL:
                log(
                    f'{piece_id}: completed {completed}/{restarts} restarts in parallel, '
                    f'best_loss={best_loss:.3e}, best_max_abs={best_max_abs:.3e}, exact_hits={exact_hits}'
                )

    require(best_vector is not None, f'{piece_id}: parallel restart batch produced no candidate vectors.')
    summary = {
        'piece_id': piece_id,
        'rank': rank,
        'restart_count_used': len(restart_rows),
        'best_verified_loss': best_loss,
        'best_max_abs_residual': best_max_abs,
        'exact_hit_count': exact_hits,
        'verified_exact': bool(best_loss < EXACT_TOL),
    }
    restart_rows.sort(key=lambda row: row['seed'])
    return best_vector, restart_rows, summary


def optimize_reduced_piece(piece_id: str, outputs: tuple[tuple[int, int], ...], rank: int, restarts: int, workers: int) -> tuple[list[ReducedTerm], dict, list[dict]]:
    target = reduced_output_tensor(outputs)
    best_vector, restart_rows, summary = run_parallel_restart_batch(piece_id, target, rank, restarts, workers)
    alpha, beta, gamma = unpack_factors(best_vector, rank, target.shape[2])
    terms = [ReducedTerm(alpha=alpha[idx].copy(), beta=beta[idx].copy(), gamma_reduced=gamma[idx].copy()) for idx in range(rank)]
    return terms, summary, restart_rows


def fresh_piece_rank_scan(piece_id: str, outputs: tuple[tuple[int, int], ...], ranks: list[int], restarts: int, workers: int) -> tuple[list[dict], dict]:
    target = reduced_output_tensor(outputs)
    rank_rows: list[dict] = []
    best_rank = ''
    best_loss = float('inf')
    best_abs = float('inf')
    exact_rank = ''
    for rank in ranks:
        log(f'{piece_id}: fresh piece check at rank {rank} with {restarts} parallel restarts on {workers} workers')
        _, restart_rows, summary = run_parallel_restart_batch(f'{piece_id}_R{rank}', target, rank, restarts, workers)
        rank_rows.append({
            'piece_id': piece_id,
            'piece_outputs': cells_to_str(outputs),
            'rank_tested': rank,
            'restarts': len(restart_rows),
            'best_verified_loss': f"{summary['best_verified_loss']:.16e}",
            'best_max_abs_residual': f"{summary['best_max_abs_residual']:.16e}",
            'exact_hit_count': summary['exact_hit_count'],
            'verified_exact': str(summary['verified_exact']),
            'provenance': 'MEASURED_FROM_CODE',
        })
        if summary['best_verified_loss'] < best_loss:
            best_loss = summary['best_verified_loss']
            best_abs = summary['best_max_abs_residual']
            best_rank = str(rank)
        if summary['verified_exact'] and not exact_rank:
            exact_rank = str(rank)
            break
    verdict = {
        'piece_id': piece_id,
        'piece_outputs': cells_to_str(outputs),
        'flattening_rank_A_BC': '',
        'flattening_rank_B_AC': '',
        'flattening_rank_C_AB': '',
        'flattening_lower_bound': '',
        'best_rank_tested': best_rank,
        'best_verified_loss': f'{best_loss:.16e}',
        'best_max_abs_residual': f'{best_abs:.16e}',
        'exact_rank_upper_bound': exact_rank,
        'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
    }
    return rank_rows, verdict


def embed_reduced_term(tiling_id: str, piece_id: str, term_index: int, outputs: tuple[tuple[int, int], ...], term: ReducedTerm) -> FullTerm:
    gamma = np.zeros(9, dtype=np.float64)
    for local_idx, cell in enumerate(outputs):
        gamma[CELL_TO_INDEX[cell]] = term.gamma_reduced[local_idx]
    return FullTerm(
        tiling_id=tiling_id,
        piece_id=piece_id,
        term_index=term_index,
        alpha=term.alpha.copy(),
        beta=term.beta.copy(),
        gamma=gamma,
    )


def monomino_terms(tiling_id: str, piece_id: str, output_cell: tuple[int, int]) -> list[FullTerm]:
    row_idx, col_idx = output_cell
    gamma = np.zeros(9, dtype=np.float64)
    gamma[CELL_TO_INDEX[output_cell]] = 1.0
    terms: list[FullTerm] = []
    for sum_idx in range(3):
        alpha = np.zeros(9, dtype=np.float64)
        beta = np.zeros(9, dtype=np.float64)
        alpha[3 * row_idx + sum_idx] = 1.0
        beta[3 * sum_idx + col_idx] = 1.0
        terms.append(FullTerm(tiling_id=tiling_id, piece_id=piece_id, term_index=sum_idx + 1, alpha=alpha, beta=beta, gamma=gamma.copy()))
    return terms


def reconstruct_from_full_terms(terms: list[FullTerm]) -> np.ndarray:
    if not terms:
        return np.zeros((9, 9, 9), dtype=np.float64)
    alpha = np.stack([term.alpha for term in terms], axis=0)
    beta = np.stack([term.beta for term in terms], axis=0)
    gamma = np.stack([term.gamma for term in terms], axis=0)
    return np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)


def count_failed_equations(residual: np.ndarray, threshold: float) -> tuple[int, list[tuple[int, int, int, float]]]:
    failing: list[tuple[int, int, int, float]] = []
    for a_idx in range(9):
        for b_idx in range(9):
            for c_idx in range(9):
                value = float(residual[a_idx, b_idx, c_idx])
                if abs(value) > threshold:
                    failing.append((a_idx, b_idx, c_idx, value))
    return len(failing), failing


def output_cell_from_c_index(c_idx: int) -> tuple[int, int]:
    return divmod(c_idx, 3)


def failure_piece_label(cell: tuple[int, int], tiling: dict[str, tuple[tuple[int, int], ...]]) -> str:
    for piece_id, outputs in tiling.items():
        if cell in outputs:
            return piece_id
    return 'outside'


def verify_piece_terms(outputs: tuple[tuple[int, int], ...], terms: list[FullTerm]) -> dict:
    target = masked_output_tensor(outputs)
    reconstruction = reconstruct_from_full_terms(terms)
    residual = target - reconstruction
    failure_count, failing = count_failed_equations(residual, EXACT_TOL)
    return {
        'masked_max_abs': float(np.max(np.abs(residual))),
        'masked_fro_norm': float(np.linalg.norm(residual)),
        'masked_failure_count': failure_count,
        'failing_equations': failing[:20],
    }


def verify_tiling(tiling_id: str, tiling: dict[str, tuple[tuple[int, int], ...]], terms: list[FullTerm]) -> dict:
    target = matrix_multiplication_tensor()
    reconstruction = reconstruct_from_full_terms(terms)
    residual = target - reconstruction
    failure_count, failing = count_failed_equations(residual, EXACT_TOL)
    failing_rows = []
    for a_idx, b_idx, c_idx, value in failing[:50]:
        output_cell = output_cell_from_c_index(c_idx)
        failing_rows.append({
            'tiling_id': tiling_id,
            'a_index': a_idx,
            'b_index': b_idx,
            'c_index': c_idx,
            'output_entry': f'({output_cell[0]},{output_cell[1]})',
            'piece_id': failure_piece_label(output_cell, tiling),
            'residual_value': f'{value:.16e}',
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })
    return {
        'tiling_id': tiling_id,
        'term_count': len(terms),
        'residual_max_abs': float(np.max(np.abs(residual))),
        'residual_fro_norm': float(np.linalg.norm(residual)),
        'failure_count': failure_count,
        'verified_exact': bool(float(np.max(np.abs(residual))) < EXACT_TOL),
        'failing_rows': failing_rows,
    }


def leakage_rows_for_piece(tiling_id: str, piece_id: str, outputs: tuple[tuple[int, int], ...], terms: list[FullTerm]) -> list[dict]:
    piece_indices = {CELL_TO_INDEX[cell] for cell in outputs}
    rows: list[dict] = []
    for term in terms:
        outside = np.array([term.gamma[idx] for idx in range(9) if idx not in piece_indices], dtype=np.float64)
        rows.append({
            'tiling_id': tiling_id,
            'piece_id': piece_id,
            'term_index': term.term_index,
            'piece_outputs': cells_to_str(outputs),
            'gamma_outside_l1': f'{float(np.sum(np.abs(outside))):.16e}',
            'gamma_outside_l2': f'{float(np.linalg.norm(outside)):.16e}',
            'gamma_outside_max_abs': f'{float(np.max(np.abs(outside))) if outside.size else 0.0:.16e}',
            'has_leakage': str(bool(outside.size and float(np.max(np.abs(outside))) > EXACT_TOL)),
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })
    return rows


def build_markdown(summary_rows: list[dict], piece_rows: list[dict], tiling_rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 66: R=21 Polyomino Compatibility Verification')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Tiling Verification Summary')
    w('')
    w('| tiling | term count | max residual | Frobenius residual | failing equations | verified |')
    w('|--------|------------|--------------|--------------------|-------------------|----------|')
    for row in tiling_rows:
        w(f"| {row['tiling_id']} | {row['term_count']} | {row['residual_max_abs']} | {row['residual_fro_norm']} | {row['failure_count']} | {row['verified_exact']} |")
    w('')
    w('## Piece Decomposition Summary')
    w('')
    w('| tiling | piece | outputs | rank | best reduced loss | best reduced max abs | masked full max abs | masked failures | verified exact |')
    w('|--------|-------|---------|------|-------------------|----------------------|---------------------|----------------|----------------|')
    for row in piece_rows:
        w(f"| {row['tiling_id']} | {row['piece_id']} | {row['piece_outputs']} | {row['rank_used']} | {row['best_reduced_loss']} | {row['best_reduced_max_abs']} | {row['masked_full_max_abs']} | {row['masked_failure_count']} | {row['verified_exact']} |")
    w('')
    verified_rows = [row for row in tiling_rows if row['verified_exact'] == 'True']
    if verified_rows:
        w('## Main Result')
        w('')
        for row in verified_rows:
            w(f"- Tiling {row['tiling_id']} verifies a 21-term full 3x3 algorithm with max residual {row['residual_max_abs']}.")
    else:
        w('## Main Result')
        w('')
        w('- No candidate tiling verified exactly in the current run.')
    w('')
    for row in summary_rows:
        w(f"- {row['summary_name']}: {row['summary_value']}")
    return '\n'.join(lines)


def build_fresh_piece_markdown(verdict: dict, rank_rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 66: Fresh Piece Ground-Truth Scan')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w(f"- piece outputs: {verdict['piece_outputs']}")
    w(f"- flattening ranks: A|BC={verdict['flattening_rank_A_BC']}, B|AC={verdict['flattening_rank_B_AC']}, C|AB={verdict['flattening_rank_C_AB']}, LB={verdict['flattening_lower_bound']}")
    w(f"- best rank tested: {verdict['best_rank_tested']}")
    w(f"- best verified loss: {verdict['best_verified_loss']}")
    w(f"- exact rank upper bound: {verdict['exact_rank_upper_bound'] or 'none found'}")
    w('')
    w('| rank | restarts | best verified loss | best max abs residual | exact hits | verified exact |')
    w('|------|----------|--------------------|-----------------------|------------|----------------|')
    for row in rank_rows:
        w(f"| {row['rank_tested']} | {row['restarts']} | {row['best_verified_loss']} | {row['best_max_abs_residual']} | {row['exact_hit_count']} | {row['verified_exact']} |")
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    log('Step 66 start')

    if args.fresh_piece_check:
        outputs = parse_cells_arg(args.piece_cells)
        require(bool(outputs), '--fresh-piece-check requires --piece-cells.')
        ranks = parse_ranks_arg(args.ranks)
        require(bool(ranks), '--fresh-piece-check requires at least one rank in --ranks.')
        label = slugify_label(args.fresh_label or cells_to_str(outputs))
        rank_a, rank_b, rank_c, rank_lb = flattening_ranks(reduced_output_tensor(outputs))
        rank_rows, verdict = fresh_piece_rank_scan('fresh_piece_check', outputs, ranks, args.restarts, args.workers)
        verdict['flattening_rank_A_BC'] = str(rank_a)
        verdict['flattening_rank_B_AC'] = str(rank_b)
        verdict['flattening_rank_C_AB'] = str(rank_c)
        verdict['flattening_lower_bound'] = str(rank_lb)
        write_csv(EXPORTS / f'step66_{label}_rank_scan.csv', rank_rows, ['piece_id', 'piece_outputs', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'provenance'])
        write_csv(EXPORTS / f'step66_{label}_verdict.csv', [verdict], ['piece_id', 'piece_outputs', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'best_rank_tested', 'best_verified_loss', 'best_max_abs_residual', 'exact_rank_upper_bound', 'provenance'])
        write_text(EXPORTS / f'step66_{label}_ground_truth.md', build_fresh_piece_markdown(verdict, rank_rows))
        log(
            f"Fresh piece verdict: outputs={verdict['piece_outputs']}, LB={verdict['flattening_lower_bound']}, "
            f"best_rank={verdict['best_rank_tested']}, exact_rank_upper={verdict['exact_rank_upper_bound'] or 'none'}, "
            f"best_loss={verdict['best_verified_loss']}"
        )
        return

    requested_tilings = [tiling_id for tiling_id in args.tilings if tiling_id in TILING_DEFS]
    require(bool(requested_tilings), 'No valid tiling identifiers were requested. Use A and/or B.')

    piece_export_rows: list[dict] = []
    restart_rows: list[dict] = []
    leakage_rows: list[dict] = []
    term_rows: list[dict] = []
    tiling_rows: list[dict] = []
    failure_rows: list[dict] = []
    summary_rows: list[dict] = []

    for tiling_id in requested_tilings:
        tiling = TILING_DEFS[tiling_id]
        log(f'Verifying tiling {tiling_id}')
        assembled_terms: list[FullTerm] = []

        mono_terms = monomino_terms(tiling_id, 'monomino', tiling['monomino'][0])
        assembled_terms.extend(mono_terms)
        mono_piece_check = verify_piece_terms(tiling['monomino'], mono_terms)
        piece_export_rows.append({
            'tiling_id': tiling_id,
            'piece_id': 'monomino',
            'piece_outputs': cells_to_str(tiling['monomino']),
            'rank_used': 3,
            'restart_count_used': 0,
            'best_reduced_loss': f'{0.0:.16e}',
            'best_reduced_max_abs': f'{0.0:.16e}',
            'masked_full_max_abs': f"{mono_piece_check['masked_max_abs']:.16e}",
            'masked_full_fro_norm': f"{mono_piece_check['masked_fro_norm']:.16e}",
            'masked_failure_count': mono_piece_check['masked_failure_count'],
            'exact_hit_count': 1,
            'verified_exact': str(bool(mono_piece_check['masked_max_abs'] < EXACT_TOL)),
            'unconstrained_rank': 3,
            'min_leakage_rank': 3,
            'constrained_rank': 3,
            'provenance': 'EXACT_DERIVED',
        })
        leakage_rows.extend(leakage_rows_for_piece(tiling_id, 'monomino', tiling['monomino'], mono_terms))

        for piece_id in ('s_z_tetromino', 'l_tetromino'):
            outputs = tiling[piece_id]
            log(f'{tiling_id}/{piece_id}: recomputing rank-9 reduced decomposition with up to {args.restarts} parallel restarts on {args.workers} workers')
            reduced_terms, reduced_summary, piece_restart_rows = optimize_reduced_piece(f'{tiling_id}_{piece_id}', outputs, rank=9, restarts=args.restarts, workers=args.workers)
            full_terms = [embed_reduced_term(tiling_id, piece_id, idx + 1, outputs, term) for idx, term in enumerate(reduced_terms)]
            piece_check = verify_piece_terms(outputs, full_terms)
            piece_export_rows.append({
                'tiling_id': tiling_id,
                'piece_id': piece_id,
                'piece_outputs': cells_to_str(outputs),
                'rank_used': 9,
                'restart_count_used': reduced_summary['restart_count_used'],
                'best_reduced_loss': f"{reduced_summary['best_verified_loss']:.16e}",
                'best_reduced_max_abs': f"{reduced_summary['best_max_abs_residual']:.16e}",
                'masked_full_max_abs': f"{piece_check['masked_max_abs']:.16e}",
                'masked_full_fro_norm': f"{piece_check['masked_fro_norm']:.16e}",
                'masked_failure_count': piece_check['masked_failure_count'],
                'exact_hit_count': reduced_summary['exact_hit_count'],
                'verified_exact': str(bool(piece_check['masked_max_abs'] < EXACT_TOL)),
                'unconstrained_rank': 9,
                'min_leakage_rank': 9,
                'constrained_rank': 9,
                'provenance': 'MEASURED_FROM_CODE',
            })
            restart_rows.extend(piece_restart_rows)
            leakage_rows.extend(leakage_rows_for_piece(tiling_id, piece_id, outputs, full_terms))
            assembled_terms.extend(full_terms)

        require(len(assembled_terms) == 21, f'Tiling {tiling_id} should assemble 21 terms, found {len(assembled_terms)}.')
        tiling_check = verify_tiling(tiling_id, tiling, assembled_terms)
        tiling_rows.append({
            'tiling_id': tiling_id,
            'term_count': tiling_check['term_count'],
            'residual_max_abs': f"{tiling_check['residual_max_abs']:.16e}",
            'residual_fro_norm': f"{tiling_check['residual_fro_norm']:.16e}",
            'failure_count': tiling_check['failure_count'],
            'verified_exact': str(tiling_check['verified_exact']),
            'pieces': ' | '.join(f'{piece_id}:{cells_to_str(outputs)}' for piece_id, outputs in tiling.items()),
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })
        failure_rows.extend(tiling_check['failing_rows'])

        for term in assembled_terms:
            term_rows.append({
                'tiling_id': term.tiling_id,
                'piece_id': term.piece_id,
                'term_index': term.term_index,
                'alpha_vector': vector_to_json(term.alpha),
                'beta_vector': vector_to_json(term.beta),
                'gamma_vector': vector_to_json(term.gamma),
                'alpha_matrix': float_matrix_to_json(term.alpha.reshape(3, 3)),
                'beta_matrix': float_matrix_to_json(term.beta.reshape(3, 3)),
                'gamma_matrix': float_matrix_to_json(term.gamma.reshape(3, 3)),
                'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
            })

    any_verified = any(row['verified_exact'] == 'True' for row in tiling_rows)
    summary_rows.extend([
        {'summary_name': 'candidate_tilings_checked', 'summary_value': str(len(requested_tilings)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of explicit 21-cost tilings checked.'},
        {'summary_name': 'any_rank21_verification', 'summary_value': str(any_verified), 'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE', 'note': 'Whether any checked tiling verified as an exact 21-term full algorithm.'},
    ])
    for row in tiling_rows:
        summary_rows.append({'summary_name': f"tiling_{row['tiling_id']}_max_residual", 'summary_value': row['residual_max_abs'], 'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE', 'note': 'Maximum absolute full-tensor residual for this tiling.'})
        summary_rows.append({'summary_name': f"tiling_{row['tiling_id']}_failure_count", 'summary_value': str(row['failure_count']), 'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE', 'note': 'Number of failing equations above 1e-10 for this tiling.'})

    log('Writing Step 66 exports')
    write_csv(EXPORTS / 'step66_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step66_piece_decompositions.csv', piece_export_rows, ['tiling_id', 'piece_id', 'piece_outputs', 'rank_used', 'restart_count_used', 'best_reduced_loss', 'best_reduced_max_abs', 'masked_full_max_abs', 'masked_full_fro_norm', 'masked_failure_count', 'exact_hit_count', 'verified_exact', 'unconstrained_rank', 'min_leakage_rank', 'constrained_rank', 'provenance'])
    write_csv(EXPORTS / 'step66_piece_restart_log.csv', restart_rows, ['piece_id', 'restart_index', 'seed', 'verified_loss', 'max_abs_residual', 'optimizer_message', 'provenance'])
    write_csv(EXPORTS / 'step66_leakage_report.csv', leakage_rows, ['tiling_id', 'piece_id', 'term_index', 'piece_outputs', 'gamma_outside_l1', 'gamma_outside_l2', 'gamma_outside_max_abs', 'has_leakage', 'provenance'])
    write_csv(EXPORTS / 'step66_global_terms.csv', term_rows, ['tiling_id', 'piece_id', 'term_index', 'alpha_vector', 'beta_vector', 'gamma_vector', 'alpha_matrix', 'beta_matrix', 'gamma_matrix', 'provenance'])
    write_csv(EXPORTS / 'step66_tiling_verification.csv', tiling_rows, ['tiling_id', 'term_count', 'residual_max_abs', 'residual_fro_norm', 'failure_count', 'verified_exact', 'pieces', 'provenance'])
    write_csv(EXPORTS / 'step66_failure_equations.csv', failure_rows, ['tiling_id', 'a_index', 'b_index', 'c_index', 'output_entry', 'piece_id', 'residual_value', 'provenance'])
    write_text(EXPORTS / 'step66_r21_polyomino_compatibility_verification.md', build_markdown(summary_rows, piece_export_rows, tiling_rows))

    verified_tilings = [row['tiling_id'] for row in tiling_rows if row['verified_exact'] == 'True']
    if verified_tilings:
        log(f'MAJOR RESULT: verified rank-21 tilings: {", ".join(verified_tilings)}')
    else:
        log('No exact rank-21 tiling verified in this run')
    log('Step 66 complete')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as exc:
        log(f'Step 66 failed: {exc}')
        traceback.print_exc()
        raise