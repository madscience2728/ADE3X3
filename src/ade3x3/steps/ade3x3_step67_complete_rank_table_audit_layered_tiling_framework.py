"""
ade3x3_step67_complete_rank_table_audit_layered_tiling_framework.py

Step 67: Complete Rank Table Audit + Layered Tiling Framework.

This step repairs the remaining unstable Step 65 tetromino/L-tromino rank rows,
rebuilds the flat tiling search using only corrected verified piece costs, and
analyzes the AlphaTensor rank-23 factorization through the Step 51/52 layered
signal-vs-nuisance decomposition.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    build_mode_matrices,
    load_public_rank23_terms,
)
from src.ade3x3.steps.ade3x3_step65_polyomino_subtensor_ranks_tiling_analysis import (
    build_piece_classes,
    canonical_cells,
    classify_cells,
    enumerate_general_tilings,
    enumerate_piece_instances,
)

EXPORTS = Path('outputs/exports')
EXACT_TOL = 1e-10
MAXITER = 400
DEFAULT_WORKERS = max(8, os.cpu_count() or 8)
PART_A_RANKS = list(range(6, 15))
PART_A_RESTARTS = 500

GRID_CELLS = tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3))
PART_A_TARGET_IDS = [
    'L_tetromino_class_01',
    'S_Z_tetromino_class_01',
    'T_tetromino_class_01',
    'T_tetromino_class_02',
    'L_tromino_class_01',
]

KNOWN_VERIFIED_RANKS = {
    'monomino_class_01': ('3', 'known_exact'),
    'domino_row_class_01': ('6', 'known_exact'),
    'domino_col_class_01': ('6', 'known_exact'),
    'tromino_row_class_01': ('9', 'known_exact'),
    'tromino_col_class_01': ('9', 'known_exact'),
    'square_tetromino_class_01': ('11', 'known_exact'),
    'L_tetromino_class_02': ('12', 'step66_fresh_scan'),
    'S_Z_tetromino_class_02': ('12', 'step66_fresh_scan'),
    'P4_anti_diagonal_missing_class_01': ('18', 'step62_exact_upper_bound'),
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{timestamp}] {message}', flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def cells_to_str(cells: tuple[tuple[int, int], ...]) -> str:
    return '; '.join(f'({row_idx},{col_idx})' for row_idx, col_idx in cells)


def reduced_output_tensor(selected_outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    tensor = np.zeros((9, 9, len(selected_outputs)), dtype=np.float64)
    for output_idx, (row_idx, col_idx) in enumerate(selected_outputs):
        for sum_idx in range(3):
            a_idx = 3 * row_idx + sum_idx
            b_idx = 3 * sum_idx + col_idx
            tensor[a_idx, b_idx, output_idx] = 1.0
    return tensor


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


def single_restart_worker(payload: dict) -> dict:
    outputs = tuple(tuple(cell) for cell in payload['outputs'])
    tensor = reduced_output_tensor(outputs)
    rank = int(payload['rank'])
    seed = int(payload['seed'])
    x0 = random_initialization(rank, tensor.shape[2], seed=seed)
    result = minimize(
        lambda x: cp_objective(x, tensor, rank),
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': MAXITER, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50},
    )
    alpha, beta, gamma = unpack_factors(result.x, rank, tensor.shape[2])
    approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    residual = approx - tensor
    return {
        'piece_id': payload['piece_id'],
        'rank': rank,
        'seed': seed,
        'verified_loss': float(np.sum(residual * residual)),
        'best_max_abs_residual': float(np.max(np.abs(residual))),
    }


def run_piece_audit_rounds(piece_targets: list[tuple[str, str, tuple[tuple[int, int], ...]]], ranks: list[int], restarts: int, workers: int) -> tuple[list[dict], list[dict]]:
    flattening_cache = {
        piece_id: flattening_ranks(reduced_output_tensor(outputs))
        for piece_id, _, outputs in piece_targets
    }
    active = {
        piece_id: {
            'family': family,
            'outputs': outputs,
            'min_rank': flattening_cache[piece_id][3],
        }
        for piece_id, family, outputs in piece_targets
    }
    verdict_rows: list[dict] = []
    restart_rows: list[dict] = []

    for rank in ranks:
        rank_active = {piece_id: info for piece_id, info in active.items() if rank >= info['min_rank']}
        if not active:
            break
        if not rank_active:
            continue
        log(f'Part A: launching rank {rank} round for {len(rank_active)} active pieces with {restarts} restarts each on {workers} workers')
        payloads = []
        for piece_id, info in rank_active.items():
            outputs = info['outputs']
            base_seed = sum(ord(ch) for ch in piece_id)
            for restart in range(restarts):
                payloads.append({
                    'piece_id': piece_id,
                    'outputs': outputs,
                    'rank': rank,
                    'seed': 100000 * rank + 701 * restart + base_seed,
                })

        aggregate = {
            piece_id: {
                'best_loss': float('inf'),
                'best_abs': float('inf'),
                'exact_hits': 0,
                'restart_count': 0,
            }
            for piece_id in rank_active
        }

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(single_restart_worker, payload) for payload in payloads]
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                piece_id = result['piece_id']
                bucket = aggregate[piece_id]
                bucket['restart_count'] += 1
                bucket['best_loss'] = min(bucket['best_loss'], float(result['verified_loss']))
                bucket['best_abs'] = min(bucket['best_abs'], float(result['best_max_abs_residual']))
                if float(result['verified_loss']) < EXACT_TOL:
                    bucket['exact_hits'] += 1
                restart_rows.append({
                    'piece_id': piece_id,
                    'rank_tested': rank,
                    'seed': result['seed'],
                    'verified_loss': f"{float(result['verified_loss']):.16e}",
                    'best_max_abs_residual': f"{float(result['best_max_abs_residual']):.16e}",
                    'provenance': 'MEASURED_FROM_CODE',
                })
                if completed == 1 or completed % 250 == 0:
                    status = '; '.join(
                        f"{pid}:best={aggregate[pid]['best_loss']:.3e},hits={aggregate[pid]['exact_hits']}"
                        for pid in sorted(aggregate)
                    )
                    log(f'Part A rank {rank}: completed {completed}/{len(payloads)} restart tasks | {status}')

        finished_ids: list[str] = []
        for piece_id, info in rank_active.items():
            family = info['family']
            outputs = info['outputs']
            rank_a, rank_b, rank_c, rank_lb = flattening_cache[piece_id]
            bucket = aggregate[piece_id]
            verified_exact = bucket['best_loss'] < EXACT_TOL
            verdict_rows.append({
                'piece_id': piece_id,
                'family': family,
                'piece_outputs': cells_to_str(outputs),
                'flattening_rank_A_BC': rank_a,
                'flattening_rank_B_AC': rank_b,
                'flattening_rank_C_AB': rank_c,
                'flattening_lower_bound': rank_lb,
                'rank_tested': rank,
                'restarts': bucket['restart_count'],
                'best_verified_loss': f"{bucket['best_loss']:.16e}",
                'best_max_abs_residual': f"{bucket['best_abs']:.16e}",
                'exact_hit_count': bucket['exact_hits'],
                'verified_exact': str(verified_exact),
                'exact_rank_upper_bound': str(rank) if verified_exact else '',
                'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
            })
            log(
                f"Part A verdict rank {rank} {piece_id}: best_loss={bucket['best_loss']:.3e}, "
                f"best_abs={bucket['best_abs']:.3e}, exact_hits={bucket['exact_hits']}"
            )
            if verified_exact:
                finished_ids.append(piece_id)

        for piece_id in finished_ids:
            active.pop(piece_id, None)

    return verdict_rows, restart_rows


def latest_exact_verdicts(verdict_rows: list[dict]) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for row in verdict_rows:
        if row['exact_rank_upper_bound']:
            resolved[row['piece_id']] = row
    return resolved


def load_step66_audits() -> dict[str, dict]:
    audits = {}
    file_map = {
        'L_tetromino_class_02': EXPORTS / 'step66_l_tetromino_class_02_verdict.csv',
        'S_Z_tetromino_class_02': EXPORTS / 'step66_s_z_tetromino_class_02_verdict.csv',
    }
    for class_id, path in file_map.items():
        rows = read_csv_rows(path)
        if rows:
            audits[class_id] = rows[0]
    return audits


def build_piece_class_lookup() -> dict[str, dict]:
    piece_instances = enumerate_piece_instances()
    classes, _ = build_piece_classes(piece_instances)
    return {
        cls.class_id: {
            'family': cls.family,
            'representative_entries': cells_to_str(cls.representative),
            'representative': cls.representative,
            'size': cls.size,
            'count_on_3x3': cls.instance_count,
            'symmetry_class_size': cls.symmetry_class_size,
            'row_profile': cls.row_profile,
            'col_profile': cls.col_profile,
        }
        for cls in classes
    }


def build_part_a_targets(piece_class_lookup: dict[str, dict]) -> list[tuple[str, str, tuple[tuple[int, int], ...]]]:
    targets = []
    for class_id in PART_A_TARGET_IDS:
        require(class_id in piece_class_lookup, f'Missing authoritative geometry for {class_id}.')
        info = piece_class_lookup[class_id]
        targets.append((class_id, info['family'], info['representative']))
    return targets


def build_corrected_rank_table(part_a_exact: dict[str, dict], piece_class_lookup: dict[str, dict]) -> tuple[list[dict], dict[str, int]]:
    base_rows = read_csv_rows(EXPORTS / 'step65_polyomino_rank_table.csv')
    require(bool(base_rows), 'Missing step65_polyomino_rank_table.csv needed for Step 67.')
    step66_audits = load_step66_audits()
    corrected_rows: list[dict] = []
    class_costs: dict[str, int] = {}

    for row in base_rows:
        class_id = row['class_id']
        verified_rank = ''
        verified_source = ''
        note = row['note']

        if class_id in KNOWN_VERIFIED_RANKS:
            verified_rank, verified_source = KNOWN_VERIFIED_RANKS[class_id]
        elif class_id in step66_audits:
            verified_rank = step66_audits[class_id]['exact_rank_upper_bound']
            verified_source = 'step66_fresh_scan'
        elif class_id in part_a_exact:
            verified_rank = part_a_exact[class_id]['exact_rank_upper_bound']
            verified_source = 'step67_partA_fresh_scan'

        if class_id == 'S_Z_tetromino_class_02':
            note = 'Step 66 fresh scan corrects the corrupted Step 65 recovery export: exact numerical upper bound 12.'
        elif class_id == 'L_tetromino_class_02':
            note = 'Step 66 fresh scan corrects the corrupted Step 65 recovery export: exact numerical upper bound 12.'
        elif class_id in part_a_exact:
            note = f'Step 67 fresh scan verified exact numerical upper bound {verified_rank}.'

        corrected = dict(row)
        if class_id in piece_class_lookup:
            corrected.update({
                'family': piece_class_lookup[class_id]['family'],
                'representative_entries': piece_class_lookup[class_id]['representative_entries'],
                'size': piece_class_lookup[class_id]['size'],
                'count_on_3x3': piece_class_lookup[class_id]['count_on_3x3'],
                'symmetry_class_size': piece_class_lookup[class_id]['symmetry_class_size'],
                'row_profile': piece_class_lookup[class_id]['row_profile'],
                'col_profile': piece_class_lookup[class_id]['col_profile'],
            })
        corrected['verified_rank_upper_bound'] = verified_rank
        corrected['verified_rank_source'] = verified_source
        corrected['corrected_note'] = note
        corrected_rows.append(corrected)
        if verified_rank:
            class_costs[class_id] = int(verified_rank)

    return corrected_rows, class_costs


def build_piece_to_class_map(piece_class_lookup: dict[str, dict]) -> dict[tuple[str, tuple[tuple[int, int], ...]], str]:
    mapping: dict[tuple[str, tuple[tuple[int, int], ...]], str] = {}
    for class_id, info in piece_class_lookup.items():
        mapping[(info['family'], canonical_cells(info['representative']))] = class_id
    return mapping


def tiling_shape_signature(tiling: tuple[tuple[tuple[int, int], ...], ...]) -> str:
    counts = Counter(classify_cells(piece) for piece in tiling)
    return ' + '.join(f'{count}x {family}' for family, count in sorted(counts.items()))


def compute_corrected_tilings(piece_class_lookup: dict[str, dict], class_costs: dict[str, int]) -> tuple[list[dict], dict, list[dict]]:
    piece_instances = enumerate_piece_instances()
    allowed_families = {'monomino', 'domino_row', 'domino_col', 'L_tromino', 'tromino_row', 'tromino_col', 'square_tetromino', 'L_tetromino', 'T_tetromino', 'S_Z_tetromino', 'P4_anti_diagonal_missing'}
    tilings = enumerate_general_tilings(piece_instances, allowed_families)
    piece_to_class = build_piece_to_class_map(piece_class_lookup)

    rows: list[dict] = []
    for tiling_idx, tiling in enumerate(tilings, start=1):
        class_ids = []
        valid = True
        for piece in tiling:
            key = (classify_cells(piece), canonical_cells(piece))
            class_id = piece_to_class.get(key)
            if not class_id or class_id not in class_costs:
                valid = False
                break
            class_ids.append(class_id)
        if not valid:
            continue
        total_cost = sum(class_costs[class_id] for class_id in class_ids)
        rows.append({
            'tiling_id': tiling_idx,
            'piece_count': len(tiling),
            'tiling_signature': tiling_shape_signature(tiling),
            'pieces': ' | '.join(cells_to_str(piece) for piece in tiling),
            'class_ids': ';'.join(class_ids),
            'total_verified_cost': total_cost,
            'beats_23': str(total_cost < 23),
            'provenance': 'EXACT_DERIVED',
        })

    require(bool(rows), 'No tilings had fully verified piece costs after the Step 67 audit.')
    best_row = min(rows, key=lambda row: (int(row['total_verified_cost']), row['tiling_signature'], row['pieces']))
    low_cost_rows = [row for row in rows if int(row['total_verified_cost']) <= 24]
    return rows, best_row, low_cost_rows


def compute_signal_only_rows(corrected_rows: list[dict], piece_class_lookup: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for row in corrected_rows:
        outputs = piece_class_lookup[row['class_id']]['representative']
        signal_only_rank = len(outputs)
        full_rank = row['verified_rank_upper_bound']
        diff = ''
        if full_rank:
            diff = str(int(full_rank) - signal_only_rank)
        rows.append({
            'class_id': row['class_id'],
            'family': row['family'],
            'representative_entries': row['representative_entries'],
            'signal_only_rank': signal_only_rank,
            'verified_full_rank_upper_bound': full_rank,
            'rank_difference': diff,
            'signal_only_rank_reason': 'Exact Sigma-only restriction gives 3 I_m on the selected output fibers, so the signal layer rank is exactly the number of covered outputs.',
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def support_cells_from_mask(mask: list[bool]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(cell for cell, active in zip(GRID_CELLS, mask) if active))


def delta_column_indices_for_output(row_idx: int, col_idx: int) -> list[int]:
    indices: list[int] = []
    column_idx = 0
    for outer_row in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for outer_col in range(3):
                    if outer_row == row_idx and outer_col == col_idx:
                        indices.append(column_idx)
                    column_idx += 1
    return indices


def term_gamma_support(term) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3) if int(term.gamma[row_idx, col_idx]) != 0))


def analyze_alphatensor_layers() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    terms, _, _ = load_public_rank23_terms()
    sigma, eta1, eta2, delta, nuisance = build_mode_matrices(terms, 3)
    sigma_np = np.array(sigma.tolist(), dtype=np.float64)
    eta1_np = np.array(eta1.tolist(), dtype=np.float64)
    eta2_np = np.array(eta2.tolist(), dtype=np.float64)
    delta_np = np.array(delta.tolist(), dtype=np.float64)
    nuisance_np = np.array(nuisance.tolist(), dtype=np.float64)

    term_rows: list[dict] = []
    support_cluster_map: defaultdict[tuple[tuple[int, int], ...], list[str]] = defaultdict(list)
    corrector_rows: list[dict] = []

    category_counts = Counter()
    signal_primary = 0
    anis_primary = 0
    dead_primary = 0
    mixed_count = 0

    for idx, term in enumerate(terms):
        sigma_vec = sigma_np[idx, :]
        eta_vec = np.concatenate([eta1_np[idx, :], eta2_np[idx, :]])
        dead_vec = delta_np[idx, :]
        sigma_sq = float(np.dot(sigma_vec, sigma_vec))
        eta_sq = float(np.dot(eta_vec, eta_vec))
        dead_sq = float(np.dot(dead_vec, dead_vec))
        total_sq = sigma_sq + eta_sq + dead_sq
        signal_fraction = sigma_sq / total_sq if total_sq else 0.0
        anis_fraction = eta_sq / total_sq if total_sq else 0.0
        dead_fraction = dead_sq / total_sq if total_sq else 0.0
        if signal_fraction > 0.5:
            dominant = 'signal'
            signal_primary += 1
        elif anis_fraction > 0.5:
            dominant = 'anisotropy'
            anis_primary += 1
        elif dead_fraction > 0.5:
            dominant = 'dead_x'
            dead_primary += 1
        else:
            dominant = 'mixed'
            mixed_count += 1
        category_counts[dominant] += 1

        nuisance_output_mask = []
        for output_idx, cell in enumerate(GRID_CELLS):
            delta_indices = delta_column_indices_for_output(cell[0], cell[1])
            delta_block = dead_vec[delta_indices]
            active = abs(eta1_np[idx, output_idx]) > 0 or abs(eta2_np[idx, output_idx]) > 0 or np.any(delta_block != 0)
            nuisance_output_mask.append(bool(active))
        nuisance_support = support_cells_from_mask(nuisance_output_mask)
        gamma_support = term_gamma_support(term)
        support_cluster_map[nuisance_support].append(term.term_id)

        row = {
            'term_id': term.term_id,
            'gamma_support': cells_to_str(gamma_support),
            'gamma_support_shape': classify_cells(gamma_support),
            'nuisance_output_support': cells_to_str(nuisance_support),
            'nuisance_output_support_shape': classify_cells(nuisance_support) if nuisance_support else 'empty',
            'sigma_norm_sq': f'{sigma_sq:.16e}',
            'anisotropy_norm_sq': f'{eta_sq:.16e}',
            'dead_x_norm_sq': f'{dead_sq:.16e}',
            'signal_fraction': f'{signal_fraction:.16e}',
            'anisotropy_fraction': f'{anis_fraction:.16e}',
            'dead_x_fraction': f'{dead_fraction:.16e}',
            'dominant_layer': dominant,
            'signal_primary': str(signal_fraction > 0.5),
            'anisotropy_primary': str(anis_fraction > 0.5),
            'dead_x_primary': str(dead_fraction > 0.5),
            'is_anisotropy_corrector': str(signal_fraction < 0.1),
            'is_dead_x_corrector': str(dead_fraction > 0.5),
            'provenance': 'MEASURED_FROM_CODE',
        }
        term_rows.append(row)
        if signal_fraction < 0.1 or dead_fraction > 0.5:
            corrector_rows.append(row)

    output_block_rows: list[dict] = []
    nuisance_column_rows: list[dict] = []
    delta_pairs = [(sum_left, sum_right) for sum_left in range(3) for sum_right in range(3) if sum_left != sum_right]
    for output_idx, cell in enumerate(GRID_CELLS):
        delta_indices = delta_column_indices_for_output(cell[0], cell[1])
        block = np.concatenate([
            eta1_np[:, output_idx:output_idx + 1],
            eta2_np[:, output_idx:output_idx + 1],
            delta_np[:, delta_indices],
        ], axis=1)
        block_rank = int(np.linalg.matrix_rank(block))
        active_terms = sorted(term_rows[idx]['term_id'] for idx in range(len(terms)) if np.any(block[idx, :] != 0))
        output_block_rows.append({
            'output_entry': f'({cell[0]},{cell[1]})',
            'block_rank': block_rank,
            'active_term_count': len(active_terms),
            'active_terms': ';'.join(active_terms),
            'eta1_nonzero_terms': int(np.count_nonzero(eta1_np[:, output_idx])),
            'eta2_nonzero_terms': int(np.count_nonzero(eta2_np[:, output_idx])),
            'dead_block_nonzero_terms': int(np.sum(np.any(block[:, 2:] != 0, axis=1))),
            'provenance': 'EXACT_DERIVED',
        })
        nuisance_column_rows.append({
            'coordinate_id': f'eta1_{cell[0]}_{cell[1]}',
            'output_entry': f'({cell[0]},{cell[1]})',
            'coordinate_type': 'eta1',
            'served_by_terms': ';'.join(sorted(term_rows[idx]['term_id'] for idx in range(len(terms)) if eta1_np[idx, output_idx] != 0)),
            'provenance': 'EXACT_DERIVED',
        })
        nuisance_column_rows.append({
            'coordinate_id': f'eta2_{cell[0]}_{cell[1]}',
            'output_entry': f'({cell[0]},{cell[1]})',
            'coordinate_type': 'eta2',
            'served_by_terms': ';'.join(sorted(term_rows[idx]['term_id'] for idx in range(len(terms)) if eta2_np[idx, output_idx] != 0)),
            'provenance': 'EXACT_DERIVED',
        })
        for pair_offset, (sum_left, sum_right) in enumerate(delta_pairs):
            column_idx = delta_indices[pair_offset]
            nuisance_column_rows.append({
                'coordinate_id': f'delta_{cell[0]}_{sum_left}_{sum_right}_{cell[1]}',
                'output_entry': f'({cell[0]},{cell[1]})',
                'coordinate_type': 'dead_x',
                'served_by_terms': ';'.join(sorted(term_rows[idx]['term_id'] for idx in range(len(terms)) if delta_np[idx, column_idx] != 0)),
                'provenance': 'EXACT_DERIVED',
            })

    support_cluster_rows: list[dict] = []
    for cluster_idx, support in enumerate(sorted(support_cluster_map), start=1):
        support_cluster_rows.append({
            'cluster_id': f'nuisance_cluster_{cluster_idx:02d}',
            'nuisance_output_support': cells_to_str(support),
            'shape': classify_cells(support) if support else 'empty',
            'term_count': len(support_cluster_map[support]),
            'term_ids': ';'.join(sorted(support_cluster_map[support])),
            'provenance': 'EXACT_DERIVED',
        })

    summary = {
        'term_count': len(terms),
        'signal_primary_count': signal_primary,
        'anisotropy_primary_count': anis_primary,
        'dead_primary_count': dead_primary,
        'mixed_count': mixed_count,
        'nuisance_rank': int(np.linalg.matrix_rank(nuisance_np)),
        'signal_rank': int(np.linalg.matrix_rank(sigma_np)),
        'eta_rank': int(np.linalg.matrix_rank(np.concatenate([eta1_np, eta2_np], axis=1))),
        'delta_rank': int(np.linalg.matrix_rank(delta_np)),
    }

    return term_rows, corrector_rows, output_block_rows, nuisance_column_rows, support_cluster_rows, summary


def build_summary_rows(part_a_exact: dict[str, dict], best_tiling: dict, low_cost_rows: list[dict], layer_summary: dict) -> list[dict]:
    rows = []
    rows.append({'summary_name': 'part_a_piece_count', 'summary_value': str(len(part_a_exact)), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Number of formerly blank/uncertain Step 65 classes that now have fresh exact numerical upper bounds from Step 67.'})
    rows.append({'summary_name': 'best_verified_flat_tiling_cost', 'summary_value': str(best_tiling['total_verified_cost']), 'provenance': 'EXACT_DERIVED', 'note': 'Best flat exact-cover tiling cost using corrected verified piece ranks.'})
    rows.append({'summary_name': 'best_verified_flat_tiling_signature', 'summary_value': best_tiling['tiling_signature'], 'provenance': 'EXACT_DERIVED', 'note': 'Shape signature of the corrected best flat tiling.'})
    rows.append({'summary_name': 'flat_tilings_cost_leq_24', 'summary_value': str(len(low_cost_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of flat exact-cover tilings with corrected verified cost at most 24.'})
    rows.append({'summary_name': 'any_flat_tiling_beats_23', 'summary_value': str(any(int(row['total_verified_cost']) < 23 for row in low_cost_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Whether any corrected flat tiling costs strictly less than 23.'})
    rows.append({'summary_name': 'alphatensor_signal_primary_terms', 'summary_value': str(layer_summary['signal_primary_count']), 'provenance': 'MEASURED_FROM_CODE', 'note': 'AlphaTensor terms with signal fraction > 0.5.'})
    rows.append({'summary_name': 'alphatensor_anisotropy_primary_terms', 'summary_value': str(layer_summary['anisotropy_primary_count']), 'provenance': 'MEASURED_FROM_CODE', 'note': 'AlphaTensor terms with anisotropy fraction > 0.5.'})
    rows.append({'summary_name': 'alphatensor_dead_primary_terms', 'summary_value': str(layer_summary['dead_primary_count']), 'provenance': 'MEASURED_FROM_CODE', 'note': 'AlphaTensor terms with dead fraction > 0.5.'})
    rows.append({'summary_name': 'alphatensor_mixed_terms', 'summary_value': str(layer_summary['mixed_count']), 'provenance': 'MEASURED_FROM_CODE', 'note': 'AlphaTensor terms with no dominant layer fraction > 0.5.'})
    rows.append({'summary_name': 'alphatensor_nuisance_rank', 'summary_value': str(layer_summary['nuisance_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Rank of the 23 x 72 nuisance matrix [Eta1 | Eta2 | Delta].'})
    return rows


def build_markdown(summary_rows: list[dict], part_a_rows: list[dict], corrected_rows: list[dict], best_tiling: dict, low_cost_rows: list[dict], signal_rows: list[dict], layer_rows: list[dict], layer_summary: dict, nuisance_blocks: list[dict]) -> str:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append
    w('# Step 67: Complete Rank Table Audit + Layered Tiling Framework')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Part A: Fresh Rank Audits')
    w('')
    w('| piece | outputs | flattening LB | minimum exact numerical upper bound | best loss at that rank |')
    w('|-------|---------|---------------|-----------------------------------|------------------------|')
    latest = latest_exact_verdicts(part_a_rows)
    for piece_id in sorted(latest):
        row = latest[piece_id]
        w(f"| {piece_id} | {row['piece_outputs']} | {row['flattening_lower_bound']} | {row['exact_rank_upper_bound']} | {row['best_verified_loss']} |")
    w('')
    w('## Part B: Corrected Flat Tilings')
    w('')
    w(f"- Best corrected flat tiling cost: {summary['best_verified_flat_tiling_cost']}")
    w(f"- Best corrected flat tiling signature: {summary['best_verified_flat_tiling_signature']}")
    w(f"- Flat tilings with cost <= 24: {summary['flat_tilings_cost_leq_24']}")
    w(f"- Any flat tiling beats 23: {summary['any_flat_tiling_beats_23']}")
    w('')
    if low_cost_rows:
        w('| tiling signature | pieces | class ids | verified cost |')
        w('|------------------|--------|-----------|---------------|')
        for row in low_cost_rows:
            w(f"| {row['tiling_signature']} | {row['pieces']} | {row['class_ids']} | {row['total_verified_cost']} |")
    else:
        w('- No corrected flat tilings cost at most 24.')
    w('')
    w('## Part C: AlphaTensor Layers')
    w('')
    w(f"- Signal-primary terms: {summary['alphatensor_signal_primary_terms']}")
    w(f"- Anisotropy-primary terms: {summary['alphatensor_anisotropy_primary_terms']}")
    w(f"- Dead-primary terms: {summary['alphatensor_dead_primary_terms']}")
    w(f"- Mixed terms: {summary['alphatensor_mixed_terms']}")
    w(f"- Nuisance rank: {summary['alphatensor_nuisance_rank']}")
    w('')
    w('| term | signal fraction | anisotropy fraction | dead fraction | dominant layer | nuisance support |')
    w('|------|-----------------|---------------------|---------------|----------------|-----------------|')
    for row in sorted(layer_rows, key=lambda item: (item['dominant_layer'], item['term_id'])):
        w(f"| {row['term_id']} | {row['signal_fraction']} | {row['anisotropy_fraction']} | {row['dead_x_fraction']} | {row['dominant_layer']} | {row['nuisance_output_support']} |")
    w('')
    w('## Signal-Only Ranks')
    w('')
    w('| class | full verified rank | signal-only rank | difference |')
    w('|-------|--------------------|------------------|------------|')
    for row in signal_rows:
        w(f"| {row['class_id']} | {row['verified_full_rank_upper_bound']} | {row['signal_only_rank']} | {row['rank_difference']} |")
    w('')
    w('## Nuisance Output Blocks')
    w('')
    w('| output entry | block rank | active term count |')
    w('|--------------|------------|-------------------|')
    for row in nuisance_blocks:
        w(f"| {row['output_entry']} | {row['block_rank']} | {row['active_term_count']} |")
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    log('Step 67 start')

    piece_class_lookup = build_piece_class_lookup()
    part_a_targets = build_part_a_targets(piece_class_lookup)

    log('Part A: running shared-pool fresh rank audits for the remaining Step 65 blank rows')
    part_a_rows, restart_rows = run_piece_audit_rounds(part_a_targets, PART_A_RANKS, PART_A_RESTARTS, DEFAULT_WORKERS)
    part_a_exact = latest_exact_verdicts(part_a_rows)
    require(len(part_a_exact) == len(part_a_targets), f'Part A did not resolve all targets: {sorted(set(piece_id for piece_id, _, _ in part_a_targets) - set(part_a_exact))}')

    log('Part B: rebuilding the corrected rank table and flat tiling costs')
    corrected_rows, class_costs = build_corrected_rank_table(part_a_exact, piece_class_lookup)
    tiling_rows, best_tiling, low_cost_rows = compute_corrected_tilings(piece_class_lookup, class_costs)

    log('Part C: computing signal-only ranks and AlphaTensor layered fractions')
    signal_rows = compute_signal_only_rows(corrected_rows, piece_class_lookup)
    layer_rows, corrector_rows, nuisance_blocks, nuisance_columns, nuisance_clusters, layer_summary = analyze_alphatensor_layers()

    summary_rows = build_summary_rows(part_a_exact, best_tiling, low_cost_rows, layer_summary)

    log('Writing Step 67 exports')
    write_csv(EXPORTS / 'step67_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step67_partA_rank_audit.csv', part_a_rows, ['piece_id', 'family', 'piece_outputs', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'exact_rank_upper_bound', 'provenance'])
    write_csv(EXPORTS / 'step67_partA_restart_log.csv', restart_rows, ['piece_id', 'rank_tested', 'seed', 'verified_loss', 'best_max_abs_residual', 'provenance'])
    write_csv(EXPORTS / 'step67_corrected_rank_table.csv', corrected_rows, ['family', 'class_id', 'representative_entries', 'size', 'count_on_3x3', 'symmetry_class_size', 'row_profile', 'col_profile', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'numerical_rank_upper_bound', 'alpha_tensor_upper_bound', 'exact_rank_if_determined', 'best_verified_loss', 'note', 'provenance', 'verified_rank_upper_bound', 'verified_rank_source', 'corrected_note'])
    write_csv(EXPORTS / 'step67_corrected_tiling_table.csv', tiling_rows, ['tiling_id', 'piece_count', 'tiling_signature', 'pieces', 'class_ids', 'total_verified_cost', 'beats_23', 'provenance'])
    write_csv(EXPORTS / 'step67_low_cost_tilings_leq24.csv', low_cost_rows, ['tiling_id', 'piece_count', 'tiling_signature', 'pieces', 'class_ids', 'total_verified_cost', 'beats_23', 'provenance'])
    write_csv(EXPORTS / 'step67_signal_only_rank_table.csv', signal_rows, ['class_id', 'family', 'representative_entries', 'signal_only_rank', 'verified_full_rank_upper_bound', 'rank_difference', 'signal_only_rank_reason', 'provenance'])
    write_csv(EXPORTS / 'step67_alphatensor_layer_fractions.csv', layer_rows, ['term_id', 'gamma_support', 'gamma_support_shape', 'nuisance_output_support', 'nuisance_output_support_shape', 'sigma_norm_sq', 'anisotropy_norm_sq', 'dead_x_norm_sq', 'signal_fraction', 'anisotropy_fraction', 'dead_x_fraction', 'dominant_layer', 'signal_primary', 'anisotropy_primary', 'dead_x_primary', 'is_anisotropy_corrector', 'is_dead_x_corrector', 'provenance'])
    write_csv(EXPORTS / 'step67_alphatensor_corrector_terms.csv', corrector_rows, ['term_id', 'gamma_support', 'gamma_support_shape', 'nuisance_output_support', 'nuisance_output_support_shape', 'sigma_norm_sq', 'anisotropy_norm_sq', 'dead_x_norm_sq', 'signal_fraction', 'anisotropy_fraction', 'dead_x_fraction', 'dominant_layer', 'signal_primary', 'anisotropy_primary', 'dead_x_primary', 'is_anisotropy_corrector', 'is_dead_x_corrector', 'provenance'])
    write_csv(EXPORTS / 'step67_nuisance_output_blocks.csv', nuisance_blocks, ['output_entry', 'block_rank', 'active_term_count', 'active_terms', 'eta1_nonzero_terms', 'eta2_nonzero_terms', 'dead_block_nonzero_terms', 'provenance'])
    write_csv(EXPORTS / 'step67_nuisance_columns.csv', nuisance_columns, ['coordinate_id', 'output_entry', 'coordinate_type', 'served_by_terms', 'provenance'])
    write_csv(EXPORTS / 'step67_nuisance_support_clusters.csv', nuisance_clusters, ['cluster_id', 'nuisance_output_support', 'shape', 'term_count', 'term_ids', 'provenance'])
    write_text(EXPORTS / 'step67_complete_rank_table_audit_layered_tiling_framework.md', build_markdown(summary_rows, part_a_rows, corrected_rows, best_tiling, low_cost_rows, signal_rows, layer_rows, layer_summary, nuisance_blocks))

    log(f"Part A resolved pieces: {', '.join(sorted(part_a_exact))}")
    log(f"Corrected best flat tiling cost: {best_tiling['total_verified_cost']} ({best_tiling['tiling_signature']})")
    log(f"AlphaTensor layer split: signal={layer_summary['signal_primary_count']}, anisotropy={layer_summary['anisotropy_primary_count']}, dead={layer_summary['dead_primary_count']}, mixed={layer_summary['mixed_count']}")
    log('Step 67 complete')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as exc:
        log(f'Step 67 failed: {exc}')
        traceback.print_exc()
        raise