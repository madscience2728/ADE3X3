"""
ade3x3_step68_layered_correction_tiling.py

Step 68: Layered Correction Tiling.

This step separates the exact 3x3 tensor into a chosen 9-term signal layer and a
remaining correction tensor. It then measures the polyomino-piece ranks of that
correction tensor, rebuilds flat exact-cover tilings from those correction costs,
and probes how the correction budget changes when the 9-term signal layer is varied.
"""

from __future__ import annotations

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

from src.ade3x3.steps.ade3x3_step51_symbolic_fiber_mode_decomposition import target_rhs
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
CANONICAL_RESTARTS = 500
CANONICAL_MAX_RANK = 18
NAMED_SIGNAL_RESTARTS = 80
RANDOM_TRIALS = 100
RANDOM_REFINEMENT_COUNT = 10
RANDOM_REFINEMENT_RESTARTS = 40
OMEGA = np.exp(2j * np.pi / 3.0)

GRID_CELLS = tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3))
CELL_TO_INDEX = {cell: idx for idx, cell in enumerate(GRID_CELLS)}
CONNECTED_FAMILIES = {
    'monomino',
    'domino_row',
    'domino_col',
    'tromino_row',
    'tromino_col',
    'L_tromino',
    'square_tetromino',
    'L_tetromino',
    'S_Z_tetromino',
    'T_tetromino',
    'P4_anti_diagonal_missing',
}


@dataclass(frozen=True)
class SignalTerm:
    output_cell: tuple[int, int]
    left_weights: tuple[float, float, float]
    right_weights: tuple[float, float, float]


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


def vector_to_json(vector: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return json.dumps([float(value) for value in vector], separators=(',', ':'))


def complex_to_json(value: complex) -> str:
    return json.dumps({'real': float(np.real(value)), 'imag': float(np.imag(value))}, separators=(',', ':'))


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
    return {
        'seed': seed,
        'verified_loss': float(np.sum(residual * residual)),
        'best_max_abs_residual': float(np.max(np.abs(residual))),
    }


def rank_scan_target(piece_id: str, target: np.ndarray, ranks: list[int], restarts: int, workers: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    verdict = {
        'piece_id': piece_id,
        'exact_rank_upper_bound': '',
        'best_loss_overall': float('inf'),
        'best_rank_tested': '',
    }
    for rank in ranks:
        seeds = [100000 * rank + 701 * restart + sum(ord(ch) for ch in piece_id) for restart in range(restarts)]
        payloads = [{'target': target.tolist(), 'rank': rank, 'seed': seed} for seed in seeds]
        best_loss = float('inf')
        best_abs = float('inf')
        exact_hits = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(single_restart_worker, payload) for payload in payloads]
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                loss = float(result['verified_loss'])
                best_loss = min(best_loss, loss)
                best_abs = min(best_abs, float(result['best_max_abs_residual']))
                if loss < EXACT_TOL:
                    exact_hits += 1
                if completed == 1 or completed % 250 == 0 or completed == len(payloads):
                    log(f'{piece_id} rank {rank}: completed {completed}/{len(payloads)} | best_loss={best_loss:.3e} exact_hits={exact_hits}')
        rows.append({
            'piece_id': piece_id,
            'rank_tested': rank,
            'restarts': restarts,
            'best_verified_loss': f'{best_loss:.16e}',
            'best_max_abs_residual': f'{best_abs:.16e}',
            'exact_hit_count': exact_hits,
            'verified_exact': str(best_loss < EXACT_TOL),
            'exact_rank_upper_bound': str(rank) if best_loss < EXACT_TOL else '',
            'provenance': 'MEASURED_FROM_CODE',
        })
        if best_loss < verdict['best_loss_overall']:
            verdict['best_loss_overall'] = best_loss
            verdict['best_rank_tested'] = str(rank)
        if best_loss < EXACT_TOL:
            verdict['exact_rank_upper_bound'] = str(rank)
            break
    return rows, verdict


def normalize_trace_three(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dot = float(np.dot(left, right))
    require(abs(dot) > 1e-9, 'Signal weight vectors must have nonzero channel pairing.')
    scale = 3.0 / dot
    return left.astype(np.float64), (right.astype(np.float64) * scale)


def canonical_signal_terms() -> list[SignalTerm]:
    terms = []
    left = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    right = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    for cell in GRID_CELLS:
        terms.append(SignalTerm(cell, tuple(left.tolist()), tuple(right.tolist())))
    return terms


def named_signal_families() -> dict[str, list[SignalTerm]]:
    families: dict[str, list[SignalTerm]] = {}
    specs = {
        'canonical_balanced': (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
        'single_channel_0': (np.array([3.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        'asymmetric_real': (np.array([1.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.5])),
        'dft_real_proxy': (np.array([1.0, -0.5, -0.5]), np.array([2.0, -1.0, -1.0])),
    }
    for family_id, (left_raw, right_raw) in specs.items():
        left, right = normalize_trace_three(left_raw, right_raw)
        families[family_id] = [SignalTerm(cell, tuple(left.tolist()), tuple(right.tolist())) for cell in GRID_CELLS]
    return families


def random_signal_family(seed: int, *, shared_weights: bool) -> list[SignalTerm]:
    rng = np.random.default_rng(seed)
    terms: list[SignalTerm] = []
    shared_left: np.ndarray | None = None
    shared_right: np.ndarray | None = None
    if shared_weights:
        left_raw = rng.standard_normal(3)
        right_raw = rng.standard_normal(3)
        while abs(float(np.dot(left_raw, right_raw))) < 1e-9:
            left_raw = rng.standard_normal(3)
            right_raw = rng.standard_normal(3)
        shared_left, shared_right = normalize_trace_three(left_raw, right_raw)
    for cell in GRID_CELLS:
        if shared_weights:
            left = shared_left
            right = shared_right
        else:
            left_raw = rng.standard_normal(3)
            right_raw = rng.standard_normal(3)
            while abs(float(np.dot(left_raw, right_raw))) < 1e-9:
                left_raw = rng.standard_normal(3)
                right_raw = rng.standard_normal(3)
            left, right = normalize_trace_three(left_raw, right_raw)
        terms.append(SignalTerm(cell, tuple(left.tolist()), tuple(right.tolist())))
    return terms


def signal_term_matrices(term: SignalTerm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_idx, col_idx = term.output_cell
    alpha = np.zeros((3, 3), dtype=np.float64)
    beta = np.zeros((3, 3), dtype=np.float64)
    gamma = np.zeros((3, 3), dtype=np.float64)
    alpha[row_idx, :] = np.array(term.left_weights, dtype=np.float64)
    beta[:, col_idx] = np.array(term.right_weights, dtype=np.float64)
    gamma[row_idx, col_idx] = 1.0
    return alpha, beta, gamma


def tensor_from_signal_terms(terms: list[SignalTerm]) -> np.ndarray:
    tensor = np.zeros((9, 9, 9), dtype=np.float64)
    for term in terms:
        alpha, beta, gamma = signal_term_matrices(term)
        tensor += np.einsum('a,b,c->abc', alpha.reshape(-1), beta.reshape(-1), gamma.reshape(-1), optimize=True)
    return tensor


def correction_tensor_from_signal_terms(terms: list[SignalTerm]) -> np.ndarray:
    return matrix_multiplication_tensor() - tensor_from_signal_terms(terms)


def canonical_correction_slice() -> np.ndarray:
    slice_matrix = np.full((3, 3), -1.0, dtype=np.complex128)
    np.fill_diagonal(slice_matrix, 0.0)
    return slice_matrix


def phase_j1_signal_slice() -> np.ndarray:
    left = np.array([1.0, OMEGA, OMEGA ** 2], dtype=np.complex128)
    right = np.array([1.0, OMEGA ** 2, OMEGA], dtype=np.complex128)
    signal = np.outer(left, right)
    return np.eye(3, dtype=np.complex128) - signal


def fourier_coefficients_from_slice(slice_matrix: np.ndarray) -> dict[tuple[int, int], complex]:
    coeffs: dict[tuple[int, int], complex] = {}
    for freq_left in range(3):
        for freq_right in range(3):
            total = 0.0 + 0.0j
            for sum_left in range(3):
                for sum_right in range(3):
                    total += (OMEGA ** (freq_left * sum_left + freq_right * sum_right)) * slice_matrix[sum_left, sum_right]
            coeffs[(freq_left, freq_right)] = complex(np.round(np.real(total), 12), np.round(np.imag(total), 12))
    return coeffs


def normalized_fourier_basis_slice(freq_left: int, freq_right: int) -> np.ndarray:
    basis = np.zeros((3, 3), dtype=np.complex128)
    for sum_left in range(3):
        for sum_right in range(3):
            basis[sum_left, sum_right] = (OMEGA ** (-(freq_left * sum_left + freq_right * sum_right))) / 9.0
    return basis


def correction_tensor_from_slice(slice_matrix: np.ndarray) -> np.ndarray:
    tensor = np.zeros((9, 9, 9), dtype=np.complex128)
    for row_idx in range(3):
        for out_col in range(3):
            output_idx = CELL_TO_INDEX[(row_idx, out_col)]
            for sum_left in range(3):
                a_idx = 3 * row_idx + sum_left
                for sum_right in range(3):
                    b_idx = 3 * sum_right + out_col
                    tensor[a_idx, b_idx, output_idx] = slice_matrix[sum_left, sum_right]
    return tensor


def component_rank_row(family_id: str, freq_left: int, freq_right: int) -> dict:
    component_tensor = correction_tensor_from_slice(normalized_fourier_basis_slice(freq_left, freq_right))
    rank_a, rank_b, rank_c, rank_lb = flattening_ranks(component_tensor)
    slice_ub = sum(int(np.linalg.matrix_rank(component_tensor[:, :, idx])) for idx in range(9))
    exact_rank = str(rank_lb) if rank_lb == slice_ub else ''
    return {
        'family_id': family_id,
        'mode_j': freq_left,
        'mode_k': freq_right,
        'flattening_rank_A_BC': rank_a,
        'flattening_rank_B_AC': rank_b,
        'flattening_rank_C_AB': rank_c,
        'flattening_lower_bound': rank_lb,
        'exact_slice_rank_upper_bound': slice_ub,
        'exact_rank_if_determined': exact_rank,
        'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
    }


def fourier_mode_rows_for_family(family_id: str, slice_matrix: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    coeffs = fourier_coefficients_from_slice(slice_matrix)
    mode_rows: list[dict] = []
    component_rows: list[dict] = []
    nonzero_modes = []
    for freq_left in range(3):
        for freq_right in range(3):
            coeff = coeffs[(freq_left, freq_right)]
            abs_coeff = abs(coeff)
            is_nonzero = abs_coeff > 1e-9
            mode_rows.append({
                'family_id': family_id,
                'mode_j': freq_left,
                'mode_k': freq_right,
                'coefficient_real': f'{float(np.real(coeff)):.16e}',
                'coefficient_imag': f'{float(np.imag(coeff)):.16e}',
                'coefficient_abs': f'{float(abs_coeff):.16e}',
                'coefficient_json': complex_to_json(coeff),
                'is_nonzero': str(is_nonzero),
                'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
            })
            if is_nonzero:
                nonzero_modes.append((freq_left, freq_right, coeff))
                component_rows.append(component_rank_row(family_id, freq_left, freq_right))
    total_component_rank = sum(int(row['exact_rank_if_determined'] or row['exact_slice_rank_upper_bound']) for row in component_rows)
    summary = {
        'family_id': family_id,
        'nonzero_mode_count': len(nonzero_modes),
        'nonzero_modes': ';'.join(f'({freq_left},{freq_right})' for freq_left, freq_right, _ in nonzero_modes),
        'coefficients': ';'.join(f'({freq_left},{freq_right})={coeff.real:.6g}{coeff.imag:+.6g}i' for freq_left, freq_right, coeff in nonzero_modes),
        'component_rank_sum_upper_bound': total_component_rank,
        'any_component_sum_below_14': str(total_component_rank < 14),
        'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
    }
    return mode_rows, component_rows, summary


def run_fourier_extension_exports() -> tuple[list[dict], list[dict], list[dict], str]:
    canonical_modes, canonical_components, canonical_summary = fourier_mode_rows_for_family('canonical_balanced_dead_residual', canonical_correction_slice())
    phase_modes, phase_components, phase_summary = fourier_mode_rows_for_family('phase_j1_dead_residual', phase_j1_signal_slice())
    summary_rows = [
        {
            'summary_name': 'canonical_fourier_nonzero_modes',
            'summary_value': canonical_summary['nonzero_modes'],
            'provenance': canonical_summary['provenance'],
            'note': 'Nonzero DFT modes of the canonical all-ones dead residual.',
        },
        {
            'summary_name': 'canonical_fourier_component_rank_sum_upper_bound',
            'summary_value': str(canonical_summary['component_rank_sum_upper_bound']),
            'provenance': canonical_summary['provenance'],
            'note': 'Sum of exact component ranks for the canonical dead residual Fourier support.',
        },
        {
            'summary_name': 'phase_j1_fourier_nonzero_modes',
            'summary_value': phase_summary['nonzero_modes'],
            'provenance': phase_summary['provenance'],
            'note': 'Nonzero DFT modes of the j=1 phase-weighted dead residual.',
        },
        {
            'summary_name': 'phase_j1_fourier_component_rank_sum_upper_bound',
            'summary_value': str(phase_summary['component_rank_sum_upper_bound']),
            'provenance': phase_summary['provenance'],
            'note': 'Sum of exact component ranks for the j=1 phase-weighted dead residual Fourier support.',
        },
        {
            'summary_name': 'any_fourier_component_sum_below_14',
            'summary_value': str(canonical_summary['component_rank_sum_upper_bound'] < 14 or phase_summary['component_rank_sum_upper_bound'] < 14),
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
            'note': 'Immediate flag requested by the steering note.',
        },
    ]
    markdown_lines = [
        '# Step 68 Steering: Fourier-Encoded Correction Layer',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '[EXACT_DERIVED] + [MEASURED_FROM_CODE]',
        '',
        '## Canonical Dead Residual Modes',
        '',
        f"- Nonzero modes: {canonical_summary['nonzero_modes']}",
        f"- Coefficients: {canonical_summary['coefficients']}",
        f"- Component-rank sum upper bound: {canonical_summary['component_rank_sum_upper_bound']}",
        '',
        '## Phase-j1 Dead Residual Modes',
        '',
        f"- Nonzero modes: {phase_summary['nonzero_modes']}",
        f"- Coefficients: {phase_summary['coefficients']}",
        f"- Component-rank sum upper bound: {phase_summary['component_rank_sum_upper_bound']}",
        '',
        '| family | mode | coefficient | exact component rank |',
        '|--------|------|-------------|----------------------|',
    ]
    component_map = {
        (row['family_id'], int(row['mode_j']), int(row['mode_k'])): row for row in canonical_components + phase_components
    }
    for row in canonical_modes + phase_modes:
        if row['is_nonzero'] != 'True':
            continue
        key = (row['family_id'], int(row['mode_j']), int(row['mode_k']))
        component_row = component_map[key]
        markdown_lines.append(
            f"| {row['family_id']} | ({row['mode_j']},{row['mode_k']}) | {row['coefficient_real']} + {row['coefficient_imag']} i | {component_row['exact_rank_if_determined'] or component_row['exact_slice_rank_upper_bound']} |"
        )
    markdown = '\n'.join(markdown_lines)
    return summary_rows, canonical_modes + phase_modes, canonical_components + phase_components, markdown


def reduced_tensor_from_outputs(full_tensor: np.ndarray, outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    result = np.zeros((9, 9, len(outputs)), dtype=np.float64)
    for output_idx, cell in enumerate(outputs):
        result[:, :, output_idx] = full_tensor[:, :, CELL_TO_INDEX[cell]]
    return result


def output_slice_matrix(signal_terms: list[SignalTerm], cell: tuple[int, int]) -> np.ndarray:
    correction = correction_tensor_from_signal_terms(signal_terms)
    return correction[:, :, CELL_TO_INDEX[cell]]


def slice_rank_upper_bound(full_tensor: np.ndarray, outputs: tuple[tuple[int, int], ...]) -> int:
    return sum(int(np.linalg.matrix_rank(full_tensor[:, :, CELL_TO_INDEX[cell]])) for cell in outputs)


def verify_signal_family(family_id: str, terms: list[SignalTerm]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    dead_nonzero_count = 0
    signal_tensor = tensor_from_signal_terms(terms)
    correction_tensor = matrix_multiplication_tensor() - signal_tensor
    for out_row in range(3):
        for out_col in range(3):
            for row_idx in range(3):
                for col_idx in range(3):
                    sigma_total = 0.0
                    eta1_total = 0.0
                    eta2_total = 0.0
                    for term in terms:
                        alpha, beta, gamma = signal_term_matrices(term)
                        gamma_weight = gamma[out_row, out_col]
                        lambda0 = alpha[row_idx, 0] * beta[0, col_idx]
                        lambda1 = alpha[row_idx, 1] * beta[1, col_idx]
                        lambda2 = alpha[row_idx, 2] * beta[2, col_idx]
                        sigma_total += gamma_weight * (lambda0 + lambda1 + lambda2)
                        eta1_total += gamma_weight * (lambda0 - lambda1)
                        eta2_total += gamma_weight * (lambda1 - lambda2)
                    rows.append({
                        'family_id': family_id,
                        'output_c': f'C[{out_row},{out_col}]',
                        'fiber': f'({row_idx},{col_idx})',
                        'sigma_total': f'{sigma_total:.16e}',
                        'sigma_expected': f'{3.0 * target_rhs(row_idx, col_idx, out_row, out_col):.16e}',
                        'eta1_total': f'{eta1_total:.16e}',
                        'eta2_total': f'{eta2_total:.16e}',
                        'provenance': 'EXACT_DERIVED',
                    })
            for row_idx in range(3):
                for sum_left in range(3):
                    for sum_right in range(3):
                        if sum_left == sum_right:
                            continue
                        for col_idx in range(3):
                            if abs(correction_tensor[3 * row_idx + sum_left, 3 * sum_right + col_idx, 3 * out_row + out_col]) > 1e-12:
                                dead_nonzero_count += 1
    live_positions = []
    dead_positions = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                for col_idx in range(3):
                    value = correction_tensor[3 * row_idx + sum_left, 3 * sum_right + col_idx, 3 * row_idx + col_idx]
                    if sum_left == sum_right:
                        live_positions.append(value)
                    else:
                        dead_positions.append(value)
    summary = {
        'family_id': family_id,
        'signal_term_count': len(terms),
        'signal_trace_min': f'{min(float(np.dot(term.left_weights, term.right_weights)) for term in terms):.16e}',
        'signal_trace_max': f'{max(float(np.dot(term.left_weights, term.right_weights)) for term in terms):.16e}',
        'correction_live_min': f'{float(np.min(live_positions)):.16e}',
        'correction_live_max': f'{float(np.max(live_positions)):.16e}',
        'correction_dead_min': f'{float(np.min(dead_positions)):.16e}',
        'correction_dead_max': f'{float(np.max(dead_positions)):.16e}',
        'correction_frobenius_norm_sq': f'{float(np.sum(correction_tensor * correction_tensor)):.16e}',
        'correction_flattening_lower_bound': str(flattening_ranks(correction_tensor)[3]),
        'correction_slice_rank_upper_bound': str(sum(int(np.linalg.matrix_rank(correction_tensor[:, :, idx])) for idx in range(9))),
        'dead_equation_nonzero_count': str(dead_nonzero_count),
        'provenance': 'EXACT_DERIVED',
    }
    return rows, summary


def build_piece_lookup() -> dict[str, dict]:
    piece_instances = enumerate_piece_instances()
    classes, _ = build_piece_classes(piece_instances)
    lookup = {
        cls.class_id: {
            'family': cls.family,
            'representative': cls.representative,
            'representative_entries': cells_to_str(cls.representative),
            'size': cls.size,
            'count_on_3x3': cls.instance_count,
            'symmetry_class_size': cls.symmetry_class_size,
            'row_profile': cls.row_profile,
            'col_profile': cls.col_profile,
        }
        for cls in classes
        if cls.family in CONNECTED_FAMILIES
    }
    lookup['full_board_class_01'] = {
        'family': 'full_board',
        'representative': GRID_CELLS,
        'representative_entries': cells_to_str(GRID_CELLS),
        'size': 9,
        'count_on_3x3': 1,
        'symmetry_class_size': 1,
        'row_profile': '3,3,3',
        'col_profile': '3,3,3',
    }
    return lookup


def canonical_correction_rank_table(piece_lookup: dict[str, dict], workers: int) -> tuple[list[dict], list[dict], dict[str, int]]:
    signal_terms = canonical_signal_terms()
    full_correction_tensor = correction_tensor_from_signal_terms(signal_terms)
    rank_rows: list[dict] = []
    scan_rows: list[dict] = []
    class_costs: dict[str, int] = {}
    for class_id, info in sorted(piece_lookup.items()):
        outputs = info['representative']
        target = reduced_tensor_from_outputs(full_correction_tensor, outputs)
        rank_a, rank_b, rank_c, rank_lb = flattening_ranks(target)
        exact_slice_ub = slice_rank_upper_bound(full_correction_tensor, outputs)
        if class_id == 'full_board_class_01':
            rows = []
            verdict = {
                'piece_id': class_id,
                'exact_rank_upper_bound': '',
                'best_loss_overall': float('nan'),
                'best_rank_tested': '',
            }
            verified_rank = str(exact_slice_ub)
            verified_source = 'deferred_to_signal_sensitivity_exact_slice_upper_bound'
        else:
            max_rank = min(CANONICAL_MAX_RANK, exact_slice_ub)
            ranks = list(range(max(1, rank_lb), max_rank + 1))
            log(f'Canonical correction scan {class_id}: ranks={ranks[0]}..{ranks[-1]}, restarts={CANONICAL_RESTARTS}, slice_ub={exact_slice_ub}')
            rows, verdict = rank_scan_target(class_id, target, ranks, CANONICAL_RESTARTS, workers)
            scan_rows.extend(rows)
            verified_rank = verdict['exact_rank_upper_bound'] or str(exact_slice_ub)
            verified_source = 'step68_canonical_correction_scan' if verdict['exact_rank_upper_bound'] else 'exact_slice_upper_bound'
        class_costs[class_id] = int(verified_rank)
        rank_rows.append({
            'family': info['family'],
            'class_id': class_id,
            'representative_entries': info['representative_entries'],
            'size': info['size'],
            'count_on_3x3': info['count_on_3x3'],
            'symmetry_class_size': info['symmetry_class_size'],
            'row_profile': info['row_profile'],
            'col_profile': info['col_profile'],
            'flattening_rank_A_BC': rank_a,
            'flattening_rank_B_AC': rank_b,
            'flattening_rank_C_AB': rank_c,
            'flattening_lower_bound': rank_lb,
            'best_rank_tested': verdict['best_rank_tested'],
            'best_verified_loss': '' if str(verdict['best_loss_overall']) == 'nan' else f"{float(verdict['best_loss_overall']):.16e}",
            'exact_scan_rank_upper_bound': verdict['exact_rank_upper_bound'],
            'exact_slice_rank_upper_bound': exact_slice_ub,
            'verified_correction_rank_upper_bound': verified_rank,
            'verified_rank_source': verified_source,
            'signal_family': 'canonical_balanced',
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })
    return rank_rows, scan_rows, class_costs


def build_piece_to_class_map(piece_lookup: dict[str, dict]) -> dict[tuple[str, tuple[tuple[int, int], ...]], str]:
    return {
        (info['family'], canonical_cells(info['representative'])): class_id
        for class_id, info in piece_lookup.items()
        if info['family'] != 'full_board'
    }


def tiling_shape_signature(tiling: tuple[tuple[tuple[int, int], ...], ...]) -> str:
    counts = {}
    for piece in tiling:
        family = classify_cells(piece)
        counts[family] = counts.get(family, 0) + 1
    return ' + '.join(f'{count}x {family}' for family, count in sorted(counts.items()))


def compute_correction_tilings(piece_lookup: dict[str, dict], class_costs: dict[str, int]) -> tuple[list[dict], dict, list[dict]]:
    piece_instances = enumerate_piece_instances()
    tilings = enumerate_general_tilings(piece_instances, CONNECTED_FAMILIES)
    piece_to_class = build_piece_to_class_map(piece_lookup)
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
        correction_cost = sum(class_costs[class_id] for class_id in class_ids)
        total_cost = 9 + correction_cost
        rows.append({
            'tiling_id': tiling_idx,
            'piece_count': len(tiling),
            'tiling_signature': tiling_shape_signature(tiling),
            'pieces': ' | '.join(cells_to_str(piece) for piece in tiling),
            'class_ids': ';'.join(class_ids),
            'correction_cost_upper_bound': correction_cost,
            'total_rank_upper_bound': total_cost,
            'beats_23': str(total_cost < 23),
            'provenance': 'EXACT_DERIVED',
        })
    require(bool(rows), 'No exact-cover tilings were produced for Step 68.')
    best = min(rows, key=lambda row: (int(row['total_rank_upper_bound']), row['tiling_signature'], row['pieces']))
    low = [row for row in rows if int(row['total_rank_upper_bound']) < 23]
    return rows, best, low


def refine_signal_family_rank(family_id: str, signal_terms: list[SignalTerm], restarts: int, workers: int, max_rank_cap: int = 18) -> dict:
    correction = correction_tensor_from_signal_terms(signal_terms)
    rank_a, rank_b, rank_c, rank_lb = flattening_ranks(correction)
    slice_ub = sum(int(np.linalg.matrix_rank(correction[:, :, idx])) for idx in range(9))
    max_rank = min(max_rank_cap, slice_ub)
    ranks = list(range(rank_lb, max_rank + 1))
    rows, verdict = rank_scan_target(f'{family_id}_full_correction', correction, ranks, restarts, workers)
    best_exact = verdict['exact_rank_upper_bound']
    return {
        'family_id': family_id,
        'flattening_rank_A_BC': rank_a,
        'flattening_rank_B_AC': rank_b,
        'flattening_rank_C_AB': rank_c,
        'flattening_lower_bound': rank_lb,
        'exact_slice_rank_upper_bound': slice_ub,
        'scan_restarts': restarts,
        'scan_max_rank': max_rank,
        'best_rank_tested': verdict['best_rank_tested'],
        'best_verified_loss': f"{float(verdict['best_loss_overall']):.16e}",
        'exact_scan_rank_upper_bound': best_exact,
        'verified_correction_rank_upper_bound': best_exact or str(slice_ub),
        'verified_rank_source': 'step68_signal_sensitivity_scan' if best_exact else 'exact_slice_upper_bound',
        'total_rank_upper_bound_with_signal': str(9 + int(best_exact or slice_ub)),
        'beats_23': str(9 + int(best_exact or slice_ub) < 23),
        'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        'scan_rows': rows,
    }


def random_signal_trial_rows(workers: int) -> tuple[list[dict], list[dict], dict]:
    proxy_rows: list[dict] = []
    refined_rows: list[dict] = []
    scan_rows: list[dict] = []
    families: list[tuple[str, list[SignalTerm]]] = []
    for trial_idx in range(1, RANDOM_TRIALS + 1):
        signal_terms = random_signal_family(680000 + trial_idx, shared_weights=False)
        family_id = f'random_trial_{trial_idx:03d}'
        families.append((family_id, signal_terms))
        correction = correction_tensor_from_signal_terms(signal_terms)
        rank_a, rank_b, rank_c, rank_lb = flattening_ranks(correction)
        slice_ub = sum(int(np.linalg.matrix_rank(correction[:, :, idx])) for idx in range(9))
        proxy_rows.append({
            'family_id': family_id,
            'flattening_rank_A_BC': rank_a,
            'flattening_rank_B_AC': rank_b,
            'flattening_rank_C_AB': rank_c,
            'flattening_lower_bound': rank_lb,
            'exact_slice_rank_upper_bound': slice_ub,
            'proxy_total_rank_upper_bound_with_signal': 9 + slice_ub,
            'provenance': 'EXACT_DERIVED',
        })
    candidates = sorted(proxy_rows, key=lambda row: (int(row['proxy_total_rank_upper_bound_with_signal']), int(row['flattening_lower_bound']), row['family_id']))[:RANDOM_REFINEMENT_COUNT]
    family_map = {family_id: terms for family_id, terms in families}
    for row in candidates:
        refined = refine_signal_family_rank(row['family_id'], family_map[row['family_id']], RANDOM_REFINEMENT_RESTARTS, workers)
        scan_rows.extend(refined.pop('scan_rows'))
        refined_rows.append(refined)
    best_row = min(refined_rows, key=lambda row: (int(row['total_rank_upper_bound_with_signal']), int(row['flattening_lower_bound']), row['family_id'])) if refined_rows else {}
    return proxy_rows, refined_rows, {'best_random_refined_family': best_row.get('family_id', ''), 'best_random_refined_total_rank_upper_bound': best_row.get('total_rank_upper_bound_with_signal', ''), 'scan_rows': scan_rows}


def build_summary_rows(canonical_summary: dict, best_tiling: dict, low_tilings: list[dict], named_rows: list[dict], random_best: dict) -> list[dict]:
    rows = []
    rows.append({'summary_name': 'canonical_signal_term_count', 'summary_value': '9', 'provenance': 'EXACT_DERIVED', 'note': 'The Step 68 base signal layer uses nine rank-1 terms, one per output entry.'})
    rows.append({'summary_name': 'canonical_correction_flattening_lower_bound', 'summary_value': canonical_summary['correction_flattening_lower_bound'], 'provenance': 'EXACT_DERIVED', 'note': 'Flattening lower bound of the full 9-output correction tensor for the canonical balanced signal layer.'})
    rows.append({'summary_name': 'best_canonical_correction_tiling_total_rank_upper_bound', 'summary_value': str(best_tiling['total_rank_upper_bound']), 'provenance': 'EXACT_DERIVED', 'note': 'Best flat exact-cover total cost using Step 68 canonical correction-piece ranks.'})
    rows.append({'summary_name': 'best_canonical_correction_tiling_signature', 'summary_value': best_tiling['tiling_signature'], 'provenance': 'EXACT_DERIVED', 'note': 'Shape signature of the best canonical correction tiling.'})
    rows.append({'summary_name': 'canonical_correction_tilings_beating_23', 'summary_value': str(len(low_tilings)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of canonical correction tilings with total cost strictly below 23.'})
    best_named = min(named_rows, key=lambda row: (int(row['total_rank_upper_bound_with_signal']), row['family_id'])) if named_rows else None
    if best_named is not None:
        rows.append({'summary_name': 'best_named_signal_family', 'summary_value': best_named['family_id'], 'provenance': 'MEASURED_FROM_CODE', 'note': 'Best named 9-term signal family among the Step 68 sensitivity scans.'})
        rows.append({'summary_name': 'best_named_total_rank_upper_bound', 'summary_value': best_named['total_rank_upper_bound_with_signal'], 'provenance': 'MEASURED_FROM_CODE', 'note': 'Best measured total-rank upper bound among named signal families.'})
    rows.append({'summary_name': 'best_random_refined_signal_family', 'summary_value': random_best.get('best_random_refined_family', ''), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Best sampled random signal family after refinement scans.'})
    rows.append({'summary_name': 'best_random_refined_total_rank_upper_bound', 'summary_value': str(random_best.get('best_random_refined_total_rank_upper_bound', '')), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Best measured total-rank upper bound among refined random signal families.'})
    rows.append({'summary_name': 'any_signal_family_under_23', 'summary_value': str(any(int(row['total_rank_upper_bound_with_signal']) < 23 for row in named_rows if row['total_rank_upper_bound_with_signal']) or (str(random_best.get('best_random_refined_total_rank_upper_bound', '')).isdigit() and int(str(random_best.get('best_random_refined_total_rank_upper_bound', '999'))) < 23) or any(int(row['total_rank_upper_bound']) < 23 for row in low_tilings)), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Immediate alarm flag requested by the Step 68 spec.'})
    return rows


def build_markdown(summary_rows: list[dict], canonical_rank_rows: list[dict], best_tiling: dict, low_tilings: list[dict], named_rows: list[dict], random_refined_rows: list[dict]) -> str:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append
    w('# Step 68: Layered Correction Tiling')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Canonical Correction Rank Table')
    w('')
    w('| class | outputs | flattening LB | exact scan rank | slice upper bound | verified correction rank |')
    w('|-------|---------|---------------|-----------------|------------------|--------------------------|')
    for row in canonical_rank_rows:
        w(f"| {row['class_id']} | {row['representative_entries']} | {row['flattening_lower_bound']} | {row['exact_scan_rank_upper_bound']} | {row['exact_slice_rank_upper_bound']} | {row['verified_correction_rank_upper_bound']} |")
    w('')
    w('## Canonical Flat Tilings')
    w('')
    w(f"- Best canonical total cost: {summary['best_canonical_correction_tiling_total_rank_upper_bound']}")
    w(f"- Best canonical tiling signature: {summary['best_canonical_correction_tiling_signature']}")
    w(f"- Canonical tilings beating 23: {summary['canonical_correction_tilings_beating_23']}")
    if low_tilings:
        w('')
        w('| tiling signature | pieces | class ids | correction cost | total cost |')
        w('|------------------|--------|-----------|-----------------|------------|')
        for row in low_tilings:
            w(f"| {row['tiling_signature']} | {row['pieces']} | {row['class_ids']} | {row['correction_cost_upper_bound']} | {row['total_rank_upper_bound']} |")
    w('')
    w('## Signal Sensitivity')
    w('')
    w('| family | flattening LB | slice upper bound | measured correction rank | total cost |')
    w('|--------|---------------|------------------|---------------------------|------------|')
    for row in named_rows:
        w(f"| {row['family_id']} | {row['flattening_lower_bound']} | {row['exact_slice_rank_upper_bound']} | {row['verified_correction_rank_upper_bound']} | {row['total_rank_upper_bound_with_signal']} |")
    for row in random_refined_rows:
        w(f"| {row['family_id']} | {row['flattening_lower_bound']} | {row['exact_slice_rank_upper_bound']} | {row['verified_correction_rank_upper_bound']} | {row['total_rank_upper_bound_with_signal']} |")
    w('')
    w(f"- Any signal family under 23: {summary['any_signal_family_under_23']}")
    if 'canonical_fourier_nonzero_modes' in summary:
        w('')
        w('## Fourier Steering')
        w('')
        w(f"- Canonical Fourier modes: {summary['canonical_fourier_nonzero_modes']}")
        w(f"- Canonical component-rank sum upper bound: {summary['canonical_fourier_component_rank_sum_upper_bound']}")
        w(f"- Phase-j1 Fourier modes: {summary['phase_j1_fourier_nonzero_modes']}")
        w(f"- Phase-j1 component-rank sum upper bound: {summary['phase_j1_fourier_component_rank_sum_upper_bound']}")
        w(f"- Any Fourier component sum below 14: {summary['any_fourier_component_sum_below_14']}")
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    log('Step 68 start')

    log('Task 1-2: constructing exact signal and correction tensors')
    canonical_terms = canonical_signal_terms()
    canonical_verification_rows, canonical_summary = verify_signal_family('canonical_balanced', canonical_terms)

    log('Task 3: scanning canonical correction-piece ranks and rebuilding flat tilings')
    piece_lookup = build_piece_lookup()
    canonical_rank_rows, canonical_scan_rows, canonical_costs = canonical_correction_rank_table(piece_lookup, DEFAULT_WORKERS)
    canonical_tiling_rows, best_tiling, low_tilings = compute_correction_tilings(piece_lookup, canonical_costs)

    log('Task 4: running named signal-family sensitivity scans')
    named_rows: list[dict] = []
    named_scan_rows: list[dict] = []
    for family_id, terms in named_signal_families().items():
        verify_rows, verify_summary = verify_signal_family(family_id, terms)
        result = refine_signal_family_rank(family_id, terms, NAMED_SIGNAL_RESTARTS, DEFAULT_WORKERS)
        named_scan_rows.extend(result.pop('scan_rows'))
        result.update({
            'correction_live_min': verify_summary['correction_live_min'],
            'correction_live_max': verify_summary['correction_live_max'],
            'correction_dead_min': verify_summary['correction_dead_min'],
            'correction_dead_max': verify_summary['correction_dead_max'],
        })
        named_rows.append(result)
        canonical_verification_rows.extend(verify_rows)

    log('Task 5: sampling random signal layers and refining the best candidates')
    random_proxy_rows, random_refined_rows, random_best = random_signal_trial_rows(DEFAULT_WORKERS)
    random_scan_rows = random_best.pop('scan_rows')

    summary_rows = build_summary_rows(canonical_summary, best_tiling, low_tilings, named_rows, random_best)
    fourier_summary_rows, fourier_mode_rows, fourier_component_rows, fourier_markdown = run_fourier_extension_exports()
    summary_rows.extend(fourier_summary_rows)

    log('Writing Step 68 exports')
    write_csv(EXPORTS / 'step68_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step68_signal_layer_verification.csv', canonical_verification_rows, ['family_id', 'output_c', 'fiber', 'sigma_total', 'sigma_expected', 'eta1_total', 'eta2_total', 'provenance'])
    write_csv(EXPORTS / 'step68_canonical_signal_rank_table.csv', canonical_rank_rows, ['family', 'class_id', 'representative_entries', 'size', 'count_on_3x3', 'symmetry_class_size', 'row_profile', 'col_profile', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'best_rank_tested', 'best_verified_loss', 'exact_scan_rank_upper_bound', 'exact_slice_rank_upper_bound', 'verified_correction_rank_upper_bound', 'verified_rank_source', 'signal_family', 'provenance'])
    write_csv(EXPORTS / 'step68_canonical_signal_rank_scan.csv', canonical_scan_rows, ['piece_id', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'exact_rank_upper_bound', 'provenance'])
    write_csv(EXPORTS / 'step68_canonical_correction_tilings.csv', canonical_tiling_rows, ['tiling_id', 'piece_count', 'tiling_signature', 'pieces', 'class_ids', 'correction_cost_upper_bound', 'total_rank_upper_bound', 'beats_23', 'provenance'])
    write_csv(EXPORTS / 'step68_canonical_correction_tilings_lt23.csv', low_tilings, ['tiling_id', 'piece_count', 'tiling_signature', 'pieces', 'class_ids', 'correction_cost_upper_bound', 'total_rank_upper_bound', 'beats_23', 'provenance'])
    write_csv(EXPORTS / 'step68_named_signal_sensitivity.csv', named_rows, ['family_id', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'exact_slice_rank_upper_bound', 'scan_restarts', 'scan_max_rank', 'best_rank_tested', 'best_verified_loss', 'exact_scan_rank_upper_bound', 'verified_correction_rank_upper_bound', 'verified_rank_source', 'total_rank_upper_bound_with_signal', 'beats_23', 'provenance', 'correction_live_min', 'correction_live_max', 'correction_dead_min', 'correction_dead_max'])
    write_csv(EXPORTS / 'step68_named_signal_rank_scans.csv', named_scan_rows, ['piece_id', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'exact_rank_upper_bound', 'provenance'])
    write_csv(EXPORTS / 'step68_random_signal_proxy_rows.csv', random_proxy_rows, ['family_id', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'exact_slice_rank_upper_bound', 'proxy_total_rank_upper_bound_with_signal', 'provenance'])
    write_csv(EXPORTS / 'step68_random_signal_refined_rows.csv', random_refined_rows, ['family_id', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'exact_slice_rank_upper_bound', 'scan_restarts', 'scan_max_rank', 'best_rank_tested', 'best_verified_loss', 'exact_scan_rank_upper_bound', 'verified_correction_rank_upper_bound', 'verified_rank_source', 'total_rank_upper_bound_with_signal', 'beats_23', 'provenance'])
    write_csv(EXPORTS / 'step68_random_signal_rank_scans.csv', random_scan_rows, ['piece_id', 'rank_tested', 'restarts', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'exact_rank_upper_bound', 'provenance'])
    write_csv(EXPORTS / 'step68_fourier_summary.csv', fourier_summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step68_fourier_mode_decomposition.csv', fourier_mode_rows, ['family_id', 'mode_j', 'mode_k', 'coefficient_real', 'coefficient_imag', 'coefficient_abs', 'coefficient_json', 'is_nonzero', 'provenance'])
    write_csv(EXPORTS / 'step68_fourier_component_ranks.csv', fourier_component_rows, ['family_id', 'mode_j', 'mode_k', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'exact_slice_rank_upper_bound', 'exact_rank_if_determined', 'provenance'])
    write_text(EXPORTS / 'step68_layered_correction_tiling.md', build_markdown(summary_rows, canonical_rank_rows, best_tiling, low_tilings, named_rows, random_refined_rows))
    write_text(EXPORTS / 'step68_fourier_encoded_correction_layer.md', fourier_markdown)

    log(f"Canonical best tiling: total={best_tiling['total_rank_upper_bound']} signature={best_tiling['tiling_signature']}")
    log(f"Named best signal family: {min(named_rows, key=lambda row: (int(row['total_rank_upper_bound_with_signal']), row['family_id']))['family_id']}")
    log(f"Random best refined family: {random_best.get('best_random_refined_family', '')} total={random_best.get('best_random_refined_total_rank_upper_bound', '')}")
    log('Step 68 complete')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as exc:
        log(f'Step 68 failed: {exc}')
        traceback.print_exc()
        raise