"""
ade3x3_step69_interlocking_mechanism_analysis.py

Step 69: Interlocking Mechanism Analysis.

This step studies how the public rank-23 AlphaTensor 3x3 algorithm reuses its
dead-X directions across the three Step 68 Fourier modes. It then tests two
follow-up construction ideas: mode-routed 3-fiber corrections and a simple
three-mode, two-level residual scheme.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import sympy as sp

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
)
from src.ade3x3.steps.ade3x3_step64_small_integer_coefficient_enumeration import (
    ternary_projective_vectors,
)

EXPORTS = Path('outputs/exports')
OMEGA = np.exp(2j * np.pi / 3.0)
SYM_OMEGA = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
FOURIER_MODES = ((0, 0), (1, 2), (2, 1))
MODE_LABELS = {(0, 0): 'mode_00', (1, 2): 'mode_12', (2, 1): 'mode_21'}
NUMERIC_TOL = 1e-10
GREEDY_MAX_TERMS = 20


@dataclass(frozen=True)
class CandidatePair:
    candidate_id: str
    alpha: np.ndarray
    beta: np.ndarray
    source: str


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


def matrix_to_json(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix).tolist(), separators=(',', ':'))


def complex_value_to_json(value: complex) -> str:
    return json.dumps({'real': float(np.real(value)), 'imag': float(np.imag(value))}, separators=(',', ':'))


def complex_vector_to_json(vector: np.ndarray) -> str:
    return json.dumps(
        [{'real': float(np.real(value)), 'imag': float(np.imag(value))} for value in vector],
        separators=(',', ':'),
    )


def parse_matrix(text: str) -> np.ndarray:
    return np.array(ast.literal_eval(text), dtype=np.int8)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


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


def term_dead_mode_projection_vector(alpha: np.ndarray, beta: np.ndarray, freq_left: int, freq_right: int) -> np.ndarray:
    vector = np.zeros(9, dtype=np.complex128)
    for row_idx in range(3):
        for out_col in range(3):
            total = 0.0 + 0.0j
            for sum_left in range(3):
                for sum_right in range(3):
                    if sum_left == sum_right:
                        continue
                    total += (OMEGA ** (freq_left * sum_left + freq_right * sum_right)) * alpha[row_idx, sum_left] * beta[sum_right, out_col]
            vector[3 * row_idx + out_col] = total
    return vector


def term_dead_mode_projection_vector_sym(alpha: np.ndarray, beta: np.ndarray, freq_left: int, freq_right: int) -> list[sp.Expr]:
    vector: list[sp.Expr] = []
    for row_idx in range(3):
        for out_col in range(3):
            total = sp.Integer(0)
            for sum_left in range(3):
                for sum_right in range(3):
                    if sum_left == sum_right:
                        continue
                    total += (SYM_OMEGA ** (freq_left * sum_left + freq_right * sum_right)) * int(alpha[row_idx, sum_left]) * int(beta[sum_right, out_col])
            vector.append(sp.expand(total))
    return vector


def build_mode_matrices(terms: list[Term]) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], sp.Matrix]]:
    numeric: dict[tuple[int, int], np.ndarray] = {}
    symbolic: dict[tuple[int, int], sp.Matrix] = {}
    for mode in FOURIER_MODES:
        numeric_rows = []
        symbolic_rows = []
        for term in terms:
            numeric_rows.append(term_dead_mode_projection_vector(term.alpha, term.beta, *mode))
            symbolic_rows.append(term_dead_mode_projection_vector_sym(term.alpha, term.beta, *mode))
        numeric[mode] = np.vstack(numeric_rows)
        symbolic[mode] = sp.Matrix(symbolic_rows)
    return numeric, symbolic


def mode_norm_classification(norms: dict[tuple[int, int], float], dead_norm: float) -> str:
    if dead_norm < NUMERIC_TOL:
        return 'silent'
    ordered = sorted(norms.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] > 2.0 * max(ordered[1][1], ordered[2][1]):
        mode = ordered[0][0]
        return {
            (0, 0): 'DC-dominant',
            (1, 2): 'Mode12-dominant',
            (2, 1): 'Mode21-dominant',
        }[mode]
    return 'multi-mode'


def build_term_projection_rows(terms: list[Term], numeric_modes: dict[tuple[int, int], np.ndarray]) -> tuple[list[dict], list[dict]]:
    corrector_rows = {row['term_id']: row for row in read_csv_rows(EXPORTS / 'step67_alphatensor_corrector_terms.csv')}
    projection_rows: list[dict] = []
    heatmap_rows: list[dict] = []
    for idx, term in enumerate(terms):
        mode_norms = {mode: float(np.linalg.norm(numeric_modes[mode][idx])) for mode in FOURIER_MODES}
        dead_norm = float(np.sqrt(sum(value * value for value in mode_norms.values())))
        corrector = corrector_rows.get(term.term_id, {})
        projection_rows.append({
            'term_id': term.term_id,
            'source_label': term.source_label,
            'mode00_norm': f'{mode_norms[(0, 0)]:.16e}',
            'mode12_norm': f'{mode_norms[(1, 2)]:.16e}',
            'mode21_norm': f'{mode_norms[(2, 1)]:.16e}',
            'dead_x_mode_norm_total': f'{dead_norm:.16e}',
            'classification': mode_norm_classification(mode_norms, dead_norm),
            'mode00_vector_json': complex_vector_to_json(numeric_modes[(0, 0)][idx]),
            'mode12_vector_json': complex_vector_to_json(numeric_modes[(1, 2)][idx]),
            'mode21_vector_json': complex_vector_to_json(numeric_modes[(2, 1)][idx]),
            'step67_dominant_layer': corrector.get('dominant_layer', ''),
            'step67_dead_x_corrector': corrector.get('is_dead_x_corrector', ''),
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })
        for mode in FOURIER_MODES:
            heatmap_rows.append({
                'term_id': term.term_id,
                'mode_label': MODE_LABELS[mode],
                'value': f'{mode_norms[mode]:.16e}',
                'provenance': 'MEASURED_FROM_CODE',
            })
    return projection_rows, heatmap_rows


def columnspace_basis_matrix(matrix: sp.Matrix) -> sp.Matrix:
    basis = matrix.columnspace()
    if not basis:
        return sp.zeros(matrix.rows, 0)
    return sp.Matrix.hstack(*basis)


def intersection_basis_matrix(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    left_basis = columnspace_basis_matrix(left)
    right_basis = columnspace_basis_matrix(right)
    if left_basis.cols == 0 or right_basis.cols == 0:
        return sp.zeros(left.rows, 0)
    joined = left_basis.row_join(-right_basis)
    null_basis = joined.nullspace()
    if not null_basis:
        return sp.zeros(left.rows, 0)
    columns = []
    for vec in null_basis:
        coeff_left = vec[:left_basis.cols, :]
        intersection_col = sp.simplify(left_basis * coeff_left)
        if any(entry != 0 for entry in intersection_col):
            columns.append(intersection_col)
    if not columns:
        return sp.zeros(left.rows, 0)
    return columnspace_basis_matrix(sp.Matrix.hstack(*columns))


def build_intersection_rows(symbolic_modes: dict[tuple[int, int], sp.Matrix]) -> tuple[list[dict], dict[str, int]]:
    mode_ranks = {mode: int(symbolic_modes[mode].rank()) for mode in FOURIER_MODES}
    pair_00_12 = int(symbolic_modes[(0, 0)].rank() + symbolic_modes[(1, 2)].rank() - symbolic_modes[(0, 0)].row_join(symbolic_modes[(1, 2)]).rank())
    pair_00_21 = int(symbolic_modes[(0, 0)].rank() + symbolic_modes[(2, 1)].rank() - symbolic_modes[(0, 0)].row_join(symbolic_modes[(2, 1)]).rank())
    pair_12_21 = int(symbolic_modes[(1, 2)].rank() + symbolic_modes[(2, 1)].rank() - symbolic_modes[(1, 2)].row_join(symbolic_modes[(2, 1)]).rank())
    pair_basis = intersection_basis_matrix(symbolic_modes[(0, 0)], symbolic_modes[(1, 2)])
    triple_basis = intersection_basis_matrix(pair_basis, symbolic_modes[(2, 1)])
    triple_dim = int(triple_basis.rank())
    union_dim = int(symbolic_modes[(0, 0)].row_join(symbolic_modes[(1, 2)]).row_join(symbolic_modes[(2, 1)]).rank())
    summary = {
        'mode00_rank': mode_ranks[(0, 0)],
        'mode12_rank': mode_ranks[(1, 2)],
        'mode21_rank': mode_ranks[(2, 1)],
        'pair_00_12': pair_00_12,
        'pair_00_21': pair_00_21,
        'pair_12_21': pair_12_21,
        'triple': triple_dim,
        'union': union_dim,
    }
    rows = [
        {'subspace_name': 'V_00', 'dimension': str(mode_ranks[(0, 0)]), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_12', 'dimension': str(mode_ranks[(1, 2)]), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_21', 'dimension': str(mode_ranks[(2, 1)]), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_00_intersect_V_12', 'dimension': str(pair_00_12), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_00_intersect_V_21', 'dimension': str(pair_00_21), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_12_intersect_V_21', 'dimension': str(pair_12_21), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_00_intersect_V_12_intersect_V_21', 'dimension': str(triple_dim), 'provenance': 'EXACT_DERIVED'},
        {'subspace_name': 'V_00_union_V_12_union_V_21', 'dimension': str(union_dim), 'provenance': 'EXACT_DERIVED'},
    ]
    return rows, summary


def heatmap_svg(terms: list[str], values: np.ndarray, path: Path) -> None:
    cell_w = 76
    cell_h = 24
    left_pad = 72
    top_pad = 42
    width = left_pad + 3 * cell_w + 12
    height = top_pad + len(terms) * cell_h + 12
    max_val = float(np.max(values)) if values.size else 1.0
    if max_val < NUMERIC_TOL:
        max_val = 1.0
    header = ['mode_00', 'mode_12', 'mode_21']
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Consolas, monospace; font-size:11px; fill:#111} .small{font-size:10px}</style>',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text x="8" y="18">Step 69 mode participation heatmap</text>',
    ]
    for col_idx, label in enumerate(header):
        x = left_pad + col_idx * cell_w + 6
        lines.append(f'<text x="{x}" y="32">{label}</text>')
    for row_idx, term_id in enumerate(terms):
        y = top_pad + row_idx * cell_h
        lines.append(f'<text x="8" y="{y + 16}" class="small">{term_id}</text>')
        for col_idx in range(3):
            value = float(values[row_idx, col_idx])
            scaled = min(max(value / max_val, 0.0), 1.0)
            red = int(255 - 40 * scaled)
            green = int(255 - 120 * scaled)
            blue = int(255 - 220 * scaled)
            x = left_pad + col_idx * cell_w
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="rgb({red},{green},{blue})" stroke="#ddd"/>')
            lines.append(f'<text x="{x + 4}" y="{y + 16}" class="small">{value:.3g}</text>')
    lines.append('</svg>')
    write_text(path, '\n'.join(lines))


def dead_mode_slice(freq_left: int, freq_right: int) -> np.ndarray:
    slice_matrix = np.zeros((3, 3), dtype=np.complex128)
    for sum_left in range(3):
        for sum_right in range(3):
            if sum_left == sum_right:
                continue
            slice_matrix[sum_left, sum_right] = OMEGA ** (freq_left * sum_left + freq_right * sum_right)
    return slice_matrix


def tensor_from_group_slice(cells: tuple[tuple[int, int], ...], slice_matrix: np.ndarray) -> np.ndarray:
    tensor = np.zeros((9, 9, len(cells)), dtype=np.complex128)
    for local_idx, (row_idx, out_col) in enumerate(cells):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                a_idx = 3 * row_idx + sum_left
                b_idx = 3 * sum_right + out_col
                tensor[a_idx, b_idx, local_idx] = slice_matrix[sum_left, sum_right]
    return tensor


def slice_rank_upper_bound(tensor: np.ndarray) -> int:
    return sum(int(np.linalg.matrix_rank(tensor[:, :, idx])) for idx in range(tensor.shape[2]))


def canonicalize_three_cell_set(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    best: tuple[tuple[int, int], ...] | None = None
    for row_perm in permutations(range(3)):
        for col_perm in permutations(range(3)):
            moved = tuple(sorted((row_perm[row_idx], col_perm[col_idx]) for row_idx, col_idx in cells))
            if best is None or moved < best:
                best = moved
    assert best is not None
    return best


def classify_three_cell_classes() -> dict[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]]:
    groups: dict[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]] = {}
    all_cells = tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3))
    for cells in combinations(all_cells, 3):
        canonical = canonicalize_three_cell_set(tuple(sorted(cells)))
        groups.setdefault(canonical, []).append(tuple(sorted(cells)))
    return groups


def mode_routed_rows() -> tuple[list[dict], dict[str, str]]:
    class_map = classify_three_cell_classes()
    rows: list[dict] = []
    best_total = None
    for canonical_cells, orbit_members in sorted(class_map.items()):
        for mode in ((1, 2), (2, 1)):
            tensor = tensor_from_group_slice(canonical_cells, dead_mode_slice(*mode))
            rank_a, rank_b, rank_c, rank_lb = flattening_ranks(tensor)
            slice_ub = slice_rank_upper_bound(tensor)
            exact_rank = str(rank_lb) if rank_lb == slice_ub else ''
            total_cost = 9 + 3 * int(exact_rank or slice_ub)
            best_total = total_cost if best_total is None else min(best_total, total_cost)
            rows.append({
                'cell_class': '; '.join(f'({r},{u})' for r, u in canonical_cells),
                'orbit_size': str(len(orbit_members)),
                'mode_label': MODE_LABELS[mode],
                'flattening_rank_A_BC': str(rank_a),
                'flattening_rank_B_AC': str(rank_b),
                'flattening_rank_C_AB': str(rank_c),
                'flattening_lower_bound': str(rank_lb),
                'slice_rank_upper_bound': str(slice_ub),
                'exact_rank_if_determined': exact_rank,
                'total_algorithm_cost_if_repeated_three_times': str(total_cost),
                'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
            })
    summary = {
        'mode_routed_classes_tested': str(len(rows)),
        'mode_routed_best_total_cost_upper_bound': str(best_total),
        'mode_routed_dc_only_achievable_by_single_phase_family': 'False',
    }
    return rows, summary


def best_rank1_cp_als(target: np.ndarray, restarts: int = 12, iterations: int = 80, seed_base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    best_loss = float('inf')
    best = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed_base + 97 * restart)
        alpha = rng.standard_normal(target.shape[0]) + 1j * rng.standard_normal(target.shape[0])
        beta = rng.standard_normal(target.shape[1]) + 1j * rng.standard_normal(target.shape[1])
        gamma = rng.standard_normal(target.shape[2]) + 1j * rng.standard_normal(target.shape[2])
        for _ in range(iterations):
            denom = np.vdot(beta, beta) * np.vdot(gamma, gamma)
            alpha = np.einsum('abc,b,c->a', target, np.conjugate(beta), np.conjugate(gamma), optimize=True) / denom
            denom = np.vdot(alpha, alpha) * np.vdot(gamma, gamma)
            beta = np.einsum('abc,a,c->b', target, np.conjugate(alpha), np.conjugate(gamma), optimize=True) / denom
            denom = np.vdot(alpha, alpha) * np.vdot(beta, beta)
            gamma = np.einsum('abc,a,b->c', target, np.conjugate(alpha), np.conjugate(beta), optimize=True) / denom
        approx = np.einsum('a,b,c->abc', alpha, beta, gamma, optimize=True)
        loss = float(np.sum(np.abs(target - approx) ** 2))
        if loss < best_loss:
            best_loss = loss
            best = (alpha.copy(), beta.copy(), gamma.copy())
    assert best is not None
    return best[0], best[1], best[2], best_loss


def multilevel_rows() -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    residual = np.zeros((9, 9, 9), dtype=np.complex128)
    for mode in FOURIER_MODES:
        tensor = tensor_from_group_slice(tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3)), dead_mode_slice(*mode))
        alpha, beta, gamma, loss = best_rank1_cp_als(tensor, seed_base=100 * mode[0] + 13 * mode[1])
        approx = np.einsum('a,b,c->abc', alpha, beta, gamma, optimize=True)
        mode_residual = tensor - approx
        residual += mode_residual
        rank_a, rank_b, rank_c, rank_lb = flattening_ranks(mode_residual)
        slice_ub = slice_rank_upper_bound(mode_residual)
        rows.append({
            'family': MODE_LABELS[mode],
            'rank1_als_loss': f'{loss:.16e}',
            'residual_flattening_rank_A_BC': str(rank_a),
            'residual_flattening_rank_B_AC': str(rank_b),
            'residual_flattening_rank_C_AB': str(rank_c),
            'residual_flattening_lower_bound': str(rank_lb),
            'residual_slice_rank_upper_bound': str(slice_ub),
            'provenance': 'MEASURED_FROM_CODE',
        })
    comb_rank_a, comb_rank_b, comb_rank_c, comb_rank_lb = flattening_ranks(residual)
    comb_slice_ub = slice_rank_upper_bound(residual)
    summary = {
        'multilevel_combined_residual_flattening_lower_bound': str(comb_rank_lb),
        'multilevel_combined_residual_slice_rank_upper_bound': str(comb_slice_ub),
        'multilevel_total_cost_upper_bound_via_slice_rank': str(9 + 3 + comb_slice_ub),
        'multilevel_total_cost_lower_bound_via_flattening': str(9 + 3 + comb_rank_lb),
    }
    rows.append({
        'family': 'combined_three_mode_residual',
        'rank1_als_loss': '',
        'residual_flattening_rank_A_BC': str(comb_rank_a),
        'residual_flattening_rank_B_AC': str(comb_rank_b),
        'residual_flattening_rank_C_AB': str(comb_rank_c),
        'residual_flattening_lower_bound': str(comb_rank_lb),
        'residual_slice_rank_upper_bound': str(comb_slice_ub),
        'provenance': 'MEASURED_FROM_CODE',
    })
    return rows, summary


def permute_pair(alpha: np.ndarray, beta: np.ndarray, row_perm: tuple[int, int, int], shared_perm: tuple[int, int, int], col_perm: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    moved_alpha = np.zeros_like(alpha)
    moved_beta = np.zeros_like(beta)
    for row_idx in range(3):
        for sum_idx in range(3):
            moved_alpha[row_perm[row_idx], shared_perm[sum_idx]] = alpha[row_idx, sum_idx]
    for sum_idx in range(3):
        for out_col in range(3):
            moved_beta[shared_perm[sum_idx], col_perm[out_col]] = beta[sum_idx, out_col]
    return moved_alpha, moved_beta


def build_reduced_candidate_pool(terms: list[Term]) -> tuple[list[CandidatePair], str]:
    candidates: dict[bytes, CandidatePair] = {}
    top_rows = read_csv_rows(EXPORTS / 'step64_top100_usefulness_profiles.csv')
    for row_idx, row in enumerate(top_rows):
        alpha = parse_matrix(row['alpha'])
        beta = parse_matrix(row['beta'])
        key = alpha.tobytes() + b'|' + beta.tobytes()
        candidates.setdefault(key, CandidatePair(f'step64_top_{row_idx + 1:03d}', alpha, beta, 'step64_top100_usefulness_profiles'))
    perms = list(permutations(range(3)))
    for term in terms:
        for row_perm in perms[:3]:
            for shared_perm in perms[:3]:
                for col_perm in perms[:3]:
                    moved_alpha, moved_beta = permute_pair(term.alpha.astype(np.int8), term.beta.astype(np.int8), row_perm, shared_perm, col_perm)
                    key = moved_alpha.tobytes() + b'|' + moved_beta.tobytes()
                    candidates.setdefault(key, CandidatePair(f'{term.term_id}_{len(candidates):04d}', moved_alpha, moved_beta, 'alphatensor_term_orbit_seed'))
    return list(candidates.values()), 'reduced_shortlist_from_step64_top100_plus_alphatensor_orbit_seeds'


def candidate_mode_vectors(candidate: CandidatePair) -> dict[tuple[int, int], np.ndarray]:
    return {mode: term_dead_mode_projection_vector(candidate.alpha, candidate.beta, *mode) for mode in FOURIER_MODES}


def row_rank_with_candidate(existing: list[np.ndarray], candidate: np.ndarray) -> tuple[int, float]:
    if not existing:
        residual = candidate.copy()
        return (1 if np.linalg.norm(residual) > NUMERIC_TOL else 0), float(np.linalg.norm(residual))
    matrix = np.vstack(existing + [candidate])
    rank = numeric_rank(matrix)
    previous_rank = numeric_rank(np.vstack(existing))
    if previous_rank == rank:
        q, _, _, _ = np.linalg.lstsq(np.vstack(existing).T, candidate, rcond=None)
        residual = candidate - np.vstack(existing).T @ q
    else:
        residual = candidate.copy()
    return rank, float(np.linalg.norm(residual))


def run_reduced_greedy_cover(candidates: list[CandidatePair]) -> tuple[list[dict], dict[str, str]]:
    selected_rows: list[dict] = []
    chosen_vectors = {mode: [] for mode in FOURIER_MODES}
    chosen_ids: set[str] = set()
    reached_step = ''
    for step_idx in range(1, GREEDY_MAX_TERMS + 1):
        best = None
        best_score = None
        for candidate in candidates:
            if candidate.candidate_id in chosen_ids:
                continue
            vectors = candidate_mode_vectors(candidate)
            proposed_ranks = []
            proposed_norms = []
            for mode in FOURIER_MODES:
                rank_after, residual_norm = row_rank_with_candidate(chosen_vectors[mode], vectors[mode])
                proposed_ranks.append(rank_after)
                proposed_norms.append(residual_norm if rank_after > len(chosen_vectors[mode]) else 0.0)
            score = (min(proposed_ranks), sum(proposed_ranks), min(proposed_norms), sum(proposed_norms))
            if best_score is None or score > best_score:
                best_score = score
                best = (candidate, vectors, proposed_ranks, proposed_norms)
        if best is None:
            break
        candidate, vectors, proposed_ranks, proposed_norms = best
        chosen_ids.add(candidate.candidate_id)
        for mode in FOURIER_MODES:
            chosen_vectors[mode].append(vectors[mode])
        all_full = all(numeric_rank(np.vstack(chosen_vectors[mode])) >= 9 for mode in FOURIER_MODES)
        selected_rows.append({
            'step': str(step_idx),
            'candidate_id': candidate.candidate_id,
            'candidate_source': candidate.source,
            'mode00_rank_after': str(numeric_rank(np.vstack(chosen_vectors[(0, 0)]))),
            'mode12_rank_after': str(numeric_rank(np.vstack(chosen_vectors[(1, 2)]))),
            'mode21_rank_after': str(numeric_rank(np.vstack(chosen_vectors[(2, 1)]))),
            'mode00_residual_gain_norm': f'{proposed_norms[0]:.16e}',
            'mode12_residual_gain_norm': f'{proposed_norms[1]:.16e}',
            'mode21_residual_gain_norm': f'{proposed_norms[2]:.16e}',
            'all_three_modes_rank9': str(all_full),
            'alpha': matrix_to_json(candidate.alpha.astype(int)),
            'beta': matrix_to_json(candidate.beta.astype(int)),
            'provenance': 'MEASURED_FROM_CODE',
        })
        if all_full and not reached_step:
            reached_step = str(step_idx)
            break
    summary = {
        'reduced_greedy_pool_scope': 'reduced_shortlist_only',
        'reduced_greedy_candidate_count': str(len(candidates)),
        'reduced_greedy_first_rank9_cover_step': reached_step or 'not_reached',
    }
    return selected_rows, summary


def markdown_report(
    intersection_summary: dict[str, int],
    routed_summary: dict[str, str],
    multilevel_summary: dict[str, str],
    greedy_summary: dict[str, str],
    projection_rows: list[dict],
) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 69: Interlocking Mechanism Analysis')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Task 1: AlphaTensor Fourier-mode interlocking')
    w('')
    w(f"- Mode ranks in term space: V_00={intersection_summary['mode00_rank']}, V_12={intersection_summary['mode12_rank']}, V_21={intersection_summary['mode21_rank']}")
    w(f"- Pair intersections: dim(V_00 ∩ V_12)={intersection_summary['pair_00_12']}, dim(V_00 ∩ V_21)={intersection_summary['pair_00_21']}, dim(V_12 ∩ V_21)={intersection_summary['pair_12_21']}")
    w(f"- Triple intersection: dim(V_00 ∩ V_12 ∩ V_21)={intersection_summary['triple']}")
    w(f"- Union dimension: dim(V_00 ∪ V_12 ∪ V_21)={intersection_summary['union']}")
    w('')
    w('### Dominant / multi-mode term classes')
    w('')
    for row in projection_rows:
        w(f"- {row['term_id']}: {row['classification']} | ||00||={float(row['mode00_norm']):.3g} ||12||={float(row['mode12_norm']):.3g} ||21||={float(row['mode21_norm']):.3g}")
    w('')
    w('## Task 3: Mode-routed 3-fiber correction probes')
    w('')
    w(f"- Best tested repeated-three-group total cost upper bound: {routed_summary['mode_routed_best_total_cost_upper_bound']}")
    w(f"- DC-only single-phase routing achievable by the reciprocal phase family: {routed_summary['mode_routed_dc_only_achievable_by_single_phase_family']}")
    w('')
    w('## Task 4: Three-mode two-level residual probe')
    w('')
    w(f"- Combined residual flattening lower bound after one rank-1 ALS mega-corrector per mode: {multilevel_summary['multilevel_combined_residual_flattening_lower_bound']}")
    w(f"- Combined residual slice-rank upper bound: {multilevel_summary['multilevel_combined_residual_slice_rank_upper_bound']}")
    w(f"- Total cost window from this probe: {multilevel_summary['multilevel_total_cost_lower_bound_via_flattening']} .. {multilevel_summary['multilevel_total_cost_upper_bound_via_slice_rank']}")
    w('')
    w('## Task 2: Reduced ternary greedy covering')
    w('')
    w(f"- Pool scope: {greedy_summary['reduced_greedy_pool_scope']}")
    w(f"- Candidate count: {greedy_summary['reduced_greedy_candidate_count']}")
    w(f"- First step where all three mode spans reached rank 9: {greedy_summary['reduced_greedy_first_rank9_cover_step']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The exact Step 69 interlocking object is not the 9-dimensional target-space mode image; it is the column space of each 23x9 term-participation matrix inside term space. That is the right space for inclusion-exclusion against the measured rank-14 nuisance budget. The pairwise and triple intersections therefore directly measure how many AlphaTensor coefficient directions are reused across the three Fourier modes.')
    w('The mode-routed probes test the optimistic single-mode dead-routing idea only on the dead block itself. The multilevel probe is also intentionally modest: one complex rank-1 ALS mega-corrector per mode plus the measured residual bounds. The reduced ternary greedy is heuristic because Step 64 exports the exact orbit counts but not a full 570,521-representative candidate table.')
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)

    terms, _, _ = load_public_rank23_terms()
    numeric_modes, symbolic_modes = build_mode_matrices(terms)
    projection_rows, heatmap_rows = build_term_projection_rows(terms, numeric_modes)
    intersection_rows, intersection_summary = build_intersection_rows(symbolic_modes)

    rank_rows = []
    for mode in FOURIER_MODES:
        matrix = numeric_modes[mode]
        rank_rows.append({
            'mode_label': MODE_LABELS[mode],
            'matrix_shape': f'{matrix.shape[0]}x{matrix.shape[1]}',
            'numeric_rank': str(numeric_rank(matrix)),
            'exact_rank': str(symbolic_modes[mode].rank()),
            'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE',
        })

    heatmap_matrix = np.array(
        [[float(row['mode00_norm']), float(row['mode12_norm']), float(row['mode21_norm'])] for row in projection_rows],
        dtype=np.float64,
    )
    heatmap_svg([row['term_id'] for row in projection_rows], heatmap_matrix, EXPORTS / 'step69_mode_participation_heatmap.svg')

    routed_rows, routed_summary = mode_routed_rows()
    multilevel_detail_rows, multilevel_summary = multilevel_rows()
    candidates, candidate_scope = build_reduced_candidate_pool(terms)
    greedy_rows, greedy_summary = run_reduced_greedy_cover(candidates)
    greedy_summary['reduced_greedy_pool_scope'] = candidate_scope

    summary_rows = [
        {'summary_name': 'alphatensor_mode00_rank', 'summary_value': str(intersection_summary['mode00_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact column-space rank of D^{00} inside the 23-dimensional term space.'},
        {'summary_name': 'alphatensor_mode12_rank', 'summary_value': str(intersection_summary['mode12_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact column-space rank of D^{12} inside the 23-dimensional term space.'},
        {'summary_name': 'alphatensor_mode21_rank', 'summary_value': str(intersection_summary['mode21_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact column-space rank of D^{21} inside the 23-dimensional term space.'},
        {'summary_name': 'alphatensor_pair_intersection_00_12', 'summary_value': str(intersection_summary['pair_00_12']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact dimension of V_00 ∩ V_12.'},
        {'summary_name': 'alphatensor_pair_intersection_00_21', 'summary_value': str(intersection_summary['pair_00_21']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact dimension of V_00 ∩ V_21.'},
        {'summary_name': 'alphatensor_pair_intersection_12_21', 'summary_value': str(intersection_summary['pair_12_21']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact dimension of V_12 ∩ V_21.'},
        {'summary_name': 'alphatensor_triple_intersection_dimension', 'summary_value': str(intersection_summary['triple']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact dimension of V_00 ∩ V_12 ∩ V_21.'},
        {'summary_name': 'alphatensor_three_mode_union_dimension', 'summary_value': str(intersection_summary['union']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact dimension of V_00 + V_12 + V_21 inside the 23-dimensional term space.'},
        {'summary_name': 'mode_routed_best_total_cost_upper_bound', 'summary_value': routed_summary['mode_routed_best_total_cost_upper_bound'], 'provenance': 'EXACT_DERIVED + MEASURED_FROM_CODE', 'note': 'Best repeated-three-group total cost from the tested 3-fiber single-mode routing classes.'},
        {'summary_name': 'multilevel_total_cost_lower_bound_via_flattening', 'summary_value': multilevel_summary['multilevel_total_cost_lower_bound_via_flattening'], 'provenance': 'MEASURED_FROM_CODE', 'note': '9 signal terms + 3 mega-correctors + combined residual flattening lower bound.'},
        {'summary_name': 'multilevel_total_cost_upper_bound_via_slice_rank', 'summary_value': multilevel_summary['multilevel_total_cost_upper_bound_via_slice_rank'], 'provenance': 'MEASURED_FROM_CODE', 'note': '9 signal terms + 3 mega-correctors + combined residual slice-rank upper bound.'},
        {'summary_name': 'reduced_greedy_pool_scope', 'summary_value': greedy_summary['reduced_greedy_pool_scope'], 'provenance': 'MEASURED_FROM_CODE', 'note': 'Scope of the ternary-pool shortlist used in the reduced greedy covering probe.'},
        {'summary_name': 'reduced_greedy_first_rank9_cover_step', 'summary_value': greedy_summary['reduced_greedy_first_rank9_cover_step'], 'provenance': 'MEASURED_FROM_CODE', 'note': 'First greedy step where the reduced shortlist spans rank 9 in all three modes, if reached.'},
    ]

    report = markdown_report(intersection_summary, routed_summary, multilevel_summary, greedy_summary, projection_rows)

    write_csv(
        EXPORTS / 'step69_summary.csv',
        summary_rows,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step69_term_mode_projections.csv',
        projection_rows,
        [
            'term_id', 'source_label', 'mode00_norm', 'mode12_norm', 'mode21_norm', 'dead_x_mode_norm_total',
            'classification', 'mode00_vector_json', 'mode12_vector_json', 'mode21_vector_json',
            'step67_dominant_layer', 'step67_dead_x_corrector', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step69_mode_participation_matrix.csv',
        heatmap_rows,
        ['term_id', 'mode_label', 'value', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step69_mode_subspace_dimensions.csv',
        intersection_rows,
        ['subspace_name', 'dimension', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step69_mode_matrix_ranks.csv',
        rank_rows,
        ['mode_label', 'matrix_shape', 'numeric_rank', 'exact_rank', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step69_mode_routed_three_fiber_ranks.csv',
        routed_rows,
        [
            'cell_class', 'orbit_size', 'mode_label', 'flattening_rank_A_BC', 'flattening_rank_B_AC',
            'flattening_rank_C_AB', 'flattening_lower_bound', 'slice_rank_upper_bound',
            'exact_rank_if_determined', 'total_algorithm_cost_if_repeated_three_times', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step69_multilevel_residual_probe.csv',
        multilevel_detail_rows,
        [
            'family', 'rank1_als_loss', 'residual_flattening_rank_A_BC', 'residual_flattening_rank_B_AC',
            'residual_flattening_rank_C_AB', 'residual_flattening_lower_bound',
            'residual_slice_rank_upper_bound', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step69_reduced_ternary_greedy_cover.csv',
        greedy_rows,
        [
            'step', 'candidate_id', 'candidate_source', 'mode00_rank_after', 'mode12_rank_after',
            'mode21_rank_after', 'mode00_residual_gain_norm', 'mode12_residual_gain_norm',
            'mode21_residual_gain_norm', 'all_three_modes_rank9', 'alpha', 'beta', 'provenance',
        ],
    )
    write_text(EXPORTS / 'step69_interlocking_mechanism_analysis.md', report)


if __name__ == '__main__':
    main()