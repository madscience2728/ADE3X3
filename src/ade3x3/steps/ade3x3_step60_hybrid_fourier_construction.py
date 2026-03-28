"""
ade3x3_step60_hybrid_fourier_construction.py

Step 60: Hybrid Fourier Construction.

This step treats the hybrid ansatz as a direct sum of two blocks:

1. Same-fiber Fourier triples, which are completely modular and solve their
   selected fibers exactly without leaking to other output fibers.
2. A residual spreader problem on the remaining output fibers, analyzed here
   through exact sub-tensor construction and flattening-rank lower bounds.
"""

from __future__ import annotations

import csv
from datetime import datetime
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

EXPORTS = Path('outputs/exports')
OMEGA = np.exp(2j * np.pi / 3)
TOL = 1e-8
ALL_FIBERS = [(row_idx, col_idx) for row_idx in range(3) for col_idx in range(3)]
ROW_PERMS = list(permutations(range(3)))
COL_PERMS = list(permutations(range(3)))
TARGET_RANKS = [19, 20, 21, 22, 23]
KNOWN_RECTANGLE_RANKS = {
    (1, 1): 3,
    (1, 2): 6,
    (2, 1): 6,
    (1, 3): 9,
    (3, 1): 9,
    (2, 2): 11,
    (2, 3): 15,
    (3, 2): 15,
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_markdown(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f"  Wrote markdown -> {path}")


def numeric_rank(matrix: np.ndarray, tol: float | None = None) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    if tol is None:
        tol = max(matrix.shape) * np.finfo(float).eps * float(np.max(singular_values)) * 10.0
    return int(np.sum(singular_values > tol))


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def fiber_str(fiber: tuple[int, int]) -> str:
    return f"({fiber[0]},{fiber[1]})"


def fibers_str(fibers: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> str:
    return '; '.join(fiber_str(fiber) for fiber in fibers)


def make_samefiber_term(row_idx: int, col_idx: int, channel_j: int, gamma_scale: complex = 1.0) -> dict[str, np.ndarray]:
    alpha = np.zeros((3, 3), dtype=np.complex128)
    beta = np.zeros((3, 3), dtype=np.complex128)
    gamma = np.zeros((3, 3), dtype=np.complex128)
    for sum_idx in range(3):
        alpha[row_idx, sum_idx] = OMEGA ** (sum_idx * channel_j)
        beta[sum_idx, col_idx] = OMEGA ** (-sum_idx * channel_j)
    gamma[row_idx, col_idx] = gamma_scale
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma}


def evaluate_terms(terms: list[dict[str, np.ndarray]]) -> np.ndarray:
    result = np.zeros((3, 3, 3, 3, 3, 3), dtype=np.complex128)
    for term in terms:
        result += np.einsum('ab,cd,ef->efabcd', term['alpha'], term['beta'], term['gamma'], optimize=True)
    return result


def restricted_target_tensor(selected_outputs: set[tuple[int, int]]) -> np.ndarray:
    tensor = np.zeros((3, 3, 3, 3, 3, 3), dtype=np.complex128)
    for out_row, out_col in selected_outputs:
        for sum_idx in range(3):
            tensor[out_row, out_col, out_row, sum_idx, sum_idx, out_col] = 1.0
    return tensor


def fourier_terms_for_subset(selected_outputs: tuple[tuple[int, int], ...]) -> list[dict[str, np.ndarray]]:
    terms: list[dict[str, np.ndarray]] = []
    for row_idx, col_idx in selected_outputs:
        for channel_j in range(3):
            terms.append(make_samefiber_term(row_idx, col_idx, channel_j, gamma_scale=1.0 / 3.0))
    return terms


def modularity_rows() -> tuple[list[dict], list[dict], dict[str, str]]:
    single_rows: list[dict] = []
    subset_rows: list[dict] = []

    for fiber in ALL_FIBERS:
        subset = (fiber,)
        actual = evaluate_terms(fourier_terms_for_subset(subset))
        target = restricted_target_tensor(set(subset))
        single_rows.append({
            'fiber': fiber_str(fiber),
            'term_count': 3,
            'max_abs_residual': f'{max_abs_difference(actual, target):.10f}',
            'max_abs_leakage_to_other_outputs': f'{max_abs_difference(actual, restricted_target_tensor(set(subset))):.10f}',
            'passes': str(max_abs_difference(actual, target) <= TOL),
            'provenance': 'EXACT_DERIVED',
        })

    subset_index = 0
    by_size = {size: {'subset_count': 0, 'passing_count': 0, 'max_residual': 0.0} for size in range(10)}
    for size in range(10):
        for subset in combinations(ALL_FIBERS, size):
            subset_index += 1
            actual = evaluate_terms(fourier_terms_for_subset(subset))
            target = restricted_target_tensor(set(subset))
            residual = max_abs_difference(actual, target)
            passes = residual <= TOL
            by_size[size]['subset_count'] += 1
            by_size[size]['max_residual'] = max(by_size[size]['max_residual'], residual)
            if passes:
                by_size[size]['passing_count'] += 1
            subset_rows.append({
                'subset_id': subset_index,
                'subset_size': size,
                'fiber_subset': fibers_str(subset),
                'term_count': 3 * size,
                'max_abs_residual': f'{residual:.10f}',
                'passes': str(passes),
                'provenance': 'EXACT_DERIVED',
            })

    summary = {
        'modularity_total_subsets_checked': str(sum(by_size[size]['subset_count'] for size in range(10))),
        'modularity_total_passing_subsets': str(sum(by_size[size]['passing_count'] for size in range(10))),
        'modularity_max_abs_residual': f"{max(by_size[size]['max_residual'] for size in range(10)):.10f}",
    }
    summary_rows = []
    for size in range(10):
        summary_rows.append({
            'subset_size': size,
            'subset_count': by_size[size]['subset_count'],
            'passing_count': by_size[size]['passing_count'],
            'max_abs_residual': f"{by_size[size]['max_residual']:.10f}",
            'provenance': 'EXACT_DERIVED',
        })
    return single_rows, summary_rows, summary


def reduced_output_tensor(selected_outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    tensor = np.zeros((9, 9, len(selected_outputs)), dtype=np.float64)
    for output_idx, (row_idx, col_idx) in enumerate(selected_outputs):
        for sum_idx in range(3):
            a_idx = 3 * row_idx + sum_idx
            b_idx = 3 * sum_idx + col_idx
            tensor[a_idx, b_idx, output_idx] = 1.0
    return tensor


def flattening_ranks(selected_outputs: tuple[tuple[int, int], ...]) -> dict[str, int]:
    tensor = reduced_output_tensor(selected_outputs)
    rank_a = numeric_rank(tensor.reshape(9, -1))
    rank_b = numeric_rank(np.transpose(tensor, (1, 0, 2)).reshape(9, -1))
    rank_c = numeric_rank(np.transpose(tensor, (2, 0, 1)).reshape(len(selected_outputs), -1))
    return {
        'flattening_rank_A_BC': rank_a,
        'flattening_rank_B_AC': rank_b,
        'flattening_rank_C_AB': rank_c,
        'flattening_rank_max': max(rank_a, rank_b, rank_c),
    }


def output_perm_key(selected_outputs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    best = None
    for row_perm in ROW_PERMS:
        for col_perm in COL_PERMS:
            transformed = tuple(sorted((row_perm[row_idx], col_perm[col_idx]) for row_idx, col_idx in selected_outputs))
            if best is None or transformed < best:
                best = transformed
    assert best is not None
    return best


def fiber_profiles(selected_outputs: tuple[tuple[int, int], ...]) -> dict[str, str | int | bool]:
    row_counts = [0, 0, 0]
    col_counts = [0, 0, 0]
    row_support = set()
    col_support = set()
    for row_idx, col_idx in selected_outputs:
        row_counts[row_idx] += 1
        col_counts[col_idx] += 1
        row_support.add(row_idx)
        col_support.add(col_idx)
    row_support_list = sorted(row_support)
    col_support_list = sorted(col_support)
    rectangle = tuple(sorted((row_idx, col_idx) for row_idx in row_support_list for col_idx in col_support_list))
    is_rectangle = rectangle == tuple(sorted(selected_outputs))
    exact_rank = ''
    exact_rank_source = ''
    dims = (len(row_support_list), len(col_support_list))
    if is_rectangle and dims in KNOWN_RECTANGLE_RANKS:
        exact_rank = str(KNOWN_RECTANGLE_RANKS[dims])
        exact_rank_source = f'known_exact_rank_<{dims[0]},3,{dims[1]}>'
    return {
        'row_support_size': len(row_support_list),
        'col_support_size': len(col_support_list),
        'row_profile': ','.join(str(value) for value in sorted(row_counts, reverse=True)),
        'col_profile': ','.join(str(value) for value in sorted(col_counts, reverse=True)),
        'is_rectangle': is_rectangle,
        'exact_rank_if_known': exact_rank,
        'exact_rank_source': exact_rank_source,
    }


def orbit_tables() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    subset_rows: list[dict] = []
    orbit_map: dict[tuple[tuple[int, int], ...], dict] = {}

    for size in range(1, 10):
        for subset in combinations(ALL_FIBERS, size):
            subset_tuple = tuple(sorted(subset))
            rep = output_perm_key(subset_tuple)
            ranks = flattening_ranks(subset_tuple)
            profiles = fiber_profiles(subset_tuple)
            known_exact = int(profiles['exact_rank_if_known']) if profiles['exact_rank_if_known'] else None
            subset_row = {
                'remaining_fiber_count': size,
                'remaining_fibers': fibers_str(subset_tuple),
                'orbit_representative': fibers_str(rep),
                'row_profile': profiles['row_profile'],
                'col_profile': profiles['col_profile'],
                'is_rectangle': str(profiles['is_rectangle']),
                'flattening_rank_A_BC': ranks['flattening_rank_A_BC'],
                'flattening_rank_B_AC': ranks['flattening_rank_B_AC'],
                'flattening_rank_C_AB': ranks['flattening_rank_C_AB'],
                'flattening_rank_max': ranks['flattening_rank_max'],
                'exact_rank_if_known': profiles['exact_rank_if_known'],
                'exact_rank_source': profiles['exact_rank_source'],
                'provenance': 'EXACT_DERIVED',
            }
            for total_rank in TARGET_RANKS:
                available = total_rank - 3 * (9 - size)
                subset_row[f'available_spreaders_R{total_rank}'] = available
                subset_row[f'flattening_rules_out_R{total_rank}'] = str(ranks['flattening_rank_max'] > available)
                subset_row[f'exact_rank_rules_out_R{total_rank}'] = str(known_exact is not None and known_exact > available)
            subset_rows.append(subset_row)
            if rep not in orbit_map:
                orbit_row = {
                    'remaining_fiber_count': size,
                    'orbit_representative': fibers_str(rep),
                    'orbit_size': 0,
                    'row_profile': profiles['row_profile'],
                    'col_profile': profiles['col_profile'],
                    'is_rectangle': str(profiles['is_rectangle']),
                    'flattening_rank_A_BC': ranks['flattening_rank_A_BC'],
                    'flattening_rank_B_AC': ranks['flattening_rank_B_AC'],
                    'flattening_rank_C_AB': ranks['flattening_rank_C_AB'],
                    'flattening_rank_max': ranks['flattening_rank_max'],
                    'exact_rank_if_known': profiles['exact_rank_if_known'],
                    'exact_rank_source': profiles['exact_rank_source'],
                    'provenance': 'EXACT_DERIVED',
                }
                for total_rank in TARGET_RANKS:
                    available = total_rank - 3 * (9 - size)
                    orbit_row[f'available_spreaders_R{total_rank}'] = available
                    orbit_row[f'flattening_rules_out_R{total_rank}'] = str(ranks['flattening_rank_max'] > available)
                    orbit_row[f'exact_rank_rules_out_R{total_rank}'] = str(known_exact is not None and known_exact > available)
                orbit_map[rep] = orbit_row
            orbit_map[rep]['orbit_size'] += 1

    orbit_rows = sorted(orbit_map.values(), key=lambda row: (row['remaining_fiber_count'], row['orbit_representative']))
    size_rows: list[dict] = []
    for size in range(1, 10):
        size_subset_rows = [row for row in subset_rows if row['remaining_fiber_count'] == size]
        size_orbit_rows = [row for row in orbit_rows if row['remaining_fiber_count'] == size]
        size_row = {
            'remaining_fiber_count': size,
            'fourier_fiber_count': 9 - size,
            'subset_count': len(size_subset_rows),
            'orbit_count': len(size_orbit_rows),
            'min_flattening_rank_max': min(row['flattening_rank_max'] for row in size_orbit_rows),
            'max_flattening_rank_max': max(row['flattening_rank_max'] for row in size_orbit_rows),
            'provenance': 'EXACT_DERIVED',
        }
        for total_rank in TARGET_RANKS:
            available = total_rank - 3 * (9 - size)
            size_row[f'available_spreaders_R{total_rank}'] = available
            size_row[f'candidate_orbits_R{total_rank}_by_flattening'] = sum(row['flattening_rank_max'] <= available for row in size_orbit_rows)
            size_row[f'known_exact_ruled_out_orbits_R{total_rank}'] = sum(
                row['exact_rank_if_known'] != '' and int(row['exact_rank_if_known']) > available
                for row in size_orbit_rows
            )
        size_rows.append(size_row)

    sixfiber_rows = [row for row in orbit_rows if row['remaining_fiber_count'] == 6]
    summary = {
        'sixfiber_orbit_count': str(len(sixfiber_rows)),
        'sixfiber_min_flattening_rank_max': str(min(row['flattening_rank_max'] for row in sixfiber_rows)),
        'sixfiber_max_flattening_rank_max': str(max(row['flattening_rank_max'] for row in sixfiber_rows)),
    }
    for total_rank in TARGET_RANKS:
        available = total_rank - 3 * (9 - 6)
        summary[f'sixfiber_candidate_orbits_R{total_rank}_by_flattening'] = str(
            sum(row['flattening_rank_max'] <= available for row in sixfiber_rows)
        )
    return subset_rows, orbit_rows, size_rows, summary


def reduced_system_rows() -> list[dict]:
    rows = []
    for remaining_fibers in range(1, 10):
        row = {
            'remaining_fiber_count': remaining_fibers,
            'fourier_fiber_count': 9 - remaining_fibers,
            'reduced_output_coordinates': remaining_fibers,
            'fiber_sum_equations': 9 * remaining_fibers,
            'live_anisotropy_equations': 18 * remaining_fibers,
            'dead_x_equations': 54 * remaining_fibers,
            'total_reduced_equations': 81 * remaining_fibers,
            'provenance': 'EXACT_DERIVED',
        }
        for total_rank in TARGET_RANKS:
            row[f'available_spreaders_R{total_rank}'] = total_rank - 3 * (9 - remaining_fibers)
        rows.append(row)
    return rows


def explicit_case_rows() -> list[dict]:
    cases = [
        ('R23_two_fiber_row_case', ((2, 1), (2, 2)), 23, 'known_exact_rank_1x3x2=6'),
        ('R22_three_fiber_row_case', ((2, 0), (2, 1), (2, 2)), 22, 'known_exact_rank_1x3x3=9'),
        ('R22_three_fiber_column_case', ((0, 2), (1, 2), (2, 2)), 22, 'known_exact_rank_3x3x1=9'),
        ('R22_three_fiber_diagonal_case', ((0, 0), (1, 1), (2, 2)), 22, 'flattening_lower_bound_only'),
        ('R22_six_fiber_rectangle_case', ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)), 22, 'known_exact_rank_2x3x3=15'),
    ]
    rows: list[dict] = []
    for case_id, remaining, total_rank, source in cases:
        remaining_tuple = tuple(sorted(remaining))
        remaining_count = len(remaining_tuple)
        spreaders = total_rank - 3 * (9 - remaining_count)
        ranks = flattening_ranks(remaining_tuple)
        profiles = fiber_profiles(remaining_tuple)
        known_exact = int(profiles['exact_rank_if_known']) if profiles['exact_rank_if_known'] else None
        if source == 'flattening_lower_bound_only':
            lower_bound = ranks['flattening_rank_max']
            verdict = 'ruled_out_by_flattening' if lower_bound > spreaders else 'not_ruled_out_by_flattening'
            justification = f'flattening lower bound {lower_bound} versus available spreaders {spreaders}'
        else:
            exact_rank = known_exact if known_exact is not None else ranks['flattening_rank_max']
            verdict = 'ruled_out_by_known_exact_rank' if exact_rank > spreaders else 'not_ruled_out'
            justification = f'known exact rank {exact_rank} versus available spreaders {spreaders}'
        rows.append({
            'case_id': case_id,
            'R_total': total_rank,
            'remaining_fibers': fibers_str(remaining_tuple),
            'remaining_fiber_count': remaining_count,
            'available_spreaders': spreaders,
            'flattening_rank_A_BC': ranks['flattening_rank_A_BC'],
            'flattening_rank_B_AC': ranks['flattening_rank_B_AC'],
            'flattening_rank_C_AB': ranks['flattening_rank_C_AB'],
            'flattening_rank_max': ranks['flattening_rank_max'],
            'known_exact_rank_if_any': profiles['exact_rank_if_known'],
            'rank_source': source,
            'verdict': verdict,
            'justification': justification,
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def feasibility_rows(size_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in size_rows:
        remaining = int(row['remaining_fiber_count'])
        fourier = 9 - remaining
        feasibility_row = {
            'fourier_fiber_count': fourier,
            'fourier_term_count': 3 * fourier,
            'remaining_fiber_count': remaining,
            'min_flattening_lower_bound': row['min_flattening_rank_max'],
            'max_flattening_lower_bound': row['max_flattening_rank_max'],
            'provenance': 'EXACT_DERIVED',
        }
        for total_rank in TARGET_RANKS:
            feasibility_row[f'available_spreaders_R{total_rank}'] = row[f'available_spreaders_R{total_rank}']
            feasibility_row[f'candidate_orbits_R{total_rank}_by_flattening'] = row[f'candidate_orbits_R{total_rank}_by_flattening']
            feasibility_row[f'known_exact_ruled_out_orbits_R{total_rank}'] = row[f'known_exact_ruled_out_orbits_R{total_rank}']
            feasibility_row[f'interpretation_R{total_rank}'] = (
                'all_flattening_ruled_out'
                if row[f'candidate_orbits_R{total_rank}_by_flattening'] == 0
                else 'flattening_candidates_remain'
            )
        rows.append(feasibility_row)
    return rows


def summary_rows(
    modularity_summary: dict[str, str],
    orbit_summary: dict[str, str],
    size_rows: list[dict],
    explicit_rows: list[dict],
) -> list[dict]:
    size_map = {int(row['remaining_fiber_count']): row for row in size_rows}
    explicit_map = {row['case_id']: row for row in explicit_rows}
    return [
        {'summary_name': 'modularity_total_subsets_checked', 'summary_value': modularity_summary['modularity_total_subsets_checked'], 'provenance': 'EXACT_DERIVED', 'note': 'All same-fiber Fourier subset combinations checked against their restricted targets.'},
        {'summary_name': 'modularity_total_passing_subsets', 'summary_value': modularity_summary['modularity_total_passing_subsets'], 'provenance': 'EXACT_DERIVED', 'note': 'Every checked Fourier subset passed exact modularity verification.'},
        {'summary_name': 'modularity_max_abs_residual', 'summary_value': modularity_summary['modularity_max_abs_residual'], 'provenance': 'EXACT_DERIVED', 'note': 'Maximum residual over all checked Fourier subsets.'},
        {'summary_name': 'sixfiber_orbit_count', 'summary_value': orbit_summary['sixfiber_orbit_count'], 'provenance': 'EXACT_DERIVED', 'note': 'Number of S3 x S3 orbits on 6-fiber residual output patterns.'},
        {'summary_name': 'sixfiber_min_flattening_rank_max', 'summary_value': orbit_summary['sixfiber_min_flattening_rank_max'], 'provenance': 'EXACT_DERIVED', 'note': 'Minimum max-flattening lower bound among 6-fiber residual patterns.'},
        {'summary_name': 'sixfiber_max_flattening_rank_max', 'summary_value': orbit_summary['sixfiber_max_flattening_rank_max'], 'provenance': 'EXACT_DERIVED', 'note': 'Maximum max-flattening lower bound among 6-fiber residual patterns.'},
        {'summary_name': 'sixfiber_candidate_orbits_R19_by_flattening', 'summary_value': orbit_summary['sixfiber_candidate_orbits_R19_by_flattening'], 'provenance': 'EXACT_DERIVED', 'note': 'Six-fiber residual orbits not ruled out by flattening against 10 spreaders at R=19.'},
        {'summary_name': 'sixfiber_candidate_orbits_R20_by_flattening', 'summary_value': orbit_summary['sixfiber_candidate_orbits_R20_by_flattening'], 'provenance': 'EXACT_DERIVED', 'note': 'Six-fiber residual orbits not ruled out by flattening against 11 spreaders at R=20.'},
        {'summary_name': 'sixfiber_candidate_orbits_R21_by_flattening', 'summary_value': orbit_summary['sixfiber_candidate_orbits_R21_by_flattening'], 'provenance': 'EXACT_DERIVED', 'note': 'Six-fiber residual orbits not ruled out by flattening against 12 spreaders at R=21.'},
        {'summary_name': 'sixfiber_candidate_orbits_R22_by_flattening', 'summary_value': orbit_summary['sixfiber_candidate_orbits_R22_by_flattening'], 'provenance': 'EXACT_DERIVED', 'note': 'Six-fiber residual orbits not ruled out by flattening against 13 spreaders at R=22.'},
        {'summary_name': 'sixfiber_candidate_orbits_R23_by_flattening', 'summary_value': orbit_summary['sixfiber_candidate_orbits_R23_by_flattening'], 'provenance': 'EXACT_DERIVED', 'note': 'Six-fiber residual orbits not ruled out by flattening against 14 spreaders at R=23.'},
        {'summary_name': 'remaining4_candidate_orbits_R19_by_flattening', 'summary_value': str(size_map[4]['candidate_orbits_R19_by_flattening']), 'provenance': 'EXACT_DERIVED', 'note': 'Four-fiber residual orbits not ruled out by flattening against 4 spreaders.'},
        {'summary_name': 'remaining4_candidate_orbits_R20_by_flattening', 'summary_value': str(size_map[4]['candidate_orbits_R20_by_flattening']), 'provenance': 'EXACT_DERIVED', 'note': 'Four-fiber residual orbits not ruled out by flattening against 5 spreaders.'},
        {'summary_name': 'remaining4_candidate_orbits_R21_by_flattening', 'summary_value': str(size_map[4]['candidate_orbits_R21_by_flattening']), 'provenance': 'EXACT_DERIVED', 'note': 'Four-fiber residual orbits not ruled out by flattening against 6 spreaders.'},
        {'summary_name': 'remaining3_candidate_orbits_R22_by_flattening', 'summary_value': str(size_map[3]['candidate_orbits_R22_by_flattening']), 'provenance': 'EXACT_DERIVED', 'note': 'Three-fiber residual orbits not ruled out by flattening against 4 spreaders.'},
        {'summary_name': 'remaining4_candidate_orbits_R22_by_flattening', 'summary_value': str(size_map[4]['candidate_orbits_R22_by_flattening']), 'provenance': 'EXACT_DERIVED', 'note': 'Four-fiber residual orbits not ruled out by flattening against 7 spreaders.'},
        {'summary_name': 'two_fiber_row_case_verdict', 'summary_value': explicit_map['R23_two_fiber_row_case']['verdict'], 'provenance': 'EXACT_DERIVED', 'note': 'Explicit two-fiber row residual at R=23.'},
        {'summary_name': 'three_fiber_row_case_verdict', 'summary_value': explicit_map['R22_three_fiber_row_case']['verdict'], 'provenance': 'EXACT_DERIVED', 'note': 'Explicit three-fiber row residual at R=22.'},
        {'summary_name': 'three_fiber_diagonal_case_verdict', 'summary_value': explicit_map['R22_three_fiber_diagonal_case']['verdict'], 'provenance': 'EXACT_DERIVED', 'note': 'Explicit three-fiber diagonal residual at R=22.'},
    ]


def write_markdown_summary(
    path: Path,
    summary: list[dict],
    modularity_size_rows: list[dict],
    sixfiber_rows: list[dict],
    feasibility: list[dict],
    reduced_system: list[dict],
    explicit_rows: list[dict],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append
    w('# Step 60: Hybrid Fourier Construction')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('Step 60 formalizes the hybrid split suggested by Step 59: same-fiber Fourier triples are exact modular blocks, while the remaining spreader part is a reduced output-fiber tensor problem. The main computational filter is the symmetry-reduced flattening-rank scan of residual fiber sets.')
    w('')
    w(f"Fourier subsets checked: {summary_map['modularity_total_subsets_checked']}")
    w(f"Fourier subsets passing exactly: {summary_map['modularity_total_passing_subsets']}")
    w(f"Six-fiber orbit count: {summary_map['sixfiber_orbit_count']}")
    w(f"Six-fiber max-flattening lower bound range: {summary_map['sixfiber_min_flattening_rank_max']}..{summary_map['sixfiber_max_flattening_rank_max']}")
    w(f"Six-fiber flattening candidates at R=19..23: {summary_map['sixfiber_candidate_orbits_R19_by_flattening']}, {summary_map['sixfiber_candidate_orbits_R20_by_flattening']}, {summary_map['sixfiber_candidate_orbits_R21_by_flattening']}, {summary_map['sixfiber_candidate_orbits_R22_by_flattening']}, {summary_map['sixfiber_candidate_orbits_R23_by_flattening']}")
    w('')
    w('## Task 1: Fourier Block Modularity')
    w('')
    w('| subset size | subset count | passing count | max abs residual |')
    w('|-------------|--------------|---------------|------------------|')
    for row in modularity_size_rows:
        w(f"| {row['subset_size']} | {row['subset_count']} | {row['passing_count']} | {row['max_abs_residual']} |")
    w('')
    w('Conclusion: the same-fiber Fourier block is perfectly modular. Covering some fibers with Fourier triples leaves the tensor on the complementary fibers unchanged.')
    w('')
    w('## Task 2d: Reduced Residual Systems')
    w('')
    w('| remaining fibers | Fourier fibers | reduced outputs | fiber-sum eqs | anisotropy eqs | dead-X eqs | total reduced eqs | R=19 spreaders | R=20 spreaders | R=21 spreaders | R=22 spreaders | R=23 spreaders |')
    w('|------------------|----------------|-----------------|---------------|----------------|------------|-------------------|---------------|---------------|---------------|---------------|---------------|')
    for row in reduced_system:
        w(f"| {row['remaining_fiber_count']} | {row['fourier_fiber_count']} | {row['reduced_output_coordinates']} | {row['fiber_sum_equations']} | {row['live_anisotropy_equations']} | {row['dead_x_equations']} | {row['total_reduced_equations']} | {row['available_spreaders_R19']} | {row['available_spreaders_R20']} | {row['available_spreaders_R21']} | {row['available_spreaders_R22']} | {row['available_spreaders_R23']} |")
    w('')
    w('## Task 4c: Six-Fiber Sub-Tensor Flattening Orbits')
    w('')
    w('| representative residual fibers | orbit size | row profile | col profile | rectangle | rank A|(BC) | rank B|(AC) | rank C|(AB) | max lower bound | known exact rank if any |')
    w('|-------------------------------|------------|-------------|-------------|-----------|------------|------------|------------|-----------------|-------------------------|')
    for row in sixfiber_rows:
        exact_cell = row['exact_rank_if_known'] if row['exact_rank_if_known'] else ''
        w(f"| {row['orbit_representative']} | {row['orbit_size']} | {row['row_profile']} | {row['col_profile']} | {row['is_rectangle']} | {row['flattening_rank_A_BC']} | {row['flattening_rank_B_AC']} | {row['flattening_rank_C_AB']} | {row['flattening_rank_max']} | {exact_cell} |")
    w('')
    w('## Task 3f: Hybrid Feasibility Table')
    w('')
    w('| Fourier fibers f | Fourier terms | remaining fibers | R=19 spreaders | R=20 spreaders | R=21 spreaders | R=22 spreaders | R=23 spreaders | flattening lower-bound range over residual orbits | R19 cand. | R20 cand. | R21 cand. | R22 cand. | R23 cand. |')
    w('|------------------|---------------|------------------|---------------|---------------|---------------|---------------|---------------|-------------------------------------------|-----------|-----------|-----------|-----------|-----------|')
    for row in feasibility:
        lower_range = f"{row['min_flattening_lower_bound']}..{row['max_flattening_lower_bound']}"
        w(f"| {row['fourier_fiber_count']} | {row['fourier_term_count']} | {row['remaining_fiber_count']} | {row['available_spreaders_R19']} | {row['available_spreaders_R20']} | {row['available_spreaders_R21']} | {row['available_spreaders_R22']} | {row['available_spreaders_R23']} | {lower_range} | {row['candidate_orbits_R19_by_flattening']} | {row['candidate_orbits_R20_by_flattening']} | {row['candidate_orbits_R21_by_flattening']} | {row['candidate_orbits_R22_by_flattening']} | {row['candidate_orbits_R23_by_flattening']} |")
    w('')
    w('## Explicit Cases')
    w('')
    w('| case | R total | remaining fibers | available spreaders | max flattening lower bound | known exact rank | verdict | justification |')
    w('|------|---------|------------------|---------------------|----------------------------|------------------|---------|---------------|')
    for row in explicit_rows:
        exact_cell = row['known_exact_rank_if_any'] if row['known_exact_rank_if_any'] else ''
        w(f"| {row['case_id']} | {row['R_total']} | {row['remaining_fibers']} | {row['available_spreaders']} | {row['flattening_rank_max']} | {exact_cell} | {row['verdict']} | {row['justification']} |")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The clean result is modularity: the Fourier block really does decouple. That means the hybrid question is exactly the rank question for the residual output-fiber tensor, not a coupled correction problem. Extending the same flattening filter down to R=19, R=20, and R=21 does not shrink the six-fiber survivor set at all: every six-fiber orbit still has lower bound 9, so all six remain compatible with spreader budgets 10, 11, 12, 13, and 14. But the rectangular 2x3 case is already known to need rank 15, so flattening is too weak there. At the small end, the explicit two-fiber and three-fiber row/column/diagonal residuals are ruled out immediately, and four-fiber residuals only begin to survive at the flattening level once the budget reaches 7. The remaining unresolved hybrid territory therefore still sits in non-rectangular 4-, 5-, 6-, 7-, and 8-fiber residual patterns, where flattening lower bounds alone do not decide feasibility.')
    write_markdown(path, '\n'.join(lines))


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print('=== Step 60: Hybrid Fourier Construction ===')
    print()

    modularity_single_rows, modularity_size_rows, modularity_summary = modularity_rows()
    subset_rows, orbit_rows, size_rows, orbit_summary = orbit_tables()
    reduced_system = reduced_system_rows()
    explicit_rows = explicit_case_rows()
    feasibility = feasibility_rows(size_rows)
    sixfiber_rows = [row for row in orbit_rows if row['remaining_fiber_count'] == 6]
    summary = summary_rows(modularity_summary, orbit_summary, size_rows, explicit_rows)

    write_csv(
        EXPORTS / 'step60_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step60_fourier_single_fiber_verification.csv',
        modularity_single_rows,
        ['fiber', 'term_count', 'max_abs_residual', 'max_abs_leakage_to_other_outputs', 'passes', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step60_fourier_modularity_summary.csv',
        modularity_size_rows,
        ['subset_size', 'subset_count', 'passing_count', 'max_abs_residual', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step60_residual_subtensor_orbits.csv',
        orbit_rows,
        [
            'remaining_fiber_count',
            'orbit_representative',
            'orbit_size',
            'row_profile',
            'col_profile',
            'is_rectangle',
            'flattening_rank_A_BC',
            'flattening_rank_B_AC',
            'flattening_rank_C_AB',
            'flattening_rank_max',
            'available_spreaders_R19',
            'available_spreaders_R20',
            'available_spreaders_R21',
            'available_spreaders_R22',
            'available_spreaders_R23',
            'flattening_rules_out_R19',
            'flattening_rules_out_R20',
            'flattening_rules_out_R21',
            'flattening_rules_out_R22',
            'flattening_rules_out_R23',
            'exact_rank_if_known',
            'exact_rank_source',
            'exact_rank_rules_out_R19',
            'exact_rank_rules_out_R20',
            'exact_rank_rules_out_R21',
            'exact_rank_rules_out_R22',
            'exact_rank_rules_out_R23',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step60_residual_subtensor_all_subsets.csv',
        subset_rows,
        [
            'remaining_fiber_count',
            'remaining_fibers',
            'orbit_representative',
            'row_profile',
            'col_profile',
            'is_rectangle',
            'flattening_rank_A_BC',
            'flattening_rank_B_AC',
            'flattening_rank_C_AB',
            'flattening_rank_max',
            'available_spreaders_R19',
            'available_spreaders_R20',
            'available_spreaders_R21',
            'available_spreaders_R22',
            'available_spreaders_R23',
            'flattening_rules_out_R19',
            'flattening_rules_out_R20',
            'flattening_rules_out_R21',
            'flattening_rules_out_R22',
            'flattening_rules_out_R23',
            'exact_rank_if_known',
            'exact_rank_source',
            'exact_rank_rules_out_R19',
            'exact_rank_rules_out_R20',
            'exact_rank_rules_out_R21',
            'exact_rank_rules_out_R22',
            'exact_rank_rules_out_R23',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step60_residual_subtensor_size_summary.csv',
        size_rows,
        [
            'remaining_fiber_count',
            'fourier_fiber_count',
            'subset_count',
            'orbit_count',
            'available_spreaders_R19',
            'available_spreaders_R20',
            'available_spreaders_R21',
            'available_spreaders_R22',
            'available_spreaders_R23',
            'min_flattening_rank_max',
            'max_flattening_rank_max',
            'candidate_orbits_R19_by_flattening',
            'candidate_orbits_R20_by_flattening',
            'candidate_orbits_R21_by_flattening',
            'candidate_orbits_R22_by_flattening',
            'candidate_orbits_R23_by_flattening',
            'known_exact_ruled_out_orbits_R19',
            'known_exact_ruled_out_orbits_R20',
            'known_exact_ruled_out_orbits_R21',
            'known_exact_ruled_out_orbits_R22',
            'known_exact_ruled_out_orbits_R23',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step60_reduced_spreader_systems.csv',
        reduced_system,
        [
            'remaining_fiber_count',
            'fourier_fiber_count',
            'reduced_output_coordinates',
            'fiber_sum_equations',
            'live_anisotropy_equations',
            'dead_x_equations',
            'total_reduced_equations',
            'available_spreaders_R19',
            'available_spreaders_R20',
            'available_spreaders_R21',
            'available_spreaders_R22',
            'available_spreaders_R23',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step60_hybrid_feasibility_table.csv',
        feasibility,
        [
            'fourier_fiber_count',
            'fourier_term_count',
            'remaining_fiber_count',
            'available_spreaders_R19',
            'available_spreaders_R20',
            'available_spreaders_R21',
            'available_spreaders_R22',
            'available_spreaders_R23',
            'min_flattening_lower_bound',
            'max_flattening_lower_bound',
            'candidate_orbits_R19_by_flattening',
            'candidate_orbits_R20_by_flattening',
            'candidate_orbits_R21_by_flattening',
            'candidate_orbits_R22_by_flattening',
            'candidate_orbits_R23_by_flattening',
            'known_exact_ruled_out_orbits_R19',
            'known_exact_ruled_out_orbits_R20',
            'known_exact_ruled_out_orbits_R21',
            'known_exact_ruled_out_orbits_R22',
            'known_exact_ruled_out_orbits_R23',
            'interpretation_R19',
            'interpretation_R20',
            'interpretation_R21',
            'interpretation_R22',
            'interpretation_R23',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step60_explicit_cases.csv',
        explicit_rows,
        [
            'case_id',
            'R_total',
            'remaining_fibers',
            'remaining_fiber_count',
            'available_spreaders',
            'flattening_rank_A_BC',
            'flattening_rank_B_AC',
            'flattening_rank_C_AB',
            'flattening_rank_max',
            'known_exact_rank_if_any',
            'rank_source',
            'verdict',
            'justification',
            'provenance',
        ],
    )
    write_markdown_summary(
        EXPORTS / 'step60_hybrid_fourier_construction.md',
        summary,
        modularity_size_rows,
        sixfiber_rows,
        feasibility,
        reduced_system,
        explicit_rows,
    )

    print('\nSummary:')
    for row in summary:
        print(f"  {row['summary_name']} = {row['summary_value']}")


if __name__ == '__main__':
    main()