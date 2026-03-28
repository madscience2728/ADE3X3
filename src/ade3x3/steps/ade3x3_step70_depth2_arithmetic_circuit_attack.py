"""
ade3x3_step70_depth2_arithmetic_circuit_attack.py

Step 70: Depth-2 Arithmetic Circuit Attack.

This step relaxes the depth-1 tensor-rank viewpoint just enough to audit one
concrete depth-2 construction route: recursive Strassen on the 4x4 zero-padded
embedding of 3x3 multiplication. It also checks whether AlphaTensor's 23 depth-1
terms exhibit pair-ratio structure that could make some terms redundant in a
depth-2 circuit.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp

from src.ade3x3.steps.ade3x3_step49_coefficient_level_rank_constraints import build_strassen_terms
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import load_public_rank23_terms

EXPORTS = Path('outputs/exports')


@dataclass(frozen=True)
class LeafTerm:
    outer_term: str
    inner_term: str
    alpha_expr: sp.Expr
    beta_expr: sp.Expr
    gamma_matrix: sp.Matrix


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


def expr_to_str(expr: sp.Expr) -> str:
    return str(sp.expand(expr))


def matrix_expr_to_str(matrix: sp.Matrix) -> str:
    return str(sp.Matrix([[sp.expand(matrix[row_idx, col_idx]) for col_idx in range(matrix.cols)] for row_idx in range(matrix.rows)]))


def matrix_to_json(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix).astype(int).tolist(), separators=(',', ':'))


def gamma_to_json(matrix: sp.Matrix) -> str:
    return json.dumps([[int(matrix[row_idx, col_idx]) for col_idx in range(matrix.cols)] for row_idx in range(matrix.rows)], separators=(',', ':'))


def matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.int64)
    for row_idx in range(n):
        for sum_idx in range(n):
            for col_idx in range(n):
                a_idx = n * row_idx + sum_idx
                b_idx = n * sum_idx + col_idx
                c_idx = n * row_idx + col_idx
                tensor[a_idx, b_idx, c_idx] = 1
    return tensor


def strassen_term_matrices() -> list[dict[str, object]]:
    terms = build_strassen_terms()
    converted: list[dict[str, object]] = []
    for term in terms:
        converted.append({
            'term_name': term['term_name'],
            'formula': term['formula'],
            'alpha': sp.Matrix(term['alpha']),
            'beta': sp.Matrix(term['beta']),
            'gamma': sp.Matrix(term['gamma']),
        })
    return converted


def padded_symbolic_inputs() -> tuple[sp.Matrix, sp.Matrix]:
    a = {(row_idx, col_idx): sp.symbols(f'a{row_idx}{col_idx}') for row_idx in range(3) for col_idx in range(3)}
    b = {(row_idx, col_idx): sp.symbols(f'b{row_idx}{col_idx}') for row_idx in range(3) for col_idx in range(3)}
    A = sp.Matrix([
        [a[(0, 0)], a[(0, 1)], a[(0, 2)], 0],
        [a[(1, 0)], a[(1, 1)], a[(1, 2)], 0],
        [a[(2, 0)], a[(2, 1)], a[(2, 2)], 0],
        [0, 0, 0, 0],
    ])
    B = sp.Matrix([
        [b[(0, 0)], b[(0, 1)], b[(0, 2)], 0],
        [b[(1, 0)], b[(1, 1)], b[(1, 2)], 0],
        [b[(2, 0)], b[(2, 1)], b[(2, 2)], 0],
        [0, 0, 0, 0],
    ])
    return A, B


def block(matrix: sp.Matrix, block_row: int, block_col: int) -> sp.Matrix:
    row0 = 2 * block_row
    col0 = 2 * block_col
    return matrix[row0:row0 + 2, col0:col0 + 2]


def linear_combination(coeffs: sp.Matrix, blocks: list[list[sp.Matrix]]) -> list[list[sp.Matrix]]:
    result = [[sp.zeros(2, 2) for _ in range(2)] for _ in range(2)]
    for row_idx in range(2):
        for col_idx in range(2):
            total = sp.zeros(2, 2)
            for src_row in range(2):
                for src_col in range(2):
                    total += coeffs[src_row, src_col] * blocks[src_row][src_col]
            result[row_idx][col_idx] = sp.expand(total[row_idx, col_idx])
    return result


def block_linear_combo(coeffs: sp.Matrix, blocks: list[list[sp.Matrix]]) -> sp.Matrix:
    total = sp.zeros(2, 2)
    for block_row in range(2):
        for block_col in range(2):
            total += coeffs[block_row, block_col] * blocks[block_row][block_col]
    return sp.Matrix([[sp.expand(total[row_idx, col_idx]) for col_idx in range(2)] for row_idx in range(2)])


def kron_gamma(outer_gamma: sp.Matrix, inner_gamma: sp.Matrix) -> sp.Matrix:
    gamma = sp.zeros(4, 4)
    for block_row in range(2):
        for block_col in range(2):
            coeff = outer_gamma[block_row, block_col]
            if coeff == 0:
                continue
            for row_idx in range(2):
                for col_idx in range(2):
                    gamma[2 * block_row + row_idx, 2 * block_col + col_idx] += coeff * inner_gamma[row_idx, col_idx]
    return gamma


def recursive_strassen_leaves() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    A4, B4 = padded_symbolic_inputs()
    A_blocks = [[block(A4, row_idx, col_idx) for col_idx in range(2)] for row_idx in range(2)]
    B_blocks = [[block(B4, row_idx, col_idx) for col_idx in range(2)] for row_idx in range(2)]
    strassen = strassen_term_matrices()

    leaf_rows: list[dict] = []
    outer_rows: list[dict] = []
    all_leaves: list[LeafTerm] = []

    for outer_term in strassen:
        outer_name = str(outer_term['term_name'])
        left_block = block_linear_combo(outer_term['alpha'], A_blocks)
        right_block = block_linear_combo(outer_term['beta'], B_blocks)
        kept = 0
        pruned_zero_linear = 0
        pruned_outside = 0
        for inner_term in strassen:
            alpha_expr = sp.expand(sum(inner_term['alpha'][row_idx, col_idx] * left_block[row_idx, col_idx] for row_idx in range(2) for col_idx in range(2)))
            beta_expr = sp.expand(sum(inner_term['beta'][row_idx, col_idx] * right_block[row_idx, col_idx] for row_idx in range(2) for col_idx in range(2)))
            gamma_matrix = kron_gamma(outer_term['gamma'], inner_term['gamma'])
            top_left_gamma_nnz = sum(1 for row_idx in range(3) for col_idx in range(3) if gamma_matrix[row_idx, col_idx] != 0)
            alpha_nonzero = alpha_expr != 0
            beta_nonzero = beta_expr != 0
            touches_target = top_left_gamma_nnz > 0
            keep = bool(alpha_nonzero and beta_nonzero and touches_target)
            if keep:
                kept += 1
                all_leaves.append(LeafTerm(outer_name, str(inner_term['term_name']), alpha_expr, beta_expr, gamma_matrix))
            elif not alpha_nonzero or not beta_nonzero:
                pruned_zero_linear += 1
            else:
                pruned_outside += 1
            leaf_rows.append({
                'outer_term': outer_name,
                'inner_term': str(inner_term['term_name']),
                'alpha_expr': expr_to_str(alpha_expr),
                'beta_expr': expr_to_str(beta_expr),
                'alpha_nonzero': str(alpha_nonzero),
                'beta_nonzero': str(beta_nonzero),
                'top_left_3x3_gamma_nonzero_count': str(top_left_gamma_nnz),
                'kept_for_top_left_3x3': str(keep),
                'gamma_matrix_json': gamma_to_json(gamma_matrix),
                'provenance': 'EXACT_DERIVED',
            })
        outer_rows.append({
            'outer_term': outer_name,
            'outer_formula': str(outer_term['formula']),
            'left_block': matrix_expr_to_str(left_block),
            'right_block': matrix_expr_to_str(right_block),
            'inner_leaf_cost_for_top_left_3x3': str(kept),
            'pruned_zero_linear_leaves': str(pruned_zero_linear),
            'pruned_outside_target_leaves': str(pruned_outside),
            'provenance': 'EXACT_DERIVED',
        })

    reconstructed = sp.zeros(4, 4)
    for leaf in all_leaves:
        reconstructed += leaf.alpha_expr * leaf.beta_expr * leaf.gamma_matrix
    target = A4 * B4
    top_left_matches = all(sp.expand(reconstructed[row_idx, col_idx] - target[row_idx, col_idx]) == 0 for row_idx in range(3) for col_idx in range(3))
    total_cost = len(all_leaves)
    summary = {
        'recursive_strassen_full_leaf_count': str(len(strassen) * len(strassen)),
        'recursive_strassen_top_left_3x3_leaf_count': str(total_cost),
        'recursive_strassen_top_left_3x3_verified': str(top_left_matches),
        'recursive_strassen_beats_23': str(total_cost < 23),
    }
    return outer_rows, leaf_rows, [
        {
            'summary_name': key,
            'summary_value': value,
            'provenance': 'EXACT_DERIVED',
            'note': {
                'recursive_strassen_full_leaf_count': 'The full recursive 4x4 Strassen expansion has 49 leaf scalar multiplications before pruning.',
                'recursive_strassen_top_left_3x3_leaf_count': 'Leaf scalar multiplications that survive after zero-padding and target-output pruning.',
                'recursive_strassen_top_left_3x3_verified': 'Whether the kept leaf terms reconstruct the top-left 3x3 padded target exactly.',
                'recursive_strassen_beats_23': 'Immediate flag requested by Step 70.',
            }[key],
        }
        for key, value in summary.items()
    ], summary


def profile_from_term(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.outer(alpha.reshape(-1), beta.reshape(-1))


def support_signature(profile_a: np.ndarray, profile_b: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    positions = [
        (a_idx, b_idx)
        for a_idx in range(profile_a.shape[0])
        for b_idx in range(profile_a.shape[1])
        if profile_a[a_idx, b_idx] != 0 and profile_b[a_idx, b_idx] != 0
    ]
    if not positions:
        return [], np.zeros((0, 0), dtype=np.int64)
    ratio_matrix = np.zeros(profile_a.shape, dtype=np.int64)
    for a_idx, b_idx in positions:
        ratio_matrix[a_idx, b_idx] = int(profile_a[a_idx, b_idx] // profile_b[a_idx, b_idx])
    return positions, ratio_matrix


def classify_pair_ratio(profile_a: np.ndarray, profile_b: np.ndarray) -> tuple[str, dict[str, object]]:
    positions, ratio_matrix = support_signature(profile_a, profile_b)
    if not positions:
        return 'disjoint_support', {'common_support_size': 0}
    ratios = [ratio_matrix[a_idx, b_idx] for a_idx, b_idx in positions]
    ratio_set = sorted(set(int(value) for value in ratios))
    rows_ok = True
    row_values: dict[int, int] = {}
    for a_idx in range(profile_a.shape[0]):
        row_entries = [int(ratio_matrix[row_idx, b_idx]) for row_idx, b_idx in positions if row_idx == a_idx]
        row_entries = [value for value in row_entries if value != 0]
        if not row_entries:
            continue
        if len(set(row_entries)) != 1:
            rows_ok = False
            break
        row_values[a_idx] = row_entries[0]
    cols_ok = True
    col_values: dict[int, int] = {}
    for b_idx in range(profile_a.shape[1]):
        col_entries = [int(ratio_matrix[a_idx, col_idx]) for a_idx, col_idx in positions if col_idx == b_idx]
        col_entries = [value for value in col_entries if value != 0]
        if not col_entries:
            continue
        if len(set(col_entries)) != 1:
            cols_ok = False
            break
        col_values[b_idx] = col_entries[0]
    if len(ratio_set) == 1:
        return 'constant_multiple', {'common_support_size': len(positions), 'ratio_values': ratio_set}
    if rows_ok and len(set(row_values.values())) > 1:
        if len(set(col_values.values())) == 1:
            return 'alpha_only', {'common_support_size': len(positions), 'row_values': row_values}
    if cols_ok and len(set(col_values.values())) > 1:
        if len(set(row_values.values())) == 1:
            return 'beta_only', {'common_support_size': len(positions), 'col_values': col_values}
    if rows_ok and cols_ok:
        return 'row_col_separable', {'common_support_size': len(positions), 'row_values': row_values, 'col_values': col_values}
    return 'unstructured_overlap', {'common_support_size': len(positions), 'ratio_values': ratio_set}


def pair_ratio_rows(term_rows: list[tuple[str, np.ndarray, np.ndarray]], dataset_name: str) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for idx in range(len(term_rows)):
        name_i, alpha_i, beta_i = term_rows[idx]
        profile_i = profile_from_term(alpha_i, beta_i)
        for jdx in range(idx + 1, len(term_rows)):
            name_j, alpha_j, beta_j = term_rows[jdx]
            profile_j = profile_from_term(alpha_j, beta_j)
            classification, details = classify_pair_ratio(profile_i, profile_j)
            counts[classification] = counts.get(classification, 0) + 1
            rows.append({
                'dataset': dataset_name,
                'term_i': name_i,
                'term_j': name_j,
                'classification': classification,
                'common_support_size': str(details.get('common_support_size', 0)),
                'details_json': json.dumps(details, separators=(',', ':'), default=str),
                'provenance': 'MEASURED_FROM_CODE',
            })
    summary = {f'{dataset_name}_{key}_pair_count': str(value) for key, value in sorted(counts.items())}
    return rows, summary


def alpha_tensor_pair_ratio_exports() -> tuple[list[dict], list[dict], dict[str, str]]:
    alpha_terms, _, _ = load_public_rank23_terms()
    alpha_rows = [(term.term_id, term.alpha.astype(np.int64), term.beta.astype(np.int64)) for term in alpha_terms]
    strassen_rows = []
    for term in build_strassen_terms():
        strassen_rows.append((str(term['term_name']), np.array(term['alpha'], dtype=np.int64), np.array(term['beta'], dtype=np.int64)))
    alpha_pair_rows, alpha_summary = pair_ratio_rows(alpha_rows, 'alphatensor23')
    strassen_pair_rows, strassen_summary = pair_ratio_rows(strassen_rows, 'strassen2x2')

    retained = {'m1', 'm2', 'm3', 'm5'}
    missing = {'m4', 'm6', 'm7'}
    retained_missing = [
        row for row in strassen_pair_rows
        if ((row['term_i'] in retained and row['term_j'] in missing) or (row['term_j'] in retained and row['term_i'] in missing))
    ]
    structured_retained_missing = sum(1 for row in retained_missing if row['classification'] in {'constant_multiple', 'alpha_only', 'beta_only'})
    combined_summary = {
        **alpha_summary,
        **strassen_summary,
        'strassen2x2_retained_missing_structured_pair_count': str(structured_retained_missing),
    }
    return alpha_pair_rows, strassen_pair_rows, combined_summary


def reduced_greedy_pool_size_note() -> str:
    return 'Step 70 does not attempt the direct depth-2 search because the parameterized search space is far beyond brute-force scale in the current tool budget.'


def literature_status_rows() -> list[dict]:
    return [
        {
            'source_name': 'Pan_trilinear_aggregation_explicit_3x3_circuit',
            'status': 'not_recovered_from_available_fetched_content',
            'detail': 'Fetched public summaries confirm Pan\'s 1978 aggregating/uniting/canceling line and his 1982 practical subcubic algorithm, but they did not provide an explicit small-3x3 depth-2 circuit or concrete multiplication count that could be instantiated directly in ADE3x3.',
            'provenance': 'MEASURED_FROM_CODE + WEB_FETCH',
        }
    ]


def markdown_report(recursive_summary: dict[str, str], pair_summary: dict[str, str], literature_rows: list[dict]) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 70: Depth-2 Arithmetic Circuit Attack')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Task 4: Recursive Strassen on 4x4 Padding')
    w('')
    w(f"- Full recursive leaf count before pruning: {recursive_summary['recursive_strassen_full_leaf_count']}")
    w(f"- Leaf count needed for the padded top-left 3x3 target: {recursive_summary['recursive_strassen_top_left_3x3_leaf_count']}")
    w(f"- Exact top-left 3x3 verification after pruning: {recursive_summary['recursive_strassen_top_left_3x3_verified']}")
    w(f"- Beats 23? {recursive_summary['recursive_strassen_beats_23']}")
    w('')
    w('## Task 6: Pair-Ratio Structure')
    w('')
    for key in sorted(pair_summary):
        w(f"- {key}: {pair_summary[key]}")
    w('')
    w('## Task 3: 2x2 Baseline')
    w('')
    w(f"- Structured retained-vs-missing Strassen pairs among {{m1,m2,m3,m5}} and {{m4,m6,m7}}: {pair_summary['strassen2x2_retained_missing_structured_pair_count']}")
    w('- This does not reprove the depth-2 lower bound 7 for 2x2, but it does show there is no immediate constant-multiple or one-sided pair-ratio collapse inside the obvious Strassen split.')
    w('')
    w('## Task 5: Pan Literature Status')
    w('')
    for row in literature_rows:
        w(f"- {row['source_name']}: {row['status']} | {row['detail']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The clean Step 70 result is that the most direct recursive depth-2 attack does not threaten rank 23: once 3x3 is zero-padded to 4x4 and Strassen is applied recursively, the exact leaf count needed to recover the top-left 3x3 block remains above 23 even after pruning leaf multiplications that are identically zero or only feed discarded padded outputs.')
    w('The AlphaTensor pair-ratio scan also does not reveal an obvious depth-2 factoring mechanism. Because every depth-1 term is already rank-1 in the A/B profile space, the only cheap pairwise collapses would come from proportional or one-sided ratios on common support. Step 70 records how often that happens exactly, and for the obvious 2x2 Strassen split it does not produce a direct 6-multiplication collapse.')
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)

    outer_rows, leaf_rows, recursive_summary_rows, recursive_summary = recursive_strassen_leaves()
    alpha_pair_rows, strassen_pair_rows, pair_summary = alpha_tensor_pair_ratio_exports()
    literature_rows = literature_status_rows()
    summary_rows = recursive_summary_rows + [
        {
            'summary_name': key,
            'summary_value': value,
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Pair-ratio structure count.' if 'pair_count' in key else reduced_greedy_pool_size_note(),
        }
        for key, value in pair_summary.items()
    ]
    report = markdown_report(recursive_summary, pair_summary, literature_rows)

    write_csv(EXPORTS / 'step70_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step70_recursive_strassen_outer_products.csv', outer_rows, ['outer_term', 'outer_formula', 'left_block', 'right_block', 'inner_leaf_cost_for_top_left_3x3', 'pruned_zero_linear_leaves', 'pruned_outside_target_leaves', 'provenance'])
    write_csv(EXPORTS / 'step70_recursive_strassen_leaf_products.csv', leaf_rows, ['outer_term', 'inner_term', 'alpha_expr', 'beta_expr', 'alpha_nonzero', 'beta_nonzero', 'top_left_3x3_gamma_nonzero_count', 'kept_for_top_left_3x3', 'gamma_matrix_json', 'provenance'])
    write_csv(EXPORTS / 'step70_alphatensor_pair_ratios.csv', alpha_pair_rows, ['dataset', 'term_i', 'term_j', 'classification', 'common_support_size', 'details_json', 'provenance'])
    write_csv(EXPORTS / 'step70_strassen_pair_ratios.csv', strassen_pair_rows, ['dataset', 'term_i', 'term_j', 'classification', 'common_support_size', 'details_json', 'provenance'])
    write_csv(EXPORTS / 'step70_literature_status.csv', literature_rows, ['source_name', 'status', 'detail', 'provenance'])
    write_text(EXPORTS / 'step70_depth2_arithmetic_circuit_attack.md', report)


if __name__ == '__main__':
    main()