"""
ade3x3_step49_coefficient_level_rank_constraints.py

Step 49: Coefficient-Level Rank Constraints in the Same-Fiber Algebra.

This step keeps the analysis at the exact coordinate level. It records the 8
equation types induced by the 216-element symmetry on the 729 tensor equations,
checks the representative equations explicitly, verifies the standard 27-term
algorithm on those representatives, rewrites the equations in a transparent
coefficient form, analyzes the 2x2 Strassen algorithm in the same framework,
and records search-space dimensions for future solver work.

Outputs:
  - outputs/exports/step49_equation_types.csv
  - outputs/exports/step49_representative_orbit_verification.csv
  - outputs/exports/step49_standard_algorithm_verification.csv
  - outputs/exports/step49_tensor_slice_summary.csv
  - outputs/exports/step49_strassen_2x2_terms.csv
  - outputs/exports/step49_strassen_2x2_term_orbits.csv
  - outputs/exports/step49_strassen_2x2_equation_orbits.csv
  - outputs/exports/step49_search_space_dimensions.csv
  - outputs/exports/step49_summary.csv
  - outputs/exports/step49_coefficient_level_rank_constraints.md

Provenance:
  [EXACT_DERIVED] for all enumerated/verifiable algebraic facts and counts.
  [INTERPRETATION] only in the markdown summary where future search claims or
  solver implications are discussed.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations, product
from math import comb
from pathlib import Path

EXPORTS = Path('outputs/exports')

S3 = list(permutations((0, 1, 2)))
S2 = list(permutations((0, 1)))

ACTIONS_3 = [(p1, p2, p3) for p1 in S3 for p2 in S3 for p3 in S3]
ACTIONS_2 = [(p1, p2, p3) for p1 in S2 for p2 in S2 for p3 in S2]

REPRESENTATIVE_3X3 = [
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 1),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0, 1),
    (0, 0, 1, 0, 1, 0),
    (0, 0, 1, 0, 1, 1),
]

TYPE_CONDITIONS = {
    0: "s=t, r=r', u=u'",
    1: "s=t, r=r', u!=u'",
    2: "s=t, r!=r', u=u'",
    3: "s=t, r!=r', u!=u'",
    4: "s!=t, r=r', u=u'",
    5: "s!=t, r=r', u!=u'",
    6: "s!=t, r!=r', u=u'",
    7: "s!=t, r!=r', u!=u'",
}

TYPE_LABELS = {
    0: 'Type 0',
    1: 'Type 1',
    2: 'Type 2',
    3: 'Type 3',
    4: 'Type 4',
    5: 'Type 5',
    6: 'Type 6',
    7: 'Type 7',
}

TYPE_DESCRIPTIONS = {
    0: 'live X, correct output',
    1: 'live X, same row wrong column in C',
    2: 'live X, wrong row same column in C',
    3: 'live X, wrong row and wrong column in C',
    4: 'dead X, correct output position in C',
    5: 'dead X, same row wrong column in C',
    6: 'dead X, wrong row same column in C',
    7: 'dead X, wrong row and wrong column in C',
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def matrix_to_string(matrix: list[list[int]]) -> str:
    return '/'.join(' '.join(str(cell) for cell in row) for row in matrix)


def flatten_matrix(matrix: list[list[int]]) -> tuple[int, ...]:
    return tuple(cell for row in matrix for cell in row)


def flatten_term(term: dict[str, list[list[int]]]) -> tuple[int, ...]:
    return flatten_matrix(term['alpha']) + flatten_matrix(term['beta']) + flatten_matrix(term['gamma'])


def image_matrix(matrix: list[list[int]], row_perm: tuple[int, ...], col_perm: tuple[int, ...]) -> list[list[int]]:
    out = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            out[row_perm[row_idx]][col_perm[col_idx]] = value
    return out


def apply_term_action(
    term: dict[str, list[list[int]]],
    action: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> dict[str, list[list[int]]]:
    p1, p2, p3 = action
    return {
        'alpha': image_matrix(term['alpha'], p1, p2),
        'beta': image_matrix(term['beta'], p2, p3),
        'gamma': image_matrix(term['gamma'], p1, p3),
    }


def encode_x_3x3(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> int:
    return 9 * (3 * row_idx + sum_left) + (3 * sum_right + col_idx)


def x_name_3x3(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> str:
    return f"X[{row_idx},{sum_left}|{sum_right},{col_idx}]"


def c_name_3x3(row_idx: int, col_idx: int) -> str:
    return f"C[{row_idx},{col_idx}]"


def a_name_3x3(row_idx: int, col_idx: int) -> str:
    return f"A[{row_idx},{col_idx}]"


def b_name_3x3(row_idx: int, col_idx: int) -> str:
    return f"B[{row_idx},{col_idx}]"


def equation_rhs(eq: tuple[int, int, int, int, int, int]) -> int:
    row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
    return int(sum_left == sum_right and row_idx == c_row and col_idx == c_col)


def equation_type_id(eq: tuple[int, int, int, int, int, int]) -> int:
    row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
    return 4 * int(sum_left != sum_right) + 2 * int(row_idx != c_row) + int(col_idx != c_col)


def equation_type_condition(eq: tuple[int, int, int, int, int, int]) -> str:
    return TYPE_CONDITIONS[equation_type_id(eq)]


def representative_label(type_id: int) -> str:
    return f"E{type_id}"


def equation_string(eq: tuple[int, int, int, int, int, int]) -> str:
    row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
    return (
        f"sum_k alpha_k[{row_idx},{sum_left}] * beta_k[{sum_right},{col_idx}] * "
        f"gamma_k[{c_row},{c_col}] = {equation_rhs(eq)}"
    )


def reduced_equation_string(type_id: int) -> str:
    c_row = 0 if type_id in (0, 1, 4, 5) else 1
    c_col = 0 if type_id in (0, 2, 4, 6) else 1
    coeff = 'p_k' if type_id < 4 else 'q_k'
    return f"sum_k {coeff} * gamma_k[{c_row},{c_col}] = {1 if type_id == 0 else 0}"


def apply_equation_action(
    eq: tuple[int, int, int, int, int, int],
    action: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[int, int, int, int, int, int]:
    p1, p2, p3 = action
    row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
    return (p1[row_idx], p2[sum_left], p2[sum_right], p3[col_idx], p1[c_row], p3[c_col])


def build_equation_type_rows() -> tuple[list[dict], list[dict], dict[str, str]]:
    all_equations = list(product(range(3), repeat=6))
    count_by_type = Counter(equation_type_id(eq) for eq in all_equations)
    expected = {0: 27, 1: 54, 2: 54, 3: 108, 4: 54, 5: 108, 6: 108, 7: 216}
    assert dict(count_by_type) == expected, count_by_type

    step48_rows = read_csv_rows(EXPORTS / 'step48_xc_orbit_equation_summary.csv')
    step48_by_id = {int(row['xc_orbit_id']): row for row in step48_rows}

    equation_rows: list[dict] = []
    verification_rows: list[dict] = []
    covered: set[tuple[int, int, int, int, int, int]] = set()

    for type_id, eq in enumerate(REPRESENTATIVE_3X3):
        orbit = {apply_equation_action(eq, action) for action in ACTIONS_3}
        expected_members = {candidate for candidate in all_equations if equation_type_id(candidate) == type_id}
        assert orbit == expected_members
        assert all(equation_rhs(candidate) == equation_rhs(eq) for candidate in orbit)
        covered |= orbit

        row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
        rep_config_id = 9 * encode_x_3x3(row_idx, sum_left, sum_right, col_idx) + 3 * c_row + c_col

        step48_rep_match = True
        if type_id in step48_by_id:
            step48_rep_match = (
                int(step48_by_id[type_id]['rep_config_id']) == rep_config_id
                and int(step48_by_id[type_id]['orbit_size']) == len(orbit)
                and int(step48_by_id[type_id]['tensor_rhs_constant']) == equation_rhs(eq)
            )

        equation_rows.append({
            'equation_label': representative_label(type_id),
            'xc_orbit_id': type_id,
            'rep_config_id': rep_config_id,
            'rep_x_atom': x_name_3x3(row_idx, sum_left, sum_right, col_idx),
            'rep_c_atom': c_name_3x3(c_row, c_col),
            'r': row_idx,
            's': sum_left,
            't': sum_right,
            'u': col_idx,
            'r_prime': c_row,
            'u_prime': c_col,
            'orbit_size': len(orbit),
            'rhs': equation_rhs(eq),
            'structure_type': TYPE_LABELS[type_id],
            'structure_condition': equation_type_condition(eq),
            'description': TYPE_DESCRIPTIONS[type_id],
            'representative_equation': equation_string(eq),
            'transparent_rewrite': reduced_equation_string(type_id),
            'p_or_q_definition': (
                'p_k = alpha_k[0,0] * beta_k[0,0]' if type_id < 4 else 'q_k = alpha_k[0,0] * beta_k[1,0]'
            ),
            'provenance': 'EXACT_DERIVED',
        })
        verification_rows.append({
            'equation_label': representative_label(type_id),
            'xc_orbit_id': type_id,
            'computed_orbit_size': len(orbit),
            'expected_orbit_size': expected[type_id],
            'orbit_size_matches': len(orbit) == expected[type_id],
            'rhs_invariant_on_orbit': True,
            'type_partition_matches_full_729': orbit == expected_members,
            'step48_roster_match': step48_rep_match,
            'sample_orbit_member_1': equation_string(sorted(orbit)[0]),
            'sample_orbit_member_2': equation_string(sorted(orbit)[min(1, len(orbit) - 1)]),
            'provenance': 'EXACT_DERIVED',
        })

    assert len(covered) == 729
    summary = {
        'n_equation_types_3x3': '8',
        'equation_type_sizes_3x3': str([expected[i] for i in range(8)]),
        'step48_roster_crosscheck_present': str(bool(step48_rows)),
        'step48_roster_crosscheck_passed': str(all(bool(row['step48_roster_match']) for row in verification_rows) if step48_rows else True),
    }
    return equation_rows, verification_rows, summary


def verify_standard_basis_algorithm() -> tuple[list[dict], dict[str, str]]:
    basis_terms = [(row_idx, sum_idx, col_idx) for row_idx in range(3) for sum_idx in range(3) for col_idx in range(3)]
    rows: list[dict] = []
    mismatches_8 = 0
    mismatches_729 = 0

    for type_id, eq in enumerate(REPRESENTATIVE_3X3):
        contributions = [
            (term, int(eq == (term[0], term[1], term[1], term[2], term[0], term[2])))
            for term in basis_terms
        ]
        total = sum(value for _, value in contributions)
        expected = equation_rhs(eq)
        if total != expected:
            mismatches_8 += 1
        rows.append({
            'equation_label': representative_label(type_id),
            'representative_equation': equation_string(eq),
            'expected_rhs': expected,
            'observed_sum': total,
            'contributing_basis_terms': str([term for term, value in contributions if value]),
            'status': 'PASS' if total == expected else 'FAIL',
            'provenance': 'EXACT_DERIVED',
        })

    for eq in product(range(3), repeat=6):
        total = sum(int(eq == (term[0], term[1], term[1], term[2], term[0], term[2])) for term in basis_terms)
        if total != equation_rhs(eq):
            mismatches_729 += 1

    summary = {
        'standard_27_term_equations_checked': '729',
        'standard_27_term_representative_mismatches': str(mismatches_8),
        'standard_27_term_full_tensor_mismatches': str(mismatches_729),
    }
    assert mismatches_8 == 0
    assert mismatches_729 == 0
    return rows, summary


def build_tensor_slice_rows() -> list[dict]:
    a_basis = [a_name_3x3(row_idx, col_idx) for row_idx in range(3) for col_idx in range(3)]
    b_basis = [b_name_3x3(row_idx, col_idx) for row_idx in range(3) for col_idx in range(3)]
    rows: list[dict] = []
    for c_row in range(3):
        for c_col in range(3):
            matrix = [[0 for _ in range(9)] for _ in range(9)]
            nonzeros: list[str] = []
            for sum_idx in range(3):
                a_pos = 3 * c_row + sum_idx
                b_pos = 3 * sum_idx + c_col
                matrix[a_pos][b_pos] = 1
                nonzeros.append(f"({a_basis[a_pos]},{b_basis[b_pos]})")
            rows.append({
                'c_atom': c_name_3x3(c_row, c_col),
                'slice_formula': f"T[:,:,{c_name_3x3(c_row, c_col)}] has ones at A[{c_row},s] x B[s,{c_col}] for s in {{0,1,2}}",
                'nonzero_count': 3,
                'nonzero_positions': ', '.join(nonzeros),
                'matrix_row_major': matrix_to_string(matrix),
                'provenance': 'EXACT_DERIVED',
            })
    return rows


def build_strassen_terms() -> list[dict[str, object]]:
    return [
        {
            'term_name': 'm1',
            'alpha': [[1, 0], [0, 1]],
            'beta': [[1, 0], [0, 1]],
            'gamma': [[1, 0], [0, 1]],
            'formula': '(A[0,0] + A[1,1]) * (B[0,0] + B[1,1])',
        },
        {
            'term_name': 'm2',
            'alpha': [[0, 0], [1, 1]],
            'beta': [[1, 0], [0, 0]],
            'gamma': [[0, 0], [1, -1]],
            'formula': '(A[1,0] + A[1,1]) * B[0,0]',
        },
        {
            'term_name': 'm3',
            'alpha': [[1, 0], [0, 0]],
            'beta': [[0, 1], [0, -1]],
            'gamma': [[0, 1], [0, 1]],
            'formula': 'A[0,0] * (B[0,1] - B[1,1])',
        },
        {
            'term_name': 'm4',
            'alpha': [[0, 0], [0, 1]],
            'beta': [[-1, 0], [1, 0]],
            'gamma': [[1, 0], [1, 0]],
            'formula': 'A[1,1] * (B[1,0] - B[0,0])',
        },
        {
            'term_name': 'm5',
            'alpha': [[1, 1], [0, 0]],
            'beta': [[0, 0], [0, 1]],
            'gamma': [[-1, 1], [0, 0]],
            'formula': '(A[0,0] + A[0,1]) * B[1,1]',
        },
        {
            'term_name': 'm6',
            'alpha': [[-1, 0], [1, 0]],
            'beta': [[1, 1], [0, 0]],
            'gamma': [[0, 0], [0, 1]],
            'formula': '(A[1,0] - A[0,0]) * (B[0,0] + B[0,1])',
        },
        {
            'term_name': 'm7',
            'alpha': [[0, 1], [0, -1]],
            'beta': [[0, 0], [1, 1]],
            'gamma': [[1, 0], [0, 0]],
            'formula': '(A[0,1] - A[1,1]) * (B[1,0] + B[1,1])',
        },
    ]


def strassen_term_stats(term: dict[str, object]) -> tuple[int, int, list[int]]:
    alpha = term['alpha']
    beta = term['beta']
    gamma = term['gamma']
    live_count = 0
    dead_count = 0
    active_types: set[int] = set()
    for row_idx, sum_left, sum_right, col_idx, c_row, c_col in product(range(2), repeat=6):
        value = alpha[row_idx][sum_left] * beta[sum_right][col_idx] * gamma[c_row][c_col]
        if value != 0:
            active_types.add(4 * int(sum_left != sum_right) + 2 * int(row_idx != c_row) + int(col_idx != c_col))
        if alpha[row_idx][sum_left] * beta[sum_right][col_idx] != 0:
            if sum_left == sum_right:
                live_count += 1
            else:
                dead_count += 1
    return live_count, dead_count, sorted(active_types)


def canonicalize_term(
    term: dict[str, list[list[int]]],
    actions: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[int, ...], int]:
    orbit = {flatten_term(apply_term_action(term, action)) for action in actions}
    return min(orbit), len(orbit)


def analyze_strassen_2x2() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    terms = build_strassen_terms()
    tensor_mismatches = 0
    for eq in product(range(2), repeat=6):
        row_idx, sum_left, sum_right, col_idx, c_row, c_col = eq
        lhs = sum(
            term['alpha'][row_idx][sum_left] * term['beta'][sum_right][col_idx] * term['gamma'][c_row][c_col]
            for term in terms
        )
        rhs = equation_rhs(eq)
        if lhs != rhs:
            tensor_mismatches += 1
    assert tensor_mismatches == 0

    equation_orbit_rows: list[dict] = []
    counts_2x2 = Counter(equation_type_id(eq) for eq in product(range(2), repeat=6))
    for type_id, eq in enumerate([(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 1, 1), (0, 0, 1, 0, 0, 0), (0, 0, 1, 0, 0, 1), (0, 0, 1, 0, 1, 0), (0, 0, 1, 0, 1, 1)]):
        orbit = {apply_equation_action(eq, action) for action in ACTIONS_2}
        expected_members = {candidate for candidate in product(range(2), repeat=6) if equation_type_id(candidate) == type_id}
        assert orbit == expected_members
        equation_orbit_rows.append({
            'equation_type': TYPE_LABELS[type_id],
            'structure_condition': TYPE_CONDITIONS[type_id],
            'orbit_size': len(orbit),
            'rhs': equation_rhs(eq),
            'provenance': 'EXACT_DERIVED',
        })
    assert all(count == 8 for count in counts_2x2.values())

    canonical_map: dict[tuple[int, ...], int] = {}
    term_rows: list[dict] = []
    orbit_member_names: defaultdict[int, list[str]] = defaultdict(list)
    orbit_size_by_id: dict[int, int] = {}

    for term in terms:
        live_count, dead_count, active_types = strassen_term_stats(term)
        canonical, orbit_size = canonicalize_term(term, ACTIONS_2)
        if canonical not in canonical_map:
            canonical_map[canonical] = len(canonical_map)
        orbit_id = canonical_map[canonical]
        orbit_member_names[orbit_id].append(term['term_name'])
        orbit_size_by_id[orbit_id] = orbit_size
        term_rows.append({
            'term_name': term['term_name'],
            'formula': term['formula'],
            'alpha_matrix': matrix_to_string(term['alpha']),
            'beta_matrix': matrix_to_string(term['beta']),
            'gamma_matrix': matrix_to_string(term['gamma']),
            'coefficient_orbit_id': orbit_id,
            'coefficient_orbit_size': orbit_size,
            'live_x_nonzero_entries': live_count,
            'dead_x_nonzero_entries': dead_count,
            'active_equation_types': str(active_types),
            'provenance': 'EXACT_DERIVED',
        })

    term_orbit_rows = [
        {
            'coefficient_orbit_id': orbit_id,
            'coefficient_orbit_size': orbit_size_by_id[orbit_id],
            'strassen_terms': str(sorted(names)),
            'n_strassen_terms_in_orbit': len(names),
            'provenance': 'EXACT_DERIVED',
        }
        for orbit_id, names in sorted(orbit_member_names.items())
    ]

    summary = {
        'strassen_2x2_terms': str(len(terms)),
        'strassen_2x2_full_tensor_mismatches': str(tensor_mismatches),
        'strassen_2x2_equation_orbits': str(len(equation_orbit_rows)),
        'strassen_2x2_term_coefficient_orbits_used': str(len(term_orbit_rows)),
    }
    return term_rows, term_orbit_rows, equation_orbit_rows, summary


def search_space_rows() -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    inputs = [(23, 9), (23, 3), (23, 2), (22, 9), (21, 9), (20, 9), (19, 9)]
    for rank, sparsity in inputs:
        dense_variables = 27 * rank
        sparse_upper_bound = 3 * sparsity * rank
        exact_support_patterns_per_term = comb(9, sparsity) ** 3
        at_most_support_patterns_per_term = sum(comb(9, j) for j in range(sparsity + 1)) ** 3
        rows.append({
            'R': rank,
            'K': sparsity,
            'dense_variables_27R': dense_variables,
            'sparse_variable_upper_bound_3KR': sparse_upper_bound,
            'constraints': 729,
            'linear_balance_dense': 'over' if dense_variables < 729 else ('square' if dense_variables == 729 else 'under'),
            'linear_balance_sparse_upper_bound': 'over' if sparse_upper_bound < 729 else ('square' if sparse_upper_bound == 729 else 'under'),
            'exact_support_patterns_per_term': exact_support_patterns_per_term,
            'support_patterns_per_term_at_most_K': at_most_support_patterns_per_term,
            'exact_support_patterns_div_216_floor': exact_support_patterns_per_term // 216,
            'provenance': 'EXACT_DERIVED',
        })
    summary = {
        'search_constraint_count': '729',
        'search_variable_count_formula_dense': '27R',
        'search_variable_count_formula_sparse': '3KR',
    }
    return rows, summary


def write_markdown_summary(
    path: Path,
    equation_rows: list[dict],
    verification_rows: list[dict],
    standard_rows: list[dict],
    tensor_slice_rows: list[dict],
    strassen_term_rows: list[dict],
    strassen_term_orbit_rows: list[dict],
    strassen_equation_orbits: list[dict],
    search_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append

    w('# Step 49: Coefficient-Level Rank Constraints in the Same-Fiber Algebra')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('## Task 3c: The 8 Equation Types')
    w('')
    w('| equation | structure | orbit_size | RHS | representative | transparent rewrite |')
    w('|----------|-----------|------------|-----|----------------|---------------------|')
    for row in equation_rows:
        w(
            f"| {row['equation_label']} | {row['structure_condition']} | {row['orbit_size']} | {row['rhs']} | "
            f"{row['representative_equation']} | {row['transparent_rewrite']} |"
        )
    w('')
    w('These 8 structural types partition all 729 tensor equations.')
    w('')
    w('## Task 1a-1b: Orbit Verification and Standard 27-Term Check')
    w('')
    w('| equation | orbit_size_match | step48_match | observed_standard_sum | expected_rhs |')
    w('|----------|------------------|--------------|-----------------------|--------------|')
    for verify_row, standard_row in zip(verification_rows, standard_rows, strict=True):
        w(
            f"| {verify_row['equation_label']} | {verify_row['orbit_size_matches']} | {verify_row['step48_roster_match']} | "
            f"{standard_row['observed_sum']} | {standard_row['expected_rhs']} |"
        )
    w('')
    w(f"Full 729-equation standard-basis mismatches: {summary['standard_27_term_full_tensor_mismatches']}")
    w('')
    w('## Task 2: Transparent Rewriting and Matrix Form')
    w('')
    w('Define p_k = alpha_k[0,0] * beta_k[0,0] and q_k = alpha_k[0,0] * beta_k[1,0].')
    w('Then E0-E3 are the p_k equations against gamma_k[0,0], gamma_k[0,1], gamma_k[1,0], gamma_k[1,1].')
    w('E4-E7 are the q_k equations against those same four gamma coordinates.')
    w('')
    c00_row = next(row for row in tensor_slice_rows if row['c_atom'] == 'C[0,0]')
    w(f"T[:,:,C[0,0]] nonzero positions: {c00_row['nonzero_positions']}")
    w('')
    w('## Task 4: Strassen 2x2 Worked Example')
    w('')
    w(f"Strassen full 64-equation mismatches: {summary['strassen_2x2_full_tensor_mismatches']}")
    w(f"2x2 equation orbit classes under S2 x S2 x S2: {summary['strassen_2x2_equation_orbits']}")
    w(f"Coefficient orbits used by the 7 Strassen terms: {summary['strassen_2x2_term_coefficient_orbits_used']}")
    w('')
    w('| term | coefficient_orbit_id | orbit_size | live_X_nonzero | dead_X_nonzero | active_equation_types |')
    w('|------|----------------------|------------|----------------|----------------|-----------------------|')
    for row in strassen_term_rows:
        w(
            f"| {row['term_name']} | {row['coefficient_orbit_id']} | {row['coefficient_orbit_size']} | "
            f"{row['live_x_nonzero_entries']} | {row['dead_x_nonzero_entries']} | {row['active_equation_types']} |"
        )
    w('')
    w('| coefficient_orbit_id | coefficient_orbit_size | strassen_terms |')
    w('|----------------------|------------------------|----------------|')
    for row in strassen_term_orbit_rows:
        w(f"| {row['coefficient_orbit_id']} | {row['coefficient_orbit_size']} | {row['strassen_terms']} |")
    w('')
    w('| equation_type | structure | orbit_size | RHS |')
    w('|---------------|-----------|------------|-----|')
    for row in strassen_equation_orbits:
        w(f"| {row['equation_type']} | {row['structure_condition']} | {row['orbit_size']} | {row['rhs']} |")
    w('')
    w('## Task 5: 3x3 Search Space Dimensions')
    w('')
    w('| R | K | dense_variables_27R | sparse_upper_bound_3KR | constraints | dense_balance | sparse_balance |')
    w('|---|---|---------------------|-----------------------|-------------|---------------|----------------|')
    for row in search_rows:
        w(
            f"| {row['R']} | {row['K']} | {row['dense_variables_27R']} | {row['sparse_variable_upper_bound_3KR']} | "
            f"{row['constraints']} | {row['linear_balance_dense']} | {row['linear_balance_sparse_upper_bound']} |"
        )
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The 8 equation types are the core structural reduction, but they only replace all 729 equations when the full term multiset is closed under the S3 x S3 x S3 action.')
    w('This is not a lower bound on general algorithms: non-group-closed decompositions, including known fast algorithms, still have to satisfy all 729 instantiated equations directly.')
    w('Strassen 2x2 confirms the cancellation pattern explicitly: dead-X activations do occur, and the weighted gamma outputs must cancel them.')
    w('The search-space table shows that even sparse ansatze remain heavily overdetermined in raw equation count, but because the system is trilinear that count alone does not settle feasibility or emptiness of the solution variety.')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 49: Coefficient-Level Rank Constraints in the Same-Fiber Algebra ===')
    print()

    print('Task 1 / 3c: building the 8 equation types and verifying their orbits...')
    equation_rows, verification_rows, equation_summary = build_equation_type_rows()
    print(f"  equation types={equation_summary['n_equation_types_3x3']}  sizes={equation_summary['equation_type_sizes_3x3']}")
    print()

    print('Task 1b: verifying the standard 27-term basis algorithm...')
    standard_rows, standard_summary = verify_standard_basis_algorithm()
    print(f"  representative mismatches={standard_summary['standard_27_term_representative_mismatches']}  full mismatches={standard_summary['standard_27_term_full_tensor_mismatches']}")
    print()

    print('Task 2c: building tensor slice summaries...')
    tensor_slice_rows = build_tensor_slice_rows()
    print(f"  tensor slices exported={len(tensor_slice_rows)}")
    print()

    print('Task 4: analyzing Strassen 2x2 in the same framework...')
    strassen_term_rows, strassen_term_orbit_rows, strassen_equation_orbits, strassen_summary = analyze_strassen_2x2()
    print(f"  Strassen term orbits used={strassen_summary['strassen_2x2_term_coefficient_orbits_used']}")
    print()

    print('Task 5: recording search-space dimensions...')
    search_rows, search_summary = search_space_rows()
    print(f"  search rows={len(search_rows)}")
    print()

    summary_rows = [
        {
            'summary_name': 'n_equation_types_3x3',
            'summary_value': equation_summary['n_equation_types_3x3'],
            'provenance': 'EXACT_DERIVED',
            'note': 'The 729 tensor equations split into 8 structural XC equation types.',
        },
        {
            'summary_name': 'equation_type_sizes_3x3',
            'summary_value': equation_summary['equation_type_sizes_3x3'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Orbit sizes of the 8 structural equation types.',
        },
        {
            'summary_name': 'step48_roster_crosscheck_passed',
            'summary_value': equation_summary['step48_roster_crosscheck_passed'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Representative equations match the Step 48 XC orbit roster when that export is present.',
        },
        {
            'summary_name': 'standard_27_term_representative_mismatches',
            'summary_value': standard_summary['standard_27_term_representative_mismatches'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Standard 27-term basis algorithm mismatches on E0-E7.',
        },
        {
            'summary_name': 'standard_27_term_full_tensor_mismatches',
            'summary_value': standard_summary['standard_27_term_full_tensor_mismatches'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Standard 27-term basis algorithm mismatches on all 729 equations.',
        },
        {
            'summary_name': 'group_closed_reduction_scope',
            'summary_value': 'Representative equations imply all 729 only for group-closed multisets of terms.',
            'provenance': 'EXACT_DERIVED',
            'note': 'This is a symmetry statement, not a general lower bound for arbitrary algorithms.',
        },
        {
            'summary_name': 'non_group_closed_algorithms_require_all_729',
            'summary_value': 'True',
            'provenance': 'EXACT_DERIVED',
            'note': 'Known fast algorithms are not generally group-closed and must satisfy every instantiated equation.',
        },
        {
            'summary_name': 'strassen_2x2_terms',
            'summary_value': strassen_summary['strassen_2x2_terms'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Number of rank-1 terms in Strassen 2x2.',
        },
        {
            'summary_name': 'strassen_2x2_full_tensor_mismatches',
            'summary_value': strassen_summary['strassen_2x2_full_tensor_mismatches'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Strassen verification on the full 64-equation 2x2 tensor system.',
        },
        {
            'summary_name': 'strassen_2x2_equation_orbits',
            'summary_value': strassen_summary['strassen_2x2_equation_orbits'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Number of equation orbit types in the 2x2 analog under S2 x S2 x S2.',
        },
        {
            'summary_name': 'strassen_2x2_term_coefficient_orbits_used',
            'summary_value': strassen_summary['strassen_2x2_term_coefficient_orbits_used'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Number of coefficient-level term orbits used by the 7 Strassen terms.',
        },
        {
            'summary_name': 'search_constraint_count',
            'summary_value': search_summary['search_constraint_count'],
            'provenance': 'EXACT_DERIVED',
            'note': 'The exact coordinate system remains 729 equations for 3x3.',
        },
        {
            'summary_name': 'search_variable_count_formula_dense',
            'summary_value': search_summary['search_variable_count_formula_dense'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Dense parameter count formula.',
        },
        {
            'summary_name': 'search_variable_count_formula_sparse',
            'summary_value': search_summary['search_variable_count_formula_sparse'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Sparse upper-bound variable count formula when each factor has at most K nonzeros.',
        },
    ]

    print('Writing outputs...')
    write_csv(
        EXPORTS / 'step49_equation_types.csv',
        equation_rows,
        [
            'equation_label', 'xc_orbit_id', 'rep_config_id', 'rep_x_atom', 'rep_c_atom', 'r', 's', 't', 'u', 'r_prime', 'u_prime',
            'orbit_size', 'rhs', 'structure_type', 'structure_condition', 'description', 'representative_equation',
            'transparent_rewrite', 'p_or_q_definition', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step49_representative_orbit_verification.csv',
        verification_rows,
        [
            'equation_label', 'xc_orbit_id', 'computed_orbit_size', 'expected_orbit_size', 'orbit_size_matches',
            'rhs_invariant_on_orbit', 'type_partition_matches_full_729', 'step48_roster_match',
            'sample_orbit_member_1', 'sample_orbit_member_2', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step49_standard_algorithm_verification.csv',
        standard_rows,
        ['equation_label', 'representative_equation', 'expected_rhs', 'observed_sum', 'contributing_basis_terms', 'status', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step49_tensor_slice_summary.csv',
        tensor_slice_rows,
        ['c_atom', 'slice_formula', 'nonzero_count', 'nonzero_positions', 'matrix_row_major', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step49_strassen_2x2_terms.csv',
        strassen_term_rows,
        [
            'term_name', 'formula', 'alpha_matrix', 'beta_matrix', 'gamma_matrix', 'coefficient_orbit_id',
            'coefficient_orbit_size', 'live_x_nonzero_entries', 'dead_x_nonzero_entries', 'active_equation_types', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step49_strassen_2x2_term_orbits.csv',
        strassen_term_orbit_rows,
        ['coefficient_orbit_id', 'coefficient_orbit_size', 'strassen_terms', 'n_strassen_terms_in_orbit', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step49_strassen_2x2_equation_orbits.csv',
        strassen_equation_orbits,
        ['equation_type', 'structure_condition', 'orbit_size', 'rhs', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step49_search_space_dimensions.csv',
        search_rows,
        [
            'R', 'K', 'dense_variables_27R', 'sparse_variable_upper_bound_3KR', 'constraints', 'linear_balance_dense',
            'linear_balance_sparse_upper_bound', 'exact_support_patterns_per_term', 'support_patterns_per_term_at_most_K',
            'exact_support_patterns_div_216_floor', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step49_summary.csv',
        summary_rows,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_markdown_summary(
        EXPORTS / 'step49_coefficient_level_rank_constraints.md',
        equation_rows,
        verification_rows,
        standard_rows,
        tensor_slice_rows,
        strassen_term_rows,
        strassen_term_orbit_rows,
        strassen_equation_orbits,
        search_rows,
        summary_rows,
    )
    print()
    print('=== SUMMARY ===')
    print(f"Equation types: {equation_summary['n_equation_types_3x3']} -> {equation_summary['equation_type_sizes_3x3']}")
    print(f"Standard 27-term mismatches: {standard_summary['standard_27_term_full_tensor_mismatches']}")
    print(f"Strassen 2x2 coefficient orbits used: {strassen_summary['strassen_2x2_term_coefficient_orbits_used']}")
    print(f"Search rows: {len(search_rows)}")


if __name__ == '__main__':
    main()