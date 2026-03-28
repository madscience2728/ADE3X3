"""
ade3x3_step48_tensor_profile_constraint_model.py

Step 48: Tensor Profile Constraint Model.

This step turns the Step 47 same-fiber tensor profile into an explicit rank-1
constraint model at three levels.

Task 3a
  Decompose the full 729 tensor equations into XC orbit classes.

Task 2a-2b
  Write the exact 729-equation rank-R tensor system in coordinates and compare it
  with the support-level same-fiber counts induced by a single rank-1 term.

Task 3b-3c
  Form the 8-equation XC orbit-sum linearization, record the exact orbit-sum
  formula for a single rank-1 term, and test whether this coarse model can force
  any nontrivial lower bound on R.

Task 1
  Verify the generic live-X activation pattern of a rank-1 term and enumerate all
  support-level fiber-activation profiles up to row/column symmetry.

Outputs:
  - outputs/exports/step48_generic_rank1_live_atoms.csv
  - outputs/exports/step48_generic_rank1_fiber_summary.csv
  - outputs/exports/step48_activation_profile_orbits.csv
  - outputs/exports/step48_support_contribution_summary.csv
  - outputs/exports/step48_tensor_equations.csv
  - outputs/exports/step48_xc_orbit_equation_summary.csv
  - outputs/exports/step48_rank1_xc_orbit_sum_formulas.csv
  - outputs/exports/step48_orbit_sum_constraint_system.csv
  - outputs/exports/step48_orbit_sum_counterexample.csv
  - outputs/exports/step48_constraint_model_summary.csv
  - outputs/exports/step48_tensor_profile_constraint_model.md

Provenance:
  [EXACT_DERIVED] for all enumerated/exported data.
  [INTERPRETATION] only in the markdown summary.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations
from pathlib import Path

EXPORTS = Path('outputs/exports')

S3 = list(permutations((0, 1, 2)))
PROFILE_PERMS = [(row_perm, col_perm) for row_perm in S3 for col_perm in S3]
ACTION_TRIPLES = [(row_perm, mid_perm, col_perm) for row_perm in S3 for mid_perm in S3 for col_perm in S3]


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


def decode_x(x_atom: int) -> tuple[int, int, int, int]:
    return (x_atom // 9) // 3, (x_atom // 9) % 3, (x_atom % 9) // 3, (x_atom % 9) % 3


def encode_x(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> int:
    return 9 * (3 * row_idx + sum_left) + (3 * sum_right + col_idx)


def x_name(x_atom: int) -> str:
    row_idx, sum_left, sum_right, col_idx = decode_x(x_atom)
    return f"X[{row_idx},{sum_left}|{sum_right},{col_idx}]"


def c_name(c_atom: int) -> str:
    return f"C[{c_atom // 3},{c_atom % 3}]"


def a_name(a_atom: int) -> str:
    return f"A[{a_atom // 3},{a_atom % 3}]"


def b_name(b_atom: int) -> str:
    return f"B[{b_atom // 3},{b_atom % 3}]"


def support_mask_to_matrix(mask: int) -> list[list[int]]:
    return [[(mask >> (3 * row_idx + sum_idx)) & 1 for sum_idx in range(3)] for row_idx in range(3)]


def support_matrix_to_string(matrix: list[list[int]]) -> str:
    return '/'.join(''.join(str(cell) for cell in row) for row in matrix)


def fiber_profile_from_support_masks(a_mask: int, b_mask: int) -> tuple[int, ...]:
    a_support = support_mask_to_matrix(a_mask)
    b_support = support_mask_to_matrix(b_mask)
    return tuple(
        sum(a_support[row_idx][sum_idx] & b_support[sum_idx][col_idx] for sum_idx in range(3))
        for row_idx in range(3)
        for col_idx in range(3)
    )


def canonicalize_profile(profile: tuple[int, ...]) -> tuple[int, ...]:
    matrix = [[profile[3 * row_idx + col_idx] for col_idx in range(3)] for row_idx in range(3)]
    best: tuple[int, ...] | None = None
    for row_perm, col_perm in PROFILE_PERMS:
        candidate = tuple(matrix[row_perm[row_idx]][col_perm[col_idx]] for row_idx in range(3) for col_idx in range(3))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def support_contribution_totals(profile: tuple[int, ...]) -> tuple[int, int]:
    cxxc_orbit0_support = sum(profile)
    cxxc_orbit30_support = sum(value * (value - 1) // 2 for value in profile)
    return cxxc_orbit0_support, cxxc_orbit30_support


def build_generic_rank1_rows() -> tuple[list[dict], list[dict]]:
    live_atom_rows: list[dict] = []
    fiber_rows: list[dict] = []
    for row_idx in range(3):
        for col_idx in range(3):
            fiber_id = 3 * row_idx + col_idx
            monomials: list[str] = []
            atoms: list[str] = []
            for sum_idx in range(3):
                x_atom = encode_x(row_idx, sum_idx, sum_idx, col_idx)
                coeff = f"a[{row_idx},{sum_idx}]*b[{sum_idx},{col_idx}]"
                atoms.append(x_name(x_atom))
                monomials.append(coeff)
                live_atom_rows.append({
                    'fiber_id': fiber_id,
                    'target_c': fiber_id,
                    'target_c_name': c_name(fiber_id),
                    'row_index': row_idx,
                    'sum_index': sum_idx,
                    'col_index': col_idx,
                    'x_atom': x_atom,
                    'x_atom_name': x_name(x_atom),
                    'coefficient_monomial': coeff,
                })
            fiber_rows.append({
                'fiber_id': fiber_id,
                'target_c': fiber_id,
                'target_c_name': c_name(fiber_id),
                'activated_live_atoms_generic': 3,
                'activated_atoms': str(atoms),
                'coefficient_monomials': str(monomials),
            })
    return live_atom_rows, fiber_rows


def enumerate_activation_profile_orbits() -> tuple[list[dict], list[dict], dict[str, object]]:
    profile_stats: dict[tuple[int, ...], dict[str, object]] = {}
    for a_mask in range(1 << 9):
        for b_mask in range(1 << 9):
            profile = canonicalize_profile(fiber_profile_from_support_masks(a_mask, b_mask))
            entry = profile_stats.setdefault(
                profile,
                {
                    'support_pair_count': 0,
                    'sample_a_mask': a_mask,
                    'sample_b_mask': b_mask,
                },
            )
            entry['support_pair_count'] += 1

    profile_rows: list[dict] = []
    contribution_summary: defaultdict[tuple[int, int], dict[str, object]] = defaultdict(
        lambda: {'n_profile_orbits': 0, 'support_pair_count': 0, 'example_profile': ''}
    )
    generic_profile = (3, 3, 3, 3, 3, 3, 3, 3, 3)

    for profile_orbit_id, profile in enumerate(sorted(profile_stats), start=1):
        stats = profile_stats[profile]
        a_support = support_mask_to_matrix(int(stats['sample_a_mask']))
        b_support = support_mask_to_matrix(int(stats['sample_b_mask']))
        cxxc_orbit0_support, cxxc_orbit30_support = support_contribution_totals(profile)
        row = {
            'profile_orbit_id': profile_orbit_id,
            'canonical_profile': str(profile),
            'profile_matrix': str([list(profile[3 * row_idx:3 * row_idx + 3]) for row_idx in range(3)]),
            'cxxc_orbit0_support': cxxc_orbit0_support,
            'cxxc_orbit30_support': cxxc_orbit30_support,
            'support_pair_count': int(stats['support_pair_count']),
            'max_fiber_activation': max(profile),
            'nonzero_fibers': sum(1 for value in profile if value > 0),
            'fibers_with_0': sum(1 for value in profile if value == 0),
            'fibers_with_1': sum(1 for value in profile if value == 1),
            'fibers_with_2': sum(1 for value in profile if value == 2),
            'fibers_with_3': sum(1 for value in profile if value == 3),
            'sample_a_support': support_matrix_to_string(a_support),
            'sample_b_support': support_matrix_to_string(b_support),
            'is_generic_full_support_profile': profile == generic_profile,
        }
        profile_rows.append(row)
        pair_key = (cxxc_orbit0_support, cxxc_orbit30_support)
        contribution_summary[pair_key]['n_profile_orbits'] += 1
        contribution_summary[pair_key]['support_pair_count'] += int(stats['support_pair_count'])
        if not contribution_summary[pair_key]['example_profile']:
            contribution_summary[pair_key]['example_profile'] = str(profile)

    contribution_rows = [
        {
            'cxxc_orbit0_support': pair_key[0],
            'cxxc_orbit30_support': pair_key[1],
            'n_profile_orbits': value['n_profile_orbits'],
            'support_pair_count': value['support_pair_count'],
            'example_profile': value['example_profile'],
        }
        for pair_key, value in sorted(contribution_summary.items())
    ]
    generic_support_pairs = contribution_summary[(27, 27)]['n_profile_orbits']
    summary = {
        'activation_profile_orbit_types': len(profile_rows),
        'support_contribution_pairs': len(contribution_rows),
        'generic_profile_unique': generic_support_pairs == 1,
        'generic_profile_support_pair_count': int(profile_stats[generic_profile]['support_pair_count']),
        'generic_cxxc_orbit0_support': 27,
        'generic_cxxc_orbit30_support': 27,
    }
    return profile_rows, contribution_rows, summary


def act_xc(cfg: int, row_perm: tuple[int, int, int], mid_perm: tuple[int, int, int], col_perm: tuple[int, int, int]) -> int:
    x_atom, c_atom = divmod(cfg, 9)
    row_idx, sum_left, sum_right, col_idx = decode_x(x_atom)
    c_row, c_col = divmod(c_atom, 3)
    new_x = encode_x(
        row_perm[row_idx],
        mid_perm[sum_left],
        mid_perm[sum_right],
        col_perm[col_idx],
    )
    new_c = 3 * row_perm[c_row] + col_perm[c_col]
    return 9 * new_x + new_c


def build_xc_orbit_map() -> list[int]:
    min_rep: list[int] = []
    for cfg in range(81 * 9):
        best = cfg
        for row_perm, mid_perm, col_perm in ACTION_TRIPLES:
            candidate = act_xc(cfg, row_perm, mid_perm, col_perm)
            if candidate < best:
                best = candidate
        min_rep.append(best)
    rep_to_orbit_id = {rep: orbit_id for orbit_id, rep in enumerate(sorted(set(min_rep)))}
    orbit_map = [rep_to_orbit_id[rep] for rep in min_rep]
    assert len(set(orbit_map)) == 8, f"XC orbit count mismatch: {len(set(orbit_map))}"
    return orbit_map


def relation_to_target(row_idx: int, col_idx: int, c_row: int, c_col: int) -> str:
    if row_idx == c_row and col_idx == c_col:
        return 'exact_target'
    if row_idx == c_row:
        return 'same_row_diff_col'
    if col_idx == c_col:
        return 'diff_row_same_col'
    return 'diff_row_diff_col'


def build_tensor_equation_rows() -> tuple[list[dict], list[dict]]:
    orbit_map = build_xc_orbit_map()
    orbit_meta_rows = {
        int(row['orbit_id']): row
        for row in read_csv_rows(EXPORTS / 'orbits_XC.csv')
        if row.get('schema') == 'XC'
    }

    equation_rows: list[dict] = []
    orbit_rhs_values: defaultdict[int, set[int]] = defaultdict(set)
    orbit_rhs_sum: Counter[int] = Counter()
    orbit_size: Counter[int] = Counter()

    for row_idx in range(3):
        for sum_left in range(3):
            a_atom = 3 * row_idx + sum_left
            for sum_right in range(3):
                for col_idx in range(3):
                    b_atom = 3 * sum_right + col_idx
                    x_atom = encode_x(row_idx, sum_left, sum_right, col_idx)
                    for c_row in range(3):
                        for c_col in range(3):
                            c_atom = 3 * c_row + c_col
                            xc_cfg = 9 * x_atom + c_atom
                            orbit_id = orbit_map[xc_cfg]
                            rhs = int(sum_left == sum_right and row_idx == c_row and col_idx == c_col)
                            orbit_rhs_values[orbit_id].add(rhs)
                            orbit_rhs_sum[orbit_id] += rhs
                            orbit_size[orbit_id] += 1
                            equation_rows.append({
                                'a_atom': a_atom,
                                'a_name': a_name(a_atom),
                                'b_atom': b_atom,
                                'b_name': b_name(b_atom),
                                'c_atom': c_atom,
                                'c_name': c_name(c_atom),
                                'x_atom': x_atom,
                                'x_atom_name': x_name(x_atom),
                                'xc_config_id': xc_cfg,
                                'xc_orbit_id': orbit_id,
                                'x_live': sum_left == sum_right,
                                'c_relation_to_target': relation_to_target(row_idx, col_idx, c_row, c_col),
                                'tensor_rhs': rhs,
                                'equation_template': (
                                    f"sum_k a_k[{row_idx},{sum_left}]*b_k[{sum_right},{col_idx}]"
                                    f"*c_k[{c_row},{c_col}] = {rhs}"
                                ),
                            })

    orbit_rows: list[dict] = []
    for orbit_id in sorted(orbit_size):
        meta = orbit_meta_rows[orbit_id]
        rep_config_id = int(meta['rep_config_id'])
        rep_x, rep_c = divmod(rep_config_id, 9)
        row_idx, sum_left, sum_right, col_idx = decode_x(rep_x)
        c_row, c_col = divmod(rep_c, 3)
        rhs_values = orbit_rhs_values[orbit_id]
        assert len(rhs_values) == 1, f"XC orbit {orbit_id} has nonconstant RHS: {rhs_values}"
        rhs_constant = next(iter(rhs_values))
        orbit_rows.append({
            'xc_orbit_id': orbit_id,
            'orbit_size': orbit_size[orbit_id],
            'rep_config_id': rep_config_id,
            'rep_readable': meta['rep_readable'],
            'rep_x_atom_name': x_name(rep_x),
            'rep_c_atom_name': c_name(rep_c),
            'x_live': sum_left == sum_right,
            'c_relation_to_target': relation_to_target(row_idx, col_idx, c_row, c_col),
            'tensor_rhs_constant': rhs_constant,
            'tensor_rhs_sum_on_orbit': orbit_rhs_sum[orbit_id],
            'tensor_rhs_average_on_orbit': f"{orbit_rhs_sum[orbit_id] / orbit_size[orbit_id]:.6f}",
        })

    assert sum(int(row['tensor_rhs']) for row in equation_rows) == 27
    assert [int(row['xc_orbit_id']) for row in orbit_rows if int(row['tensor_rhs_constant']) == 1] == [0]
    return equation_rows, orbit_rows


def build_rank1_orbit_sum_formula_rows(orbit_rows: list[dict]) -> list[dict]:
    formula_by_relation = {
        ('True', 'exact_target'): 'q_k,0 = sum_(r,u) L_k[r,u] * c_k[r,u]',
        ('True', 'same_row_diff_col'): 'q_k,1 = sum_(r,u) L_k[r,u] * (R_k[r] - c_k[r,u])',
        ('True', 'diff_row_same_col'): 'q_k,2 = sum_(r,u) L_k[r,u] * (U_k[u] - c_k[r,u])',
        ('True', 'diff_row_diff_col'): 'q_k,3 = sum_(r,u) L_k[r,u] * (S_k - R_k[r] - U_k[u] + c_k[r,u])',
        ('False', 'exact_target'): 'q_k,4 = sum_(r,u) D_k[r,u] * c_k[r,u]',
        ('False', 'same_row_diff_col'): 'q_k,5 = sum_(r,u) D_k[r,u] * (R_k[r] - c_k[r,u])',
        ('False', 'diff_row_same_col'): 'q_k,6 = sum_(r,u) D_k[r,u] * (U_k[u] - c_k[r,u])',
        ('False', 'diff_row_diff_col'): 'q_k,7 = sum_(r,u) D_k[r,u] * (S_k - R_k[r] - U_k[u] + c_k[r,u])',
    }
    rows: list[dict] = []
    for orbit_row in orbit_rows:
        key = (str(orbit_row['x_live']), orbit_row['c_relation_to_target'])
        rows.append({
            'xc_orbit_id': orbit_row['xc_orbit_id'],
            'orbit_size': orbit_row['orbit_size'],
            'x_live': orbit_row['x_live'],
            'c_relation_to_target': orbit_row['c_relation_to_target'],
            'notation': 'L_k[r,u] = sum_s a_k[r,s]*b_k[s,u]; D_k[r,u] = alpha_k[r]*beta_k[u] - L_k[r,u]; alpha_k[r] = sum_s a_k[r,s]; beta_k[u] = sum_t b_k[t,u]; R_k[r] = sum_v c_k[r,v]; U_k[u] = sum_w c_k[w,u]; S_k = sum_(r,u) c_k[r,u]',
            'orbit_sum_formula': formula_by_relation[key],
            'target_rhs_sum': orbit_row['tensor_rhs_sum_on_orbit'],
            'target_rhs_average': orbit_row['tensor_rhs_average_on_orbit'],
        })
    return rows


def build_orbit_sum_constraint_rows(orbit_rows: list[dict]) -> list[dict]:
    return [
        {
            'xc_orbit_id': orbit_row['xc_orbit_id'],
            'constraint_lhs': f"sum_k q_(k,{orbit_row['xc_orbit_id']})",
            'constraint_rhs_sum': orbit_row['tensor_rhs_sum_on_orbit'],
            'constraint_rhs_average': orbit_row['tensor_rhs_average_on_orbit'],
            'exact_constraint': f"sum_k q_(k,{orbit_row['xc_orbit_id']}) = {orbit_row['tensor_rhs_sum_on_orbit']}",
            'provenance': 'EXACT_DERIVED',
        }
        for orbit_row in orbit_rows
    ]


def build_orbit_sum_counterexample(orbit_map: list[int]) -> list[dict]:
    a_coeff = [[0, 0, 0] for _ in range(3)]
    b_coeff = [[0, 0, 0] for _ in range(3)]
    c_coeff = [[0, 0, 0] for _ in range(3)]
    a_coeff[0][0] = 1
    b_coeff[0][0] = 1
    c_coeff[0][0] = 27

    orbit_sum = [0] * 8
    mismatches = 0
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                for col_idx in range(3):
                    x_atom = encode_x(row_idx, sum_left, sum_right, col_idx)
                    for c_row in range(3):
                        for c_col in range(3):
                            c_atom = 3 * c_row + c_col
                            value = a_coeff[row_idx][sum_left] * b_coeff[sum_right][col_idx] * c_coeff[c_row][c_col]
                            rhs = int(sum_left == sum_right and row_idx == c_row and col_idx == c_col)
                            orbit_sum[orbit_map[9 * x_atom + c_atom]] += value
                            if value != rhs:
                                mismatches += 1

    assert orbit_sum == [27, 0, 0, 0, 0, 0, 0, 0], orbit_sum
    return [{
        'a_support': '100/000/000',
        'b_support': '100/000/000',
        'c_support': '270/000/000',
        'orbit_sum_vector': str(orbit_sum),
        'matches_8_orbit_sum_system': True,
        'full_tensor_equation_mismatches': mismatches,
        'interpretation': 'A single sparse rank-1 term can satisfy the 8 aggregated orbit-sum equations while failing the full 729-equation tensor system.',
    }]


def load_and_verify_step47_tensor_profile() -> dict[str, object]:
    rows = read_csv_rows(EXPORTS / 'step47_tensor_orbit_profile.csv')
    if not rows:
        return {'step47_tensor_profile_present': False}
    observed = {(int(row['orbit_id']), int(row['pair_count'])) for row in rows}
    assert observed == {(0, 27), (30, 27)}, f"Unexpected Step 47 tensor profile: {observed}"
    return {
        'step47_tensor_profile_present': True,
        'step47_same_fiber_orbits': '[0, 30]',
        'step47_same_fiber_pair_total': sum(int(row['pair_count']) for row in rows),
    }


def write_markdown_summary(
    path: Path,
    generic_fiber_rows: list[dict],
    profile_rows: list[dict],
    contribution_rows: list[dict],
    tensor_orbit_rows: list[dict],
    formula_rows: list[dict],
    constraint_rows: list[dict],
    counterexample_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    summary = {row['constraint_name']: row['constraint_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append

    w('# Step 48: Tensor Profile Constraint Model')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('## Task 3a: XC-Orbit Decomposition of the 729 Tensor Equations')
    w('')
    w(f"XC orbit classes: {summary['n_xc_equation_orbits']}")
    w(f"Positive XC orbit ids: {summary['positive_xc_orbits']}")
    w(f"Positive tensor equations: {summary['positive_tensor_equations']} of 729")
    w('')
    w('| xc_orbit_id | orbit_size | x_live | c_relation_to_target | rhs_constant | rhs_sum |')
    w('|-------------|------------|--------|----------------------|--------------|---------|')
    for row in tensor_orbit_rows:
        w(
            f"| {row['xc_orbit_id']} | {row['orbit_size']} | {row['x_live']} | "
            f"{row['c_relation_to_target']} | {row['tensor_rhs_constant']} | {row['tensor_rhs_sum_on_orbit']} |"
        )
    w('')
    w('Only XC orbit 0 carries tensor RHS 1. The other 7 XC orbits are exact zero classes.')
    w('')
    w('## Task 2b: Exact Coordinate Tensor System')
    w('')
    w('For each rank-R decomposition, the exact equations are:')
    w('sum_k a_k[r,s] * b_k[t,u] * c_k[r\',u\'] = delta_(s=t) * delta_(r=r\') * delta_(u=u\')')
    w('')
    w('This is a 729-equation trilinear system on 27R scalar unknowns.')
    w('')
    w('## Task 1 and 2a: Rank-1 Fiber Activation Profiles')
    w('')
    w('A generic rank-1 term activates exactly 3 live X atoms in every output fiber C[r,u].')
    w('')
    w('| fiber_id | target_c_name | activated_live_atoms_generic | coefficient_monomials |')
    w('|----------|---------------|------------------------------|-----------------------|')
    for row in generic_fiber_rows[:9]:
        w(
            f"| {row['fiber_id']} | {row['target_c_name']} | {row['activated_live_atoms_generic']} | "
            f"{row['coefficient_monomials']} |"
        )
    w('')
    w(f"Support-profile orbit types: {summary['activation_profile_orbit_types']}")
    w(f"Distinct support-level (CXXC orbit 0, CXXC orbit 30) contribution pairs: {summary['support_contribution_pairs']}")
    w(f"Generic support profile unique: {summary['generic_profile_unique']}")
    w(f"Generic support profile: {summary['generic_profile']} -> (orbit 0 support, orbit 30 support) = ({summary['generic_cxxc_orbit0_support']}, {summary['generic_cxxc_orbit30_support']})")
    w('')
    w('| profile_orbit_id | canonical_profile | cxxc_orbit0_support | cxxc_orbit30_support | support_pair_count |')
    w('|------------------|-------------------|---------------------|----------------------|--------------------|')
    for row in profile_rows[:12]:
        w(
            f"| {row['profile_orbit_id']} | {row['canonical_profile']} | {row['cxxc_orbit0_support']} | "
            f"{row['cxxc_orbit30_support']} | {row['support_pair_count']} |"
        )
    w('')
    w('At the weighted same-fiber level, with lambda_[r,u,s] = a[r,s]*b[s,u], the exact one-term polynomials are:')
    w('- weighted orbit-0 = sum_(r,u,s) lambda_[r,u,s]^2')
    w('- weighted orbit-30 = sum_(r,u) sum_(s1<s2) lambda_[r,u,s1] * lambda_[r,u,s2]')
    w('')
    w('## Task 3b-3c: XC Orbit-Sum Constraint Model')
    w('')
    w('For one rank-1 term k define:')
    w('- L_k[r,u] = sum_s a_k[r,s]*b_k[s,u]')
    w('- D_k[r,u] = alpha_k[r]*beta_k[u] - L_k[r,u]')
    w('- alpha_k[r] = sum_s a_k[r,s], beta_k[u] = sum_t b_k[t,u]')
    w('- R_k[r] = sum_v c_k[r,v], U_k[u] = sum_w c_k[w,u], S_k = sum_(r,u) c_k[r,u]')
    w('')
    w('| xc_orbit_id | orbit_sum_formula | target_rhs_sum |')
    w('|-------------|-------------------|----------------|')
    for row in formula_rows:
        w(f"| {row['xc_orbit_id']} | {row['orbit_sum_formula']} | {row['target_rhs_sum']} |")
    w('')
    w('The exact aggregated necessary condition is:')
    for row in constraint_rows:
        w(f"- {row['exact_constraint']}")
    w('')
    counterexample = counterexample_rows[0]
    w(f"Orbit-sum counterexample vector: {counterexample['orbit_sum_vector']}")
    w(f"Full tensor equation mismatches for that one-term witness: {counterexample['full_tensor_equation_mismatches']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The orbit-sum model is exact as a necessary condition, but it is far too coarse to force a rank lower bound by itself.')
    w('A single sparse rank-1 term already matches the 8 XC-orbit sums ((27,0,0,0,0,0,0,0)) while failing the full 729 tensor equations.')
    w('So the Step 48 payoff is a clean obstruction statement: any lower-bound attack has to use structure finer than the XC orbit-sum linearization, together with real coefficient constraints rather than support counts alone.')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 48: Tensor Profile Constraint Model ===')
    print()

    print('Verifying Step 47 tensor same-fiber profile...')
    step47_summary = load_and_verify_step47_tensor_profile()
    if step47_summary['step47_tensor_profile_present']:
        print(f"  Step 47 profile verified: {step47_summary['step47_same_fiber_orbits']} with {step47_summary['step47_same_fiber_pair_total']} unordered same-fiber pairs")
    else:
        print('  Step 47 tensor profile export not found; continuing with direct Step 48 derivations')
    print()

    print('Task 1a: building generic rank-1 activation rows...')
    generic_live_atom_rows, generic_fiber_rows = build_generic_rank1_rows()
    print(f"  live atoms={len(generic_live_atom_rows)}  fibers={len(generic_fiber_rows)}")
    print()

    print('Task 1b / 2a: enumerating support-level fiber activation profiles...')
    profile_rows, contribution_rows, profile_summary = enumerate_activation_profile_orbits()
    print(f"  profile orbit types={profile_summary['activation_profile_orbit_types']}")
    print(f"  support contribution pairs={profile_summary['support_contribution_pairs']}")
    print()

    print('Task 3a / 2b: decomposing the 729 tensor equations by XC orbit...')
    equation_rows, tensor_orbit_rows = build_tensor_equation_rows()
    orbit_map = build_xc_orbit_map()
    print(f"  equations={len(equation_rows)}  XC orbits={len(tensor_orbit_rows)}")
    print()

    print('Task 3b / 3c: building orbit-sum formulas and counterexample...')
    formula_rows = build_rank1_orbit_sum_formula_rows(tensor_orbit_rows)
    constraint_rows = build_orbit_sum_constraint_rows(tensor_orbit_rows)
    counterexample_rows = build_orbit_sum_counterexample(orbit_map)
    print(f"  orbit-sum counterexample={counterexample_rows[0]['orbit_sum_vector']}")
    print()

    summary_rows = [
        {
            'constraint_name': 'n_xc_equation_orbits',
            'constraint_value': str(len(tensor_orbit_rows)),
            'provenance': 'EXACT_DERIVED',
            'note': 'The 729 tensor equations decompose into exactly 8 XC orbit classes.',
        },
        {
            'constraint_name': 'positive_xc_orbits',
            'constraint_value': str([int(row['xc_orbit_id']) for row in tensor_orbit_rows if int(row['tensor_rhs_constant']) == 1]),
            'provenance': 'EXACT_DERIVED',
            'note': 'Only these XC orbit classes carry tensor RHS 1.',
        },
        {
            'constraint_name': 'positive_tensor_equations',
            'constraint_value': '27',
            'provenance': 'EXACT_DERIVED',
            'note': 'Exactly 27 of the 729 coordinate equations have RHS 1.',
        },
        {
            'constraint_name': 'activation_profile_orbit_types',
            'constraint_value': str(profile_summary['activation_profile_orbit_types']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Support-level fiber activation profiles up to row/column symmetry.',
        },
        {
            'constraint_name': 'support_contribution_pairs',
            'constraint_value': str(profile_summary['support_contribution_pairs']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Distinct support-level (CXXC orbit 0, CXXC orbit 30) pairs realized by the 531 profile orbits.',
        },
        {
            'constraint_name': 'generic_profile_unique',
            'constraint_value': str(profile_summary['generic_profile_unique']),
            'provenance': 'EXACT_DERIVED',
            'note': 'The full-support profile (3 in all 9 fibers) is unique at the profile-orbit level.',
        },
        {
            'constraint_name': 'generic_profile',
            'constraint_value': str((3, 3, 3, 3, 3, 3, 3, 3, 3)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Generic rank-1 support activates all 27 live X atoms, 3 in each output fiber.',
        },
        {
            'constraint_name': 'generic_cxxc_orbit0_support',
            'constraint_value': str(profile_summary['generic_cxxc_orbit0_support']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Support-level self-pair count for the generic rank-1 activation profile.',
        },
        {
            'constraint_name': 'generic_cxxc_orbit30_support',
            'constraint_value': str(profile_summary['generic_cxxc_orbit30_support']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Support-level distinct same-fiber pair count for the generic rank-1 activation profile.',
        },
        {
            'constraint_name': 'orbit_sum_linearization_nontrivial_rank_bound',
            'constraint_value': 'False',
            'provenance': 'EXACT_DERIVED',
            'note': 'The 8-equation XC orbit-sum model alone does not force any positive lower bound on R.',
        },
        {
            'constraint_name': 'orbit_sum_counterexample_vector',
            'constraint_value': counterexample_rows[0]['orbit_sum_vector'],
            'provenance': 'EXACT_DERIVED',
            'note': 'One sparse rank-1 term can match the full orbit-sum RHS vector while failing the coordinate tensor equations.',
        },
        {
            'constraint_name': 'orbit_sum_counterexample_full_tensor_mismatches',
            'constraint_value': str(counterexample_rows[0]['full_tensor_equation_mismatches']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Number of coordinate tensor equations missed by that single-term orbit-sum witness.',
        },
    ]

    print('Writing outputs...')
    write_csv(
        EXPORTS / 'step48_generic_rank1_live_atoms.csv',
        generic_live_atom_rows,
        ['fiber_id', 'target_c', 'target_c_name', 'row_index', 'sum_index', 'col_index', 'x_atom', 'x_atom_name', 'coefficient_monomial'],
    )
    write_csv(
        EXPORTS / 'step48_generic_rank1_fiber_summary.csv',
        generic_fiber_rows,
        ['fiber_id', 'target_c', 'target_c_name', 'activated_live_atoms_generic', 'activated_atoms', 'coefficient_monomials'],
    )
    write_csv(
        EXPORTS / 'step48_activation_profile_orbits.csv',
        profile_rows,
        ['profile_orbit_id', 'canonical_profile', 'profile_matrix', 'cxxc_orbit0_support', 'cxxc_orbit30_support', 'support_pair_count', 'max_fiber_activation', 'nonzero_fibers', 'fibers_with_0', 'fibers_with_1', 'fibers_with_2', 'fibers_with_3', 'sample_a_support', 'sample_b_support', 'is_generic_full_support_profile'],
    )
    write_csv(
        EXPORTS / 'step48_support_contribution_summary.csv',
        contribution_rows,
        ['cxxc_orbit0_support', 'cxxc_orbit30_support', 'n_profile_orbits', 'support_pair_count', 'example_profile'],
    )
    write_csv(
        EXPORTS / 'step48_tensor_equations.csv',
        equation_rows,
        ['a_atom', 'a_name', 'b_atom', 'b_name', 'c_atom', 'c_name', 'x_atom', 'x_atom_name', 'xc_config_id', 'xc_orbit_id', 'x_live', 'c_relation_to_target', 'tensor_rhs', 'equation_template'],
    )
    write_csv(
        EXPORTS / 'step48_xc_orbit_equation_summary.csv',
        tensor_orbit_rows,
        ['xc_orbit_id', 'orbit_size', 'rep_config_id', 'rep_readable', 'rep_x_atom_name', 'rep_c_atom_name', 'x_live', 'c_relation_to_target', 'tensor_rhs_constant', 'tensor_rhs_sum_on_orbit', 'tensor_rhs_average_on_orbit'],
    )
    write_csv(
        EXPORTS / 'step48_rank1_xc_orbit_sum_formulas.csv',
        formula_rows,
        ['xc_orbit_id', 'orbit_size', 'x_live', 'c_relation_to_target', 'notation', 'orbit_sum_formula', 'target_rhs_sum', 'target_rhs_average'],
    )
    write_csv(
        EXPORTS / 'step48_orbit_sum_constraint_system.csv',
        constraint_rows,
        ['xc_orbit_id', 'constraint_lhs', 'constraint_rhs_sum', 'constraint_rhs_average', 'exact_constraint', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step48_orbit_sum_counterexample.csv',
        counterexample_rows,
        ['a_support', 'b_support', 'c_support', 'orbit_sum_vector', 'matches_8_orbit_sum_system', 'full_tensor_equation_mismatches', 'interpretation'],
    )
    write_csv(
        EXPORTS / 'step48_constraint_model_summary.csv',
        summary_rows,
        ['constraint_name', 'constraint_value', 'provenance', 'note'],
    )
    write_markdown_summary(
        EXPORTS / 'step48_tensor_profile_constraint_model.md',
        generic_fiber_rows,
        profile_rows,
        contribution_rows,
        tensor_orbit_rows,
        formula_rows,
        constraint_rows,
        counterexample_rows,
        summary_rows,
    )
    print()

    print('=== SUMMARY ===')
    print(f"XC equation orbits: {len(tensor_orbit_rows)}")
    print(f"Positive XC orbit ids: {[int(row['xc_orbit_id']) for row in tensor_orbit_rows if int(row['tensor_rhs_constant']) == 1]}")
    print(f"Profile orbit types: {profile_summary['activation_profile_orbit_types']}")
    print(f"Orbit-sum counterexample: {counterexample_rows[0]['orbit_sum_vector']}")


if __name__ == '__main__':
    main()