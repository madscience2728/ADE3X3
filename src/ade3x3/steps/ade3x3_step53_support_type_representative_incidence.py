"""
ade3x3_step53_support_type_representative_incidence.py

Step 53: Support-Type Representative Incidence.

This step asks whether support geometry alone can force any of the 8 exact
representative tensor equation types from Step 49 to be absent for a single
rank-1 term. A support type records only which row/column indices appear in the
alpha, beta, and gamma factors:

  (supp_row(alpha), supp_col(alpha), supp_row(beta), supp_col(beta),
   supp_row(gamma), supp_col(gamma))

modulo the natural S3 x S3 x S3 action.

The result is exact and negative: once Type 0 is support-feasible, there exist
support classes that realize all 8 representative equation types. Therefore
support-only incidence cannot yield a universal lower bound at the
representative-equation level.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations, product
from pathlib import Path

EXPORTS = Path('outputs/exports')

S3 = list(permutations((0, 1, 2)))
SUBSET_MASKS = list(range(8))
ACTIONS = [(left_idx, mid_idx, right_idx) for left_idx in range(6) for mid_idx in range(6) for right_idx in range(6)]
REPRESENTATIVE_TYPES = [
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 1),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0, 1),
    (0, 0, 1, 0, 1, 0),
    (0, 0, 1, 0, 1, 1),
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def subset_mask_to_string(mask: int) -> str:
    entries = [str(index) for index in range(3) if (mask >> index) & 1]
    return '{' + ','.join(entries) + '}'


PERMUTED_MASKS = [
    [sum(((mask >> src_idx) & 1) << perm[src_idx] for src_idx in range(3)) for mask in SUBSET_MASKS]
    for perm in S3
]


def transform_state(state: tuple[int, int, int, int, int, int], action: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    left_idx, mid_idx, right_idx = action
    return (
        PERMUTED_MASKS[left_idx][state[0]],
        PERMUTED_MASKS[mid_idx][state[1]],
        PERMUTED_MASKS[mid_idx][state[2]],
        PERMUTED_MASKS[right_idx][state[3]],
        PERMUTED_MASKS[left_idx][state[4]],
        PERMUTED_MASKS[right_idx][state[5]],
    )


def canonicalize_state(state: tuple[int, int, int, int, int, int]) -> tuple[tuple[int, int, int, int, int, int], int]:
    orbit: set[tuple[int, int, int, int, int, int]] = set()
    best: tuple[int, int, int, int, int, int] | None = None
    for action in ACTIONS:
        candidate = transform_state(state, action)
        orbit.add(candidate)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best, len(orbit)


def representative_present(state: tuple[int, int, int, int, int, int], type_id: int) -> bool:
    row_idx, sum_left, sum_right, col_idx, out_row, out_col = REPRESENTATIVE_TYPES[type_id]
    return all(
        ((mask >> target) & 1) == 1
        for mask, target in zip(state, (row_idx, sum_left, sum_right, col_idx, out_row, out_col))
    )


def build_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    class_map: dict[tuple[int, int, int, int, int, int], dict[str, object]] = {}

    for state in product(SUBSET_MASKS, repeat=6):
        canonical_state, orbit_size = canonicalize_state(state)
        if canonical_state in class_map:
            continue
        active = [representative_present(canonical_state, type_id) for type_id in range(8)]
        class_map[canonical_state] = {
            'orbit_size': orbit_size,
            'active_count': sum(active),
            'auto_zero_count': 8 - sum(active),
            'orbit0_possible': active[0],
            'all8_active': sum(active) == 8,
            'active_mask': ''.join('1' if flag else '0' for flag in active),
        }

    support_rows: list[dict] = []
    mask_summary: defaultdict[str, dict[str, int]] = defaultdict(lambda: {'n_classes': 0, 'n_orbit0_possible': 0, 'n_all8_active': 0})
    orbit0_hist: Counter[int] = Counter()

    for support_type_id, canonical_state in enumerate(sorted(class_map), start=1):
        stats = class_map[canonical_state]
        row = {
            'support_type_id': support_type_id,
            'canonical_state': str(tuple(subset_mask_to_string(mask) for mask in canonical_state)),
            'alpha_row_support': subset_mask_to_string(canonical_state[0]),
            'alpha_col_support': subset_mask_to_string(canonical_state[1]),
            'beta_row_support': subset_mask_to_string(canonical_state[2]),
            'beta_col_support': subset_mask_to_string(canonical_state[3]),
            'gamma_row_support': subset_mask_to_string(canonical_state[4]),
            'gamma_col_support': subset_mask_to_string(canonical_state[5]),
            'orbit_size': stats['orbit_size'],
            'active_count': stats['active_count'],
            'auto_zero_count': stats['auto_zero_count'],
            'orbit0_possible': stats['orbit0_possible'],
            'all8_active': stats['all8_active'],
            'active_mask': stats['active_mask'],
            'provenance': 'EXACT_DERIVED',
        }
        support_rows.append(row)
        mask_summary[row['active_mask']]['n_classes'] += 1
        mask_summary[row['active_mask']]['n_orbit0_possible'] += int(bool(stats['orbit0_possible']))
        mask_summary[row['active_mask']]['n_all8_active'] += int(bool(stats['all8_active']))
        if stats['orbit0_possible']:
            orbit0_hist[int(stats['active_count'])] += 1

    mask_rows = [
        {
            'active_mask': active_mask,
            'active_count': active_mask.count('1'),
            'auto_zero_count': 8 - active_mask.count('1'),
            'n_classes': counts['n_classes'],
            'n_orbit0_possible': counts['n_orbit0_possible'],
            'n_all8_active': counts['n_all8_active'],
            'provenance': 'EXACT_DERIVED',
        }
        for active_mask, counts in sorted(mask_summary.items())
    ]

    orbit0_rows = [
        {
            'active_count': active_count,
            'auto_zero_count': 8 - active_count,
            'n_support_classes': orbit0_hist[active_count],
            'provenance': 'EXACT_DERIVED',
        }
        for active_count in sorted(orbit0_hist)
    ]

    orbit_hist = Counter(int(stats['orbit_size']) for stats in class_map.values())
    valid_rows = [row for row in support_rows if row['orbit0_possible'] == True]
    all8_rows = [row for row in valid_rows if row['all8_active'] == True]

    summary_rows = [
        {'summary_name': 'n_support_type_classes', 'summary_value': str(len(support_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Distinct six-support classes modulo S3 x S3 x S3.'},
        {'summary_name': 'support_type_orbit_sizes', 'summary_value': ','.join(str(size) for size in sorted(orbit_hist)), 'provenance': 'EXACT_DERIVED', 'note': 'Possible orbit sizes of support classes.'},
        {'summary_name': 'support_type_orbit_size_histogram', 'summary_value': '; '.join(f'{size}:{orbit_hist[size]}' for size in sorted(orbit_hist)), 'provenance': 'EXACT_DERIVED', 'note': 'Class counts by support-orbit size.'},
        {'summary_name': 'n_with_orbit0_possible', 'summary_value': str(len(valid_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Support classes that can realize the positive Type 0 equation.'},
        {'summary_name': 'min_auto_zero_among_orbit0_possible', 'summary_value': str(min(int(row['auto_zero_count']) for row in valid_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Minimum number of representative types forced absent by support among Type-0-feasible classes.'},
        {'summary_name': 'n_all8_active_among_orbit0_possible', 'summary_value': str(len(all8_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Type-0-feasible classes whose support allows all 8 representative equation types.'},
        {'summary_name': 'support_only_representative_obstruction', 'summary_value': 'vacuous', 'provenance': 'EXACT_DERIVED', 'note': 'Support-only representative incidence cannot force any zero-RHS type to vanish universally once Type 0 is allowed.'},
        {'summary_name': 'sample_all8_active_support_type_id', 'summary_value': str(all8_rows[0]['support_type_id']), 'provenance': 'EXACT_DERIVED', 'note': 'One canonical support class with active_mask 11111111.'},
    ]
    return support_rows, mask_rows, orbit0_rows, summary_rows


def write_markdown_summary(
    path: Path,
    support_rows: list[dict],
    mask_rows: list[dict],
    orbit0_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    sample_all8 = next(row for row in support_rows if row['all8_active'] == True and row['orbit0_possible'] == True)

    lines: list[str] = []
    w = lines.append
    w('# Step 53: Support-Type Representative Incidence')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('This step keeps only support information for a single rank-1 term: which indices appear in alpha, beta, and gamma along their row/column coordinates. Those six subset supports are classified modulo the natural S3 x S3 x S3 action.')
    w('')
    w(f"Support-type classes: {summary['n_support_type_classes']}")
    w(f"Orbit sizes: {summary['support_type_orbit_sizes']}")
    w(f"Type-0-feasible classes: {summary['n_with_orbit0_possible']}")
    w(f"Minimum forced auto-zero count among Type-0-feasible classes: {summary['min_auto_zero_among_orbit0_possible']}")
    w(f"All-8-active Type-0-feasible classes: {summary['n_all8_active_among_orbit0_possible']}")
    w('')
    w('## Orbit-0-Feasible Histogram')
    w('')
    w('| active_count | auto_zero_count | n_support_classes |')
    w('|--------------|-----------------|-------------------|')
    for row in orbit0_rows:
        w(f"| {row['active_count']} | {row['auto_zero_count']} | {row['n_support_classes']} |")
    w('')
    w('## Active-Mask Summary')
    w('')
    w('| active_mask | active_count | auto_zero_count | n_classes | n_orbit0_possible | n_all8_active |')
    w('|-------------|--------------|-----------------|-----------|-------------------|---------------|')
    for row in mask_rows[:16]:
        w(f"| {row['active_mask']} | {row['active_count']} | {row['auto_zero_count']} | {row['n_classes']} | {row['n_orbit0_possible']} | {row['n_all8_active']} |")
    w('')
    w('## Sample All-8-Active Support Class')
    w('')
    w(f"- support_type_id: {sample_all8['support_type_id']}")
    w(f"- active_mask: {sample_all8['active_mask']}")
    w(f"- alpha_row_support: {sample_all8['alpha_row_support']}")
    w(f"- alpha_col_support: {sample_all8['alpha_col_support']}")
    w(f"- beta_row_support: {sample_all8['beta_row_support']}")
    w(f"- beta_col_support: {sample_all8['beta_col_support']}")
    w(f"- gamma_row_support: {sample_all8['gamma_row_support']}")
    w(f"- gamma_col_support: {sample_all8['gamma_col_support']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('Step 53 closes off another support-only route. The Step 48 orbit-sum model was already too coarse; Step 53 shows that even exact support incidence against the 8 representative equation types is still vacuous. Once support is rich enough to allow the positive Type 0 equation, there are many support classes that also allow all 7 zero-RHS types. Any universal lower-bound argument must therefore use coefficient relations, quotient-space structure, or stronger algebraic constraints than support incidence alone.')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 53: Support-Type Representative Incidence ===')
    print()
    print('Enumerating support-type classes modulo S3 x S3 x S3...')
    support_rows, mask_rows, orbit0_rows, summary_rows = build_rows()
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    print(f"  support_type_classes={summary['n_support_type_classes']}")
    print(f"  type0_feasible={summary['n_with_orbit0_possible']}")
    print(f"  min_auto_zero_among_type0_feasible={summary['min_auto_zero_among_orbit0_possible']}")
    print(f"  all8_active_type0_feasible={summary['n_all8_active_among_orbit0_possible']}")
    print()
    print('Writing outputs...')
    write_csv(
        EXPORTS / 'step53_support_type_classes.csv',
        support_rows,
        [
            'support_type_id', 'canonical_state', 'alpha_row_support', 'alpha_col_support',
            'beta_row_support', 'beta_col_support', 'gamma_row_support', 'gamma_col_support',
            'orbit_size', 'active_count', 'auto_zero_count', 'orbit0_possible', 'all8_active',
            'active_mask', 'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step53_support_type_mask_summary.csv',
        mask_rows,
        ['active_mask', 'active_count', 'auto_zero_count', 'n_classes', 'n_orbit0_possible', 'n_all8_active', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step53_orbit0_feasible_histogram.csv',
        orbit0_rows,
        ['active_count', 'auto_zero_count', 'n_support_classes', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step53_summary.csv',
        summary_rows,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_markdown_summary(
        EXPORTS / 'step53_support_type_representative_incidence.md',
        support_rows,
        mask_rows,
        orbit0_rows,
        summary_rows,
    )
    print()
    print('=== SUMMARY ===')
    print(f"Support-type classes: {summary['n_support_type_classes']}")
    print(f"Type-0-feasible classes: {summary['n_with_orbit0_possible']}")
    print(f"Minimum auto-zero count among Type-0-feasible classes: {summary['min_auto_zero_among_orbit0_possible']}")
    print(f"All-8-active Type-0-feasible classes: {summary['n_all8_active_among_orbit0_possible']}")


if __name__ == '__main__':
    main()