"""
ade3x3_step57_fiber_group_partition_enumeration.py

Step 57: Fiber-group partition enumeration with orbit-budget filtering.

This step stays entirely at the discrete combinatorial level.

Track A:
1. Enumerate ordered 9-part compositions for R = 9..23.
2. Reduce by the S3 x S3 action on the 3x3 output-fiber grid.
3. Enumerate partition types (sorted 9-tuples) and their symmetry-inequivalent
   fiber assignments.

Track B:
1. Compute within-fiber orbit-30 budgets per partition type.
2. Record exact no-spreading survivors with sum_i C(n_i, 2) >= 27.
3. Run a conservative exact-3 rectangle-tiling feasibility model: each term is
   assigned a primary fiber and a rectangular spreading shape containing that
   fiber, and the resulting touched-fiber counts must be exactly 3 on all 9
   fibers.

The exact-3 tiling model is intentionally conservative. It is an exact discrete
covering model for the 9-fiber x 3-channel picture, but it is not yet the full
coefficient feasibility problem from the 729 equations.
"""

from __future__ import annotations

import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations
from math import comb, factorial
from pathlib import Path

import numpy as np

EXPORTS = Path('outputs/exports')
R_VALUES = tuple(range(9, 24))
TARGET_TOUCHES = (3,) * 9
FIBERS = tuple((row_idx, col_idx) for row_idx in range(3) for col_idx in range(3))
S3 = tuple(permutations(range(3)))
DEFAULT_WORKER_COUNT = min(len(R_VALUES), max(1, os.cpu_count() or 1))


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


def fiber_index(row_idx: int, col_idx: int) -> int:
    return 3 * row_idx + col_idx


def build_group_actions() -> tuple[tuple[int, ...], ...]:
    actions: list[tuple[int, ...]] = []
    for row_perm in S3:
        for col_perm in S3:
            mapping = [0] * 9
            for old_idx, (row_idx, col_idx) in enumerate(FIBERS):
                mapping[old_idx] = fiber_index(row_perm[row_idx], col_perm[col_idx])
            actions.append(tuple(mapping))
    return tuple(actions)


GROUP_ACTIONS = build_group_actions()


def transform_assignment(assignment: tuple[int, ...], action: tuple[int, ...]) -> tuple[int, ...]:
    transformed = [0] * 9
    for old_idx, new_idx in enumerate(action):
        transformed[new_idx] = assignment[old_idx]
    return tuple(transformed)


def canonical_assignment(assignment: tuple[int, ...]) -> tuple[int, ...]:
    return min(transform_assignment(assignment, action) for action in GROUP_ACTIONS)


def cycle_lengths(action: tuple[int, ...]) -> tuple[int, ...]:
    seen = [False] * 9
    lengths: list[int] = []
    for start in range(9):
        if seen[start]:
            continue
        idx = start
        length = 0
        while not seen[idx]:
            seen[idx] = True
            idx = action[idx]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths))


GROUP_CYCLES = tuple(cycle_lengths(action) for action in GROUP_ACTIONS)


def ordered_assignment_count(partition_type: tuple[int, ...]) -> int:
    multiplicities = Counter(partition_type)
    result = factorial(9)
    for count in multiplicities.values():
        result //= factorial(count)
    return result


def within_fiber_pairs(partition_type: tuple[int, ...]) -> int:
    return sum(part * (part - 1) // 2 for part in partition_type)


def partition_types_exact(total: int, max_parts: int = 9, max_part: int | None = None) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def recurse(remaining: int, max_allowed: int, parts_left: int, prefix: tuple[int, ...]) -> None:
        if remaining == 0:
            padded = prefix + (0,) * (max_parts - len(prefix))
            results.append(padded)
            return
        if parts_left == 0:
            return
        local_max = min(remaining, max_allowed)
        if max_part is not None:
            local_max = min(local_max, max_part)
        for value in range(local_max, 0, -1):
            recurse(remaining - value, value, parts_left - 1, prefix + (value,))

    recurse(total, total if max_part is None else max_part, max_parts, tuple())
    return results


def fixed_assignments_for_type(partition_type: tuple[int, ...], cycles: tuple[int, ...]) -> int:
    counts = Counter(partition_type)
    count_items = tuple(sorted(counts.items()))

    @lru_cache(maxsize=None)
    def recurse(cycle_idx: int, remaining_items: tuple[tuple[int, int], ...]) -> int:
        if cycle_idx == len(cycles):
            return int(all(count == 0 for _, count in remaining_items))
        cycle_len = cycles[cycle_idx]
        total = 0
        for item_idx, (value, count) in enumerate(remaining_items):
            if count < cycle_len:
                continue
            updated_items = list(remaining_items)
            updated_items[item_idx] = (value, count - cycle_len)
            total += recurse(cycle_idx + 1, tuple(updated_items))
        return total

    return recurse(0, count_items)


def assignment_orbits_for_type(partition_type: tuple[int, ...]) -> int:
    total = 0
    for cycles in GROUP_CYCLES:
        total += fixed_assignments_for_type(partition_type, cycles)
    return total // len(GROUP_CYCLES)


def unique_assignments_for_type(partition_type: tuple[int, ...]) -> list[tuple[int, ...]]:
    counts = Counter(partition_type)
    results: list[tuple[int, ...]] = []

    def recurse(position: int, remaining: dict[int, int], prefix: list[int]) -> None:
        if position == 9:
            results.append(tuple(prefix))
            return
        for value in sorted(remaining.keys(), reverse=True):
            if remaining[value] == 0:
                continue
            remaining[value] -= 1
            prefix.append(value)
            recurse(position + 1, remaining, prefix)
            prefix.pop()
            remaining[value] += 1

    recurse(0, dict(counts), [])
    return results


def subset_masks_containing(index_value: int) -> list[tuple[int, ...]]:
    masks: list[tuple[int, ...]] = []
    base = {0, 1, 2}
    for size in (1, 2, 3):
        for subset in combinations(base, size):
            if index_value in subset:
                masks.append(tuple(sorted(subset)))
    return masks


ROW_SUBSETS = {row_idx: subset_masks_containing(row_idx) for row_idx in range(3)}
COL_SUBSETS = {col_idx: subset_masks_containing(col_idx) for col_idx in range(3)}


def shape_label(row_subset: tuple[int, ...], col_subset: tuple[int, ...]) -> str:
    return f"{len(row_subset)}x{len(col_subset)}"


def rectangle_vector(row_subset: tuple[int, ...], col_subset: tuple[int, ...]) -> tuple[int, ...]:
    vector = np.zeros(9, dtype=np.int8)
    for row_idx in row_subset:
        for col_idx in col_subset:
            vector[fiber_index(row_idx, col_idx)] = 1
    return tuple(int(value) for value in vector.tolist())


def build_rectangle_options() -> dict[int, list[tuple[tuple[int, ...], str]]]:
    options: dict[int, list[tuple[tuple[int, ...], str]]] = {}
    for primary_idx, (row_idx, col_idx) in enumerate(FIBERS):
        rects: list[tuple[tuple[int, ...], str]] = []
        for row_subset in ROW_SUBSETS[row_idx]:
            for col_subset in COL_SUBSETS[col_idx]:
                rects.append((rectangle_vector(row_subset, col_subset), shape_label(row_subset, col_subset)))
        options[primary_idx] = rects
    return options


RECTANGLE_OPTIONS = build_rectangle_options()


def build_aggregate_options(max_terms: int = 3) -> dict[int, dict[int, list[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]]]]:
    aggregate: dict[int, dict[int, list[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]]]] = {}
    for primary_idx in range(9):
        aggregate[primary_idx] = {0: [((0,) * 9, tuple())]}
        rectangles = RECTANGLE_OPTIONS[primary_idx]
        for term_count in range(1, max_terms + 1):
            coverage_map: dict[tuple[int, ...], tuple[tuple[str, int], ...]] = {}
            for choice_indices in combinations_with_replacement(range(len(rectangles)), term_count):
                total = np.zeros(9, dtype=np.int8)
                shapes = Counter()
                for choice_idx in choice_indices:
                    rect_vec, label = rectangles[choice_idx]
                    total += np.array(rect_vec, dtype=np.int8)
                    shapes[label] += 1
                coverage_key = tuple(int(value) for value in total.tolist())
                shape_profile = tuple(sorted(shapes.items()))
                incumbent = coverage_map.get(coverage_key)
                if incumbent is None or shape_profile < incumbent:
                    coverage_map[coverage_key] = shape_profile
            rows = sorted(coverage_map.items(), key=lambda item: (sum(item[0]), item[0], item[1]))
            aggregate[primary_idx][term_count] = [(coverage_key, shape_profile) for coverage_key, shape_profile in rows]
    return aggregate


AGGREGATE_OPTIONS = build_aggregate_options()


def format_shape_profile(shape_profile: tuple[tuple[str, int], ...]) -> str:
    if not shape_profile:
        return 'none'
    return '; '.join(f"{label}:{count}" for label, count in shape_profile)


def format_partition(partition_type: tuple[int, ...]) -> str:
    return '(' + ','.join(str(value) for value in partition_type) + ')'


def orbit_of_assignment(assignment: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted({transform_assignment(assignment, action) for action in GROUP_ACTIONS}))


def exact3_tiling_witness(assignment: tuple[int, ...]) -> tuple[bool, tuple[tuple[tuple[int, int], tuple[tuple[str, int], ...]], ...] | None]:
    if max(assignment) > 3:
        return False, None

    @lru_cache(maxsize=None)
    def recurse(primary_idx: int, remaining: tuple[int, ...]) -> tuple[tuple[tuple[int, int], tuple[tuple[str, int], ...]], ...] | None:
        if primary_idx == 9:
            return tuple() if all(value == 0 for value in remaining) else None
        term_count = assignment[primary_idx]
        for coverage_key, shape_profile in AGGREGATE_OPTIONS[primary_idx][term_count]:
            if any(coverage_key[fiber_idx] > remaining[fiber_idx] for fiber_idx in range(9)):
                continue
            updated = tuple(remaining[fiber_idx] - coverage_key[fiber_idx] for fiber_idx in range(9))
            suffix = recurse(primary_idx + 1, updated)
            if suffix is not None:
                return (((primary_idx, term_count), shape_profile),) + suffix
        return None

    witness = recurse(0, TARGET_TOUCHES)
    return witness is not None, witness


def summarize_witness(witness: tuple[tuple[tuple[int, int], tuple[tuple[str, int], ...]], ...] | None) -> str:
    if witness is None:
        return ''
    parts: list[str] = []
    for (primary_idx, term_count), shape_profile in witness:
        if term_count == 0:
            continue
        row_idx, col_idx = FIBERS[primary_idx]
        parts.append(f"f[{row_idx},{col_idx}]/{term_count}: {format_shape_profile(shape_profile)}")
    return ' | '.join(parts)


def summarize_cycle_classes() -> list[dict]:
    class_counter = Counter(GROUP_CYCLES)
    rows: list[dict] = []
    for cycles, count in sorted(class_counter.items()):
        rows.append({
            'cycle_lengths': str(cycles),
            'class_size': count,
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def summarize_shape_classes() -> list[dict]:
    rows: list[dict] = []
    for p_size in (1, 2, 3):
        for q_size in (1, 2, 3):
            multiplicity = comb(2, p_size - 1) * comb(2, q_size - 1)
            rows.append({
                'shape': f'{p_size}x{q_size}',
                'row_support_size': p_size,
                'col_support_size': q_size,
                'fibers_touched': p_size * q_size,
                'extra_fibers_beyond_primary': p_size * q_size - 1,
                'primary_anchored_rectangles': multiplicity,
                'provenance': 'EXACT_DERIVED',
            })
    return rows


def process_rank(total_terms: int) -> dict[str, object]:
    partition_types = partition_types_exact(total_terms)
    symmetry_total = 0
    no_spreading_types = 0
    exact3_type_survivors = 0
    exact3_assignment_survivors = 0
    first_survivor_type = ''
    first_survivor_assignment = ''
    first_survivor_witness = ''
    min_deficit = None
    max_deficit = None
    type_rows: list[dict] = []
    survivor_rows: list[dict] = []

    for partition_type in partition_types:
        within_pairs = within_fiber_pairs(partition_type)
        deficit = max(0, 27 - within_pairs)
        min_deficit = deficit if min_deficit is None else min(min_deficit, deficit)
        max_deficit = deficit if max_deficit is None else max(max_deficit, deficit)
        orbit_count = assignment_orbits_for_type(partition_type)
        symmetry_total += orbit_count
        no_spreading_pass = within_pairs >= 27
        if no_spreading_pass:
            no_spreading_types += 1

        surviving_assignment_orbits = 0
        representative_assignment = ''
        representative_witness = ''
        is_exact3_candidate = max(partition_type) <= 3

        if is_exact3_candidate:
            seen_orbits: set[tuple[tuple[int, ...], ...]] = set()
            for assignment in unique_assignments_for_type(partition_type):
                orbit = orbit_of_assignment(assignment)
                if orbit in seen_orbits:
                    continue
                seen_orbits.add(orbit)
                feasible, witness = exact3_tiling_witness(assignment)
                if feasible:
                    surviving_assignment_orbits += 1
                    if not representative_assignment:
                        representative_assignment = format_partition(assignment)
                        representative_witness = summarize_witness(witness)
            if surviving_assignment_orbits > 0:
                exact3_type_survivors += 1
                exact3_assignment_survivors += surviving_assignment_orbits
                if not first_survivor_type:
                    first_survivor_type = format_partition(partition_type)
                    first_survivor_assignment = representative_assignment
                    first_survivor_witness = representative_witness

        type_rows.append({
            'R': total_terms,
            'partition_type': format_partition(partition_type),
            'ordered_assignments': ordered_assignment_count(partition_type),
            'symmetry_assignment_orbits': orbit_count,
            'within_fiber_pairs': within_pairs,
            'spreading_deficit_to_27': deficit,
            'zero_fibers': sum(1 for value in partition_type if value == 0),
            'max_group_size': max(partition_type),
            'no_spreading_filter_pass': no_spreading_pass,
            'exact3_tiling_candidate': is_exact3_candidate,
            'exact3_tiling_surviving_assignment_orbits': surviving_assignment_orbits,
            'example_surviving_assignment': representative_assignment,
            'example_shape_witness': representative_witness,
            'provenance': 'EXACT_DERIVED',
        })

        if surviving_assignment_orbits > 0:
            survivor_rows.append({
                'R': total_terms,
                'partition_type': format_partition(partition_type),
                'surviving_assignment_orbits': surviving_assignment_orbits,
                'example_surviving_assignment': representative_assignment,
                'example_shape_witness': representative_witness,
                'provenance': 'EXACT_DERIVED',
            })

    summary_row = {
        'R': total_terms,
        'ordered_compositions': comb(total_terms + 8, 8),
        'partition_types': len(partition_types),
        'symmetry_inequivalent_compositions': symmetry_total,
        'spreading_allowed_types': len(partition_types),
        'no_spreading_survivor_types': no_spreading_types,
        'exact3_tiling_survivor_types': exact3_type_survivors,
        'exact3_tiling_surviving_assignment_orbits': exact3_assignment_survivors,
        'min_spreading_deficit': min_deficit,
        'max_spreading_deficit': max_deficit,
        'first_exact3_survivor_type': first_survivor_type,
        'first_exact3_survivor_assignment': first_survivor_assignment,
        'first_exact3_survivor_witness': first_survivor_witness,
        'provenance': 'EXACT_DERIVED',
    }
    return {
        'summary_row': summary_row,
        'type_rows': type_rows,
        'survivor_rows': survivor_rows,
    }


def resolve_worker_count() -> int:
    raw_value = os.environ.get('ADE3X3_STEP57_WORKERS', '').strip()
    if not raw_value:
        return DEFAULT_WORKER_COUNT
    try:
        requested = int(raw_value)
    except ValueError:
        return DEFAULT_WORKER_COUNT
    return max(1, min(len(R_VALUES), requested))


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)

    cycle_rows = summarize_cycle_classes()
    shape_rows = summarize_shape_classes()
    summary_rows: list[dict] = []
    type_rows: list[dict] = []
    survivor_rows: list[dict] = []

    worker_count = resolve_worker_count()
    print(f'Processing R values in parallel with {worker_count} worker processes ...')
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(process_rank, total_terms): total_terms for total_terms in R_VALUES}
        for future in as_completed(future_map):
            total_terms = future_map[future]
            result = future.result()
            print(f"  finished R={total_terms}")
            summary_rows.append(result['summary_row'])
            type_rows.extend(result['type_rows'])
            survivor_rows.extend(result['survivor_rows'])

    summary_rows.sort(key=lambda row: int(row['R']))
    type_rows.sort(key=lambda row: (int(row['R']), row['partition_type']))
    survivor_rows.sort(key=lambda row: (int(row['R']), row['partition_type']))

    summary_fieldnames = [
        'R',
        'ordered_compositions',
        'partition_types',
        'symmetry_inequivalent_compositions',
        'spreading_allowed_types',
        'no_spreading_survivor_types',
        'exact3_tiling_survivor_types',
        'exact3_tiling_surviving_assignment_orbits',
        'min_spreading_deficit',
        'max_spreading_deficit',
        'first_exact3_survivor_type',
        'first_exact3_survivor_assignment',
        'first_exact3_survivor_witness',
        'provenance',
    ]
    write_csv(EXPORTS / 'step57_summary.csv', summary_rows, summary_fieldnames)
    write_csv(
        EXPORTS / 'step57_partition_types.csv',
        type_rows,
        [
            'R',
            'partition_type',
            'ordered_assignments',
            'symmetry_assignment_orbits',
            'within_fiber_pairs',
            'spreading_deficit_to_27',
            'zero_fibers',
            'max_group_size',
            'no_spreading_filter_pass',
            'exact3_tiling_candidate',
            'exact3_tiling_surviving_assignment_orbits',
            'example_surviving_assignment',
            'example_shape_witness',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step57_exact3_survivors.csv',
        survivor_rows,
        [
            'R',
            'partition_type',
            'surviving_assignment_orbits',
            'example_surviving_assignment',
            'example_shape_witness',
            'provenance',
        ],
    )
    write_csv(EXPORTS / 'step57_symmetry_cycle_types.csv', cycle_rows, ['cycle_lengths', 'class_size', 'provenance'])
    write_csv(
        EXPORTS / 'step57_shape_classes.csv',
        shape_rows,
        [
            'shape',
            'row_support_size',
            'col_support_size',
            'fibers_touched',
            'extra_fibers_beyond_primary',
            'primary_anchored_rectangles',
            'provenance',
        ],
    )

    r9 = next(row for row in summary_rows if int(row['R']) == 9)
    r22 = next(row for row in summary_rows if int(row['R']) == 22)
    r23 = next(row for row in summary_rows if int(row['R']) == 23)
    markdown = f"""# Step 57: Fiber-Group Partition Enumeration with Orbit Budget Filter

[EXACT_DERIVED]

The fiber-partition sweep for R = 9..23 is now exact at the composition and symmetry level.
Ordered 9-part compositions are counted by C(R+8, 8), partition types are the integer partitions
of R into at most 9 parts, and symmetry-inequivalent fiber assignments are reduced under the
S3 x S3 action on the 3x3 output-fiber grid by exact Burnside counting.

The exact no-spreading filter keeps only partition types with within-fiber pair budget
sum_i C(n_i, 2) >= 27. Separately, a conservative exact-3 tiling model tests whether the
partition can be realized by rectangular spreading shapes so that every fiber is touched
exactly three times.

Key rows:

- R=9: ordered={r9['ordered_compositions']}, types={r9['partition_types']}, symmetry classes={r9['symmetry_inequivalent_compositions']}, no-spreading survivors={r9['no_spreading_survivor_types']}, exact-3 survivor assignment orbits={r9['exact3_tiling_surviving_assignment_orbits']}
- R=22: ordered={r22['ordered_compositions']}, types={r22['partition_types']}, symmetry classes={r22['symmetry_inequivalent_compositions']}, no-spreading survivors={r22['no_spreading_survivor_types']}, exact-3 survivor assignment orbits={r22['exact3_tiling_surviving_assignment_orbits']}
- R=23: ordered={r23['ordered_compositions']}, types={r23['partition_types']}, symmetry classes={r23['symmetry_inequivalent_compositions']}, no-spreading survivors={r23['no_spreading_survivor_types']}, exact-3 survivor assignment orbits={r23['exact3_tiling_surviving_assignment_orbits']}

[INTERPRETATION]

This step does not solve coefficients. It gives the exact discrete search front before the
729-equation feasibility problem. The no-spreading filter is strong only for heavily clustered
partitions; the exact-3 tiling model is stronger on the opposite side because it enforces the
9 fibers x 3 channels picture literally. If exact-3 survivors already appear at small R, then the
partition ansatz alone cannot rule out low rank. If exact-3 survivors disappear at some R, that is
only a failure of this conservative tiling model, not yet a theorem against weighted cancellation.
"""
    write_markdown(EXPORTS / 'step57_fiber_group_partition_enumeration.md', markdown)

    print('\nSummary rows:')
    for row in summary_rows:
        print(
            f"  R={row['R']}: ordered={row['ordered_compositions']}, "
            f"types={row['partition_types']}, sym={row['symmetry_inequivalent_compositions']}, "
            f"no-spread={row['no_spreading_survivor_types']}, "
            f"exact3_orbits={row['exact3_tiling_surviving_assignment_orbits']}"
        )


if __name__ == '__main__':
    main()