"""
ade3x3_step46_same_fiber_subalgebra.py

Step 46: Same-Fiber Core + 64-Orbit Sub-Algebra Structure.

This step extends the verified step 45 exports instead of rebuilding orbit
maps from scratch. It treats the step-2 64-orbit closure table as the exact
composition law for the small closed CXXC sub-algebra.

Tasks:
  1. Same-fiber composition table and same-fiber-generated closure.
  2. Full 64-orbit sub-algebra composition table and Cayley-style matrix.
  3. Focused-orbit structural and composition analysis.
  4. Generator analysis of the 64-orbit sub-algebra.

Outputs:
  - outputs/exports/step46_same_fiber_outputs.csv
  - outputs/exports/step46_same_fiber_pair_summary.csv
  - outputs/exports/step46_same_fiber_closure.csv
  - outputs/exports/step46_64_subalgebra_table.csv
  - outputs/exports/step46_64_subalgebra_cayley.csv
  - outputs/exports/step46_64_subalgebra_summary.csv
  - outputs/exports/step46_focused_orbits.csv
  - outputs/exports/step46_focused_composition.csv
  - outputs/exports/step46_focus_profile_summary.csv
  - outputs/exports/step46_identity_map.csv
  - outputs/exports/step46_single_generator_closure.csv
  - outputs/exports/step46_greedy_generators.csv
  - outputs/exports/step46_same_fiber_subalgebra.md

Provenance:
  [EXACT_DERIVED] for all CSV and summary facts.
  [INTERPRETATION] only in the markdown narrative.
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

EXPORTS = Path('outputs/exports')

SAME_FIBER_EXPECTED = {0, 1, 2, 30, 131, 132, 133, 1055, 1057, 1059}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _decode_cxxc(cfg: int) -> tuple[int, int, int, int]:
    c2 = cfg % 9
    rem = cfg // 9
    x2 = rem % 81
    rem //= 81
    x1 = rem % 81
    c1 = rem // 81
    return c1, x1, x2, c2


def _decode_x(x: int) -> tuple[int, int, int, int]:
    return (x // 9) // 3, (x // 9) % 3, (x % 9) // 3, (x % 9) % 3


def c_name(c_atom: int) -> str:
    return f"C[{c_atom // 3},{c_atom % 3}]"


def build_orbit_meta() -> dict[int, dict]:
    signatures = {
        int(row['orbit_id']): row
        for row in read_csv_rows(EXPORTS / 'signatures_CXXC_refined.csv')
    }
    result: dict[int, dict] = {}
    for row in read_csv_rows(EXPORTS / 'stabilizer_classification.csv'):
        if row['schema'] != 'CXXC':
            continue
        orbit_id = int(row['orbit_id'])
        rep_config_id = int(row['rep_config_id'])
        c1, x1, x2, c2 = _decode_cxxc(rep_config_id)
        x1_r, x1_s, x1_t, x1_u = _decode_x(x1)
        x2_r, x2_s, x2_t, x2_u = _decode_x(x2)
        live_x1 = x1_s == x1_t
        live_x2 = x2_s == x2_t
        live_count = int(live_x1) + int(live_x2)
        sig = signatures.get(orbit_id, {})
        result[orbit_id] = {
            'orbit_id': orbit_id,
            'rep_config_id': rep_config_id,
            'orbit_size': int(row['orbit_size']),
            'stabilizer_order': int(row['stabilizer_order']),
            'stabilizer_type': row['stabilizer_type'],
            'signature_key': sig.get('signature_key', ''),
            'rep_readable': sig.get('rep_readable', ''),
            'generator_action_ids': row['generator_action_ids'],
            'c1': c1,
            'c2': c2,
            'c1_name': c_name(c1),
            'c2_name': c_name(c2),
            'x1_r': x1_r,
            'x1_s': x1_s,
            'x1_t': x1_t,
            'x1_u': x1_u,
            'x2_r': x2_r,
            'x2_s': x2_s,
            'x2_t': x2_t,
            'x2_u': x2_u,
            'live_x1': live_x1,
            'live_x2': live_x2,
            'both_live': live_count == 2,
            'live_count': live_count,
            'target_x1_atom': 3 * x1_r + x1_u if live_x1 else -1,
            'target_x2_atom': 3 * x2_r + x2_u if live_x2 else -1,
            'target_of_x1': c_name(3 * x1_r + x1_u) if live_x1 else '',
            'target_of_x2': c_name(3 * x2_r + x2_u) if live_x2 else '',
            'x1_fiber_position': x1_s if live_x1 else -1,
            'x2_fiber_position': x2_s if live_x2 else -1,
            'x1_eq_x2': x1 == x2,
            'c1_eq_c2': c1 == c2,
        }
    return result


def load_step45_core_rows() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv_rows(EXPORTS / 'step45_doubly_live_core.csv'):
        rows.append({
            'orbit_id': int(row['orbit_id']),
            'same_fiber': row['same_fiber'] == 'True',
            'focused_on_boundary': row['focused_on_boundary'] == 'True',
            'target_of_x1': row['target_of_x1'],
            'target_of_x2': row['target_of_x2'],
        })
    return rows


def load_step45_pair_table() -> dict[tuple[int, int], dict]:
    pair_table: dict[tuple[int, int], dict] = {}
    for row in read_csv_rows(EXPORTS / 'step45_closure_pair_table.csv'):
        if int(row['closure_step']) != 2:
            continue
        a_orbit = int(row['A_orbit'])
        b_orbit = int(row['B_orbit'])
        output_ids = ast.literal_eval(row['output_orbit_ids'])
        output_counts = ast.literal_eval(row['output_counts'])
        pair_table[(a_orbit, b_orbit)] = {
            'A_orbit': a_orbit,
            'B_orbit': b_orbit,
            'A_stab_type': row['A_stab_type'],
            'B_stab_type': row['B_stab_type'],
            'total_pairs': int(row['total_pairs']),
            'output_orbit_ids': list(output_ids),
            'output_counts': {int(k): int(v) for k, v in output_counts.items()},
        }
    return pair_table


def load_64_closed_ids(pair_table: dict[tuple[int, int], dict]) -> list[int]:
    orbit_ids = sorted({a for a, _ in pair_table} | {b for _, b in pair_table})
    if len(orbit_ids) != 64:
        raise ValueError(f"Expected 64 closure ids, got {len(orbit_ids)}")
    return orbit_ids


def closure_from_pair_table(seed_ids: set[int], pair_table: dict[tuple[int, int], dict]) -> list[dict]:
    current = set(seed_ids)
    rows: list[dict] = []
    step = 1
    while True:
        new_outputs: set[int] = set()
        compatible_pairs = 0
        witness_pairs = 0
        for a_orbit in sorted(current):
            for b_orbit in sorted(current):
                entry = pair_table[(a_orbit, b_orbit)]
                if entry['total_pairs'] == 0:
                    continue
                compatible_pairs += 1
                witness_pairs += entry['total_pairs']
                new_outputs.update(entry['output_orbit_ids'])
        next_set = current | new_outputs
        new_ids = sorted(next_set - current)
        rows.append({
            'closure_step': step,
            'set_size': len(current),
            'compatible_orbit_pairs': compatible_pairs,
            'witness_pairs': witness_pairs,
            'new_orbits': len(new_ids),
            'new_orbit_ids': str(new_ids),
        })
        if not new_ids:
            break
        current = next_set
        step += 1
    return rows


def singleton_generator_closure(seed: int, pair_table: dict[tuple[int, int], dict]) -> list[dict]:
    current = {seed}
    seen: dict[tuple[int, ...], int] = {}
    rows: list[dict] = []
    step = 0
    while True:
        key = tuple(sorted(current))
        if key in seen:
            rows.append({
                'step': step,
                'set_size': len(current),
                'orbit_ids': str(list(key)),
                'status': f"cycle_to_step_{seen[key]}",
            })
            break
        seen[key] = step
        rows.append({
            'step': step,
            'set_size': len(current),
            'orbit_ids': str(list(key)),
            'status': 'active',
        })
        next_set = set(current)
        for a_orbit in current:
            for b_orbit in current:
                next_set.update(pair_table[(a_orbit, b_orbit)]['output_orbit_ids'])
        if next_set == current:
            rows[-1]['status'] = 'fixed_point'
            break
        current = next_set
        step += 1
    return rows


def closure_of_generators(generators: set[int], pair_table: dict[tuple[int, int], dict]) -> set[int]:
    current = set(generators)
    while True:
        next_set = set(current)
        for a_orbit in current:
            for b_orbit in current:
                next_set.update(pair_table[(a_orbit, b_orbit)]['output_orbit_ids'])
        if next_set == current:
            return current
        current = next_set


def greedy_generating_set(all_ids: list[int], pair_table: dict[tuple[int, int], dict]) -> list[dict]:
    target = set(all_ids)
    generators: set[int] = set()
    current_closure: set[int] = set()
    rows: list[dict] = []
    step = 1
    while current_closure != target:
        best_orbit = None
        best_closure: set[int] | None = None
        best_gain = -1
        for orbit_id in all_ids:
            if orbit_id in generators:
                continue
            trial_generators = set(generators)
            trial_generators.add(orbit_id)
            trial_closure = closure_of_generators(trial_generators, pair_table)
            gain = len(trial_closure - current_closure)
            if gain > best_gain or (gain == best_gain and best_orbit is not None and orbit_id < best_orbit):
                best_orbit = orbit_id
                best_closure = trial_closure
                best_gain = gain
        if best_orbit is None or best_closure is None:
            raise RuntimeError('Greedy generator search stalled')
        generators.add(best_orbit)
        current_closure = best_closure
        rows.append({
            'step': step,
            'chosen_orbit': best_orbit,
            'generator_set': str(sorted(generators)),
            'closure_size': len(current_closure),
            'new_orbits_added': best_gain,
            'closure_orbits': str(sorted(current_closure)),
        })
        step += 1
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_markdown_summary(
    path: Path,
    same_fiber_summary: dict,
    same_fiber_closure_rows: list[dict],
    focused_rows: list[dict],
    focused_summary: dict,
    subalg_summary: dict,
    greedy_rows: list[dict],
    generator_rows: list[dict],
) -> None:
    lines: list[str] = []
    w = lines.append

    w("# Step 46: Same-Fiber Core + 64-Orbit Sub-Algebra Structure")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("[EXACT_DERIVED]")
    w("")
    w("## Task 1: Same-Fiber Composition")
    w("")
    w(f"Total same-fiber x same-fiber witness pairs: {same_fiber_summary['total_pairs']:,}")
    w(f"- Same-fiber outputs: {same_fiber_summary['same_fiber_pairs']:,} ({same_fiber_summary['same_fiber_pct']:.2f}%)")
    w(f"- Doubly-live different-fiber outputs: {same_fiber_summary['diff_fiber_doubly_live_pairs']:,} ({same_fiber_summary['diff_fiber_doubly_live_pct']:.2f}%)")
    w(f"- Outside the 28-orbit core: {same_fiber_summary['outside_core_pairs']:,} ({same_fiber_summary['outside_core_pct']:.2f}%)")
    w("")
    w("| closure_step | set_size | compatible_orbit_pairs | witness_pairs | new_orbits |")
    w("|--------------|----------|------------------------|---------------|------------|")
    for row in same_fiber_closure_rows:
        w(f"| {row['closure_step']} | {row['set_size']} | {row['compatible_orbit_pairs']} | {row['witness_pairs']:,} | {row['new_orbits']} |")
    w("")

    w("## Task 3: Focused Orbits")
    w("")
    w(f"Focused orbits: {len(focused_rows)}")
    w("| orbit_id | focus_boundary | fiber_positions | rep |")
    w("|----------|----------------|-----------------|-----|")
    for row in focused_rows:
        w(f"| {row['orbit_id']} | {row['focus_boundary']} | ({row['x1_fiber_position']}, {row['x2_fiber_position']}) | {row['rep_readable']} |")
    w("")
    w(f"Focused x focused witness pairs: {focused_summary['total_pairs']:,}")
    w(f"- Focused outputs: {focused_summary['focused_pairs']:,} ({focused_summary['focused_pct']:.2f}%)")
    w(f"- Same-fiber outputs: {focused_summary['same_fiber_pairs']:,} ({focused_summary['same_fiber_pct']:.2f}%)")
    w("")

    w("## Task 2: 64-Orbit Sub-Algebra Table")
    w("")
    w(f"- Compatible orbit pairs: {subalg_summary['compatible_pairs']} / 4096")
    w(f"- Deterministic compatible pairs: {subalg_summary['deterministic_pairs']} ({subalg_summary['deterministic_pct']:.2f}%)")
    w(f"- Mixed compatible pairs: {subalg_summary['mixed_pairs']} ({subalg_summary['mixed_pct']:.2f}%)")
    w(f"- Output coverage: {subalg_summary['reachable_outputs']} / 64 orbits ({subalg_summary['reachable_outputs_pct']:.2f}%)")
    w("")

    w("## Task 4: Generator Analysis")
    w("")
    w(f"Orbit 0 identity-compatible rows behaving exactly as identity: {subalg_summary['identity_exact']} / {subalg_summary['identity_compatible']}")
    if generator_rows:
        terminal = generator_rows[-1]
        w(f"Smallest nontrivial orbit singleton-closure seed: {subalg_summary['singleton_seed']}")
        w(f"Singleton closure terminal status: {terminal['status']} at size {terminal['set_size']}")
    if greedy_rows:
        final = greedy_rows[-1]
        w(f"Greedy generating set size: {len(ast.literal_eval(final['generator_set']))}")
        w(f"Greedy generating set: {final['generator_set']}")
    w("")
    w("[INTERPRETATION]")
    w("")
    w("The same-fiber layer is more rigid than the full 28-orbit live core, but the decisive")
    w("object is the 64-orbit sub-algebra: every later restricted motif lives inside it. The")
    w("focused subset isolates the most constrained same-fiber configurations, while the full")
    w("64x64 table and greedy closure expose whether the small algebra is monogenic or requires")
    w("several independent orbit types to generate.")

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print("=== Step 46: Same-Fiber Core + 64-Orbit Sub-Algebra Structure ===")
    print()

    print("Loading verified step 45 exports...")
    orbit_meta = build_orbit_meta()
    core_rows = load_step45_core_rows()
    pair_table = load_step45_pair_table()
    closed64_ids = load_64_closed_ids(pair_table)
    doubly_live_core_ids = sorted(row['orbit_id'] for row in core_rows)
    same_fiber_ids = sorted(row['orbit_id'] for row in core_rows if row['same_fiber'])
    focused_ids = sorted(row['orbit_id'] for row in core_rows if row['focused_on_boundary'])
    if set(same_fiber_ids) != SAME_FIBER_EXPECTED:
        raise AssertionError(f"Same-fiber ids mismatch: {same_fiber_ids}")
    print(f"  64-closed ids={len(closed64_ids)}  same-fiber ids={len(same_fiber_ids)}  focused ids={len(focused_ids)}")
    print()

    print("Task 1: same-fiber composition table...")
    same_fiber_set = set(same_fiber_ids)
    core_set = set(doubly_live_core_ids)
    closed64_set = set(closed64_ids)
    same_fiber_output_rows: list[dict] = []
    same_fiber_pair_rows: list[dict] = []
    total_pairs = 0
    same_fiber_pairs = 0
    diff_fiber_doubly_live_pairs = 0
    outside_core_pairs = 0

    for a_orbit in same_fiber_ids:
        for b_orbit in same_fiber_ids:
            entry = pair_table[(a_orbit, b_orbit)]
            total = entry['total_pairs']
            outputs = entry['output_counts']
            pair_same = sum(count for orbit_id, count in outputs.items() if orbit_id in same_fiber_set)
            pair_diff_live = sum(count for orbit_id, count in outputs.items() if orbit_id in core_set and orbit_id not in same_fiber_set)
            pair_out_core = sum(count for orbit_id, count in outputs.items() if orbit_id not in core_set)
            same_fiber_pair_rows.append({
                'input_A': a_orbit,
                'input_B': b_orbit,
                'compatible': total > 0,
                'total_pairs': total,
                'output_orbits': str(sorted(entry['output_orbit_ids'])),
                'same_fiber_pairs': pair_same,
                'doubly_live_different_fiber_pairs': pair_diff_live,
                'outside_core_pairs': pair_out_core,
            })
            total_pairs += total
            same_fiber_pairs += pair_same
            diff_fiber_doubly_live_pairs += pair_diff_live
            outside_core_pairs += pair_out_core
            for out_orbit, pair_count in sorted(outputs.items()):
                same_fiber_output_rows.append({
                    'input_A': a_orbit,
                    'input_B': b_orbit,
                    'output_orbit': out_orbit,
                    'pair_count': pair_count,
                    'same_fiber': out_orbit in same_fiber_set,
                    'in_doubly_live_core': out_orbit in core_set,
                    'both_live_output': orbit_meta[out_orbit]['both_live'],
                    'in_64_closed': out_orbit in closed64_set,
                    'output_stab_type': orbit_meta[out_orbit]['stabilizer_type'],
                })

    same_fiber_summary = {
        'total_pairs': total_pairs,
        'same_fiber_pairs': same_fiber_pairs,
        'same_fiber_pct': 100.0 * same_fiber_pairs / total_pairs if total_pairs else 0.0,
        'diff_fiber_doubly_live_pairs': diff_fiber_doubly_live_pairs,
        'diff_fiber_doubly_live_pct': 100.0 * diff_fiber_doubly_live_pairs / total_pairs if total_pairs else 0.0,
        'outside_core_pairs': outside_core_pairs,
        'outside_core_pct': 100.0 * outside_core_pairs / total_pairs if total_pairs else 0.0,
    }
    same_fiber_closure_rows = closure_from_pair_table(same_fiber_set, pair_table)
    print(f"  witness pairs={total_pairs:,}  same-fiber outputs={same_fiber_summary['same_fiber_pct']:.2f}%")
    print()

    print("Task 3: focused-orbit analysis...")
    focused_rows: list[dict] = []
    focused_comp_rows: list[dict] = []
    focus_total_pairs = 0
    focus_focused_pairs = 0
    focus_same_fiber_pairs = 0
    focus_core_pairs = 0
    focused_set = set(focused_ids)
    for orbit_id in focused_ids:
        meta = orbit_meta[orbit_id]
        shared_target = meta['target_x1_atom']
        if shared_target == meta['c1'] == meta['c2']:
            focus_boundary = 'both'
        elif shared_target == meta['c1']:
            focus_boundary = 'c1'
        elif shared_target == meta['c2']:
            focus_boundary = 'c2'
        else:
            focus_boundary = 'none'
        focused_rows.append({
            'orbit_id': orbit_id,
            'rep_config_id': meta['rep_config_id'],
            'orbit_size': meta['orbit_size'],
            'stabilizer_type': meta['stabilizer_type'],
            'signature_key': meta['signature_key'],
            'rep_readable': meta['rep_readable'],
            'focus_boundary': focus_boundary,
            'target_c_atom': c_name(shared_target),
            'x1_fiber_position': meta['x1_fiber_position'],
            'x2_fiber_position': meta['x2_fiber_position'],
            'x1_eq_x2': meta['x1_eq_x2'],
            'c1_eq_c2': meta['c1_eq_c2'],
        })

    for a_orbit in focused_ids:
        for b_orbit in focused_ids:
            entry = pair_table[(a_orbit, b_orbit)]
            outputs = entry['output_counts']
            for out_orbit, pair_count in sorted(outputs.items()):
                is_focused = out_orbit in focused_set
                is_same_fiber = out_orbit in same_fiber_set
                in_core = out_orbit in core_set
                focused_comp_rows.append({
                    'input_A': a_orbit,
                    'input_B': b_orbit,
                    'output_orbit': out_orbit,
                    'pair_count': pair_count,
                    'focused_output': is_focused,
                    'same_fiber_output': is_same_fiber,
                    'in_doubly_live_core': in_core,
                    'output_stab_type': orbit_meta[out_orbit]['stabilizer_type'],
                })
                focus_total_pairs += pair_count
                focus_focused_pairs += pair_count if is_focused else 0
                focus_same_fiber_pairs += pair_count if is_same_fiber else 0
                focus_core_pairs += pair_count if in_core else 0

    focused_summary = {
        'total_pairs': focus_total_pairs,
        'focused_pairs': focus_focused_pairs,
        'focused_pct': 100.0 * focus_focused_pairs / focus_total_pairs if focus_total_pairs else 0.0,
        'same_fiber_pairs': focus_same_fiber_pairs,
        'same_fiber_pct': 100.0 * focus_same_fiber_pairs / focus_total_pairs if focus_total_pairs else 0.0,
        'core_pairs': focus_core_pairs,
        'core_pct': 100.0 * focus_core_pairs / focus_total_pairs if focus_total_pairs else 0.0,
    }
    print(f"  focused witness pairs={focus_total_pairs:,}  focused outputs={focused_summary['focused_pct']:.2f}%")
    print()

    print("Task 2: full 64-orbit sub-algebra table...")
    subalg_rows: list[dict] = []
    compatible_pairs = 0
    deterministic_pairs = 0
    mixed_pairs = 0
    output_orbit_cover: set[int] = set()
    cayley_rows: list[dict] = []
    for a_orbit in closed64_ids:
        matrix_row = {'orbit_A': a_orbit}
        for b_orbit in closed64_ids:
            entry = pair_table[(a_orbit, b_orbit)]
            output_ids = sorted(entry['output_orbit_ids'])
            compatible = entry['total_pairs'] > 0
            if compatible:
                compatible_pairs += 1
                output_orbit_cover.update(output_ids)
                if len(output_ids) == 1:
                    deterministic_pairs += 1
                else:
                    mixed_pairs += 1
            output_stab_types = sorted({orbit_meta[orbit_id]['stabilizer_type'] for orbit_id in output_ids})
            subalg_rows.append({
                'orbit_A': a_orbit,
                'orbit_B': b_orbit,
                'compatible': compatible,
                'total_pairs': entry['total_pairs'],
                'output_orbits': str(output_ids),
                'deterministic': len(output_ids) == 1,
                'stab_type_A': orbit_meta[a_orbit]['stabilizer_type'],
                'stab_type_B': orbit_meta[b_orbit]['stabilizer_type'],
                'stab_type_out': str(output_stab_types),
                'output_counts': str(entry['output_counts']),
            })
            matrix_row[f'B_{b_orbit}'] = str(output_ids) if compatible else ''
        cayley_rows.append(matrix_row)
    subalg_summary = {
        'compatible_pairs': compatible_pairs,
        'deterministic_pairs': deterministic_pairs,
        'mixed_pairs': mixed_pairs,
        'deterministic_pct': 100.0 * deterministic_pairs / compatible_pairs if compatible_pairs else 0.0,
        'mixed_pct': 100.0 * mixed_pairs / compatible_pairs if compatible_pairs else 0.0,
        'reachable_outputs': len(output_orbit_cover),
        'reachable_outputs_pct': 100.0 * len(output_orbit_cover) / len(closed64_ids),
    }
    print(f"  compatible orbit pairs={compatible_pairs}  deterministic={subalg_summary['deterministic_pct']:.2f}%")
    print()

    print("Task 4: generator analysis...")
    identity_rows: list[dict] = []
    identity_compatible = 0
    identity_exact = 0
    for orbit_id in closed64_ids:
        left_entry = pair_table[(0, orbit_id)]
        right_entry = pair_table[(orbit_id, 0)]
        left_outputs = sorted(left_entry['output_orbit_ids'])
        right_outputs = sorted(right_entry['output_orbit_ids'])
        left_compatible = left_entry['total_pairs'] > 0
        right_compatible = right_entry['total_pairs'] > 0
        if left_compatible:
            identity_compatible += 1
            if left_outputs == [orbit_id]:
                identity_exact += 1
        if right_compatible:
            identity_compatible += 1
            if right_outputs == [orbit_id]:
                identity_exact += 1
        identity_rows.append({
            'orbit_id': orbit_id,
            'left_compatible': left_compatible,
            'left_outputs': str(left_outputs),
            'left_is_identity': left_outputs == [orbit_id],
            'right_compatible': right_compatible,
            'right_outputs': str(right_outputs),
            'right_is_identity': right_outputs == [orbit_id],
        })

    singleton_seed = min(orbit_id for orbit_id in closed64_ids if orbit_id != 0)
    generator_rows = singleton_generator_closure(singleton_seed, pair_table)
    greedy_rows = greedy_generating_set(closed64_ids, pair_table)
    subalg_summary['identity_compatible'] = identity_compatible
    subalg_summary['identity_exact'] = identity_exact
    subalg_summary['singleton_seed'] = singleton_seed
    print(f"  singleton seed={singleton_seed}  greedy generators={len(ast.literal_eval(greedy_rows[-1]['generator_set']))}")
    print()

    print("Writing outputs...")
    write_csv(
        EXPORTS / 'step46_same_fiber_outputs.csv',
        same_fiber_output_rows,
        ['input_A', 'input_B', 'output_orbit', 'pair_count', 'same_fiber', 'in_doubly_live_core', 'both_live_output', 'in_64_closed', 'output_stab_type'],
    )
    write_csv(
        EXPORTS / 'step46_same_fiber_pair_summary.csv',
        same_fiber_pair_rows,
        ['input_A', 'input_B', 'compatible', 'total_pairs', 'output_orbits', 'same_fiber_pairs', 'doubly_live_different_fiber_pairs', 'outside_core_pairs'],
    )
    write_csv(
        EXPORTS / 'step46_same_fiber_closure.csv',
        same_fiber_closure_rows,
        ['closure_step', 'set_size', 'compatible_orbit_pairs', 'witness_pairs', 'new_orbits', 'new_orbit_ids'],
    )
    write_csv(
        EXPORTS / 'step46_64_subalgebra_table.csv',
        subalg_rows,
        ['orbit_A', 'orbit_B', 'compatible', 'total_pairs', 'output_orbits', 'deterministic', 'stab_type_A', 'stab_type_B', 'stab_type_out', 'output_counts'],
    )
    write_csv(
        EXPORTS / 'step46_64_subalgebra_cayley.csv',
        cayley_rows,
        ['orbit_A'] + [f'B_{orbit_id}' for orbit_id in closed64_ids],
    )
    write_csv(
        EXPORTS / 'step46_64_subalgebra_summary.csv',
        [subalg_summary],
        ['compatible_pairs', 'deterministic_pairs', 'mixed_pairs', 'deterministic_pct', 'mixed_pct', 'reachable_outputs', 'reachable_outputs_pct', 'identity_compatible', 'identity_exact', 'singleton_seed'],
    )
    write_csv(
        EXPORTS / 'step46_focused_orbits.csv',
        focused_rows,
        ['orbit_id', 'rep_config_id', 'orbit_size', 'stabilizer_type', 'signature_key', 'rep_readable', 'focus_boundary', 'target_c_atom', 'x1_fiber_position', 'x2_fiber_position', 'x1_eq_x2', 'c1_eq_c2'],
    )
    write_csv(
        EXPORTS / 'step46_focused_composition.csv',
        focused_comp_rows,
        ['input_A', 'input_B', 'output_orbit', 'pair_count', 'focused_output', 'same_fiber_output', 'in_doubly_live_core', 'output_stab_type'],
    )
    write_csv(
        EXPORTS / 'step46_focus_profile_summary.csv',
        [focused_summary],
        ['total_pairs', 'focused_pairs', 'focused_pct', 'same_fiber_pairs', 'same_fiber_pct', 'core_pairs', 'core_pct'],
    )
    write_csv(
        EXPORTS / 'step46_identity_map.csv',
        identity_rows,
        ['orbit_id', 'left_compatible', 'left_outputs', 'left_is_identity', 'right_compatible', 'right_outputs', 'right_is_identity'],
    )
    write_csv(
        EXPORTS / 'step46_single_generator_closure.csv',
        generator_rows,
        ['step', 'set_size', 'orbit_ids', 'status'],
    )
    write_csv(
        EXPORTS / 'step46_greedy_generators.csv',
        greedy_rows,
        ['step', 'chosen_orbit', 'generator_set', 'closure_size', 'new_orbits_added', 'closure_orbits'],
    )
    write_markdown_summary(
        EXPORTS / 'step46_same_fiber_subalgebra.md',
        same_fiber_summary,
        same_fiber_closure_rows,
        focused_rows,
        focused_summary,
        subalg_summary,
        greedy_rows,
        generator_rows,
    )
    print()

    print("=== SUMMARY ===")
    print(f"same-fiber ids: {len(same_fiber_ids)}")
    print(f"focused ids: {len(focused_ids)}")
    print(f"64-subalgebra compatible orbit pairs: {compatible_pairs}")


if __name__ == '__main__':
    main()