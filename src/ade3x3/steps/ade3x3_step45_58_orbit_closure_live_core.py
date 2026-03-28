"""
ade3x3_step45_58_orbit_closure_live_core.py

Step 45: 58-Orbit Closure + Doubly-Live Floor Core.

This step extends the step 44 floor-layer computation rather than rebuilding
the algebraic object from scratch.

Task 1
  Start from the 58-orbit candidate set = 40 floor orbits + 18 step-1 escapes.
  Iterate closure under the same CXC-interface composition used in steps 42/44.

Task 2
  Analyze the 28 doubly-live floor orbits (both X atoms live).
  Record target-map and C-fiber data, then measure composition restricted to
  this live core.

Task 3
  For each floor Z2xZ2 stabilizer subgroup, compute the fixed-point dimension
  on the 81-dimensional X-atom permutation representation. Because a Z2xZ2
  action splits into four sign characters rather than a single global "-1"
  eigenspace, both the fixed dimension and the full character decomposition are
  reported. The requested "-1" dimension is recorded as the total non-fixed
  dimension = 81 - fixed_dim.

Outputs:
  - outputs/exports/step45_closure_steps.csv
  - outputs/exports/step45_closure_pair_table.csv
  - outputs/exports/step45_doubly_live_core.csv
  - outputs/exports/step45_doubly_live_composition.csv
  - outputs/exports/step45_doubly_live_output_summary.csv
  - outputs/exports/step45_fixed_point_subspaces.csv
  - outputs/exports/step45_58_orbit_live_core.md

Provenance:
  [EXACT_DERIVED] for enumerated facts.
  [INTERPRETATION] for structural claims in the markdown summary only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

EXPORTS = Path('outputs/exports')

S3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
ACTIONS = [(p1, p2, p3) for p1 in S3 for p2 in S3 for p3 in S3]


def _build_tables() -> tuple[list[list[int]], list[list[int]]]:
    cmaps: list[list[int]] = []
    xmaps: list[list[int]] = []
    for p1, p2, p3 in ACTIONS:
        cmaps.append([3 * p1[c // 3] + p3[c % 3] for c in range(9)])
        xmaps.append([
            9 * (3 * p1[(x // 9) // 3] + p2[(x // 9) % 3])
            + (3 * p2[(x % 9) // 3] + p3[(x % 9) % 3])
            for x in range(81)
        ])
    return cmaps, xmaps


CMAPS, XMAPS = _build_tables()
XMAP_TO_ACTION_ID = {tuple(x_map): idx for idx, x_map in enumerate(XMAPS)}


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


def _cxxc_min_rep_chunk(start: int, stop: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for cfg in range(start, stop):
        c2 = cfg % 9
        rem = cfg // 9
        x2 = rem % 81
        rem //= 81
        x1 = rem % 81
        c1 = rem // 81
        m = cfg
        for c_map, x_map in zip(CMAPS, XMAPS):
            v = ((c_map[c1] * 81 + x_map[x1]) * 81 + x_map[x2]) * 9 + c_map[c2]
            if v < m:
                m = v
        result[cfg] = m
    return result


def _build_orbit_map_parallel(n: int, chunk_size: int, workers: int) -> list[int]:
    print(f"  Building CXXC orbit map ({n:,} configs, {workers} workers)...")
    chunks = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
    min_rep = [0] * n
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_cxxc_min_rep_chunk, start, stop): (start, stop) for start, stop in chunks}
        for future in as_completed(futures):
            partial = future.result()
            for cfg, rep in partial.items():
                min_rep[cfg] = rep
            done += len(partial)
            print(f"    {100 * done / n:5.1f}%  ({done:,}/{n:,})", end='\r')
    print()
    reps = sorted(set(min_rep))
    rep_to_oid = {rep: idx for idx, rep in enumerate(reps)}
    return [rep_to_oid[min_rep[cfg]] for cfg in range(n)]


def _scan_restricted_c1m_chunk(args: tuple[list[int], list[int], frozenset[int]]) -> dict[tuple[int, int, int], int]:
    c1_values, cxxc_oid, allowed_ids = args
    counter: Counter = Counter()

    for c1_m in c1_values:
        for x_m in range(81):
            left_base = c1_m * 81
            for c2_m in range(9):
                a_orbs = [
                    cxxc_oid[((left_base + x1) * 81 + x_m) * 9 + c2_m]
                    for x1 in range(81)
                ]
                allowed_a = [x1 for x1, orbit_id in enumerate(a_orbs) if orbit_id in allowed_ids]
                if not allowed_a:
                    continue

                b_orbs = [
                    cxxc_oid[((left_base + x_m) * 81 + x4) * 9 + c2_m]
                    for x4 in range(81)
                ]
                allowed_b = [x4 for x4, orbit_id in enumerate(b_orbs) if orbit_id in allowed_ids]
                if not allowed_b:
                    continue

                for x1 in allowed_a:
                    a_orb = a_orbs[x1]
                    out_prefix = (left_base + x1) * 81
                    for x4 in allowed_b:
                        b_orb = b_orbs[x4]
                        out_orb = cxxc_oid[(out_prefix + x4) * 9 + c2_m]
                        counter[(a_orb, b_orb, out_orb)] += 1

    return dict(counter)


def scan_restricted_parallel(
    cxxc_oid: list[int],
    allowed_ids: frozenset[int],
    workers: int,
    label: str,
) -> Counter:
    print(f"  Scanning {label} ({len(allowed_ids)} orbit ids, {workers} workers)...")
    c1_values = list(range(9))
    n_chunks = min(workers, 9)
    chunk_size = max(1, -(-9 // n_chunks))
    chunks = [c1_values[idx:idx + chunk_size] for idx in range(0, 9, chunk_size)]

    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_scan_restricted_c1m_chunk, (chunk, cxxc_oid, allowed_ids))
            for chunk in chunks
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            for key, value in future.result().items():
                total[key] += value
            print(f"    chunk {idx}/{len(chunks)} done", end='\r')
    print(f"    {len(chunks)}/{len(chunks)} chunks done.   ")
    return total


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def load_stabilizer_rows() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv_rows(EXPORTS / 'stabilizer_classification.csv'):
        if row['schema'] != 'CXXC':
            continue
        rows.append({
            'orbit_id': int(row['orbit_id']),
            'rep_config_id': int(row['rep_config_id']),
            'orbit_size': int(row['orbit_size']),
            'stabilizer_order': int(row['stabilizer_order']),
            'stabilizer_type': row['stabilizer_type'],
            'generator_action_ids': ast.literal_eval(row['generator_action_ids']),
        })
    return rows


def load_signature_map() -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(EXPORTS / 'signatures_CXXC_refined.csv'):
        orbit_id = int(row['orbit_id'])
        result[orbit_id] = {
            'signature_key': row['signature_key'],
            'rep_readable': row['rep_readable'],
            'refinement_features': row['refinement_features'],
        }
    return result


def load_floor_inventory() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv_rows(EXPORTS / 'floor_layer_inventory.csv'):
        rows.append({
            'orbit_id': int(row['orbit_id']),
            'rep_config_id': int(row['rep_config_id']),
            'orbit_size': int(row['orbit_size']),
            'stabilizer_order': int(row['stabilizer_order']),
            'stabilizer_type': row['stabilizer_type'],
            'both_live': row['both_live'] == 'True',
            'c1_eq_c2': row['c1_eq_c2'] == 'True',
            'x1_eq_x2': row['x1_eq_x2'] == 'True',
        })
    return rows


def load_step44_baseline(floor_ids: set[int]) -> dict[str, object]:
    rows = read_csv_rows(EXPORTS / 'floor_layer_composition_table.csv')
    pair_total = sum(int(row['total_pairs']) for row in rows)
    inside = sum(int(row['output_in_floor']) for row in rows)
    outside = sum(int(row['output_outside_floor']) for row in rows)
    all_outputs: set[int] = set()
    for row in rows:
        all_outputs.update(ast.literal_eval(row['output_orbit_ids']))
    escaped = sorted(all_outputs - floor_ids)
    return {
        'pair_total': pair_total,
        'inside': inside,
        'outside': outside,
        'escaped': escaped,
    }


def c_name(c_atom: int) -> str:
    return f"C[{c_atom // 3},{c_atom % 3}]"


def build_orbit_metadata(stabilizer_rows: list[dict], signature_map: dict[int, dict[str, str]]) -> dict[int, dict]:
    orbit_meta: dict[int, dict] = {}
    for row in stabilizer_rows:
        orbit_id = row['orbit_id']
        c1, x1, x2, c2 = _decode_cxxc(row['rep_config_id'])
        x1_r, x1_s, x1_t, x1_u = _decode_x(x1)
        x2_r, x2_s, x2_t, x2_u = _decode_x(x2)
        live_x1 = x1_s == x1_t
        live_x2 = x2_s == x2_t
        live_count = int(live_x1) + int(live_x2)
        sig = signature_map.get(orbit_id, {})
        orbit_meta[orbit_id] = {
            'orbit_id': orbit_id,
            'rep_config_id': row['rep_config_id'],
            'orbit_size': row['orbit_size'],
            'stabilizer_order': row['stabilizer_order'],
            'stabilizer_type': row['stabilizer_type'],
            'generator_action_ids': row['generator_action_ids'],
            'signature_key': sig.get('signature_key', ''),
            'rep_readable': sig.get('rep_readable', ''),
            'refinement_features': sig.get('refinement_features', ''),
            'c1': c1,
            'c2': c2,
            'x1': x1,
            'x2': x2,
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
            'live_class': 'doubly_live' if live_count == 2 else ('singly_live' if live_count == 1 else 'dead'),
            'c1_eq_c2': c1 == c2,
            'x1_eq_x2': x1 == x2,
            'target_of_x1': c_name(3 * x1_r + x1_u) if live_x1 else '',
            'target_of_x2': c_name(3 * x2_r + x2_u) if live_x2 else '',
            'x1_target_atom': 3 * x1_r + x1_u if live_x1 else -1,
            'x2_target_atom': 3 * x2_r + x2_u if live_x2 else -1,
            'x1_fiber_position': x1_s if live_x1 else -1,
            'x2_fiber_position': x2_s if live_x2 else -1,
        }
    return orbit_meta


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def summarize_pair_counter(
    pair_counter: Counter,
    current_ids: list[int],
    current_set: set[int],
    orbit_meta: dict[int, dict],
    closure_step: int,
) -> list[dict]:
    ab_table: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for (a_orbit, b_orbit, out_orbit), count in pair_counter.items():
        ab_table[(a_orbit, b_orbit)][out_orbit] += count

    rows: list[dict] = []
    for a_orbit in current_ids:
        for b_orbit in current_ids:
            outputs = ab_table.get((a_orbit, b_orbit), Counter())
            total_pairs = sum(outputs.values())
            output_inside = sum(count for orbit_id, count in outputs.items() if orbit_id in current_set)
            row = {
                'closure_step': closure_step,
                'A_orbit': a_orbit,
                'B_orbit': b_orbit,
                'A_stab_type': orbit_meta[a_orbit]['stabilizer_type'],
                'B_stab_type': orbit_meta[b_orbit]['stabilizer_type'],
                'total_pairs': total_pairs,
                'output_inside_set': output_inside,
                'output_outside_set': total_pairs - output_inside,
                'output_orbit_ids': str(sorted(outputs.keys())),
                'output_counts': str(dict(sorted(outputs.items()))),
            }
            rows.append(row)
    return rows


def run_closure_iteration(
    cxxc_oid: list[int],
    seed_ids: list[int],
    orbit_meta: dict[int, dict],
    workers: int,
    max_size: int,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    closure_rows: list[dict] = []
    pair_rows: list[dict] = []
    current_ids = sorted(seed_ids)
    stop_reason = 'closed'

    for closure_step in range(1, 10):
        current_set = set(current_ids)
        pair_counter = scan_restricted_parallel(
            cxxc_oid,
            frozenset(current_ids),
            workers,
            f'closure step {closure_step}',
        )
        pairs_checked = sum(pair_counter.values())
        pairs_inside = sum(count for (_, _, out_orbit), count in pair_counter.items() if out_orbit in current_set)
        pairs_outside = pairs_checked - pairs_inside
        escaped = sorted({out_orbit for (_, _, out_orbit) in pair_counter if out_orbit not in current_set})

        closure_rows.append({
            'closure_step': closure_step,
            'set_size': len(current_ids),
            'pairs_checked': pairs_checked,
            'pairs_inside': pairs_inside,
            'pairs_outside': pairs_outside,
            'new_escapes': len(escaped),
            'new_escape_ids': str(escaped),
        })
        pair_rows.extend(summarize_pair_counter(pair_counter, current_ids, current_set, orbit_meta, closure_step))

        if not escaped:
            stop_reason = 'closed'
            break

        next_ids = sorted(current_set | set(escaped))
        if len(next_ids) > max_size:
            stop_reason = f'blow_up>{max_size}'
            break
        current_ids = next_ids
    else:
        stop_reason = 'iteration_cap'

    return closure_rows, pair_rows, {'final_ids': current_ids, 'stop_reason': stop_reason}


def classify_doubly_live_output(orbit_id: int, core_ids: set[int], orbit_meta: dict[int, dict]) -> str:
    meta = orbit_meta[orbit_id]
    if orbit_id in core_ids:
        return 'doubly_live_floor'
    if meta['both_live']:
        return 'doubly_live_nonfloor'
    if meta['live_count'] == 1:
        return 'singly_live'
    return 'dead'


def summarize_doubly_live_composition(
    pair_counter: Counter,
    core_ids: list[int],
    orbit_meta: dict[int, dict],
) -> tuple[list[dict], list[dict], dict[str, int]]:
    core_set = set(core_ids)
    output_class_counter: Counter = Counter()
    ab_table: dict[tuple[int, int], Counter] = defaultdict(Counter)

    for (a_orbit, b_orbit, out_orbit), count in pair_counter.items():
        output_class_counter[classify_doubly_live_output(out_orbit, core_set, orbit_meta)] += count
        ab_table[(a_orbit, b_orbit)][out_orbit] += count

    pair_rows: list[dict] = []
    for a_orbit in core_ids:
        for b_orbit in core_ids:
            outputs = ab_table.get((a_orbit, b_orbit), Counter())
            total_pairs = sum(outputs.values())
            doubly_live_outputs = sum(
                count for orbit_id, count in outputs.items() if orbit_meta[orbit_id]['both_live']
            )
            pair_rows.append({
                'A_orbit': a_orbit,
                'B_orbit': b_orbit,
                'A_stab_type': orbit_meta[a_orbit]['stabilizer_type'],
                'B_stab_type': orbit_meta[b_orbit]['stabilizer_type'],
                'total_pairs': total_pairs,
                'output_both_live': doubly_live_outputs,
                'output_not_both_live': total_pairs - doubly_live_outputs,
                'output_orbit_ids': str(sorted(outputs.keys())),
                'output_counts': str(dict(sorted(outputs.items()))),
            })

    summary_rows: list[dict] = []
    total_outputs = sum(output_class_counter.values())
    for output_class in ['doubly_live_floor', 'doubly_live_nonfloor', 'singly_live', 'dead']:
        count = output_class_counter.get(output_class, 0)
        summary_rows.append({
            'output_class': output_class,
            'pair_count': count,
            'fraction': f"{(count / total_outputs) if total_outputs else 0.0:.6f}",
        })

    return pair_rows, summary_rows, dict(output_class_counter)


def enrich_doubly_live_core_rows(core_ids: list[int], orbit_meta: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for orbit_id in core_ids:
        meta = orbit_meta[orbit_id]
        target_x1_equals_target_x2 = meta['x1_target_atom'] == meta['x2_target_atom']
        c1_is_target_of_x1 = meta['x1_target_atom'] == meta['c1']
        c1_is_target_of_x2 = meta['x2_target_atom'] == meta['c1']
        c2_is_target_of_x1 = meta['x1_target_atom'] == meta['c2']
        c2_is_target_of_x2 = meta['x2_target_atom'] == meta['c2']
        focused_on_boundary = target_x1_equals_target_x2 and (c1_is_target_of_x1 or c2_is_target_of_x1)
        rows.append({
            'orbit_id': orbit_id,
            'rep_config_id': meta['rep_config_id'],
            'orbit_size': meta['orbit_size'],
            'stabilizer_type': meta['stabilizer_type'],
            'signature_key': meta['signature_key'],
            'c1_eq_c2': meta['c1_eq_c2'],
            'x1_eq_x2': meta['x1_eq_x2'],
            'rep_readable': meta['rep_readable'],
            'target_of_x1': meta['target_of_x1'],
            'target_of_x2': meta['target_of_x2'],
            'c1_is_target_of_x1': c1_is_target_of_x1,
            'c1_is_target_of_x2': c1_is_target_of_x2,
            'c2_is_target_of_x1': c2_is_target_of_x1,
            'c2_is_target_of_x2': c2_is_target_of_x2,
            'target_x1_equals_target_x2': target_x1_equals_target_x2,
            'x1_fiber_position': meta['x1_fiber_position'],
            'x2_fiber_position': meta['x2_fiber_position'],
            'focused_on_boundary': focused_on_boundary,
            'same_fiber': target_x1_equals_target_x2,
        })
    return rows


def compose_xmaps(left: list[int], right: list[int]) -> list[int]:
    return [left[right[idx]] for idx in range(81)]


def trace_of_xmap(x_map: list[int]) -> int:
    return sum(1 for idx, value in enumerate(x_map) if idx == value)


def subgroup_fixed_space_rows(floor_ids: list[int], orbit_meta: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for orbit_id in floor_ids:
        meta = orbit_meta[orbit_id]
        if meta['stabilizer_type'] != 'Z2xZ2':
            continue
        gen_ids = meta['generator_action_ids']
        if len(gen_ids) != 2:
            raise ValueError(f"Orbit {orbit_id} expected 2 generators, got {gen_ids}")
        gen_a = XMAPS[gen_ids[0]]
        gen_b = XMAPS[gen_ids[1]]
        gen_ab = compose_xmaps(gen_a, gen_b)
        subgroup_action_ids = sorted({0, gen_ids[0], gen_ids[1], XMAP_TO_ACTION_ID[tuple(gen_ab)]})

        tr_e = 81
        tr_a = trace_of_xmap(gen_a)
        tr_b = trace_of_xmap(gen_b)
        tr_ab = trace_of_xmap(gen_ab)

        chi_pp = (tr_e + tr_a + tr_b + tr_ab) // 4
        chi_pm = (tr_e + tr_a - tr_b - tr_ab) // 4
        chi_mp = (tr_e - tr_a + tr_b - tr_ab) // 4
        chi_mm = (tr_e - tr_a - tr_b + tr_ab) // 4
        fixed_dim = chi_pp

        rows.append({
            'orbit_id': orbit_id,
            'rep_config_id': meta['rep_config_id'],
            'generator_action_ids': str(gen_ids),
            'subgroup_action_ids': str(subgroup_action_ids),
            'trace_id': tr_e,
            'trace_gen_a': tr_a,
            'trace_gen_b': tr_b,
            'trace_gen_ab': tr_ab,
            'fixed_dim': fixed_dim,
            'minus_dim': 81 - fixed_dim,
            'char_pp_dim': chi_pp,
            'char_pm_dim': chi_pm,
            'char_mp_dim': chi_mp,
            'char_mm_dim': chi_mm,
        })
    return rows


def write_markdown_summary(
    path: Path,
    closure_rows: list[dict],
    closure_meta: dict[str, object],
    core_rows: list[dict],
    core_output_summary: list[dict],
    fixed_rows: list[dict],
) -> None:
    lines: list[str] = []
    append = lines.append

    append("# Step 45: 58-Orbit Closure + Doubly-Live Floor Core")
    append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    append("")
    append("[EXACT_DERIVED]")
    append("")
    append("## Task 1: Closure Iteration Starting from the 58-Orbit Candidate")
    append("")
    append("| closure_step | set_size | pairs_checked | pairs_inside | pairs_outside | new_escapes |")
    append("|--------------|----------|---------------|--------------|---------------|-------------|")
    for row in closure_rows:
        append(
            f"| {row['closure_step']} | {row['set_size']} | {row['pairs_checked']:,} | "
            f"{row['pairs_inside']:,} | {row['pairs_outside']:,} | {row['new_escapes']} |"
        )
    append("")
    for row in closure_rows:
        if row['new_escapes']:
            append(f"Step {row['closure_step']} new escape ids: {row['new_escape_ids']}")
    append(f"Stop reason: {closure_meta['stop_reason']}")
    append(f"Final checked set size: {len(closure_meta['final_ids'])}")
    append("")

    append("## Task 2: Doubly-Live Floor Core")
    append("")
    append(f"Doubly-live floor orbits: {len(core_rows)}")
    same_fiber = sum(1 for row in core_rows if row['same_fiber'])
    focused = sum(1 for row in core_rows if row['focused_on_boundary'])
    append(f"- Same target fiber: {same_fiber} / {len(core_rows)}")
    append(f"- Focused on a boundary C atom: {focused} / {len(core_rows)}")
    append("")
    append("| orbit_id | stab_type | target_of_x1 | target_of_x2 | same_fiber | focused_on_boundary |")
    append("|----------|-----------|--------------|--------------|------------|---------------------|")
    for row in core_rows:
        append(
            f"| {row['orbit_id']} | {row['stabilizer_type']} | {row['target_of_x1']} | "
            f"{row['target_of_x2']} | {row['same_fiber']} | {row['focused_on_boundary']} |"
        )
    append("")

    append("## Task 2c: Doubly-Live Core Composition")
    append("")
    append("| output_class | pair_count | fraction |")
    append("|--------------|------------|----------|")
    for row in core_output_summary:
        append(f"| {row['output_class']} | {int(row['pair_count']):,} | {float(row['fraction']):.6f} |")
    append("")

    append("## Task 3: Fixed-Point Subspace Dimensions on X")
    append("")
    append(f"Computed rows: {len(fixed_rows)} Z2xZ2 floor stabilizers")
    if fixed_rows:
        fixed_dims = sorted({int(row['fixed_dim']) for row in fixed_rows})
        minus_dims = sorted({int(row['minus_dim']) for row in fixed_rows})
        append(f"- Fixed dimensions observed: {fixed_dims}")
        append(f"- Non-fixed dimensions observed: {minus_dims}")
    append("")
    append("[INTERPRETATION]")
    append("")
    append("The live-core rows separate two geometric motifs: same-fiber pairs, where both")
    append("bilinear terms feed the same output C atom, and spread pairs, where the two live")
    append("terms feed different C atoms. The closure table shows whether the 58-orbit layer")
    append("stabilizes as a small sub-algebra or continues expanding; the fixed-space data shows")
    append("how much of the 81-dimensional X space remains rigid under each floor stabilizer.")

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Parallel workers for orbit-map and restricted-composition scans',
    )
    parser.add_argument(
        '--max-closure-size',
        type=int,
        default=200,
        help='Stop closure iteration once the next closure step would exceed this size',
    )
    args = parser.parse_args()
    workers = max(1, args.workers)

    print(f"=== Step 45: 58-Orbit Closure + Doubly-Live Floor Core (workers={workers}) ===")
    print()

    print("Loading exported orbit metadata...")
    stabilizer_rows = load_stabilizer_rows()
    signature_map = load_signature_map()
    orbit_meta = build_orbit_metadata(stabilizer_rows, signature_map)
    floor_inventory = load_floor_inventory()
    floor_ids = sorted(row['orbit_id'] for row in floor_inventory)
    doubly_live_core_ids = sorted(row['orbit_id'] for row in floor_inventory if row['both_live'])
    baseline = load_step44_baseline(set(floor_ids))
    assert len(floor_ids) == 40, f"Expected 40 floor ids, got {len(floor_ids)}"
    assert len(doubly_live_core_ids) == 28, f"Expected 28 doubly-live floor ids, got {len(doubly_live_core_ids)}"
    assert len(baseline['escaped']) == 18, f"Expected 18 step-1 escapes, got {len(baseline['escaped'])}"
    print(f"  floor ids={len(floor_ids)}  doubly-live floor ids={len(doubly_live_core_ids)}  escaped={len(baseline['escaped'])}")
    print()

    print("Building CXXC orbit map...")
    cxxc_oid = _build_orbit_map_parallel(531_441, chunk_size=4096, workers=workers)
    assert len(set(cxxc_oid)) == 2744, 'CXXC orbit count mismatch'
    print("  CXXC=2744 verified")
    print()

    print("Validating restricted scanner against the known floor-layer result...")
    floor_pair_counter = scan_restricted_parallel(cxxc_oid, frozenset(floor_ids), workers, 'step 44 floor baseline')
    floor_total = sum(floor_pair_counter.values())
    floor_inside = sum(count for (_, _, out_orbit), count in floor_pair_counter.items() if out_orbit in set(floor_ids))
    floor_outside = floor_total - floor_inside
    floor_escaped = sorted({out_orbit for (_, _, out_orbit) in floor_pair_counter if out_orbit not in set(floor_ids)})
    assert floor_total == baseline['pair_total'], f"floor total mismatch: {floor_total} vs {baseline['pair_total']}"
    assert floor_inside == baseline['inside'], f"floor inside mismatch: {floor_inside} vs {baseline['inside']}"
    assert floor_outside == baseline['outside'], f"floor outside mismatch: {floor_outside} vs {baseline['outside']}"
    assert floor_escaped == baseline['escaped'], f"floor escaped mismatch: {floor_escaped} vs {baseline['escaped']}"
    print(f"  [SELF-TEST PASS] {floor_total:,} pairs, {floor_inside:,} inside, {floor_outside:,} outside, {len(floor_escaped)} escapes")
    print()

    print("Task 1: 58-orbit closure iteration...")
    seed_58 = sorted(set(floor_ids) | set(baseline['escaped']))
    closure_rows, closure_pair_rows, closure_meta = run_closure_iteration(
        cxxc_oid,
        seed_58,
        orbit_meta,
        workers,
        args.max_closure_size,
    )
    print(f"  closure stop reason: {closure_meta['stop_reason']}")
    print(f"  final set size: {len(closure_meta['final_ids'])}")
    print()

    print("Task 2: Doubly-live floor core inventory and composition...")
    core_rows = enrich_doubly_live_core_rows(doubly_live_core_ids, orbit_meta)
    core_pair_counter = scan_restricted_parallel(cxxc_oid, frozenset(doubly_live_core_ids), workers, 'doubly-live core')
    core_pair_rows, core_output_summary, core_class_counter = summarize_doubly_live_composition(
        core_pair_counter,
        doubly_live_core_ids,
        orbit_meta,
    )
    total_core_pairs = sum(core_class_counter.values())
    both_live_outputs = core_class_counter.get('doubly_live_floor', 0) + core_class_counter.get('doubly_live_nonfloor', 0)
    print(f"  core pairs={total_core_pairs:,}  outputs still both-live={both_live_outputs:,}")
    print()

    print("Task 3: Fixed-point subspace dimensions...")
    fixed_rows = subgroup_fixed_space_rows(floor_ids, orbit_meta)
    print(f"  computed {len(fixed_rows)} Z2xZ2 rows")
    print()

    print("Writing outputs...")
    write_csv(
        EXPORTS / 'step45_closure_steps.csv',
        closure_rows,
        ['closure_step', 'set_size', 'pairs_checked', 'pairs_inside', 'pairs_outside', 'new_escapes', 'new_escape_ids'],
    )
    write_csv(
        EXPORTS / 'step45_closure_pair_table.csv',
        closure_pair_rows,
        ['closure_step', 'A_orbit', 'B_orbit', 'A_stab_type', 'B_stab_type', 'total_pairs', 'output_inside_set', 'output_outside_set', 'output_orbit_ids', 'output_counts'],
    )
    write_csv(
        EXPORTS / 'step45_doubly_live_core.csv',
        core_rows,
        ['orbit_id', 'rep_config_id', 'orbit_size', 'stabilizer_type', 'signature_key', 'c1_eq_c2', 'x1_eq_x2', 'rep_readable', 'target_of_x1', 'target_of_x2', 'c1_is_target_of_x1', 'c1_is_target_of_x2', 'c2_is_target_of_x1', 'c2_is_target_of_x2', 'target_x1_equals_target_x2', 'x1_fiber_position', 'x2_fiber_position', 'focused_on_boundary', 'same_fiber'],
    )
    write_csv(
        EXPORTS / 'step45_doubly_live_composition.csv',
        core_pair_rows,
        ['A_orbit', 'B_orbit', 'A_stab_type', 'B_stab_type', 'total_pairs', 'output_both_live', 'output_not_both_live', 'output_orbit_ids', 'output_counts'],
    )
    write_csv(
        EXPORTS / 'step45_doubly_live_output_summary.csv',
        core_output_summary,
        ['output_class', 'pair_count', 'fraction'],
    )
    write_csv(
        EXPORTS / 'step45_fixed_point_subspaces.csv',
        fixed_rows,
        ['orbit_id', 'rep_config_id', 'generator_action_ids', 'subgroup_action_ids', 'trace_id', 'trace_gen_a', 'trace_gen_b', 'trace_gen_ab', 'fixed_dim', 'minus_dim', 'char_pp_dim', 'char_pm_dim', 'char_mp_dim', 'char_mm_dim'],
    )
    write_markdown_summary(
        EXPORTS / 'step45_58_orbit_live_core.md',
        closure_rows,
        closure_meta,
        core_rows,
        core_output_summary,
        fixed_rows,
    )
    print()

    print("=== SUMMARY ===")
    print(f"58-seed closure rows: {len(closure_rows)}")
    print(f"Doubly-live core size: {len(core_rows)}")
    print(f"Fixed-space rows: {len(fixed_rows)}")


if __name__ == '__main__':
    main()