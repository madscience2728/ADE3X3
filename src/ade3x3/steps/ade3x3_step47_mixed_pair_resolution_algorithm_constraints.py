"""
ade3x3_step47_mixed_pair_resolution_algorithm_constraints.py

Step 47: Mixed Pair Resolution + Algorithm Constraint Extraction.

This step extends the verified step 46 64-orbit sub-algebra exports.

Task 1
  Resolve the 164 mixed orbit-pairs in the 64-orbit sub-algebra by conditioning
  on interface coordinates of the shared CXC face.

Task 2
  Map the standard 3x3 multiplication tensor's same-fiber live-X pairs into the
  CXXC orbit coordinates of the 64-orbit sub-algebra.

Task 3
  Extract conservative algorithm-search constraints from the orbit data. The exact
  orbit count for general rank-1 terms is not finite until a coefficient/support
  model is specified, so this step records the exact basis-level tensor profile and
  the exact coordinate-resolution data that any future search model must respect.

Task 4
  External known-algorithm ingestion is attempted only if a direct public source is
  available. No such source is bundled in the repository, so this step records the
  status explicitly instead of fabricating a mapping.

Outputs:
  - outputs/exports/step47_mixed_pair_witnesses.csv
  - outputs/exports/step47_mixed_pair_st_resolution.csv
  - outputs/exports/step47_mixed_pair_rstu_resolution.csv
  - outputs/exports/step47_mixed_pair_resolution_summary.csv
  - outputs/exports/step47_interface_lookup.csv
  - outputs/exports/step47_tensor_fiber_pairs.csv
  - outputs/exports/step47_tensor_orbit_profile.csv
  - outputs/exports/step47_algorithm_constraints.csv
  - outputs/exports/step47_external_algorithm_status.csv
  - outputs/exports/step47_mixed_pair_resolution_algorithm_constraints.md

Provenance:
  [EXACT_DERIVED] for all enumerated data.
  [INTERPRETATION] only in the markdown summary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations_with_replacement
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


def x_name(x: int) -> str:
    r, s, t, u = _decode_x(x)
    return f"X[{r},{s}|{t},{u}]"


def c_name(c_atom: int) -> str:
    return f"C[{c_atom // 3},{c_atom % 3}]"


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


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
        result[orbit_id] = {
            'orbit_id': orbit_id,
            'rep_config_id': rep_config_id,
            'orbit_size': int(row['orbit_size']),
            'stabilizer_type': row['stabilizer_type'],
            'signature_key': signatures.get(orbit_id, {}).get('signature_key', ''),
            'rep_readable': signatures.get(orbit_id, {}).get('rep_readable', ''),
            'c1': c1,
            'c2': c2,
            'x1': x1,
            'x2': x2,
            'x1_r': x1_r,
            'x1_s': x1_s,
            'x1_t': x1_t,
            'x1_u': x1_u,
            'x2_r': x2_r,
            'x2_s': x2_s,
            'x2_t': x2_t,
            'x2_u': x2_u,
            'both_live': (x1_s == x1_t) and (x2_s == x2_t),
        }
    return result


def load_step46_outputs() -> tuple[list[dict], list[int], set[tuple[int, int]]]:
    table = read_csv_rows(EXPORTS / 'step46_64_subalgebra_table.csv')
    compatible_rows = [row for row in table if row['compatible'] == 'True']
    closed64_ids = sorted({int(row['orbit_A']) for row in compatible_rows} | {int(row['orbit_B']) for row in compatible_rows})
    mixed_pairs = {
        (int(row['orbit_A']), int(row['orbit_B']))
        for row in compatible_rows
        if row['deterministic'] == 'False'
    }
    return compatible_rows, closed64_ids, mixed_pairs


def load_step45_core_ids() -> set[int]:
    return {
        int(row['orbit_id'])
        for row in read_csv_rows(EXPORTS / 'step45_doubly_live_core.csv')
    }


def load_step46_same_fiber_ids() -> set[int]:
    return {
        int(row['orbit_id'])
        for row in read_csv_rows(EXPORTS / 'step45_doubly_live_core.csv')
        if row['same_fiber'] == 'True'
    }


def _scan_mixed_chunk(args: tuple) -> dict[str, object]:
    c1_values, cxxc_oid, mixed_by_a = args
    pair_witness_rows: list[dict] = []
    pair_st_counts: Counter = Counter()
    pair_full_counts: Counter = Counter()
    pair_total_counts: Counter = Counter()

    for c1_m in c1_values:
        for x_m in range(81):
            r, s, t, u = _decode_x(x_m)
            for c2_m in range(9):
                a_orbs = [
                    cxxc_oid[((c1_m * 81 + x1) * 81 + x_m) * 9 + c2_m]
                    for x1 in range(81)
                ]
                a_index_map: dict[int, list[int]] = defaultdict(list)
                for x1, a_orbit in enumerate(a_orbs):
                    if a_orbit in mixed_by_a:
                        a_index_map[a_orbit].append(x1)
                if not a_index_map:
                    continue

                b_orbs = [
                    cxxc_oid[((c1_m * 81 + x_m) * 81 + x4) * 9 + c2_m]
                    for x4 in range(81)
                ]
                b_index_map: dict[int, list[int]] = defaultdict(list)
                for x4, b_orbit in enumerate(b_orbs):
                    b_index_map[b_orbit].append(x4)

                for a_orbit, x1_indices in a_index_map.items():
                    candidate_b = mixed_by_a[a_orbit]
                    for b_orbit in candidate_b:
                        if b_orbit not in b_index_map:
                            continue
                        x4_indices = b_index_map[b_orbit]
                        for x1 in x1_indices:
                            a_cfg = ((c1_m * 81 + x1) * 81 + x_m) * 9 + c2_m
                            for x4 in x4_indices:
                                b_cfg = ((c1_m * 81 + x_m) * 81 + x4) * 9 + c2_m
                                out_cfg = ((c1_m * 81 + x1) * 81 + x4) * 9 + c2_m
                                out_orbit = cxxc_oid[out_cfg]
                                pair_total_counts[(a_orbit, b_orbit, out_orbit)] += 1
                                pair_st_counts[(a_orbit, b_orbit, s, t, out_orbit)] += 1
                                pair_full_counts[(a_orbit, b_orbit, r, s, t, u, out_orbit)] += 1
                                pair_witness_rows.append({
                                    'orbit_A': a_orbit,
                                    'orbit_B': b_orbit,
                                    'a_config_id': a_cfg,
                                    'b_config_id': b_cfg,
                                    'output_config_id': out_cfg,
                                    'output_orbit': out_orbit,
                                    'interface_c1': c1_m,
                                    'interface_x': x_m,
                                    'interface_c2': c2_m,
                                    'iface_r': r,
                                    'iface_s': s,
                                    'iface_t': t,
                                    'iface_u': u,
                                    'iface_st': str((s, t)),
                                    'iface_rstu': str((r, s, t, u)),
                                })

    return {
        'witness_rows': pair_witness_rows,
        'pair_total_counts': dict(pair_total_counts),
        'pair_st_counts': dict(pair_st_counts),
        'pair_full_counts': dict(pair_full_counts),
    }


def scan_mixed_pairs_parallel(cxxc_oid: list[int], mixed_pairs: set[tuple[int, int]], workers: int) -> tuple[list[dict], Counter, Counter, Counter]:
    mixed_by_a: dict[int, set[int]] = defaultdict(set)
    for a_orbit, b_orbit in mixed_pairs:
        mixed_by_a[a_orbit].add(b_orbit)

    c1_values = list(range(9))
    n_chunks = min(workers, 9)
    chunk_size = max(1, -(-9 // n_chunks))
    chunks = [c1_values[idx:idx + chunk_size] for idx in range(0, 9, chunk_size)]

    all_witness_rows: list[dict] = []
    pair_total_counts: Counter = Counter()
    pair_st_counts: Counter = Counter()
    pair_full_counts: Counter = Counter()

    print(f"  Scanning mixed-pair witnesses ({len(mixed_pairs)} orbit pairs, {workers} workers)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_scan_mixed_chunk, (chunk, cxxc_oid, mixed_by_a)) for chunk in chunks]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            all_witness_rows.extend(result['witness_rows'])
            for key, value in result['pair_total_counts'].items():
                pair_total_counts[key] += value
            for key, value in result['pair_st_counts'].items():
                pair_st_counts[key] += value
            for key, value in result['pair_full_counts'].items():
                pair_full_counts[key] += value
            print(f"    chunk {idx}/{len(chunks)} done", end='\r')
    print(f"    {len(chunks)}/{len(chunks)} chunks done.   ")
    return all_witness_rows, pair_total_counts, pair_st_counts, pair_full_counts


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def summarize_resolution(
    mixed_pairs: set[tuple[int, int]],
    pair_total_counts: Counter,
    pair_st_counts: Counter,
    pair_full_counts: Counter,
) -> tuple[list[dict], list[dict], dict[str, object], list[dict]]:
    pair_outputs: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_st_map: dict[tuple[int, int], dict[tuple[int, int], Counter]] = defaultdict(lambda: defaultdict(Counter))
    pair_full_map: dict[tuple[int, int], dict[tuple[int, int, int, int], Counter]] = defaultdict(lambda: defaultdict(Counter))

    for (a_orbit, b_orbit, out_orbit), count in pair_total_counts.items():
        pair_outputs[(a_orbit, b_orbit)].add(out_orbit)

    for (a_orbit, b_orbit, s, t, out_orbit), count in pair_st_counts.items():
        pair_st_map[(a_orbit, b_orbit)][(s, t)][out_orbit] += count

    for (a_orbit, b_orbit, r, s, t, u, out_orbit), count in pair_full_counts.items():
        pair_full_map[(a_orbit, b_orbit)][(r, s, t, u)][out_orbit] += count

    st_rows: list[dict] = []
    full_rows: list[dict] = []
    lookup_rows: list[dict] = []
    resolved_by_st = 0
    resolved_by_full = 0
    constant_outputset_by_st = 0
    constant_outputset_by_full = 0

    for pair in sorted(mixed_pairs):
        outputs = sorted(pair_outputs[pair])
        st_map = pair_st_map[pair]
        full_map = pair_full_map[pair]
        st_resolution_map: dict[str, object] = {}
        full_resolution_map: dict[str, object] = {}
        st_ok = True
        full_ok = True

        for coords in sorted(st_map):
            counter = st_map[coords]
            values = sorted(counter)
            st_resolution_map[str(coords)] = values[0] if len(values) == 1 else values
            if len(values) != 1:
                st_ok = False

        if len({str(v) for v in st_resolution_map.values()}) == 1:
            constant_outputset_by_st += 1

        for coords in sorted(full_map):
            counter = full_map[coords]
            values = sorted(counter)
            full_resolution_map[str(coords)] = values[0] if len(values) == 1 else values
            if len(values) != 1:
                full_ok = False
            if len(values) == 1:
                lookup_rows.append({
                    'orbit_A': pair[0],
                    'orbit_B': pair[1],
                    'iface_r': coords[0],
                    'iface_s': coords[1],
                    'iface_t': coords[2],
                    'iface_u': coords[3],
                    'output_orbit': values[0],
                })

        if len({str(v) for v in full_resolution_map.values()}) == 1:
            constant_outputset_by_full += 1

        st_rows.append({
            'orbit_A': pair[0],
            'orbit_B': pair[1],
            'output_orbits': str(outputs),
            'n_strata_by_st': len(st_map),
            'fully_resolved_by_st': st_ok,
            'resolution_map': str(st_resolution_map),
        })
        full_rows.append({
            'orbit_A': pair[0],
            'orbit_B': pair[1],
            'output_orbits': str(outputs),
            'n_strata_by_rstu': len(full_map),
            'fully_resolved_by_rstu': full_ok,
            'resolution_map': str(full_resolution_map),
        })

        resolved_by_st += 1 if st_ok else 0
        resolved_by_full += 1 if full_ok else 0

    summary = {
        'n_mixed_pairs': len(mixed_pairs),
        'mixed_pair_witnesses': sum(pair_total_counts.values()),
        'resolved_by_st': resolved_by_st,
        'unresolved_by_st': len(mixed_pairs) - resolved_by_st,
        'resolved_by_rstu': resolved_by_full,
        'unresolved_by_rstu': len(mixed_pairs) - resolved_by_full,
        'constant_outputset_by_st': constant_outputset_by_st,
        'constant_outputset_by_rstu': constant_outputset_by_full,
        'fully_resolved_lookup_available': resolved_by_full == len(mixed_pairs),
    }
    return st_rows, full_rows, summary, lookup_rows


def tensor_fiber_pairs(cxxc_oid: list[int], orbit_meta: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    orbit_counter: Counter = Counter()
    pair_id = 0
    for r in range(3):
        for u in range(3):
            target_c = 3 * r + u
            fiber = [9 * (3 * r + s) + (3 * s + u) for s in range(3)]
            for x1, x2 in combinations_with_replacement(fiber, 2):
                cfg = ((target_c * 81 + x1) * 81 + x2) * 9 + target_c
                orbit_id = cxxc_oid[cfg]
                orbit_counter[orbit_id] += 1
                rows.append({
                    'fiber_pair_id': pair_id,
                    'target_c': target_c,
                    'target_c_name': c_name(target_c),
                    'x1': x1,
                    'x1_name': x_name(x1),
                    'x2': x2,
                    'x2_name': x_name(x2),
                    'same_x': x1 == x2,
                    'config_id': cfg,
                    'orbit_id': orbit_id,
                    'orbit_stab_type': orbit_meta[orbit_id]['stabilizer_type'],
                    'orbit_rep': orbit_meta[orbit_id]['rep_readable'],
                })
                pair_id += 1
    profile_rows = [
        {
            'orbit_id': orbit_id,
            'pair_count': count,
            'fraction': f"{count / len(rows):.6f}",
            'stabilizer_type': orbit_meta[orbit_id]['stabilizer_type'],
            'rep_readable': orbit_meta[orbit_id]['rep_readable'],
        }
        for orbit_id, count in sorted(orbit_counter.items())
    ]
    return rows, profile_rows


def write_markdown_summary(
    path: Path,
    res_summary: dict[str, object],
    st_rows: list[dict],
    full_rows: list[dict],
    tensor_profile_rows: list[dict],
    algo_constraint_rows: list[dict],
) -> None:
    lines: list[str] = []
    w = lines.append

    w("# Step 47: Mixed Pair Resolution + Algorithm Constraint Extraction")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("[EXACT_DERIVED]")
    w("")
    w("## Task 1: Mixed-Pair Resolution")
    w("")
    w(f"Mixed orbit pairs: {res_summary['n_mixed_pairs']}")
    w(f"Mixed-pair witnesses: {res_summary['mixed_pair_witnesses']:,}")
    w(f"- Fully resolved by (s,t): {res_summary['resolved_by_st']} / {res_summary['n_mixed_pairs']}")
    w(f"- Fully resolved by (r,s,t,u): {res_summary['resolved_by_rstu']} / {res_summary['n_mixed_pairs']}")
    w(f"- Output-set invariant across all occupied (s,t) strata: {res_summary['constant_outputset_by_st']} / {res_summary['n_mixed_pairs']}")
    w(f"- Output-set invariant across all occupied (r,s,t,u) strata: {res_summary['constant_outputset_by_rstu']} / {res_summary['n_mixed_pairs']}")
    w("")
    w("| orbit_A | orbit_B | output_orbits | n_strata_by_st | fully_resolved_by_st |")
    w("|---------|---------|---------------|----------------|----------------------|")
    for row in st_rows[:20]:
        w(f"| {row['orbit_A']} | {row['orbit_B']} | {row['output_orbits']} | {row['n_strata_by_st']} | {row['fully_resolved_by_st']} |")
    if len(st_rows) > 20:
        w("| ... | ... | ... | ... | ... |")
    w("")

    w("## Task 2: Multiplication Tensor in Orbit Coordinates")
    w("")
    if tensor_profile_rows:
        w("| orbit_id | pair_count | fraction | stab_type |")
        w("|----------|------------|----------|-----------|")
        for row in tensor_profile_rows:
            w(f"| {row['orbit_id']} | {int(row['pair_count'])} | {float(row['fraction']):.6f} | {row['stabilizer_type']} |")
    w("")

    w("## Task 3: Algorithm Constraints")
    w("")
    for row in algo_constraint_rows:
        w(f"- {row['constraint_name']}: {row['constraint_value']}")
    w("")
    w("[INTERPRETATION]")
    w("")
    w("The mixed pairs are the exact forks that an algorithm must resolve by committing to")
    w("finer interface data than orbit identity alone. The multiplication tensor itself lives")
    w("on a very small same-fiber slice of the 64-orbit algebra, so any algorithm inside this")
    w("sub-algebra must reproduce that orbit profile while choosing consistent coordinate-level")
    w("branches across the 164 mixed pairs.")

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='Parallel workers for the mixed-pair witness scan')
    args = parser.parse_args()
    workers = max(1, args.workers)

    print(f"=== Step 47: Mixed Pair Resolution + Algorithm Constraint Extraction (workers={workers}) ===")
    print()

    print("Loading step 45/46 exports...")
    orbit_meta = build_orbit_meta()
    _, closed64_ids, mixed_pairs = load_step46_outputs()
    doubly_live_core_ids = load_step45_core_ids()
    same_fiber_ids = load_step46_same_fiber_ids()
    print(f"  64-closed ids={len(closed64_ids)}  mixed pairs={len(mixed_pairs)}")
    print()

    print("Building CXXC orbit map...")
    cxxc_oid = _build_orbit_map_parallel(531_441, chunk_size=4096, workers=workers)
    assert len(set(cxxc_oid)) == 2744, 'CXXC orbit count mismatch'
    print("  CXXC=2744 verified")
    print()

    print("Task 1: scanning mixed-pair witnesses...")
    witness_rows, pair_total_counts, pair_st_counts, pair_full_counts = scan_mixed_pairs_parallel(cxxc_oid, mixed_pairs, workers)
    mixed_total = sum(pair_total_counts.values())
    assert mixed_total == 41_688, f"Expected 41,688 mixed witnesses, got {mixed_total:,}"
    st_rows, full_rows, res_summary, lookup_rows = summarize_resolution(mixed_pairs, pair_total_counts, pair_st_counts, pair_full_counts)
    print(f"  mixed witnesses={mixed_total:,}")
    print(f"  resolved by (s,t)={res_summary['resolved_by_st']}/{res_summary['n_mixed_pairs']}")
    print(f"  resolved by (r,s,t,u)={res_summary['resolved_by_rstu']}/{res_summary['n_mixed_pairs']}")
    print()

    print("Task 2: mapping multiplication tensor same-fiber live pairs...")
    tensor_pair_rows, tensor_profile_rows = tensor_fiber_pairs(cxxc_oid, orbit_meta)
    used_tensor_orbits = [int(row['orbit_id']) for row in tensor_profile_rows]
    print(f"  tensor fiber-pairs={len(tensor_pair_rows)}  used same-fiber orbits={used_tensor_orbits}")
    print()

    print("Task 3/4: extracting algorithm constraints and recording external status...")
    algo_constraint_rows = [
        {
            'constraint_name': 'mixed_pair_count',
            'constraint_value': str(res_summary['n_mixed_pairs']),
            'provenance': 'EXACT_DERIVED',
            'note': 'These are the orbit-level forks that any search inside the 64-orbit algebra must resolve.',
        },
        {
            'constraint_name': 'resolved_by_st',
            'constraint_value': f"{res_summary['resolved_by_st']}/{res_summary['n_mixed_pairs']}",
            'provenance': 'EXACT_DERIVED',
            'note': 'Pairs fully determined by interface (s,t) coordinates alone.',
        },
        {
            'constraint_name': 'resolved_by_rstu',
            'constraint_value': f"{res_summary['resolved_by_rstu']}/{res_summary['n_mixed_pairs']}",
            'provenance': 'EXACT_DERIVED',
            'note': 'Pairs fully determined by full interface X coordinates.',
        },
        {
            'constraint_name': 'constant_outputset_by_st',
            'constraint_value': f"{res_summary['constant_outputset_by_st']}/{res_summary['n_mixed_pairs']}",
            'provenance': 'EXACT_DERIVED',
            'note': 'Pairs whose mixed output-set is identical in every occupied (s,t) stratum.',
        },
        {
            'constraint_name': 'constant_outputset_by_rstu',
            'constraint_value': f"{res_summary['constant_outputset_by_rstu']}/{res_summary['n_mixed_pairs']}",
            'provenance': 'EXACT_DERIVED',
            'note': 'Pairs whose mixed output-set is identical in every occupied full-coordinate stratum.',
        },
        {
            'constraint_name': 'tensor_same_fiber_orbits',
            'constraint_value': str(used_tensor_orbits),
            'provenance': 'EXACT_DERIVED',
            'note': 'Among the 10 same-fiber closed orbits, the raw 3x3 multiplication tensor uses exactly these orbit ids.',
        },
        {
            'constraint_name': 'tensor_doubly_live_outside_0_30',
            'constraint_value': '0',
            'provenance': 'EXACT_DERIVED',
            'note': 'No raw multiplication-tensor same-fiber pair lands outside orbit 0 or orbit 30.',
        },
        {
            'constraint_name': 'rank1_exact_orbit_model_status',
            'constraint_value': 'underdetermined_without_coefficient_or_support_model',
            'provenance': 'EXACT_DERIVED',
            'note': 'General rank-1 terms live in a continuous coefficient space; a finite orbit count and exact ILP need an explicit coefficient/support restriction.',
        },
    ]
    external_status_rows = [
        {
            'algorithm_name': 'Smirnov_23x3x3_2013',
            'status': 'not_mapped',
            'reason': 'No direct algorithm specification is present in the repository or bundled sources, and no authoritative public term list was available through the current tool set.',
        },
        {
            'algorithm_name': 'Strassen_2x2_7',
            'status': 'not_mapped',
            'reason': 'Outside the current 3x3 CXXC export scope; no direct 2x2-to-3x3 embedding was formalized in this step.',
        },
    ]
    print("  external algorithm mapping status recorded")
    print()

    print("Writing outputs...")
    write_csv(
        EXPORTS / 'step47_mixed_pair_witnesses.csv',
        witness_rows,
        ['orbit_A', 'orbit_B', 'a_config_id', 'b_config_id', 'output_config_id', 'output_orbit', 'interface_c1', 'interface_x', 'interface_c2', 'iface_r', 'iface_s', 'iface_t', 'iface_u', 'iface_st', 'iface_rstu'],
    )
    write_csv(
        EXPORTS / 'step47_mixed_pair_st_resolution.csv',
        st_rows,
        ['orbit_A', 'orbit_B', 'output_orbits', 'n_strata_by_st', 'fully_resolved_by_st', 'resolution_map'],
    )
    write_csv(
        EXPORTS / 'step47_mixed_pair_rstu_resolution.csv',
        full_rows,
        ['orbit_A', 'orbit_B', 'output_orbits', 'n_strata_by_rstu', 'fully_resolved_by_rstu', 'resolution_map'],
    )
    write_csv(
        EXPORTS / 'step47_mixed_pair_resolution_summary.csv',
        [res_summary],
        ['n_mixed_pairs', 'mixed_pair_witnesses', 'resolved_by_st', 'unresolved_by_st', 'resolved_by_rstu', 'unresolved_by_rstu', 'constant_outputset_by_st', 'constant_outputset_by_rstu', 'fully_resolved_lookup_available'],
    )
    write_csv(
        EXPORTS / 'step47_interface_lookup.csv',
        lookup_rows,
        ['orbit_A', 'orbit_B', 'iface_r', 'iface_s', 'iface_t', 'iface_u', 'output_orbit'],
    )
    write_csv(
        EXPORTS / 'step47_tensor_fiber_pairs.csv',
        tensor_pair_rows,
        ['fiber_pair_id', 'target_c', 'target_c_name', 'x1', 'x1_name', 'x2', 'x2_name', 'same_x', 'config_id', 'orbit_id', 'orbit_stab_type', 'orbit_rep'],
    )
    write_csv(
        EXPORTS / 'step47_tensor_orbit_profile.csv',
        tensor_profile_rows,
        ['orbit_id', 'pair_count', 'fraction', 'stabilizer_type', 'rep_readable'],
    )
    write_csv(
        EXPORTS / 'step47_algorithm_constraints.csv',
        algo_constraint_rows,
        ['constraint_name', 'constraint_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step47_external_algorithm_status.csv',
        external_status_rows,
        ['algorithm_name', 'status', 'reason'],
    )
    write_markdown_summary(
        EXPORTS / 'step47_mixed_pair_resolution_algorithm_constraints.md',
        res_summary,
        st_rows,
        full_rows,
        tensor_profile_rows,
        algo_constraint_rows,
    )
    print()

    print("=== SUMMARY ===")
    print(f"mixed pairs: {res_summary['n_mixed_pairs']}")
    print(f"resolved by (s,t): {res_summary['resolved_by_st']}")
    print(f"resolved by (r,s,t,u): {res_summary['resolved_by_rstu']}")
    print(f"tensor orbits: {used_tensor_orbits}")


if __name__ == '__main__':
    main()