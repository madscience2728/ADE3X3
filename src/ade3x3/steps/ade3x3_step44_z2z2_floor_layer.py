"""
ade3x3_step44_z2z2_floor_layer.py  (parallel version, --workers N)

Z₂×Z₂ Floor Layer Analysis.

The 40 CXXC orbits with stabilizer order >= 4 (39 Z2xZ2 + 1 (Z2)^3)
may form a composition-stable "floor layer". This step investigates:

  2a: Do all outputs of (Z2xZ2 or (Z2)^3) o (Z2xZ2 or (Z2)^3) also land in the
      floor layer? (Full closure check from step 42 summary data.)

  2b: For floor-layer inputs, does output always have stabilizer order >= 2?
      (Floor orbits never generate Trivial output -- the "floor" property.)

  2c: List all 40 orbits with representatives, signatures, and structural
      features. Check for shared patterns (X-atom liveness, equality relations).

  2d: Compute the composition table restricted to the 40 orbits.

Composition definition (same as step 42):
  A = CXXC(c1_m, x1, x_m, c2_m)   right-CXC face = (c1_m, x_m, c2_m)
  B = CXXC(c1_m, x_m, x4, c2_m)   left-CXC  face = (c1_m, x_m, c2_m)
  Output = CXXC(c1_m, x1, x4, c2_m)

Parallelism:
  Phase 1 - CXXC orbit map: 531,441 configs split in chunks across workers.
  Phase 2 - Floor scan: 9 c1_m values split across workers.
  All worker functions are module-level (picklable under Windows spawn).

Self-test:
  After accumulation, per-(stab_type_A, stab_type_B, stab_type_output) pair
  counts are compared against hardcoded step 42 ground truth. Run aborts on
  any discrepancy.

Usage:
  python ade3x3_step44_z2z2_floor_layer.py [--workers N]  (default: cpu_count)

Provenance: [EXACT_DERIVED] for enumerated facts; [INTERPRETATION] for structural claims.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

EXPORTS = Path('outputs/exports')

# ══════════════════════════════════════════════════════════════════════════════
# Group and precomputed action tables
# ══════════════════════════════════════════════════════════════════════════════

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]   # 216 elements


def _build_tables():
    """
    For each of the 216 group actions, precompute lookup tables:
      c_map[c]  = new_c   (9 entries, acts on C atoms: c = 3r+u -> 3*p1[r]+p3[u])
      x_map[x]  = new_x   (81 entries, acts on X atoms)
    Replaces per-component arithmetic in the inner loop with a single table lookup.
    Tables are rebuild in each spawned worker by re-importing the module.
    """
    cmaps: list[list[int]] = []
    xmaps: list[list[int]] = []
    for p1, p2, p3 in ACTIONS:
        cmaps.append([3 * p1[c // 3] + p3[c % 3] for c in range(9)])
        xmaps.append([
            9 * (3 * p1[(x // 9) // 3] + p2[(x // 9) % 3])
              + (3 * p2[(x %  9) // 3] + p3[(x %  9) % 3])
            for x in range(81)
        ])
    return cmaps, xmaps


CMAPS, XMAPS = _build_tables()   # available in every worker (spawn re-imports module)

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: parallel orbit map building
# ══════════════════════════════════════════════════════════════════════════════

def _cxxc_min_rep_chunk(start: int, stop: int) -> dict[int, int]:
    """
    For each CXXC config in [start, stop), find its min-image (canonical rep)
    under all 216 group actions.  Returns {cfg: min_rep}.
    """
    result: dict[int, int] = {}
    for cfg in range(start, stop):
        c2 = cfg % 9;   rem = cfg // 9
        x2 = rem % 81;  rem //= 81
        x1 = rem % 81;  c1 = rem // 81
        m = cfg
        for c_map, x_map in zip(CMAPS, XMAPS):
            v = ((c_map[c1] * 81 + x_map[x1]) * 81 + x_map[x2]) * 9 + c_map[c2]
            if v < m:
                m = v
        result[cfg] = m
    return result


def _cxc_min_rep_chunk(start: int, stop: int) -> dict[int, int]:
    """Same for CXC (6 561 configs).  cfg = (c1*81 + x)*9 + c2."""
    result: dict[int, int] = {}
    for cfg in range(start, stop):
        c2 = cfg % 9;  rem = cfg // 9
        x  = rem % 81; c1 = rem // 81
        m = cfg
        for c_map, x_map in zip(CMAPS, XMAPS):
            v = (c_map[c1] * 81 + x_map[x]) * 9 + c_map[c2]
            if v < m:
                m = v
        result[cfg] = m
    return result


def _build_orbit_map_parallel(
    n: int,
    chunk_fn,
    chunk_size: int,
    workers: int,
    label: str,
) -> list[int]:
    """
    Build a length-n orbit map using parallel chunks.
    Returns oid[cfg] = orbit_id using canonical min-rep numbering.
    """
    print(f"  Building {label} orbit map ({n:,} configs, {workers} workers)...")
    chunks = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]
    min_rep = [0] * n
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(chunk_fn, s, e): (s, e) for s, e in chunks}
        for fut in as_completed(futs):
            for cfg, m in fut.result().items():
                min_rep[cfg] = m
            done += len(fut.result())
            print(f"    {100 * done / n:5.1f}%  ({done:,}/{n:,})", end='\r')
    print()
    reps = sorted(set(min_rep))
    r2id = {r: i for i, r in enumerate(reps)}
    return [r2id[min_rep[cfg]] for cfg in range(n)]

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: parallel floor-only composition scan
# ══════════════════════════════════════════════════════════════════════════════

def _scan_floor_c1m_chunk(args: tuple) -> dict[tuple, int]:
    """
    For a subset of c1_m values, enumerate all composition pairs where both
    A-orbit and B-orbit are floor orbits.  Count by (a_orb, b_orb, out_orb).

    args = (c1_m_values, cxxc_oid, floor_ids)
      cxxc_oid  : plain list[int]    (length 531 441)
      floor_ids : frozenset[int]
    """
    c1_m_values, cxxc_oid, floor_ids = args
    counter: Counter = Counter()

    for c1_m in c1_m_values:
        for x_m in range(81):
            for c2_m in range(9):
                # A candidates: CXXC(c1_m, x1, x_m, c2_m)
                a_orbs = [
                    cxxc_oid[((c1_m * 81 + x1) * 81 + x_m) * 9 + c2_m]
                    for x1 in range(81)
                ]
                # B candidates: CXXC(c1_m, x_m, x4, c2_m)
                b_orbs = [
                    cxxc_oid[((c1_m * 81 + x_m) * 81 + x4) * 9 + c2_m]
                    for x4 in range(81)
                ]

                for x1 in range(81):
                    a_orb = a_orbs[x1]
                    if a_orb not in floor_ids:
                        continue
                    # Output: CXXC(c1_m, x1, x4, c2_m)
                    out_orbs = [
                        cxxc_oid[((c1_m * 81 + x1) * 81 + x4) * 9 + c2_m]
                        for x4 in range(81)
                    ]
                    for x4 in range(81):
                        b_orb = b_orbs[x4]
                        if b_orb not in floor_ids:
                            continue
                        counter[(a_orb, b_orb, out_orbs[x4])] += 1

    return dict(counter)


def _scan_floor_parallel(
    cxxc_oid: list[int],
    floor_ids: frozenset[int],
    workers: int,
) -> Counter:
    """Split 9 c1_m values into chunks and scan in parallel."""
    print(f"  Scanning floor compositions (9 c1_m values, {workers} workers)...")
    c1_m_all  = list(range(9))
    n_chunks  = min(workers, 9)
    sz        = max(1, -(-9 // n_chunks))     # ceiling division
    chunks    = [c1_m_all[i : i + sz] for i in range(0, 9, sz)]

    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_scan_floor_c1m_chunk, (chunk, cxxc_oid, floor_ids))
                for chunk in chunks]
        for i, fut in enumerate(as_completed(futs)):
            for k, v in fut.result().items():
                total[k] += v
            print(f"    chunk {i+1}/{len(chunks)} done", end='\r')
    print(f"    {len(chunks)}/{len(chunks)} chunks done.   ")
    return total

# ══════════════════════════════════════════════════════════════════════════════
# Self-test: exact agreement with step 42 ground truth
# ══════════════════════════════════════════════════════════════════════════════

# Exact floor x floor pair counts from stabilizer_composition_summary.csv (step 42).
# These are the ONLY acceptable results for the new parallel computation.
STEP42_GROUND_TRUTH: dict[tuple[str, str, str], int] = {
    ('Z2xZ2',  'Z2xZ2',  'Z2'):       1_944,
    ('Z2xZ2',  'Z2xZ2',  'Z2xZ2'):    9_882,
    ('Z2xZ2',  'Z2xZ2',  '(Z2)^3'):     324,
    ('Z2xZ2',  '(Z2)^3', 'Z2xZ2'):      324,
    ('(Z2)^3', 'Z2xZ2',  'Z2xZ2'):      324,
    ('(Z2)^3', '(Z2)^3', '(Z2)^3'):      27,
}
STEP42_TOTAL = sum(STEP42_GROUND_TRUTH.values())   # 12 825


def self_test(pair_counter: Counter, stab_type_map: dict[int, str]) -> None:
    """
    Roll up pair_counter by stab-type triple and compare to STEP42_GROUND_TRUTH.
    Aborts (AssertionError) on any mismatch.
    """
    by_type: Counter = Counter()
    for (a_orb, b_orb, out_orb), cnt in pair_counter.items():
        by_type[(stab_type_map[a_orb],
                 stab_type_map[b_orb],
                 stab_type_map[out_orb])] += cnt

    total_got = sum(by_type.values())
    assert total_got == STEP42_TOTAL, (
        f"[SELF-TEST FAIL] total floor pairs: got {total_got:,}, "
        f"expected {STEP42_TOTAL:,}"
    )
    for key, expected in STEP42_GROUND_TRUTH.items():
        got = by_type.get(key, 0)
        assert got == expected, (
            f"[SELF-TEST FAIL] ({key}) got {got:,}, expected {expected:,}"
        )
    for key in by_type:
        if key not in STEP42_GROUND_TRUTH:
            raise AssertionError(
                f"[SELF-TEST FAIL] Unexpected key {key}: count={by_type[key]:,}"
            )
    print(f"  [SELF-TEST PASS] All pair counts match step 42 exactly. "
          f"Total verified: {total_got:,} ✓")

# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_stabilizer_classification() -> list[dict]:
    rows: list[dict] = []
    with open(EXPORTS / 'stabilizer_classification.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['schema'] == 'CXXC':
                rows.append({
                    'orbit_id':             int(row['orbit_id']),
                    'rep_config_id':        int(row['rep_config_id']),
                    'orbit_size':           int(row['orbit_size']),
                    'stabilizer_order':     int(row['stabilizer_order']),
                    'stabilizer_type':      row['stabilizer_type'],
                    'is_abelian':           row['is_abelian'],
                    'element_orders':       row['element_orders'],
                    'generator_action_ids': row['generator_action_ids'],
                })
    return rows


def load_signatures() -> dict[int, dict]:
    sig: dict[int, dict] = {}
    with open(EXPORTS / 'signatures_CXXC_refined.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sig[int(row['orbit_id'])] = {
                'rep_readable':        row['rep_readable'],
                'refinement_features': row['refinement_features'],
            }
    return sig


def load_floor_comp_summary() -> list[dict]:
    """Read step 42 summary restricted to floor x floor rows."""
    floor_types = {'Z2xZ2', '(Z2)^3'}
    rows: list[dict] = []
    with open(EXPORTS / 'stabilizer_composition_summary.csv',
              newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['stab_type_A'] in floor_types and row['stab_type_B'] in floor_types:
                rows.append({
                    'stab_type_A':      row['stab_type_A'],
                    'stab_type_B':      row['stab_type_B'],
                    'stab_type_output': row['stab_type_output'],
                    'pair_count':       int(row['pair_count']),
                    'fraction':         float(row['fraction']),
                })
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# Task 2c: structural analysis
# ══════════════════════════════════════════════════════════════════════════════

def _decode_cxxc(cfg: int) -> tuple[int, int, int, int]:
    c2 = cfg % 9;   rem = cfg // 9
    x2 = rem % 81;  rem //= 81
    x1 = rem % 81;  c1 = rem // 81
    return c1, x1, x2, c2


def _decode_x(x: int) -> tuple[int, int, int, int]:
    return (x // 9) // 3, (x // 9) % 3, (x % 9) // 3, (x % 9) % 3


def enrich_floor_orbits(
    floor_rows: list[dict],
    sig_map: dict[int, dict],
) -> list[dict]:
    result: list[dict] = []
    for row in floor_rows:
        oid = row['orbit_id']
        rep = row['rep_config_id']
        c1, x1, x2, c2 = _decode_cxxc(rep)
        r1, s1, t1, u1 = _decode_x(x1)
        r2, s2, t2, u2 = _decode_x(x2)
        sig = sig_map.get(oid, {})
        result.append({
            'orbit_id':             oid,
            'rep_config_id':        rep,
            'orbit_size':           row['orbit_size'],
            'stabilizer_order':     row['stabilizer_order'],
            'stabilizer_type':      row['stabilizer_type'],
            'c1': c1, 'x1': x1, 'x2': x2, 'c2': c2,
            'r1': r1, 's1': s1, 't1': t1, 'u1': u1,
            'r2': r2, 's2': s2, 't2': t2, 'u2': u2,
            'live_x1':    s1 == t1,
            'live_x2':    s2 == t2,
            'both_live':  (s1 == t1) and (s2 == t2),
            'c1_eq_c2':   c1 == c2,
            'x1_eq_x2':   x1 == x2,
            'rep_readable':         sig.get('rep_readable', ''),
            'refinement_features':  sig.get('refinement_features', ''),
        })
    return result

# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_markdown(
    floor_comp_summary: list[dict],
    floor_enriched: list[dict],
    pair_rows: list[dict],
    closure: dict,
    stab_type_map: dict[int, str],
    n_total: int,
    n_in: int,
    n_out: int,
    path: Path,
) -> None:
    lines: list[str] = []
    w = lines.append

    w("# Z2xZ2 Floor Layer Analysis")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("## Background")
    w("The 40 CXXC orbits with stabilizer order >= 4 (39 Z2xZ2 + 1 (Z2)^3) are")
    w("investigated as a potential composition-stable 'floor layer'.")
    w("")

    w("## Task 2a-2b: From step 42 summary")
    w("")
    w("| stab_type_A | stab_type_B | stab_type_output | pair_count | fraction |")
    w("|-------------|-------------|------------------|------------|----------|")
    out_types: set[str] = set()
    for r in floor_comp_summary:
        w(f"| {r['stab_type_A']} | {r['stab_type_B']} | {r['stab_type_output']} "
          f"| {r['pair_count']:,} | {r['fraction']:.6f} |")
        out_types.add(r['stab_type_output'])
    w("")
    floor_types = {'Z2xZ2', '(Z2)^3'}
    full_closure = out_types <= floor_types
    no_trivial   = 'Trivial' not in out_types
    w(f"**2a Full closure:** output types = {sorted(out_types)}")
    w(f"   => {'YES — floor layer CLOSED' if full_closure else 'NO — outputs escape floor'}")
    w("")
    w(f"**2b Floor property (no Trivial output):**")
    w(f"   => {'YES — floor layer cannot decay to generic position' if no_trivial else 'NO — Trivial is reachable'}")
    w("")

    w("## Task 2c: The 40 Floor Orbits")
    w("")
    w("| orbit_id | rep | size | stab_type | live_x1 | live_x2 | c1=c2 | x1=x2 |")
    w("|----------|-----|------|-----------|---------|---------|-------|-------|")
    for r in sorted(floor_enriched, key=lambda x: x['orbit_id']):
        w(f"| {r['orbit_id']} | {r['rep_config_id']} | {r['orbit_size']} | "
          f"{r['stabilizer_type']} | {r['live_x1']} | {r['live_x2']} | "
          f"{r['c1_eq_c2']} | {r['x1_eq_x2']} |")
    n_both = sum(1 for r in floor_enriched if r['both_live'])
    n_x1x2 = sum(1 for r in floor_enriched if r['x1_eq_x2'])
    n_c1c2 = sum(1 for r in floor_enriched if r['c1_eq_c2'])
    w("")
    w(f"Both X atoms live: {n_both}/40  |  x1=x2: {n_x1x2}/40  |  c1=c2: {n_c1c2}/40")
    w("")

    w("## Task 2d: Composition Table Restricted to Floor Orbits")
    w(f"Total floor x floor pairs: **{n_total:,}**")
    w(f"- In floor: {n_in:,} ({100*n_in/n_total:.2f}%)")
    w(f"- Outside:  {n_out:,} ({100*n_out/n_total:.2f}%)")
    w("")
    floor_ids_set = {r['orbit_id'] for r in floor_enriched}
    all_r = closure['all_reachable']
    out_fl = [o for o in all_r if o not in floor_ids_set]
    w(f"Reachable output orbits: {len(all_r)} total "
      f"({len([o for o in all_r if o in floor_ids_set])} in floor, "
      f"{len(out_fl)} outside)")
    w("")
    w("### Closure")
    w(f"New orbits added to step-1 closure: {len(closure['escaped'])}")
    if closure['escaped']:
        esc_types = sorted(set(stab_type_map[o] for o in closure['escaped']))
        w(f"  Orbit ids: {sorted(closure['escaped'])}")
        w(f"  Types:     {esc_types}")
    else:
        w("  None — the floor layer IS closed under composition (step-1).")
    w("")

    w("### Per-(A_orbit, B_orbit) Summary")
    w("| A_orbit | A_type | B_orbit | B_type | outputs | in_floor | outside |")
    w("|---------|--------|---------|--------|---------|----------|---------|")
    for r in pair_rows:
        w(f"| {r['A_orbit']} | {r['A_stab_type']} | {r['B_orbit']} | {r['B_stab_type']} "
          f"| {r['output_orbit_ids']} | {r['output_in_floor']} | {r['output_outside_floor']} |")
    w("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int,
                        default=multiprocessing.cpu_count(),
                        help='Parallel workers (default: cpu_count)')
    args = parser.parse_args()
    workers = max(1, args.workers)

    print(f"=== Z2xZ2 Floor Layer Analysis  (workers={workers}) ===")
    print()

    # ── Tasks 2a-2b from existing step 42 data (no computation needed) ────
    print("Tasks 2a-2b: Reading step 42 summary...")
    floor_comp_summary = load_floor_comp_summary()
    out_2ab = {r['stab_type_output'] for r in floor_comp_summary}
    floor_types = {'Z2xZ2', '(Z2)^3'}
    print(f"  Output types   : {sorted(out_2ab)}")
    print(f"  Full closure   : {out_2ab <= floor_types}")
    print(f"  No Trivial out : {'Trivial' not in out_2ab}")
    print()

    # ── Task 2c ──────────────────────────────────────────────────────────
    print("Task 2c: Loading floor orbits...")
    all_cxxc_rows  = load_stabilizer_classification()
    floor_rows     = [r for r in all_cxxc_rows if r['stabilizer_type'] in floor_types]
    assert len(floor_rows) == 40, f"Expected 40, got {len(floor_rows)}"
    floor_ids: frozenset[int] = frozenset(r['orbit_id'] for r in floor_rows)
    stab_type_map: dict[int, str] = {r['orbit_id']: r['stabilizer_type']
                                     for r in all_cxxc_rows}
    sig_map        = load_signatures()
    floor_enriched = enrich_floor_orbits(floor_rows, sig_map)
    n_both = sum(1 for r in floor_enriched if r['both_live'])
    n_x1x2 = sum(1 for r in floor_enriched if r['x1_eq_x2'])
    n_c1c2 = sum(1 for r in floor_enriched if r['c1_eq_c2'])
    print(f"  both_live={n_both}/40  x1=x2={n_x1x2}/40  c1=c2={n_c1c2}/40")
    print()

    # ── Phase 1: orbit maps (parallel) ───────────────────────────────────
    print("Phase 1: Building orbit maps...")
    cxc_oid  = _build_orbit_map_parallel(
        9*81*9,   _cxc_min_rep_chunk,  chunk_size=512,  workers=workers, label='CXC')
    cxxc_oid = _build_orbit_map_parallel(
        531_441,  _cxxc_min_rep_chunk, chunk_size=4096, workers=workers, label='CXXC')
    assert len(set(cxc_oid))  == 50,   f"CXC orbit count wrong"
    assert len(set(cxxc_oid)) == 2744, f"CXXC orbit count wrong"
    print(f"  CXC=50, CXXC=2744 (both verified) ✓")
    print()

    # ── Phase 2: floor composition scan (parallel) ────────────────────────
    print("Phase 2: Scanning floor x floor compositions...")
    pair_counter = _scan_floor_parallel(cxxc_oid, floor_ids, workers)
    n_total = sum(pair_counter.values())
    n_in    = sum(c for (a, b, o), c in pair_counter.items() if o in floor_ids)
    n_out   = n_total - n_in
    print(f"  floor x floor pairs: {n_total:,}")
    print(f"  output in floor    : {n_in:,}")
    print(f"  output outside     : {n_out:,}")
    print()

    # ── Self-test ─────────────────────────────────────────────────────────
    print("Self-test against step 42 ground truth...")
    self_test(pair_counter, stab_type_map)
    print()

    # ── Derive closure and pair table ─────────────────────────────────────
    all_reachable = sorted(set(o for _, _, o in pair_counter))
    escaped       = sorted(o for o in all_reachable if o not in floor_ids)
    closure = {
        'floor':         sorted(floor_ids),
        'all_reachable': all_reachable,
        'escaped':       escaped,
    }

    ab_table: dict[tuple, Counter] = defaultdict(Counter)
    for (a, b, o), cnt in pair_counter.items():
        ab_table[(a, b)][o] += cnt

    pair_rows: list[dict] = []
    for (a_orb, b_orb) in sorted(ab_table):
        ctr   = ab_table[(a_orb, b_orb)]
        total = sum(ctr.values())
        in_fl = sum(c for o, c in ctr.items() if o in floor_ids)
        pair_rows.append({
            'A_orbit':              a_orb,
            'B_orbit':              b_orb,
            'A_stab_type':          stab_type_map[a_orb],
            'B_stab_type':          stab_type_map[b_orb],
            'total_pairs':          total,
            'output_in_floor':      in_fl,
            'output_outside_floor': total - in_fl,
            'output_orbit_ids':     str(sorted(ctr.keys())),
            'output_counts':        str(dict(sorted(ctr.items()))),
        })

    # ── Write outputs ─────────────────────────────────────────────────────
    print("Writing outputs...")
    write_csv(
        sorted(floor_enriched, key=lambda x: x['orbit_id']),
        EXPORTS / 'floor_layer_inventory.csv',
        ['orbit_id', 'rep_config_id', 'orbit_size', 'stabilizer_order', 'stabilizer_type',
         'live_x1', 'live_x2', 'both_live', 'c1_eq_c2', 'x1_eq_x2',
         'c1', 'x1', 'x2', 'c2', 'r1', 's1', 't1', 'u1', 'r2', 's2', 't2', 'u2',
         'rep_readable', 'refinement_features'],
    )
    write_csv(
        pair_rows,
        EXPORTS / 'floor_layer_composition_table.csv',
        ['A_orbit', 'B_orbit', 'A_stab_type', 'B_stab_type',
         'total_pairs', 'output_in_floor', 'output_outside_floor',
         'output_orbit_ids', 'output_counts'],
    )
    write_csv(
        floor_comp_summary,
        EXPORTS / 'floor_layer_stab_composition.csv',
        ['stab_type_A', 'stab_type_B', 'stab_type_output', 'pair_count', 'fraction'],
    )
    write_markdown(
        floor_comp_summary, floor_enriched, pair_rows, closure,
        stab_type_map, n_total, n_in, n_out,
        EXPORTS / 'floor_layer_analysis.md',
    )
    print()

    print("=== SUMMARY ===")
    print(f"Floor layer:    40 orbits (39 Z2xZ2 + 1 (Z2)^3)")
    print(f"Task 2a — Full closure : {out_2ab <= floor_types}")
    print(f"Task 2b — No Trivial   : {'Trivial' not in out_2ab}")
    print(f"Task 2c — both_live={n_both}/40  x1=x2={n_x1x2}/40  c1=c2={n_c1c2}/40")
    pct = f"{100*n_out/n_total:.2f}" if n_total else "N/A"
    print(f"Task 2d — {n_total:,} pairs  exit rate={pct}%  closure adds {len(escaped)} orbit(s)")


if __name__ == '__main__':
    main()
