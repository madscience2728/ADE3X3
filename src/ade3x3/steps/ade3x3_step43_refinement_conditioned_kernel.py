"""
ade3x3_step43_refinement_conditioned_kernel.py

Refinement-Conditioned Composition Kernel:
For each of the 14 mixed CX × XC → CC orbit-pairs, stratify witnesses by
the (s, t) refinement coordinates of the shared X atom, then check whether
the CC-orbit distribution is still uniform within each stratum.

Background:
  Section 27 (Step 39) shows all 14 mixed keys split CC-orbit witnesses
  perfectly uniformly at orbit level. This step checks whether conditioning
  on the refinement coordinate of the shared X atom breaks that uniformity.

Encoding:
  X atom x ∈ {0..80}: x = 9*(3*r+s) + (3*t+u)
    r = (x // 9) // 3   (outer row index, acted on by first S3)
    s = (x // 9) % 3    (shared/left index, acted on by second S3)
    t = (x % 9)  // 3   (shared/right index, acted on by second S3)
    u = (x % 9)  % 3    (outer col index, acted on by third S3)

  The shared X atom contributes (s, t) as the "summation index" pair.
  s and t each range over {0, 1, 2}, giving 9 possible (s, t) strata.

Output columns:
  cx_orbit | xc_orbit | s_shared | t_shared | cc_orbit | witness_count
  + derived: is_uniform_within_stratum, stratum_total_witnesses

Verification:
  For each mixed key (cx_orbit, xc_orbit), sum of witness_count across all
  (s, t) strata and cc_orbits must equal Section 18 / Section 27 total.

Provenance: [EXACT_DERIVED]
"""

import csv
import ast
from pathlib import Path
from datetime import datetime
from collections import defaultdict

EXPORTS = Path('outputs/exports')

# ──────────────────────────────────────────────────────────────────────────────
# Reference data from Sections 18 / 27 (for verification)
# ──────────────────────────────────────────────────────────────────────────────

MIXED_KEYS_CANON = {
    (1, 1): 108,
    (1, 3): 216,
    (3, 5): 216,
    (3, 7): 432,
    (4, 2): 108,
    (4, 3): 216,
    (5, 1): 216,
    (5, 2): 216,
    (5, 3): 432,
    (6, 6): 216,
    (6, 7): 432,
    (7, 5): 432,
    (7, 6): 432,
    (7, 7): 864,
}
assert len(MIXED_KEYS_CANON) == 14

# ──────────────────────────────────────────────────────────────────────────────
# X atom decoding
# ──────────────────────────────────────────────────────────────────────────────

def decode_x(x: int) -> tuple[int, int, int, int]:
    """Return (r, s, t, u) coordinates of X atom index x ∈ {0..80}."""
    hi = x // 9     # = 3*r + s
    lo = x % 9      # = 3*t + u
    r = hi // 3
    s = hi % 3
    t = lo // 3
    u = lo % 3
    return r, s, t, u

# ──────────────────────────────────────────────────────────────────────────────
# Load witness data
# ──────────────────────────────────────────────────────────────────────────────

def load_witnesses() -> list[dict]:
    path = EXPORTS / 'comp_CX_XC_to_CC_witnesses.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ade3x3_step23_export_composition_cx_xc_cc.py first."
        )
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cx_o = int(row['cx_orbit_id'])
            xc_o = int(row['xc_orbit_id'])
            if (cx_o, xc_o) not in MIXED_KEYS_CANON:
                continue
            x   = int(row['x_local_id'])
            cc_o = int(row['cc_orbit_id'])
            _, s, t, _ = decode_x(x)
            rows.append({
                'cx_orbit': cx_o,
                'xc_orbit': xc_o,
                's_shared': s,
                't_shared': t,
                'cc_orbit': cc_o,
                'x_local_id': x,
            })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Aggregate
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(witnesses: list[dict]) -> list[dict]:
    """
    Group by (cx_orbit, xc_orbit, s_shared, t_shared, cc_orbit) → count.
    Then annotate each stratum with:
      - stratum_total: sum over cc_orbit for this (cx, xc, s, t)
      - is_uniform: all CC orbit counts in this stratum are equal
      - fraction: witness_count / stratum_total
    """
    # (cx, xc, s, t, cc) -> count
    counter: dict[tuple, int] = defaultdict(int)
    for w in witnesses:
        key = (w['cx_orbit'], w['xc_orbit'], w['s_shared'], w['t_shared'], w['cc_orbit'])
        counter[key] += 1

    # Stratum totals: (cx, xc, s, t) -> total witnesses
    strat_total: dict[tuple, int] = defaultdict(int)
    strat_cc_counts: dict[tuple, list] = defaultdict(list)
    for (cx, xc, s, t, cc), cnt in counter.items():
        strat_total[(cx, xc, s, t)] += cnt
        strat_cc_counts[(cx, xc, s, t)].append(cnt)

    rows = []
    for (cx, xc, s, t, cc) in sorted(counter.keys()):
        cnt = counter[(cx, xc, s, t, cc)]
        total = strat_total[(cx, xc, s, t)]
        cc_vals = strat_cc_counts[(cx, xc, s, t)]
        is_uniform = (len(set(cc_vals)) == 1)
        rows.append({
            'cx_orbit': cx,
            'xc_orbit': xc,
            's_shared': s,
            't_shared': t,
            'cc_orbit': cc,
            'witness_count': cnt,
            'stratum_total': total,
            'fraction': f"{cnt}/{total}",
            'is_uniform_within_stratum': is_uniform,
        })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

def verify(rows: list[dict]) -> None:
    key_totals: dict[tuple, int] = defaultdict(int)
    for r in rows:
        key_totals[(r['cx_orbit'], r['xc_orbit'])] += r['witness_count']

    for key, expected in MIXED_KEYS_CANON.items():
        got = key_totals.get(key, 0)
        assert got == expected, (
            f"Witness count mismatch for {key}: got {got}, expected {expected}"
        )
    print("  Verification: all 14 mixed key totals match Section 18/27 ✓")

# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(rows: list[dict]) -> dict:
    """Summarize uniformity findings across all strata."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r['cx_orbit'], r['xc_orbit'], r['s_shared'], r['t_shared'])].append(r)

    n_strata = len(strata)
    n_uniform = sum(1 for rows_s in strata.values() if rows_s[0]['is_uniform_within_stratum'])
    n_nonuniform = n_strata - n_uniform

    # Identify keys where the stratum breaks uniformity
    nonuniform_keys: set[tuple] = set()
    for (cx, xc, s, t), rows_s in strata.items():
        if not rows_s[0]['is_uniform_within_stratum']:
            nonuniform_keys.add((cx, xc))

    # Count strata per key
    strata_per_key: dict[tuple, int] = defaultdict(int)
    for (cx, xc, s, t) in strata:
        strata_per_key[(cx, xc)] += 1

    # Count empty strata (s,t) combos that have 0 witnesses for this key
    # Total (s,t) pairs = 9; some may not appear for certain keys
    occupied_strata_per_key: dict[tuple, set] = defaultdict(set)
    for (cx, xc, s, t) in strata:
        occupied_strata_per_key[(cx, xc)].add((s, t))

    return {
        'n_strata_total': n_strata,
        'n_uniform': n_uniform,
        'n_nonuniform': n_nonuniform,
        'nonuniform_keys': sorted(nonuniform_keys),
        'strata_per_key': dict(strata_per_key),
        'occupied_strata_per_key': {k: sorted(v) for k, v in occupied_strata_per_key.items()},
    }

# ──────────────────────────────────────────────────────────────────────────────
# Stratum summary table (one row per (cx, xc, s, t))
# ──────────────────────────────────────────────────────────────────────────────

def build_stratum_summary(rows: list[dict]) -> list[dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r['cx_orbit'], r['xc_orbit'], r['s_shared'], r['t_shared'])].append(r)

    summary = []
    for (cx, xc, s, t) in sorted(strata.keys()):
        rows_s = sorted(strata[(cx, xc, s, t)], key=lambda x: x['cc_orbit'])
        total = rows_s[0]['stratum_total']
        cc_counts = {r['cc_orbit']: r['witness_count'] for r in rows_s}
        n_cc = len(cc_counts)
        counts = list(cc_counts.values())
        is_uniform = (len(set(counts)) == 1)
        summary.append({
            'cx_orbit': cx,
            'xc_orbit': xc,
            's_shared': s,
            't_shared': t,
            'stratum_total': total,
            'n_cc_orbits': n_cc,
            'cc_counts': str(cc_counts),
            'is_uniform': is_uniform,
        })
    return summary

# ──────────────────────────────────────────────────────────────────────────────
# Write CSV
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Write markdown
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown(
    rows: list[dict],
    strat_summary: list[dict],
    stats: dict,
    path: Path,
) -> None:
    lines: list[str] = []
    w = lines.append

    w("# Refinement-Conditioned Composition Kernel: CX × XC → CC")
    w("")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w(f"Source: comp_CX_XC_to_CC_witnesses.csv (step 23)")
    w("")
    w("## Summary")
    w("")
    w(f"- Mixed keys: 14")
    w(f"- Total strata (cx,xc,s,t) occupied: {stats['n_strata_total']}")
    w(f"- Uniform strata: {stats['n_uniform']}")
    w(f"- Non-uniform strata: {stats['n_nonuniform']}")
    if stats['nonuniform_keys']:
        w(f"- Keys with at least one non-uniform stratum: {stats['nonuniform_keys']}")
    else:
        w("- Keys with at least one non-uniform stratum: NONE (all strata uniform)")
    w("")

    w("## Stratum Summary (one row per (cx_orbit, xc_orbit, s_shared, t_shared))")
    w("")
    w("| cx_orbit | xc_orbit | s | t | stratum_total | n_cc_orbits | cc_counts | uniform? |")
    w("|----------|----------|---|---|---------------|-------------|-----------|----------|")
    for r in strat_summary:
        flag = "YES" if r['is_uniform'] else "NO"
        w(f"| {r['cx_orbit']} | {r['xc_orbit']} | {r['s_shared']} | {r['t_shared']} | "
          f"{r['stratum_total']} | {r['n_cc_orbits']} | {r['cc_counts']} | {flag} |")
    w("")

    w("## Detailed Kernel Rows (per stratum and CC orbit)")
    w("")
    w("| cx_orbit | xc_orbit | s | t | cc_orbit | witness_count | stratum_total | fraction | uniform? |")
    w("|----------|----------|---|---|----------|---------------|---------------|----------|----------|")
    for r in rows:
        flag = "YES" if r['is_uniform_within_stratum'] else "NO"
        w(f"| {r['cx_orbit']} | {r['xc_orbit']} | {r['s_shared']} | {r['t_shared']} | "
          f"{r['cc_orbit']} | {r['witness_count']} | {r['stratum_total']} | "
          f"{r['fraction']} | {flag} |")
    w("")

    w("## Key Occupied Strata per Mixed Key")
    w("")
    for key in sorted(MIXED_KEYS_CANON.keys()):
        cx, xc = key
        occupied = stats['occupied_strata_per_key'].get((cx, xc), [])
        w(f"- CX={cx}, XC={xc}: {len(occupied)} strata — (s,t) values: {occupied}")
    w("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Wrote markdown → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Refinement-Conditioned Composition Kernel: CX × XC → CC ===")
    print()

    print("Loading witnesses from comp_CX_XC_to_CC_witnesses.csv...")
    witnesses = load_witnesses()
    print(f"  Loaded {len(witnesses)} witness rows for the 14 mixed keys")
    print()

    print("Aggregating by (cx_orbit, xc_orbit, s_shared, t_shared, cc_orbit)...")
    rows = aggregate(witnesses)
    print(f"  {len(rows)} kernel rows across all strata")
    print()

    print("Verifying against Section 18/27 totals...")
    verify(rows)
    print()

    print("Computing statistics...")
    stats = compute_stats(rows)
    strat_summary = build_stratum_summary(rows)
    print(f"  Total strata occupied: {stats['n_strata_total']}")
    print(f"  Uniform strata:     {stats['n_uniform']}")
    print(f"  Non-uniform strata: {stats['n_nonuniform']}")
    if stats['nonuniform_keys']:
        print(f"  Non-uniform keys: {stats['nonuniform_keys']}")
    else:
        print("  ALL strata uniform — conditioning on (s,t) does NOT break uniformity")
    print()

    # Write outputs
    csv_path = EXPORTS / 'refinement_conditioned_kernel.csv'
    strat_path = EXPORTS / 'refinement_conditioned_kernel_strata.csv'
    md_path  = EXPORTS / 'refinement_conditioned_kernel.md'

    write_csv(rows, csv_path, [
        'cx_orbit', 'xc_orbit', 's_shared', 't_shared',
        'cc_orbit', 'witness_count', 'stratum_total', 'fraction',
        'is_uniform_within_stratum',
    ])
    write_csv(strat_summary, strat_path, [
        'cx_orbit', 'xc_orbit', 's_shared', 't_shared',
        'stratum_total', 'n_cc_orbits', 'cc_counts', 'is_uniform',
    ])
    write_markdown(rows, strat_summary, stats, md_path)
    print()

    # Final print: uniformity verdict
    print("=== VERDICT ===")
    if stats['n_nonuniform'] == 0:
        print("ALL strata are uniform. Conditioning on the (s,t) refinement coordinate")
        print("of the shared X atom does NOT break the orbit-level CC uniformity.")
        print("=> The refinement-coordination hypothesis is CLOSED at this level.")
    else:
        print(f"{stats['n_nonuniform']} / {stats['n_strata_total']} strata are non-uniform.")
        print(f"Non-uniform keys: {stats['nonuniform_keys']}")
        print("=> Conditioning on (s,t) DOES reveal sub-orbit structure.")
    print()

    # Print a few example strata for inspection
    print("=== Sample strata (first key CX=1, XC=1) ===")
    for r in strat_summary:
        if r['cx_orbit'] == 1 and r['xc_orbit'] == 1:
            print(f"  (s={r['s_shared']}, t={r['t_shared']}): total={r['stratum_total']}, "
                  f"cc_counts={r['cc_counts']}, uniform={r['is_uniform']}")


if __name__ == '__main__':
    main()
