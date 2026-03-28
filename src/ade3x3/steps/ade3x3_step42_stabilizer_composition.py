"""
ade3x3_step42_stabilizer_composition.py

Stabilizer Composition Analysis: for each pair of CXXC orbits that compose
naturally via a shared CXC face, does the stabilizer type of the inputs
constrain the stabilizer type of the output?

## The composition

CXXC_A = (c1, x1, x2, c2) has right-CXC face = (c1, x2, c2).
CXXC_B = (c1, x2, x4, c2) with left-CXC face = (c1, x2, c2) — same config.
Output = CXXC(c1, x1, x4, c2).

For a fixed CXC interface config m = (c1_m, x_m, c2_m):
  A-configs: { CXXC(c1_m, x1, x_m, c2_m) : x1 in 0..80 } — 81 configs
  B-configs: { CXXC(c1_m, x_m, x4, c2_m) : x4 in 0..80 } — 81 configs
  Pairs: 81 × 81 = 6,561 per CXC config
  Total: 6,561 CXC configs × 6,561 = ~43M pairs

Note: A and B always share the same outer atoms (c1_m, c2_m). The interface
is the middle X atom position. This is the natural "chain" composition
for the CXXC schema.

## Output

For each (stab_type_A, stab_type_B, CXC_interface_orbit) triple, record the
distribution of stab_type of the output CXXC orbit.

Aggregate table (marginalizing over CXC_interface_orbit):
  stab_type_A | stab_type_B | stab_type_output | pair_count

Also record: is the output stabilizer type ALWAYS determined by the input
stabilizer types, or is it sometimes ambiguous?

Provenance: [EXACT_DERIVED]
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

EXPORTS = Path('outputs/exports')

# ──────────────────────────────────────────────────────────────────────────────
# Group
# ──────────────────────────────────────────────────────────────────────────────

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]

# ──────────────────────────────────────────────────────────────────────────────
# Action functions
# ──────────────────────────────────────────────────────────────────────────────

def _act_CXC(cfg, act):
    c1 = cfg // (81*9); rem = cfg % (81*9); x = rem // 9; c2 = rem % 9
    r1,u1 = divmod(c1,3); r2,s = divmod(x//9,3); t,u2 = divmod(x%9,3); r3,u3 = divmod(c2,3)
    return ((3*act[0][r1]+act[2][u1])*81 + 9*(3*act[0][r2]+act[1][s]) +
            (3*act[1][t]+act[2][u2]))*9 + (3*act[0][r3]+act[2][u3])

def _act_CXXC(cfg, act):
    c2 = cfg % 9; rem = cfg // 9; x2 = rem % 81; rem //= 81; x1 = rem % 81; c1 = rem // 81
    r1,u1 = divmod(c1,3); r2,s2 = divmod(x1//9,3); t2,u2 = divmod(x1%9,3)
    r4,s4 = divmod(x2//9,3); t4,u4 = divmod(x2%9,3); r3,u3 = divmod(c2,3)
    c1n = 3*act[0][r1]+act[2][u1]
    x1n = 9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2])
    x2n = 9*(3*act[0][r4]+act[1][s4])+(3*act[1][t4]+act[2][u4])
    c2n = 3*act[0][r3]+act[2][u3]
    return ((c1n*81+x1n)*81+x2n)*9+c2n

# ──────────────────────────────────────────────────────────────────────────────
# Orbit maps
# ──────────────────────────────────────────────────────────────────────────────

def _build_orbit_map(n: int, act_fn) -> list[int]:
    """Return list oid[cfg] (canonical min-rep orbit numbering)."""
    min_r = list(range(n))
    for cfg in range(n):
        m = cfg
        for a in ACTIONS:
            v = act_fn(cfg, a)
            if v < m:
                m = v
        min_r[cfg] = m
    reps = sorted(set(min_r))
    rep_to_oid = {rep: i for i, rep in enumerate(reps)}
    return [rep_to_oid[min_r[cfg]] for cfg in range(n)]

# ──────────────────────────────────────────────────────────────────────────────
# Load stabilizer types from CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_stab_types(schema: str) -> dict[int, str]:
    """Return {orbit_id: stabilizer_type} for the given schema."""
    path = EXPORTS / 'stabilizer_classification.csv'
    if not path.exists():
        raise FileNotFoundError(f"{path} — run step41 first")
    result = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['schema'] == schema:
                result[int(row['orbit_id'])] = row['stabilizer_type']
    return result

# ──────────────────────────────────────────────────────────────────────────────
# CXXC encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_CXXC(c1: int, x1: int, x2: int, c2: int) -> int:
    return ((c1 * 81 + x1) * 81 + x2) * 9 + c2

def encode_CXC(c1: int, x: int, c2: int) -> int:
    return (c1 * 81 + x) * 9 + c2

# ──────────────────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────────────────

TYPE_ORDER = ['Trivial', 'Z2', 'Z2xZ2', '(Z2)^3']

def main():
    print("=== Stabilizer Composition Analysis ===")
    print()

    print("Building orbit maps...")
    print("  CXC (6,561 configs × 216)...")
    cxc_oid = _build_orbit_map(9*81*9, _act_CXC)
    print("  CXXC (531,441 configs × 216)...")
    cxxc_oid = _build_orbit_map(531_441, _act_CXXC)
    n_cxc = len(set(cxc_oid))
    n_cxxc = len(set(cxxc_oid))
    assert n_cxc == 50,   f"CXC: {n_cxc}"
    assert n_cxxc == 2744, f"CXXC: {n_cxxc}"
    print(f"  CXC={n_cxc}, CXXC={n_cxxc} ✓")
    print()

    print("Loading stabilizer types...")
    cxxc_stab = load_stab_types('CXXC')
    assert len(cxxc_stab) == 2744
    print(f"  Loaded {len(cxxc_stab)} CXXC stabilizer types")
    print()

    # ── Core composition ──────────────────────────────────────────────────────
    # For each CXC interface config m = (c1_m, x_m, c2_m):
    #   A_configs: CXXC(c1_m, x1, x_m, c2_m) for x1 in 0..80
    #   B_configs: CXXC(c1_m, x_m, x4, c2_m) for x4 in 0..80
    #   Output:    CXXC(c1_m, x1, x4, c2_m)
    # ──────────────────────────────────────────────────────────────────────────

    print("Scanning all 6,561 CXC interface configs × 81×81 pairs (~43M total)...")

    # Accumulate: (stab_A, stab_B) -> Counter[stab_out]
    pair_counts: dict[tuple, Counter] = defaultdict(Counter)
    # With CXC interface: (cxc_orbit, stab_A, stab_B) -> Counter[stab_out]
    detailed: dict[tuple, Counter] = defaultdict(Counter)
    # (stab_A, stab_B, stab_out) -> distinct (A_orb, B_orb, out_orb) triple count
    triple_counter: Counter = Counter()

    n_processed = 0
    for c1_m in range(9):
        for x_m in range(81):
            for c2_m in range(9):
                cxc_mid_cfg = encode_CXC(c1_m, x_m, c2_m)
                cxc_mid_orb = cxc_oid[cxc_mid_cfg]

                # A_orb for each x1
                a_orbs = [cxxc_oid[encode_CXXC(c1_m, x1, x_m, c2_m)] for x1 in range(81)]
                # B_orb for each x4
                b_orbs = [cxxc_oid[encode_CXXC(c1_m, x_m, x4, c2_m)] for x4 in range(81)]
                # output_orb for each (x1, x4)
                out_orbs = [[cxxc_oid[encode_CXXC(c1_m, x1, x4, c2_m)]
                              for x4 in range(81)]
                             for x1 in range(81)]

                for x1 in range(81):
                    a_orb = a_orbs[x1]
                    st_a  = cxxc_stab[a_orb]
                    out_row = out_orbs[x1]
                    for x4 in range(81):
                        b_orb = b_orbs[x4]
                        st_b  = cxxc_stab[b_orb]
                        out_orb = out_row[x4]
                        st_out = cxxc_stab[out_orb]
                        pair_counts[(st_a, st_b)][st_out] += 1
                        detailed[(cxc_mid_orb, st_a, st_b)][st_out] += 1
                        triple_counter[(st_a, st_b, st_out)] += 1

                n_processed += 1
                if n_processed % 1000 == 0:
                    print(f"  {n_processed}/6561 CXC configs done...", end='\r')

    print(f"  {n_processed}/6561 CXC configs done.      ")
    print()

    # Verify total
    total_pairs = sum(c for cnt in pair_counts.values() for c in cnt.values())
    print(f"  Total composition pairs: {total_pairs:,}  (expect {6561*81*81:,})")
    assert total_pairs == 6561 * 81 * 81, f"Mismatch: {total_pairs}"

    # ── Build output rows ─────────────────────────────────────────────────────
    csv_rows = []
    for (st_a, st_b) in sorted(pair_counts.keys(), key=lambda x: (TYPE_ORDER.index(x[0]) if x[0] in TYPE_ORDER else 99, TYPE_ORDER.index(x[1]) if x[1] in TYPE_ORDER else 99)):
        cnt = pair_counts[(st_a, st_b)]
        total = sum(cnt.values())
        for st_out in sorted(cnt.keys(), key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99):
            csv_rows.append({
                'stab_type_A': st_a,
                'stab_type_B': st_b,
                'stab_type_output': st_out,
                'pair_count': cnt[st_out],
                'fraction': cnt[st_out] / total,
                'total_pairs_for_AB': total,
            })

    # Detailed rows (with CXC interface orbit)
    detailed_rows = []
    for (cxc_orb, st_a, st_b) in sorted(
        detailed.keys(),
        key=lambda x: (x[0], TYPE_ORDER.index(x[1]) if x[1] in TYPE_ORDER else 99,
                       TYPE_ORDER.index(x[2]) if x[2] in TYPE_ORDER else 99)
    ):
        cnt = detailed[(cxc_orb, st_a, st_b)]
        total = sum(cnt.values())
        for st_out, n in cnt.items():
            detailed_rows.append({
                'cxc_interface_orbit': cxc_orb,
                'stab_type_A': st_a,
                'stab_type_B': st_b,
                'stab_type_output': st_out,
                'pair_count': n,
                'fraction': n / total,
            })

    # ── Determine predictability ──────────────────────────────────────────────
    # For each (stab_A, stab_B), is the output type UNIQUE?
    predictability = {}
    for (st_a, st_b), cnt in pair_counts.items():
        output_types = set(cnt.keys())
        predictability[(st_a, st_b)] = {
            'unique': len(output_types) == 1,
            'output_types': sorted(output_types, key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99),
            'total': sum(cnt.values()),
        }

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("Writing outputs...")

    # Summary CSV
    summary_path = EXPORTS / 'stabilizer_composition_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['stab_type_A', 'stab_type_B', 'stab_type_output',
                  'pair_count', 'fraction', 'total_pairs_for_AB']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  Wrote {len(csv_rows)} rows -> {summary_path}")

    # Detailed CSV
    detailed_path = EXPORTS / 'stabilizer_composition_detailed.csv'
    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['cxc_interface_orbit', 'stab_type_A', 'stab_type_B',
                  'stab_type_output', 'pair_count', 'fraction']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detailed_rows)
    print(f"  Wrote {len(detailed_rows)} rows -> {detailed_path}")

    # Markdown
    md_path = EXPORTS / 'stabilizer_composition.md'
    _write_markdown(md_path, pair_counts, predictability, triple_counter, total_pairs)
    print(f"  Wrote markdown -> {md_path}")

    print()
    print("=== Done ===")
    print()
    print("KEY RESULT:")
    for (st_a, st_b), info in sorted(predictability.items(), key=lambda x: (TYPE_ORDER.index(x[0][0]) if x[0][0] in TYPE_ORDER else 99, TYPE_ORDER.index(x[0][1]) if x[0][1] in TYPE_ORDER else 99)):
        unique = "UNIQUE" if info['unique'] else "MIXED"
        print(f"  ({st_a}, {st_b}) -> {info['output_types']}  [{unique}]")


def _write_markdown(path, pair_counts, predictability, triple_counter, total_pairs):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# Stabilizer Composition Analysis: CXXC",
        "",
        f"Generated: {ts}",
        "",
        "[EXACT_DERIVED]",
        "",
        "## Composition Definition",
        "",
        "For a fixed CXC interface config m = (c1_m, x_m, c2_m):",
        "  - A = CXXC(c1_m, x1, x_m, c2_m)  [right-CXC face = m]",
        "  - B = CXXC(c1_m, x_m, x4, c2_m)  [left-CXC face = m]",
        "  - Output = CXXC(c1_m, x1, x4, c2_m)",
        "",
        "The composition contracts through the shared middle X atom.",
        "Total pairs: 6,561 CXC configs × 81 × 81 = 43,046,721",
        "",
        f"Verified total: {total_pairs:,}",
        "",
        "## Predictability Table",
        "",
        "For each (stab_type_A, stab_type_B) pair, what output stabilizer types occur?",
        "",
        "| stab_type_A | stab_type_B | Output Types | Predictable? |",
        "|-------------|-------------|--------------|--------------|",
    ]
    for (st_a, st_b), info in sorted(
        predictability.items(),
        key=lambda x: (TYPE_ORDER.index(x[0][0]) if x[0][0] in TYPE_ORDER else 99,
                       TYPE_ORDER.index(x[0][1]) if x[0][1] in TYPE_ORDER else 99)
    ):
        out_str = ", ".join(info['output_types'])
        pred = "Yes" if info['unique'] else "No"
        lines.append(f"| {st_a} | {st_b} | {out_str} | {pred} |")

    lines += [
        "",
        "## Full Distribution Table",
        "",
        "| stab_type_A | stab_type_B | stab_type_output | pair_count | fraction |",
        "|-------------|-------------|------------------|------------|----------|",
    ]
    for (st_a, st_b) in sorted(
        pair_counts.keys(),
        key=lambda x: (TYPE_ORDER.index(x[0]) if x[0] in TYPE_ORDER else 99,
                       TYPE_ORDER.index(x[1]) if x[1] in TYPE_ORDER else 99)
    ):
        cnt = pair_counts[(st_a, st_b)]
        total_ab = sum(cnt.values())
        for st_out in sorted(cnt.keys(), key=lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99):
            frac = cnt[st_out] / total_ab
            lines.append(f"| {st_a} | {st_b} | {st_out} | {cnt[st_out]:,} | {frac:.6f} |")

    lines += ["", "---", "*Generated by ade3x3_step42_stabilizer_composition.py*", ""]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).resolve().parents[3])
    main()
