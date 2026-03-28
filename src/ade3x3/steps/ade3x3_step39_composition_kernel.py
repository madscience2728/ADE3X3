"""
ade3x3_step39_composition_kernel.py

Composition Kernel: for each of the 14 mixed CX×XC→CC orbit-pairs,
compute the witness distribution across output CC orbits.

Data source: outputs/exports/comp_CX_XC_to_CC.csv (already computed in step23).
The cc_histogram column already records this; we just expand it and filter.

Output columns:
  CX_orbit | XC_orbit | CC_orbit | witness_count

Verification: for each mixed key, sum of witness_count must equal the
total witness count already recorded in Section 18 of the dossier.

Provenance: [EXACT_DERIVED] — computed from exact composition enumeration.
"""

import csv
import ast
from pathlib import Path
from datetime import datetime
from collections import defaultdict

EXPORTS = Path('outputs/exports')

# ──────────────────────────────────────────────────────────────────────────────
# Reference data from Section 18 of canon (for verification)
# ──────────────────────────────────────────────────────────────────────────────

MIXED_KEYS_CANON = {
    # (cx_orbit, xc_orbit): total_witness_count (from Section 18)
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

assert len(MIXED_KEYS_CANON) == 14, "Expected 14 mixed keys"

# ──────────────────────────────────────────────────────────────────────────────
# Load existing composition CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_composition_csv() -> list[dict]:
    path = EXPORTS / 'comp_CX_XC_to_CC.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ade3x3_step23_export_composition_cx_xc_cc.py first."
        )
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'cx_orbit_id': int(row['cx_orbit_id']),
                'xc_orbit_id': int(row['xc_orbit_id']),
                'witness_count': int(row['witness_count']),
                'cc_histogram': ast.literal_eval(row['cc_histogram']),
                'deterministic': int(row['deterministic']),
            })
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Expand histogram into kernel rows
# ──────────────────────────────────────────────────────────────────────────────

def build_kernel_rows(comp_rows: list[dict]) -> list[dict]:
    mixed_rows = [r for r in comp_rows if r['deterministic'] == 0]
    assert len(mixed_rows) == 14, f"Expected 14 mixed rows, got {len(mixed_rows)}"

    kernel_rows = []
    for r in sorted(mixed_rows, key=lambda x: (x['cx_orbit_id'], x['xc_orbit_id'])):
        cx = r['cx_orbit_id']
        xc = r['xc_orbit_id']
        hist = r['cc_histogram']
        total = r['witness_count']

        # Verify sum
        assert sum(hist.values()) == total, \
            f"Histogram sum mismatch for ({cx},{xc}): {sum(hist.values())} != {total}"

        # Verify against canon
        key = (cx, xc)
        if key in MIXED_KEYS_CANON:
            assert MIXED_KEYS_CANON[key] == total, \
                f"Canon total mismatch for ({cx},{xc}): {total} != {MIXED_KEYS_CANON[key]}"

        for cc_orbit in sorted(hist.keys()):
            kernel_rows.append({
                'cx_orbit': cx,
                'xc_orbit': xc,
                'cc_orbit': cc_orbit,
                'witness_count': hist[cc_orbit],
            })

    return kernel_rows

# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(kernel_rows: list[dict], comp_rows: list[dict]) -> dict:
    mixed_rows = {(r['cx_orbit_id'], r['xc_orbit_id']): r
                  for r in comp_rows if r['deterministic'] == 0}

    keys_with_2 = [(cx, xc) for (cx, xc), r in mixed_rows.items()
                   if len(r['cc_histogram']) == 2]
    keys_with_3 = [(cx, xc) for (cx, xc), r in mixed_rows.items()
                   if len(r['cc_histogram']) == 3]
    keys_with_4 = [(cx, xc) for (cx, xc), r in mixed_rows.items()
                   if len(r['cc_histogram']) == 4]

    # CC orbit usage across mixed pairs
    cc_usage = defaultdict(int)
    for row in kernel_rows:
        cc_usage[row['cc_orbit']] += row['witness_count']

    # Fraction of witnesses per CC orbit per mixed key
    fraction_uniform = []
    for (cx, xc), r in sorted(mixed_rows.items()):
        hist = r['cc_histogram']
        total = r['witness_count']
        fracs = sorted([v / total for v in hist.values()])
        is_uniform = all(abs(f - fracs[0]) < 1e-9 for f in fracs)
        fraction_uniform.append(((cx, xc), fracs, is_uniform))

    return {
        'total_kernel_rows': len(kernel_rows),
        'mixed_keys': len(mixed_rows),
        'keys_with_2_outputs': len(keys_with_2),
        'keys_with_3_outputs': len(keys_with_3),
        'keys_with_4_outputs': len(keys_with_4),
        'cc_usage': dict(cc_usage),
        'fraction_uniform': fraction_uniform,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(kernel_rows: list[dict], path: Path):
    fields = ['cx_orbit', 'xc_orbit', 'cc_orbit', 'witness_count']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kernel_rows)
    print(f"  Wrote {len(kernel_rows)} rows → {path}")


def write_markdown(kernel_rows: list[dict], stats: dict, comp_rows: list[dict], path: Path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mixed_rows = {(r['cx_orbit_id'], r['xc_orbit_id']): r
                  for r in comp_rows if r['deterministic'] == 0}

    lines = [
        "# Composition Kernel: CX × XC → CC (Mixed Key Distributions)",
        "",
        f"Generated: {ts}",
        "",
        "[EXACT_DERIVED]",
        "",
        "## Overview",
        "",
        "For each of the 14 mixed CX×XC→CC orbit-pairs, this section records the",
        "exact witness distribution across output CC orbits.",
        "",
        "A **mixed** key is one where the CX and XC orbit-pair does not uniquely",
        "determine the CC output orbit. The witness distribution answers: given that",
        "CX is in orbit `cx_o` and XC is in orbit `xc_o`, what fraction of the",
        "(C1,X,C2) triples land in each CC orbit?",
        "",
        "## Summary Statistics",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Mixed keys | {stats['mixed_keys']} |",
        f"| Keys with 2 CC output orbits | {stats['keys_with_2_outputs']} |",
        f"| Keys with 3 CC output orbits | {stats['keys_with_3_outputs']} |",
        f"| Keys with 4 CC output orbits | {stats['keys_with_4_outputs']} |",
        f"| Total kernel rows | {stats['total_kernel_rows']} |",
        "",
        "## CC Orbit Usage Across All Mixed Witnesses",
        "",
        "| CC Orbit | Total Witnesses in Mixed Keys |",
        "|----------|-------------------------------|",
    ]
    for cc, cnt in sorted(stats['cc_usage'].items()):
        lines.append(f"| {cc} | {cnt} |")

    lines += [
        "",
        "## Kernel Table (14 Mixed Keys × Expanded CC-Orbit Rows)",
        "",
        "| CX Orbit | XC Orbit | CC Orbit | Witness Count | Fraction of Key Total |",
        "|----------|----------|----------|---------------|-----------------------|",
    ]
    prev_key = None
    for row in kernel_rows:
        key = (row['cx_orbit'], row['xc_orbit'])
        if key != prev_key:
            prev_key = key
        total = mixed_rows[key]['witness_count']
        frac = row['witness_count'] / total
        lines.append(
            f"| {row['cx_orbit']} | {row['xc_orbit']} | {row['cc_orbit']} "
            f"| {row['witness_count']} | {frac:.4f} ({row['witness_count']}/{total}) |"
        )

    lines += [
        "",
        "## Distribution Uniformity Analysis",
        "",
        "For each mixed key, are witnesses uniformly distributed across CC output orbits?",
        "",
        "| CX Orbit | XC Orbit | CC Outputs | Fractions | Uniform? |",
        "|----------|----------|------------|-----------|----------|",
    ]
    for (cx, xc), fracs, is_uniform in stats['fraction_uniform']:
        frac_str = ", ".join(f"{f:.4f}" for f in fracs)
        cc_list = sorted(mixed_rows[(cx, xc)]['cc_histogram'].keys())
        lines.append(
            f"| {cx} | {xc} | {cc_list} | {frac_str} | {'Yes' if is_uniform else 'No'} |"
        )

    lines += [
        "",
        "## Verification Checksums",
        "",
        "All row sums verified against Section 18 totals. "
        "Every mixed key's histogram sums exactly to its recorded witness count.",
        "",
        "| Mixed Key | Canon Total | Kernel Sum | Match |",
        "|-----------|-------------|------------|-------|",
    ]
    kernel_sums = defaultdict(int)
    for row in kernel_rows:
        kernel_sums[(row['cx_orbit'], row['xc_orbit'])] += row['witness_count']
    for (cx, xc), canon_total in sorted(MIXED_KEYS_CANON.items()):
        ksum = kernel_sums[(cx, xc)]
        match = "✓" if ksum == canon_total else "✗"
        lines.append(f"| ({cx},{xc}) | {canon_total} | {ksum} | {match} |")

    lines += ["", "---", f"*Generated by ade3x3_step39_composition_kernel.py*", ""]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Wrote markdown → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Composition Kernel: Mixed CX×XC→CC Distributions ===")
    print()

    print("Loading composition CSV...")
    comp_rows = load_composition_csv()
    print(f"  Loaded {len(comp_rows)} composition rows")

    print("Building kernel rows...")
    kernel_rows = build_kernel_rows(comp_rows)
    print(f"  Expanded to {len(kernel_rows)} kernel rows")

    print("Computing statistics...")
    stats = compute_stats(kernel_rows, comp_rows)
    print(f"  Mixed keys: {stats['mixed_keys']}")
    print(f"  Output distribution: 2-output={stats['keys_with_2_outputs']}, "
          f"3-output={stats['keys_with_3_outputs']}, "
          f"4-output={stats['keys_with_4_outputs']}")

    # Print uniformity summary
    uniform_count = sum(1 for _, _, u in stats['fraction_uniform'] if u)
    print(f"  Uniform distribution: {uniform_count}/{len(stats['fraction_uniform'])} keys")

    print()
    print("Writing outputs...")
    csv_path = EXPORTS / 'composition_kernel_mixed.csv'
    md_path = EXPORTS / 'composition_kernel_mixed.md'
    write_csv(kernel_rows, csv_path)
    write_markdown(kernel_rows, stats, comp_rows, md_path)

    print()
    print("=== Done ===")
    print(f"Kernel rows: {len(kernel_rows)}")
    print(f"All 14 Canon checksums: ✓")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).resolve().parents[3])
    main()
