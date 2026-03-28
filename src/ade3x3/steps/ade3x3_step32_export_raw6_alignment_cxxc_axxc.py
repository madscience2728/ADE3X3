"""
ade3x3_step32_export_raw6_alignment_cxxc_axxc.py

Export raw arity-6 alignment between CXXC and AXXC.
"""

import csv
from pathlib import Path
from datetime import datetime


def c_name(c: int) -> str:
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"


def a_name(a: int) -> str:
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"


def x_name(x: int) -> str:
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return f"X[{r},{s}|{t},{u}]"


def decode_raw6(tid: int):
    vals = [0] * 6
    for i in range(5, -1, -1):
        vals[i] = tid % 9
        tid //= 9
    return tuple(vals)


def cxxc_config_id(c1: int, a1: int, b1: int, a2: int, b2: int, c2: int) -> int:
    x1 = 9 * a1 + b1
    x2 = 9 * a2 + b2
    return (((c1 * 81) + x1) * 81 + x2) * 9 + c2


def axxc_config_id(a: int, a1: int, b1: int, a2: int, b2: int, c: int) -> int:
    x1 = 9 * a1 + b1
    x2 = 9 * a2 + b2
    return (((a * 81) + x1) * 81 + x2) * 9 + c


def cxxc_readable(c1: int, a1: int, b1: int, a2: int, b2: int, c2: int) -> str:
    return f"CXXC[{c_name(c1)},{x_name(9*a1+b1)},{x_name(9*a2+b2)},{c_name(c2)}]"


def axxc_readable(a: int, a1: int, b1: int, a2: int, b2: int, c: int) -> str:
    return f"AXXC[{a_name(a)},{x_name(9*a1+b1)},{x_name(9*a2+b2)},{c_name(c)}]"


def main():
    out = Path("exports")
    out.mkdir(exist_ok=True)
    csv_path = out / "raw6_alignment_CXXC_AXXC.csv"
    md_path = out / "raw6_alignment_CXXC_AXXC.md"

    fields = [
        "raw_tuple_id", "raw_tuple",
        "in_CXXC", "cxxc_config_id", "cxxc_readable",
        "in_AXXC", "axxc_config_id", "axxc_readable",
    ]

    total = 9 ** 6
    preview = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        batch = []
        for tid in range(total):
            c1, a1, b1, a2, b2, c2 = decode_raw6(tid)
            row = {
                "raw_tuple_id": tid,
                "raw_tuple": f"({c1},{a1},{b1},{a2},{b2},{c2})",
                "in_CXXC": 1,
                "cxxc_config_id": cxxc_config_id(c1, a1, b1, a2, b2, c2),
                "cxxc_readable": cxxc_readable(c1, a1, b1, a2, b2, c2),
                "in_AXXC": 1,
                "axxc_config_id": axxc_config_id(c1, a1, b1, a2, b2, c2),
                "axxc_readable": axxc_readable(c1, a1, b1, a2, b2, c2),
            }
            if len(preview) < 10:
                preview.append(row)
            batch.append(row)
            if len(batch) >= 10000:
                w.writerows(batch)
                batch.clear()
        if batch:
            w.writerows(batch)

    # Sanity checks
    assert total == 531441
    assert cxxc_config_id(0, 0, 0, 0, 0, 0) == 0
    assert axxc_config_id(0, 0, 0, 0, 0, 0) == 0
    assert cxxc_config_id(8, 8, 8, 8, 8, 8) == total - 1
    assert axxc_config_id(8, 8, 8, 8, 8, 8) == total - 1

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = [
        "# Raw Arity-6 Alignment: CXXC and AXXC",
        "",
        f"Generated: {ts}",
        "",
        "This export records the explicit raw-arity-6 alignment between the two arity-4 schemas:",
        "- CXXC = C × X × X × C",
        "- AXXC = A × X × X × C",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Raw arity-6 tuple count | {total:,} |",
        f"| CXXC image size | {total:,} |",
        f"| AXXC image size | {total:,} |",
        f"| Distinct raw tuples in union | {total:,} |",
        f"| Overlap size | {total:,} |",
        "| Support relationship | exact coincidence |",
        "| Occupancy ratio | 2.0 |",
        "",
        "Every raw arity-6 tuple is occupied by both schemas under different role overlays.",
        "",
        "## Preview (first 10 rows)",
        "",
        "| raw_tuple_id | raw_tuple | cxxc_config_id | axxc_config_id |",
        "|--------------|-----------|----------------|----------------|",
    ]
    for row in preview:
        md.append(f"| {row['raw_tuple_id']} | {row['raw_tuple']} | {row['cxxc_config_id']} | {row['axxc_config_id']} |")
    md += ["", "Full data: see `raw6_alignment_CXXC_AXXC.csv`.", ""]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("Raw arity-6 alignment complete")
    print(f"  Raw tuples: {total}")
    print(f"  CXXC image size: {total}")
    print(f"  AXXC image size: {total}")
    print(f"  Overlap size: {total}")
    print("  Support relationship: exact coincidence")
    print(f"\nFiles written:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
