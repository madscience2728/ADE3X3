"""
ade3x3_step22_export_cross_schema_alignment.py

Export cross-schema raw alignment for CX, XC, AX, BX on raw arity-3.
"""

import csv
from pathlib import Path
from datetime import datetime

# ── Schema -> raw arity-3 mappings ─────────────────────────────────

def _x_raw(x):
    """X local id -> (a_x, b_x)."""
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return 3*r+s, 3*t+u

def _c_raw(c):
    r, u = divmod(c, 3)
    return 3*r+u  # same as c itself

def _a_raw(a):
    r, s = divmod(a, 3)
    return 3*r+s  # same as a

def _b_raw(b):
    t, u = divmod(b, 3)
    return 3*t+u  # same as b

def _x_name(x):
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    return f"X[{r},{s}|{t},{u}]"

def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"

def _a_name(a):
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"

def _b_name(b):
    t, u = divmod(b, 3)
    return f"B[{t},{u}]"

# typed config -> (raw_tuple_id, raw_tuple, readable)
def cx_to_raw(cfg):
    c, x = divmod(cfg, 81)
    ax, bx = _x_raw(x)
    return _raw3_id(c, ax, bx), (c, ax, bx), f"CX[{_c_name(c)},{_x_name(x)}]"

def xc_to_raw(cfg):
    x, c = divmod(cfg, 9)
    ax, bx = _x_raw(x)
    return _raw3_id(ax, bx, c), (ax, bx, c), f"XC[{_x_name(x)},{_c_name(c)}]"

def ax_to_raw(cfg):
    a, x = divmod(cfg, 81)
    ax, bx = _x_raw(x)
    return _raw3_id(a, ax, bx), (a, ax, bx), f"AX[{_a_name(a)},{_x_name(x)}]"

def bx_to_raw(cfg):
    b, x = divmod(cfg, 81)
    ax, bx = _x_raw(x)
    return _raw3_id(b, ax, bx), (b, ax, bx), f"BX[{_b_name(b)},{_x_name(x)}]"

def _raw3_id(s0, s1, s2):
    return s0*81 + s1*9 + s2

def _raw3_decode(tid):
    s2 = tid % 9; tid //= 9
    s1 = tid % 9; tid //= 9
    s0 = tid
    return s0, s1, s2

SCHEMAS = {
    'CX': (729, cx_to_raw, "('C','A_X','B_X')"),
    'XC': (729, xc_to_raw, "('A_X','B_X','C')"),
    'AX': (729, ax_to_raw, "('A','A_X','B_X')"),
    'BX': (729, bx_to_raw, "('B','A_X','B_X')"),
}

# ── Build alignment ────────────────────────────────────────────────

def build_alignment():
    # raw_tid -> {schema: (config_id, readable)}
    raw_map = {}
    for sch, (n, to_raw, _) in SCHEMAS.items():
        for cfg in range(n):
            tid, tup, readable = to_raw(cfg)
            if tid not in raw_map:
                raw_map[tid] = {}
            raw_map[tid][sch] = (cfg, readable)

    rows = []
    for tid in sorted(raw_map):
        s0, s1, s2 = _raw3_decode(tid)
        entry = raw_map[tid]
        row = {
            'raw_tuple_id': tid,
            'raw_tuple': f"({s0},{s1},{s2})",
        }
        for sch in ['CX','XC','AX','BX']:
            if sch in entry:
                row[f'in_{sch}'] = 1
                row[f'{sch.lower()}_config_id'] = entry[sch][0]
                row[f'{sch.lower()}_readable'] = entry[sch][1]
            else:
                row[f'in_{sch}'] = 0
                row[f'{sch.lower()}_config_id'] = ''
                row[f'{sch.lower()}_readable'] = ''
        rows.append(row)
    return rows

# ── Writers ────────────────────────────────────────────────────────

CSV_FIELDS = ['raw_tuple_id','raw_tuple',
              'in_CX','cx_config_id','cx_readable',
              'in_XC','xc_config_id','xc_readable',
              'in_AX','ax_config_id','ax_readable',
              'in_BX','bx_config_id','bx_readable']

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

def write_md(rows, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    L = [
        "# Raw Arity-3 Cross-Schema Alignment",
        "",
        f"Generated: {ts}",
        "",
        "Explicit alignment of CX, XC, AX, BX on the shared raw arity-3 substrate.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Schemas | CX, XC, AX, BX |",
        "| Distinct raw tuples | 729 |",
        "| Aggregate image count | 2,916 |",
        "| Occupancy ratio | 4.0 |",
        "| Full overlap | yes |",
        "",
        "All four schema images coincide exactly on raw arity-3 tuple support.",
        "",
        "## Preview (first 10 rows)",
        "",
        "| raw_tuple | CX | XC | AX | BX |",
        "|-----------|----|----|----|----|",
    ]
    for r in rows[:10]:
        cx = r['cx_readable'] if r['in_CX'] else '-'
        xc = r['xc_readable'] if r['in_XC'] else '-'
        ax = r['ax_readable'] if r['in_AX'] else '-'
        bx = r['bx_readable'] if r['in_BX'] else '-'
        L.append(f"| {r['raw_tuple']} | {cx} | {xc} | {ax} | {bx} |")
    L += ["", "Full data: see `raw3_alignment.csv`.", ""]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Building raw arity-3 cross-schema alignment...")
    rows = build_alignment()

    # Per-schema image sizes
    for sch in ['CX','XC','AX','BX']:
        img = sum(1 for r in rows if r[f'in_{sch}'])
        assert img == 729, f"{sch} image size {img} != 729"
        print(f"  {sch} image size: {img}")

    # Distinct raw tuple count
    distinct = len(rows)
    assert distinct == 729, f"Distinct raw tuples {distinct} != 729"
    print(f"  Distinct raw tuples: {distinct}")

    # Full overlap
    all_cover = all(r['in_CX'] and r['in_XC'] and r['in_AX'] and r['in_BX'] for r in rows)
    assert all_cover, "Not all schemas cover all raw tuples"
    print(f"  Full overlap: yes")

    # Aggregate
    agg = sum(r['in_CX']+r['in_XC']+r['in_AX']+r['in_BX'] for r in rows)
    assert agg == 2916, f"Aggregate {agg} != 2916"
    print(f"  Aggregate image count: {agg}")
    print(f"  Occupancy ratio: {agg/distinct:.1f}")

    # Schema lookup correctness
    for r in rows:
        tid = r['raw_tuple_id']
        for sch, (_, to_raw, _) in SCHEMAS.items():
            cfg = r[f'{sch.lower()}_config_id']
            if cfg != '':
                got_tid, _, _ = to_raw(cfg)
                assert got_tid == tid, \
                    f"{sch} cfg {cfg}: raw {got_tid} != {tid}"

    print("  Lookup correctness: OK")

    csv_path = out / 'raw3_alignment.csv'
    md_path = out / 'raw3_alignment.md'
    write_csv(rows, csv_path)
    write_md(rows, md_path)

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  distinct raw tuples: {distinct}")
    print(f"  aggregate image: {agg}")
    print(f"  occupancy: {agg/distinct:.1f}x")
    print(f"  full overlap: {'yes' if all_cover else 'no'}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
