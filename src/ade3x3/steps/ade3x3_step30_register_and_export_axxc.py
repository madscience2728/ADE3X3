"""
ade3x3_step30_register_and_export_axxc.py

Register and export AXXC = A × X × X × C, second arity-4 typed schema.
Full population: 9 * 81 * 81 * 9 = 531,441 rows.
"""

import csv
from pathlib import Path
from datetime import datetime

def _a_name(a):
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"

def _x_name(x):
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    return f"X[{r},{s}|{t},{u}]"

def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"

def _a_idx(v):
    r, s = divmod(v, 3)
    return 3*r+s

def _x_a_idx(x):
    r, s = divmod(x//9, 3)
    return 3*r+s

def _x_b_idx(x):
    t, u = divmod(x%9, 3)
    return 3*t+u

def _raw6_id(s0,s1,s2,s3,s4,s5):
    return ((((s0*9+s1)*9+s2)*9+s3)*9+s4)*9+s5

OVERLAY = "('A','A_X1','B_X1','A_X2','B_X2','C')"

def main():
    out = Path('exports'); out.mkdir(exist_ok=True)

    print("Building AXXC bridge (full population)...")
    FIELDS = ['axxc_config_id','axxc_readable',
              'a_local_id','a_name','x1_local_id','x1_name',
              'x2_local_id','x2_name','c_local_id','c_name',
              'raw_arity','raw_tuple','raw_tuple_id','role_overlay']

    csv_path = out / 'bridge_AXXC.csv'
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()

    batch = []; cfg_id = 0; seen = set()
    for a in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c in range(9):
                    a_raw = _a_idx(a)
                    a1 = _x_a_idx(x1); b1 = _x_b_idx(x1)
                    a2 = _x_a_idx(x2); b2 = _x_b_idx(x2)
                    c_raw = _a_idx(c)  # C_idx = 3*r+u = c itself
                    tid = _raw6_id(a_raw, a1, b1, a2, b2, c_raw)
                    assert tid not in seen, f"Duplicate tid {tid}"
                    seen.add(tid)
                    batch.append({
                        'axxc_config_id': cfg_id,
                        'axxc_readable': f"AXXC[{_a_name(a)},{_x_name(x1)},{_x_name(x2)},{_c_name(c)}]",
                        'a_local_id': a, 'a_name': _a_name(a),
                        'x1_local_id': x1, 'x1_name': _x_name(x1),
                        'x2_local_id': x2, 'x2_name': _x_name(x2),
                        'c_local_id': c, 'c_name': _c_name(c),
                        'raw_arity': 6,
                        'raw_tuple': f"({a_raw},{a1},{b1},{a2},{b2},{c_raw})",
                        'raw_tuple_id': tid,
                        'role_overlay': OVERLAY,
                    })
                    cfg_id += 1
                    if len(batch) >= 10000:
                        w.writerows(batch); batch.clear()

    if batch: w.writerows(batch)
    f.close()

    assert cfg_id == 531441
    assert len(seen) == 531441
    print(f"  Exported: {cfg_id} rows")
    print(f"  Injective: yes")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    preview = []
    with open(csv_path, 'r', encoding='utf-8') as fp:
        r = csv.DictReader(fp)
        for i, row in enumerate(r):
            if i >= 10: break
            preview.append(row)

    md = [
        "# Schema Registration: AXXC",
        "",
        f"Generated: {ts}",
        "",
        "AXXC is the second typed schema beyond the arity-3 core.",
        "",
        "## Schema Definition",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Typed schema | A × X × X × C |",
        "| Typed arity | 4 |",
        "| Raw arity | 6 |",
        "| Typed config count | 531,441 |",
        f"| Role overlay | `{OVERLAY}` |",
        "",
        "## Bridge Rule",
        "",
        "Each A and C expands to 1 raw slot. Each X expands to 2 raw slots.",
        "",
        "AXXC config (a, x1, x2, c) expands to raw tuple:",
        "- (A_idx(a), A_idx(x1), B_idx(x1), A_idx(x2), B_idx(x2), C_idx(c))",
        "",
        "## Export Scope",
        "",
        "bridge_AXXC.csv contains the **full population**: all 531,441 rows.",
        "",
        "## Preview (first 10 rows)",
        "",
        "| axxc_config_id | axxc_readable | raw_tuple | raw_tuple_id |",
        "|----------------|---------------|-----------|--------------|",
    ]
    for r in preview:
        md.append(f"| {r['axxc_config_id']} | {r['axxc_readable']} "
                  f"| {r['raw_tuple']} | {r['raw_tuple_id']} |")
    md += ["", "Full data: see `bridge_AXXC.csv`.", ""]

    md_path = out / 'schema_AXXC_summary.md'
    with open(md_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(md))

    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")

if __name__ == '__main__':
    main()
