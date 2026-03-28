"""
ade3x3_step24_register_and_export_cxxc.py

Register and export CXXC = C × X × X × C, the first arity-4 typed schema.
Full population: 9 × 81 × 81 × 9 = 531,441 rows.
"""

import csv
from pathlib import Path
from datetime import datetime

# ── Atom helpers ───────────────────────────────────────────────────

def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"

def _x_name(x):
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    return f"X[{r},{s}|{t},{u}]"

def _a_idx(x):
    r, s = divmod(x//9, 3)
    return 3*r+s

def _b_idx(x):
    t, u = divmod(x%9, 3)
    return 3*t+u

def _raw6_id(s0, s1, s2, s3, s4, s5):
    return ((((s0*9+s1)*9+s2)*9+s3)*9+s4)*9+s5

OVERLAY = "('C','A_X1','B_X1','A_X2','B_X2','C')"

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Building CXXC bridge (full population)...")
    print("  Typed count: 9 * 81 * 81 * 9 = 531,441")

    FIELDS = ['cxxc_config_id','cxxc_readable',
              'c1_local_id','c1_name','x1_local_id','x1_name',
              'x2_local_id','x2_name','c2_local_id','c2_name',
              'raw_arity','raw_tuple','raw_tuple_id','role_overlay']

    csv_path = out / 'bridge_CXXC.csv'
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    cfg_id = 0
    seen_ids = set()
    batch = []

    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    a1, b1 = _a_idx(x1), _b_idx(x1)
                    a2, b2 = _a_idx(x2), _b_idx(x2)
                    tid = _raw6_id(c1, a1, b1, a2, b2, c2)
                    row = {
                        'cxxc_config_id': cfg_id,
                        'cxxc_readable': f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
                        'c1_local_id': c1,
                        'c1_name': _c_name(c1),
                        'x1_local_id': x1,
                        'x1_name': _x_name(x1),
                        'x2_local_id': x2,
                        'x2_name': _x_name(x2),
                        'c2_local_id': c2,
                        'c2_name': _c_name(c2),
                        'raw_arity': 6,
                        'raw_tuple': f"({c1},{a1},{b1},{a2},{b2},{c2})",
                        'raw_tuple_id': tid,
                        'role_overlay': OVERLAY,
                    }
                    assert tid not in seen_ids, f"Duplicate raw_tuple_id {tid} at cfg {cfg_id}"
                    seen_ids.add(tid)
                    batch.append(row)
                    cfg_id += 1

                    if len(batch) >= 10000:
                        w.writerows(batch)
                        batch.clear()

    if batch:
        w.writerows(batch)
    f.close()

    n = cfg_id
    assert n == 531441, f"Count {n} != 531441"
    assert len(seen_ids) == 531441, f"Distinct ids {len(seen_ids)} != 531441"

    print(f"  Exported: {n} rows")
    print(f"  Injective: yes (531,441 distinct raw_tuple_ids)")
    print("  All checks passed.")

    # ── Markdown ──
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Re-read first 10 rows for preview
    preview = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= 10: break
            preview.append(r)

    md_lines = [
        "# Schema Registration: CXXC",
        "",
        f"Generated: {ts}",
        "",
        "CXXC is the first typed schema beyond the current arity-3 core.",
        "",
        "## Schema Definition",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Typed schema | C × X × X × C |",
        "| Typed arity | 4 |",
        "| Raw arity | 6 |",
        "| Typed config count | 531,441 |",
        f"| Role overlay | `{OVERLAY}` |",
        "",
        "## Bridge Rule",
        "",
        "Each C expands to 1 raw slot. Each X expands to 2 raw slots.",
        "",
        "CXXC config (c1, x1, x2, c2) expands to raw tuple:",
        "- (C_idx(c1), A_idx(x1), B_idx(x1), A_idx(x2), B_idx(x2), C_idx(c2))",
        "",
        "## Export Scope",
        "",
        "bridge_CXXC.csv contains the **full population**: all 531,441 rows.",
        "",
        "## Preview (first 10 rows)",
        "",
        "| cxxc_config_id | cxxc_readable | raw_tuple | raw_tuple_id |",
        "|----------------|---------------|-----------|--------------|",
    ]
    for r in preview:
        md_lines.append(
            f"| {r['cxxc_config_id']} | {r['cxxc_readable']} "
            f"| {r['raw_tuple']} | {r['raw_tuple_id']} |"
        )
    md_lines += [
        "",
        "Full data: see `bridge_CXXC.csv`.",
        "",
    ]

    md_path = out / 'schema_CXXC_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
