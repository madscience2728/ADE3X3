"""
ade3x3_step21_bridge_xx.py

Build the explicit typed/raw bridge for XX = X x X.
Each X atom expands to 2 raw slots, so XX expands to raw arity 4.
"""

import csv
from pathlib import Path
from datetime import datetime

# ── X atom lookup ──────────────────────────────────────────────────

def _x_decode(idx):
    """Decode X local index into (r, s, t, u)."""
    r, s = divmod(idx // 9, 3)
    t, u = divmod(idx % 9, 3)
    return r, s, t, u

def _x_name(idx):
    r, s, t, u = _x_decode(idx)
    return f"X[{r},{s}|{t},{u}]"

def _a_idx(idx):
    """A_idx = 3*r + s"""
    r, s = divmod(idx // 9, 3)
    return 3 * r + s

def _b_idx(idx):
    """B_idx = 3*t + u"""
    t, u = divmod(idx % 9, 3)
    return 3 * t + u

def _raw_id(a1, b1, a2, b2):
    """Base-9 tuple ID for arity-4 raw tuple."""
    return ((a1 * 9 + b1) * 9 + a2) * 9 + b2

def _raw_decode(tid):
    """Decode base-9 arity-4 tuple ID back to (a1,b1,a2,b2)."""
    b2 = tid % 9; tid //= 9
    a2 = tid % 9; tid //= 9
    b1 = tid % 9; tid //= 9
    a1 = tid
    return a1, b1, a2, b2

# ── Build bridge ───────────────────────────────────────────────────

ROLE_OVERLAY = "('A_X1','B_X1','A_X2','B_X2')"

def build_bridge():
    rows = []
    for x1 in range(81):
        for x2 in range(81):
            cfg_id = x1 * 81 + x2
            a1, b1 = _a_idx(x1), _b_idx(x1)
            a2, b2 = _a_idx(x2), _b_idx(x2)
            tid = _raw_id(a1, b1, a2, b2)
            rows.append({
                'xx_config_id': cfg_id,
                'xx_readable': f"XX[{_x_name(x1)},{_x_name(x2)}]",
                'x1_local_id': x1,
                'x1_name': _x_name(x1),
                'x2_local_id': x2,
                'x2_name': _x_name(x2),
                'raw_arity': 4,
                'raw_tuple': f"({a1},{b1},{a2},{b2})",
                'raw_tuple_id': tid,
                'role_overlay': ROLE_OVERLAY,
            })
    return rows

# ── CSV / Markdown writers ─────────────────────────────────────────

CSV_FIELDS = ['xx_config_id','xx_readable','x1_local_id','x1_name',
              'x2_local_id','x2_name','raw_arity','raw_tuple',
              'raw_tuple_id','role_overlay']

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

def write_md(rows, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    L = [
        "# Typed/Raw Bridge: XX",
        "",
        f"Generated: {ts}",
        "",
        "Explicit typed/raw bridge for schema XX = X x X.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Typed arity | 2 |",
        "| Raw arity | 4 |",
        "| Typed config count | 6,561 |",
        "| Image size | 6,561 |",
        "| Injective | yes |",
        f"| Role overlay | `{ROLE_OVERLAY}` |",
        "",
        "## Bridge Rule",
        "",
        "Each X[r,s|t,u] expands to raw symbols:",
        "- A_idx = 3*r + s",
        "- B_idx = 3*t + u",
        "",
        "XX config (X1, X2) expands to raw tuple:",
        "- (A_idx(X1), B_idx(X1), A_idx(X2), B_idx(X2))",
        "",
        "## Preview (first 10 rows)",
        "",
        "| xx_config_id | xx_readable | raw_tuple | raw_tuple_id |",
        "|--------------|-------------|-----------|--------------|",
    ]
    for r in rows[:10]:
        L.append(f"| {r['xx_config_id']} | {r['xx_readable']} | {r['raw_tuple']} | {r['raw_tuple_id']} |")
    L += [
        "",
        "Full data: see `bridge_XX.csv`.",
        "",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Building XX typed/raw bridge...")
    rows = build_bridge()
    n = len(rows)
    assert n == 6561, f"Expected 6561 rows, got {n}"

    # Injectivity
    ids = [r['raw_tuple_id'] for r in rows]
    assert len(set(ids)) == 6561, "Bridge is not injective!"

    # Roundtrip + correctness
    for r in rows:
        # raw tuple correctness
        x1, x2 = r['x1_local_id'], r['x2_local_id']
        a1, b1, a2, b2 = _a_idx(x1), _b_idx(x1), _a_idx(x2), _b_idx(x2)
        assert r['raw_tuple'] == f"({a1},{b1},{a2},{b2})", \
            f"cfg {r['xx_config_id']}: raw_tuple mismatch"

        # raw tuple ID correctness
        assert r['raw_tuple_id'] == _raw_id(a1, b1, a2, b2), \
            f"cfg {r['xx_config_id']}: raw_tuple_id mismatch"

        # roundtrip: raw -> typed
        ra1, rb1, ra2, rb2 = _raw_decode(r['raw_tuple_id'])
        # recover x1 from (ra1, rb1): r=ra1//3, s=ra1%3, t=rb1//3, u=rb1%3
        rx1 = 9 * (3 * (ra1 // 3) + (ra1 % 3)) + (3 * (rb1 // 3) + (rb1 % 3))
        rx2 = 9 * (3 * (ra2 // 3) + (ra2 % 3)) + (3 * (rb2 // 3) + (rb2 % 3))
        rcfg = rx1 * 81 + rx2
        assert rcfg == r['xx_config_id'], \
            f"cfg {r['xx_config_id']}: roundtrip got {rcfg}"

    print(f"  {n} rows, injective, roundtrip OK")

    csv_path = out / 'bridge_XX.csv'
    md_path = out / 'bridge_XX.md'
    write_csv(rows, csv_path)
    write_md(rows, md_path)

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  rows: {n}")
    print(f"  distinct raw_tuple_ids: {len(set(ids))}")
    print(f"  injective: yes")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
