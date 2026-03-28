"""
ade3x3_step23_export_composition_cx_xc_cc.py

Export the full CX ∘ XC -> CC composition table using corrected orbit layer.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Group of size 216 ──────────────────────────────────────────────

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]

# ── Action functions ───────────────────────────────────────────────

def _act_CX(cfg, act):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x//9, 3); t, u2 = divmod(x%9, 3)
    return (3*act[0][r]+act[2][u])*81 + 9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    r2, u2 = divmod(c, 3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def _act_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return 9*(3*act[0][r1]+act[2][u1])+(3*act[0][r2]+act[2][u2])

# ── Readable names ─────────────────────────────────────────────────

def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"

def _x_name(x):
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    return f"X[{r},{s}|{t},{u}]"

# ── Orbit computation ──────────────────────────────────────────────

def compute_orbits(n, act_fn):
    """Return dict: config_id -> orbit_id, and orbit count."""
    min_r = {}
    for cfg in range(n):
        m = cfg
        for a in ACTIONS:
            img = act_fn(cfg, a)
            if img < m: m = img
        min_r[cfg] = m
    groups = defaultdict(list)
    for cfg in range(n):
        groups[min_r[cfg]].append(cfg)
    oid_of = {}
    for oid, (rep, members) in enumerate(sorted(groups.items())):
        for m in members:
            oid_of[m] = oid
    return oid_of, len(groups)

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Computing orbits...")
    cx_oid, cx_n = compute_orbits(9*81, _act_CX)
    xc_oid, xc_n = compute_orbits(81*9, _act_XC)
    cc_oid, cc_n = compute_orbits(81, _act_CC)

    assert cx_n == 8, f"CX orbits {cx_n} != 8"
    assert xc_n == 8, f"XC orbits {xc_n} != 8"
    assert cc_n == 4, f"CC orbits {cc_n} != 4"
    print(f"  CX={cx_n}, XC={xc_n}, CC={cc_n}")

    # Composition: iterate all (c1, x, c2) triples
    comp = defaultdict(lambda: {'count': 0, 'cc_hist': defaultdict(int), 'witnesses': []})
    for c1 in range(9):
        for x in range(81):
            for c2 in range(9):
                cx_cfg = c1*81 + x
                xc_cfg = x*9 + c2
                cc_cfg = c1*9 + c2
                key = (cx_oid[cx_cfg], xc_oid[xc_cfg])
                cc_o = cc_oid[cc_cfg]
                comp[key]['count'] += 1
                comp[key]['cc_hist'][cc_o] += 1
                comp[key]['witnesses'].append((c1, x, c2, cc_o))

    # Build main CSV rows
    main_rows = []
    for (cx_o, xc_o) in sorted(comp):
        entry = comp[(cx_o, xc_o)]
        hist = dict(entry['cc_hist'])
        cc_ids = sorted(hist.keys())
        det = 1 if len(cc_ids) == 1 else 0
        main_rows.append({
            'cx_orbit_id': cx_o,
            'xc_orbit_id': xc_o,
            'realized': 1,
            'witness_count': entry['count'],
            'cc_histogram': str(hist),
            'cc_orbit_ids': str(cc_ids),
            'deterministic': det,
        })

    # Build witness CSV rows
    wit_rows = []
    for (cx_o, xc_o) in sorted(comp):
        for c1, x, c2, cc_o in comp[(cx_o, xc_o)]['witnesses']:
            wit_rows.append({
                'cx_orbit_id': cx_o,
                'xc_orbit_id': xc_o,
                'cc_orbit_id': cc_o,
                'c1_local_id': c1,
                'c1_name': _c_name(c1),
                'x_local_id': x,
                'x_name': _x_name(x),
                'c2_local_id': c2,
                'c2_name': _c_name(c2),
                'cxc_readable': f"CXC[{_c_name(c1)},{_x_name(x)},{_c_name(c2)}]",
            })

    # ── Sanity checks ──
    n_realized = len(main_rows)
    n_det = sum(r['deterministic'] for r in main_rows)
    n_mixed = n_realized - n_det

    # Witness correctness
    for wr in wit_rows:
        c1, x, c2 = wr['c1_local_id'], wr['x_local_id'], wr['c2_local_id']
        assert cx_oid[c1*81+x] == wr['cx_orbit_id']
        assert xc_oid[x*9+c2] == wr['xc_orbit_id']
        assert cc_oid[c1*9+c2] == wr['cc_orbit_id']

    # Histogram correctness
    for mr in main_rows:
        key = (mr['cx_orbit_id'], mr['xc_orbit_id'])
        hist_from_wit = defaultdict(int)
        for wr in wit_rows:
            if (wr['cx_orbit_id'], wr['xc_orbit_id']) == key:
                hist_from_wit[wr['cc_orbit_id']] += 1
        assert dict(hist_from_wit) == eval(mr['cc_histogram']), \
            f"Histogram mismatch for key {key}"
        assert sum(hist_from_wit.values()) == mr['witness_count']

    print(f"  Realized keys: {n_realized}/64")
    print(f"  Deterministic: {n_det}")
    print(f"  Mixed: {n_mixed}")
    print("  All checks passed.")

    # ── Write CSVs ──
    MF = ['cx_orbit_id','xc_orbit_id','realized','witness_count',
          'cc_histogram','cc_orbit_ids','deterministic']
    csv1 = out / 'comp_CX_XC_to_CC.csv'
    with open(csv1, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MF)
        w.writeheader()
        w.writerows(main_rows)

    WF = ['cx_orbit_id','xc_orbit_id','cc_orbit_id',
          'c1_local_id','c1_name','x_local_id','x_name',
          'c2_local_id','c2_name','cxc_readable']
    csv2 = out / 'comp_CX_XC_to_CC_witnesses.csv'
    with open(csv2, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=WF)
        w.writeheader()
        w.writerows(wit_rows)

    # ── Markdown ──
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md_lines = [
        "# Composition Export: CX ∘ XC → CC",
        "",
        f"Generated: {ts}",
        "",
        "Explicit composition table for CX ∘ XC → CC using corrected orbit layer.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| CX orbit count | {cx_n} |",
        f"| XC orbit count | {xc_n} |",
        f"| CC orbit count | {cc_n} |",
        "| Possible orbit-pairs | 64 |",
        f"| Realized keys | {n_realized} |",
        f"| Deterministic | {n_det} |",
        f"| Mixed | {n_mixed} |",
        "",
        "## Preview (first 10 realized keys)",
        "",
        "| cx_orbit | xc_orbit | witnesses | cc_histogram | det |",
        "|----------|----------|-----------|--------------|-----|",
    ]
    for r in main_rows[:10]:
        md_lines.append(
            f"| {r['cx_orbit_id']} | {r['xc_orbit_id']} | {r['witness_count']} "
            f"| {r['cc_histogram']} | {'yes' if r['deterministic'] else 'no'} |"
        )
    md_lines += [
        "",
        "## Historical Note",
        "",
        "An older type-based layer reported 256 possible keys, 64 realized,",
        "36 deterministic, 28 mixed. Those counts used a different type system.",
        "The current corrected orbit-based counts are shown above.",
        "",
        "## Files",
        "",
        "- `comp_CX_XC_to_CC.csv` — realized key summary",
        "- `comp_CX_XC_to_CC_witnesses.csv` — full witness triples",
        "",
    ]
    md_path = out / 'comp_CX_XC_to_CC.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  realized keys: {n_realized}")
    print(f"  deterministic: {n_det}")
    print(f"  mixed: {n_mixed}")
    print(f"\nFiles written:")
    print(f"  {csv1}")
    print(f"  {csv2}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
