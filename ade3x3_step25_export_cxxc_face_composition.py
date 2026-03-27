"""
ade3x3_step25_export_cxxc_face_composition.py

Export face-composition inventory of CXXC = C × X × X × C.
Projects each CXXC config down to CX, XX, XC, CC, and CXC faces.
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
    r, u = divmod(c, 3); r2, s = divmod(x//9, 3); t, u2 = divmod(x%9, 3)
    return (3*act[0][r]+act[2][u])*81 + 9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3); r2, u2 = divmod(c, 3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def _act_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return 9*(3*act[0][r1]+act[2][u1])+(3*act[0][r2]+act[2][u2])

def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return (9*(3*act[0][r1]+act[1][s1])+(3*act[1][t1]+act[2][u1]))*81 + \
           (9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2]))

def _act_CXC(cfg, act):
    c1 = cfg//(81*9); rem=cfg%(81*9); x=rem//9; c2=rem%9
    r1,u1=divmod(c1,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3); r3,u3=divmod(c2,3)
    return ((3*act[0][r1]+act[2][u1])*81+9*(3*act[0][r2]+act[1][s])+
            (3*act[1][t]+act[2][u2]))*9+(3*act[0][r3]+act[2][u3])

# ── Orbit computation ──────────────────────────────────────────────

def compute_orbit_map(n, act_fn):
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
    for oid, (_, members) in enumerate(sorted(groups.items())):
        for m in members:
            oid_of[m] = oid
    return oid_of

# ── Readable names ─────────────────────────────────────────────────

def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"

def _x_name(x):
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    return f"X[{r},{s}|{t},{u}]"

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Computing orbit maps...")
    cx_oid = compute_orbit_map(9*81, _act_CX)
    xc_oid = compute_orbit_map(81*9, _act_XC)
    cc_oid = compute_orbit_map(81, _act_CC)
    xx_oid = compute_orbit_map(81*81, _act_XX)
    cxc_oid = compute_orbit_map(9*81*9, _act_CXC)

    print(f"  CX orbits: {max(cx_oid.values())+1}")
    print(f"  XC orbits: {max(xc_oid.values())+1}")
    print(f"  CC orbits: {max(cc_oid.values())+1}")
    print(f"  XX orbits: {max(xx_oid.values())+1}")
    print(f"  CXC orbits: {max(cxc_oid.values())+1}")

    FIELDS = [
        'cxxc_config_id','cxxc_readable',
        'c1_local_id','x1_local_id','x2_local_id','c2_local_id',
        'left_cx_config_id','middle_xx_config_id','right_xc_config_id','outer_cc_config_id',
        'left_cx_orbit_id','middle_xx_orbit_id','right_xc_orbit_id','outer_cc_orbit_id',
        'left_cxc_config_id','right_cxc_config_id',
        'left_cxc_orbit_id','right_cxc_orbit_id',
        'raw_tuple_id','raw_tuple',
    ]

    csv_path = out / 'CXXC_face_inventory.csv'
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    batch = []
    n = 0
    # Track distinct orbit IDs seen per face
    seen = {k: set() for k in ['cx','xx','xc','cc','cxc_l','cxc_r']}

    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    # Face config IDs
                    left_cx = c1*81 + x1
                    mid_xx = x1*81 + x2
                    right_xc = x2*9 + c2
                    outer_cc = c1*9 + c2
                    left_cxc = (c1*81 + x1)*9 + c2   # c1*729 + x1*9 + c2
                    right_cxc = (c1*81 + x2)*9 + c2  # c1*729 + x2*9 + c2

                    # Raw tuple
                    a1 = 3*(x1//9//3) + (x1//9%3)
                    b1 = 3*(x1%9//3) + (x1%9%3)
                    a2 = 3*(x2//9//3) + (x2//9%3)
                    b2 = 3*(x2%9//3) + (x2%9%3)
                    raw_tup = (c1, a1, b1, a2, b2, c2)
                    raw_tid = ((((c1*9+a1)*9+b1)*9+a2)*9+b2)*9+c2

                    # Orbit IDs
                    cx_o = cx_oid[left_cx]
                    xx_o = xx_oid[mid_xx]
                    xc_o = xc_oid[right_xc]
                    cc_o = cc_oid[outer_cc]
                    cxc_l_o = cxc_oid[left_cxc]
                    cxc_r_o = cxc_oid[right_cxc]

                    seen['cx'].add(cx_o)
                    seen['xx'].add(xx_o)
                    seen['xc'].add(xc_o)
                    seen['cc'].add(cc_o)
                    seen['cxc_l'].add(cxc_l_o)
                    seen['cxc_r'].add(cxc_r_o)

                    row = {
                        'cxxc_config_id': n,
                        'cxxc_readable': f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
                        'c1_local_id': c1,
                        'x1_local_id': x1,
                        'x2_local_id': x2,
                        'c2_local_id': c2,
                        'left_cx_config_id': left_cx,
                        'middle_xx_config_id': mid_xx,
                        'right_xc_config_id': right_xc,
                        'outer_cc_config_id': outer_cc,
                        'left_cx_orbit_id': cx_o,
                        'middle_xx_orbit_id': xx_o,
                        'right_xc_orbit_id': xc_o,
                        'outer_cc_orbit_id': cc_o,
                        'left_cxc_config_id': left_cxc,
                        'right_cxc_config_id': right_cxc,
                        'left_cxc_orbit_id': cxc_l_o,
                        'right_cxc_orbit_id': cxc_r_o,
                        'raw_tuple_id': raw_tid,
                        'raw_tuple': str(raw_tup),
                    }
                    batch.append(row)
                    n += 1

                    if len(batch) >= 10000:
                        w.writerows(batch)
                        batch.clear()

    if batch:
        w.writerows(batch)
    f.close()

    assert n == 531441, f"Row count {n} != 531441"
    print(f"\n  Exported: {n} rows")

    # Orbit validity checks
    assert max(seen['cx']) < 8
    assert max(seen['xx']) < 56
    assert max(seen['xc']) < 8
    assert max(seen['cc']) < 4
    assert max(seen['cxc_l']) < 50
    assert max(seen['cxc_r']) < 50
    print("  All orbit IDs valid: OK")
    print("  Full coverage: OK")

    # ── Markdown ──
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Preview: re-read first 10 rows
    preview = []
    with open(csv_path, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for i, r in enumerate(reader):
            if i >= 10: break
            preview.append(r)

    md_lines = [
        "# CXXC Face Composition Inventory",
        "",
        f"Generated: {ts}",
        "",
        "Arity-4 face-composition inventory for CXXC = C × X × X × C.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Total CXXC rows | 531,441 |",
        f"| Distinct left CX orbit IDs | {len(seen['cx'])} |",
        f"| Distinct middle XX orbit IDs | {len(seen['xx'])} |",
        f"| Distinct right XC orbit IDs | {len(seen['xc'])} |",
        f"| Distinct outer CC orbit IDs | {len(seen['cc'])} |",
        f"| Distinct left CXC orbit IDs | {len(seen['cxc_l'])} |",
        f"| Distinct right CXC orbit IDs | {len(seen['cxc_r'])} |",
        "",
        "## Faces Exported",
        "",
        "- Left CX face: (c1, x1)",
        "- Middle XX face: (x1, x2)",
        "- Right XC face: (x2, c2)",
        "- Outer CC face: (c1, c2)",
        "- Left CXC face: (c1, x1, c2)",
        "- Right CXC face: (c1, x2, c2)",
        "",
        "## Preview (first 10 rows)",
        "",
        "| cxxc | CX_orb | XX_orb | XC_orb | CC_orb | CXC_L_orb | CXC_R_orb |",
        "|------|--------|--------|--------|--------|-----------|-----------|",
    ]
    for r in preview:
        md_lines.append(
            f"| {r['cxxc_config_id']} | {r['left_cx_orbit_id']} "
            f"| {r['middle_xx_orbit_id']} | {r['right_xc_orbit_id']} "
            f"| {r['outer_cc_orbit_id']} | {r['left_cxc_orbit_id']} "
            f"| {r['right_cxc_orbit_id']} |"
        )
    md_lines += ["", "Full data: see `CXXC_face_inventory.csv`.", ""]

    md_path = out / 'CXXC_face_inventory.md'
    with open(md_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(md_lines))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  rows: {n}")
    for k, label in [('cx','left CX'),('xx','middle XX'),('xc','right XC'),
                      ('cc','outer CC'),('cxc_l','left CXC'),('cxc_r','right CXC')]:
        print(f"  distinct {label} orbit IDs: {len(seen[k])}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
