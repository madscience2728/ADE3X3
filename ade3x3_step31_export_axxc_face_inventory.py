"""
ade3x3_step31_export_axxc_face_inventory.py

Export face inventory of AXXC = A × X × X × C.
Projects each AXXC config down to AX, XX, XC, AC, and AXC faces.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]


def _act_AX(cfg, act):
    a, x = divmod(cfg, 81)
    ra, sa = divmod(a, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    a2 = 3 * act[0][ra] + act[1][sa]
    x2 = 9 * (3 * act[0][r] + act[1][s]) + (3 * act[1][t] + act[2][u])
    return a2 * 81 + x2


def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    rc, uc = divmod(c, 3)
    x2 = 9 * (3 * act[0][r] + act[1][s]) + (3 * act[1][t] + act[2][u])
    c2 = 3 * act[0][rc] + act[2][uc]
    return x2 * 9 + c2


def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3)
    t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3)
    t2, u2 = divmod(x2 % 9, 3)
    y1 = 9 * (3 * act[0][r1] + act[1][s1]) + (3 * act[1][t1] + act[2][u1])
    y2 = 9 * (3 * act[0][r2] + act[1][s2]) + (3 * act[1][t2] + act[2][u2])
    return y1 * 81 + y2


def _act_AC(cfg, act):
    a, c = divmod(cfg, 9)
    ra, sa = divmod(a, 3)
    rc, uc = divmod(c, 3)
    a2 = 3 * act[0][ra] + act[1][sa]
    c2 = 3 * act[0][rc] + act[2][uc]
    return a2 * 9 + c2


def _act_AXC(cfg, act):
    a = cfg // (81 * 9)
    rem = cfg % (81 * 9)
    x = rem // 9
    c = rem % 9
    ra, sa = divmod(a, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    rc, uc = divmod(c, 3)
    a2 = 3 * act[0][ra] + act[1][sa]
    x2 = 9 * (3 * act[0][r] + act[1][s]) + (3 * act[1][t] + act[2][u])
    c2 = 3 * act[0][rc] + act[2][uc]
    return (a2 * 81 + x2) * 9 + c2


def compute_orbit_map(n, act_fn):
    mins = [0] * n
    for cfg in range(n):
        m = cfg
        for a in ACTIONS:
            img = act_fn(cfg, a)
            if img < m:
                m = img
        mins[cfg] = m
    groups = defaultdict(list)
    for cfg, rep in enumerate(mins):
        groups[rep].append(cfg)
    oid_of = {}
    for oid, (_, members) in enumerate(sorted(groups.items())):
        for m in members:
            oid_of[m] = oid
    return oid_of


def _a_name(a):
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"


def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"


def _x_name(x):
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return f"X[{r},{s}|{t},{u}]"


def _a_idx_of_x(x):
    r, s = divmod(x // 9, 3)
    return 3 * r + s


def _b_idx_of_x(x):
    t, u = divmod(x % 9, 3)
    return 3 * t + u


def _raw6_id(s0, s1, s2, s3, s4, s5):
    return ((((s0 * 9 + s1) * 9 + s2) * 9 + s3) * 9 + s4) * 9 + s5


def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print('Computing orbit maps for AX, XX, XC, AC, AXC...')
    ax_oid = compute_orbit_map(9 * 81, _act_AX)
    xx_oid = compute_orbit_map(81 * 81, _act_XX)
    xc_oid = compute_orbit_map(81 * 9, _act_XC)
    ac_oid = compute_orbit_map(9 * 9, _act_AC)
    axc_oid = compute_orbit_map(9 * 81 * 9, _act_AXC)

    print(f"  AX orbits: {max(ax_oid.values()) + 1}")
    print(f"  XX orbits: {max(xx_oid.values()) + 1}")
    print(f"  XC orbits: {max(xc_oid.values()) + 1}")
    print(f"  AC orbits: {max(ac_oid.values()) + 1}")
    print(f"  AXC orbits: {max(axc_oid.values()) + 1}")

    fields = [
        'axxc_config_id', 'axxc_readable',
        'a_local_id', 'x1_local_id', 'x2_local_id', 'c_local_id',
        'left_ax_config_id', 'middle_xx_config_id', 'right_xc_config_id', 'outer_ac_config_id',
        'left_ax_orbit_id', 'middle_xx_orbit_id', 'right_xc_orbit_id', 'outer_ac_orbit_id',
        'left_axc_config_id', 'right_axc_config_id',
        'left_axc_orbit_id', 'right_axc_orbit_id',
        'raw_tuple_id', 'raw_tuple',
    ]

    csv_path = out / 'AXXC_face_inventory.csv'
    seen_ax = set(); seen_xx = set(); seen_xc = set(); seen_ac = set(); seen_axc_l = set(); seen_axc_r = set()
    n = 0
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        batch = []
        for a in range(9):
            for x1 in range(81):
                for x2 in range(81):
                    for c in range(9):
                        left_ax = a * 81 + x1
                        mid_xx = x1 * 81 + x2
                        right_xc = x2 * 9 + c
                        outer_ac = a * 9 + c
                        left_axc = (a * 81 + x1) * 9 + c
                        right_axc = (a * 81 + x2) * 9 + c

                        ax_o = ax_oid[left_ax]
                        xx_o = xx_oid[mid_xx]
                        xc_o = xc_oid[right_xc]
                        ac_o = ac_oid[outer_ac]
                        axc_l_o = axc_oid[left_axc]
                        axc_r_o = axc_oid[right_axc]

                        seen_ax.add(ax_o); seen_xx.add(xx_o); seen_xc.add(xc_o)
                        seen_ac.add(ac_o); seen_axc_l.add(axc_l_o); seen_axc_r.add(axc_r_o)

                        a0 = a
                        a1 = _a_idx_of_x(x1); b1 = _b_idx_of_x(x1)
                        a2 = _a_idx_of_x(x2); b2 = _b_idx_of_x(x2)
                        c0 = c
                        raw_tup = (a0, a1, b1, a2, b2, c0)
                        raw_tid = _raw6_id(a0, a1, b1, a2, b2, c0)

                        batch.append({
                            'axxc_config_id': n,
                            'axxc_readable': f"AXXC[{_a_name(a)},{_x_name(x1)},{_x_name(x2)},{_c_name(c)}]",
                            'a_local_id': a,
                            'x1_local_id': x1,
                            'x2_local_id': x2,
                            'c_local_id': c,
                            'left_ax_config_id': left_ax,
                            'middle_xx_config_id': mid_xx,
                            'right_xc_config_id': right_xc,
                            'outer_ac_config_id': outer_ac,
                            'left_ax_orbit_id': ax_o,
                            'middle_xx_orbit_id': xx_o,
                            'right_xc_orbit_id': xc_o,
                            'outer_ac_orbit_id': ac_o,
                            'left_axc_config_id': left_axc,
                            'right_axc_config_id': right_axc,
                            'left_axc_orbit_id': axc_l_o,
                            'right_axc_orbit_id': axc_r_o,
                            'raw_tuple_id': raw_tid,
                            'raw_tuple': str(raw_tup),
                        })
                        n += 1
                        if len(batch) >= 10000:
                            w.writerows(batch)
                            batch.clear()
        if batch:
            w.writerows(batch)

    assert n == 531441, f'Row count {n} != 531441'
    assert max(seen_ax) < 10
    assert max(seen_xx) < 56
    assert max(seen_xc) < 8
    assert max(seen_ac) < (max(ac_oid.values()) + 1)
    assert max(seen_axc_l) < (max(axc_oid.values()) + 1)
    assert max(seen_axc_r) < (max(axc_oid.values()) + 1)

    print(f'  Exported: {n} rows')
    print('  All orbit IDs valid: OK')
    print('  Full coverage: OK')

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    preview = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= 10:
                break
            preview.append(row)

    md = [
        '# AXXC Face Inventory',
        '',
        f'Generated: {ts}',
        '',
        'Arity-4 face inventory for AXXC = A × X × X × C.',
        '',
        '## Summary',
        '',
        '| Property | Value |',
        '|----------|-------|',
        '| Total AXXC rows | 531,441 |',
        f'| Distinct left AX orbit IDs | {len(seen_ax)} |',
        f'| Distinct middle XX orbit IDs | {len(seen_xx)} |',
        f'| Distinct right XC orbit IDs | {len(seen_xc)} |',
        f'| Distinct outer AC orbit IDs | {len(seen_ac)} |',
        f'| Distinct left AXC orbit IDs | {len(seen_axc_l)} |',
        f'| Distinct right AXC orbit IDs | {len(seen_axc_r)} |',
        '',
        '## Faces Exported',
        '',
        '- Left AX face: (a, x1)',
        '- Middle XX face: (x1, x2)',
        '- Right XC face: (x2, c)',
        '- Outer AC face: (a, c)',
        '- Left AXC face: (a, x1, c)',
        '- Right AXC face: (a, x2, c)',
        '',
        '## Preview (first 10 rows)',
        '',
        '| axxc | AX_orb | XX_orb | XC_orb | AC_orb | AXC_L_orb | AXC_R_orb |',
        '|------|--------|--------|--------|--------|-----------|-----------|',
    ]
    for row in preview:
        md.append(
            f"| {row['axxc_config_id']} | {row['left_ax_orbit_id']} | {row['middle_xx_orbit_id']} | "
            f"{row['right_xc_orbit_id']} | {row['outer_ac_orbit_id']} | {row['left_axc_orbit_id']} | "
            f"{row['right_axc_orbit_id']} |"
        )
    md += ['', 'Full data: see `AXXC_face_inventory.csv`.', '']
    md_path = out / 'AXXC_face_inventory.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('\n--- EXPORT COMPLETE ---')
    print(f'  rows: {n}')
    print(f'  distinct left AX orbit IDs: {len(seen_ax)}')
    print(f'  distinct middle XX orbit IDs: {len(seen_xx)}')
    print(f'  distinct right XC orbit IDs: {len(seen_xc)}')
    print(f'  distinct outer AC orbit IDs: {len(seen_ac)}')
    print(f'  distinct left AXC orbit IDs: {len(seen_axc_l)}')
    print(f'  distinct right AXC orbit IDs: {len(seen_axc_r)}')
    print(f'\nFiles written:\n  {csv_path}\n  {md_path}')


if __name__ == '__main__':
    main()
