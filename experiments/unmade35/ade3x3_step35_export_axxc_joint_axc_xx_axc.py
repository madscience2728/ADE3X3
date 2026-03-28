"""
ade3x3_step35_export_axxc_joint_axc_xx_axc.py

Export the joint co-occurrence inventory of left AXC, middle XX, right AXC
inside AXXC.
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


def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3)
    t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3)
    t2, u2 = divmod(x2 % 9, 3)
    y1 = 9 * (3 * act[0][r1] + act[1][s1]) + (3 * act[1][t1] + act[2][u1])
    y2 = 9 * (3 * act[0][r2] + act[1][s2]) + (3 * act[1][t2] + act[2][u2])
    return y1 * 81 + y2


def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    rc, uc = divmod(c, 3)
    x2 = 9 * (3 * act[0][r] + act[1][s]) + (3 * act[1][t] + act[2][u])
    c2 = 3 * act[0][rc] + act[2][uc]
    return x2 * 9 + c2


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


def _orbit_map(n, fn):
    mins = {}
    for cfg in range(n):
        m = cfg
        for a in ACTIONS:
            img = fn(cfg, a)
            if img < m:
                m = img
        mins[cfg] = m
    groups = defaultdict(list)
    for cfg, rep in mins.items():
        groups[rep].append(cfg)
    oid_of = {}
    for oid, (_, members) in enumerate(sorted(groups.items())):
        for m in members:
            oid_of[m] = oid
    return oid_of


def _a_name(a):
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"


def _x_name(x):
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return f"X[{r},{s}|{t},{u}]"


def _c_name(c):
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"


def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print('Computing orbit maps...')
    ax_o = _orbit_map(9 * 81, _act_AX)
    xx_o = _orbit_map(81 * 81, _act_XX)
    xc_o = _orbit_map(81 * 9, _act_XC)
    ac_o = _orbit_map(9 * 9, _act_AC)
    axc_o = _orbit_map(9 * 81 * 9, _act_AXC)
    assert max(xx_o.values()) + 1 == 56
    assert max(axc_o.values()) + 1 == 50

    print('Scanning AXXC joint AXC-XX-AXC keys...')
    joint = {}
    total = 0
    cfg_id = 0
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

                    key = (axc_o[left_axc], xx_o[mid_xx], axc_o[right_axc])
                    if key not in joint:
                        joint[key] = {
                            'count': 0,
                            'rep': cfg_id,
                            'ax': set(),
                            'xc': set(),
                            'ac': set(),
                        }
                    d = joint[key]
                    d['count'] += 1
                    d['ax'].add(ax_o[left_ax])
                    d['xc'].add(xc_o[right_xc])
                    d['ac'].add(ac_o[outer_ac])
                    cfg_id += 1
                    total += 1
    assert total == 531441

    rows = []
    for pid, (key, d) in enumerate(sorted(joint.items())):
        rep = d['rep']
        tmp = rep
        c = tmp % 9; tmp //= 9
        x2 = tmp % 81; tmp //= 81
        x1 = tmp % 81; tmp //= 81
        a = tmp
        rows.append({
            'joint_id': pid,
            'count': d['count'],
            'left_axc_orbit_id': key[0],
            'middle_xx_orbit_id': key[1],
            'right_axc_orbit_id': key[2],
            'left_ax_orbit_ids': str(sorted(d['ax'])),
            'right_xc_orbit_ids': str(sorted(d['xc'])),
            'outer_ac_orbit_ids': str(sorted(d['ac'])),
            'rep_axxc_config_id': rep,
            'rep_axxc_readable': f"AXXC[{_a_name(a)},{_x_name(x1)},{_x_name(x2)},{_c_name(c)}]",
        })

    # Sanity
    assert sum(r['count'] for r in rows) == 531441
    seen = set()
    for r in rows:
        key = (r['left_axc_orbit_id'], r['middle_xx_orbit_id'], r['right_axc_orbit_id'])
        assert key not in seen
        seen.add(key)
        assert 0 <= r['left_axc_orbit_id'] < 50
        assert 0 <= r['middle_xx_orbit_id'] < 56
        assert 0 <= r['right_axc_orbit_id'] < 50
        for v in eval(r['left_ax_orbit_ids']):
            assert 0 <= v < 10
        for v in eval(r['right_xc_orbit_ids']):
            assert 0 <= v < 8
        for v in eval(r['outer_ac_orbit_ids']):
            assert 0 <= v < 2
        rep = r['rep_axxc_config_id']
        tmp = rep
        c = tmp % 9; tmp //= 9
        x2 = tmp % 81; tmp //= 81
        x1 = tmp % 81; tmp //= 81
        a = tmp
        actual = (axc_o[(a * 81 + x1) * 9 + c], xx_o[x1 * 81 + x2], axc_o[(a * 81 + x2) * 9 + c])
        assert actual == key

    csv_path = out / 'AXXC_joint_AXC_XX_AXC.csv'
    fields = [
        'joint_id','count','left_axc_orbit_id','middle_xx_orbit_id','right_axc_orbit_id',
        'left_ax_orbit_ids','right_xc_orbit_ids','outer_ac_orbit_ids',
        'rep_axxc_config_id','rep_axxc_readable'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = [r['count'] for r in rows]
    mn, mx = min(counts), max(counts)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md = [
        '# AXXC Joint Co-occurrence: left AXC, middle XX, right AXC',
        '',
        f'Generated: {ts}',
        '',
        'Joint co-occurrence inventory for the three key interior face-orbit families inside AXXC.',
        '',
        '## Summary',
        '',
        '| Property | Value |',
        '|----------|-------|',
        '| Total AXXC rows | 531,441 |',
        f'| Distinct joint keys | {len(rows)} |',
        f'| Min joint count | {mn} |',
        f'| Max joint count | {mx} |',
        '',
        '## Preview (first 10 rows)',
        '',
        '| joint_id | count | AXC_L | XX | AXC_R |',
        '|----------|-------|-------|----|-------|',
    ]
    for r in rows[:10]:
        md.append(f"| {r['joint_id']} | {r['count']} | {r['left_axc_orbit_id']} | {r['middle_xx_orbit_id']} | {r['right_axc_orbit_id']} |")
    md += ['', 'Full data: see `AXXC_joint_AXC_XX_AXC.csv`.', '']
    md_path = out / 'AXXC_joint_AXC_XX_AXC.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('  All checks passed.')
    print('\n--- EXPORT COMPLETE ---')
    print(f'  distinct joint keys: {len(rows)}')
    print(f'  count range: {mn}-{mx}')
    print(f'  sum: {sum(counts)}')
    print(f'\nFiles written:')
    print(f'  {csv_path}')
    print(f'  {md_path}')


if __name__ == '__main__':
    main()
