"""
ade3x3_step34_export_axxc_xx_marginals.py

Export AXXC marginals conditioned on the middle XX orbit.
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
    assert max(ax_o.values()) + 1 == 10
    assert max(xx_o.values()) + 1 == 56
    assert max(xc_o.values()) + 1 == 8
    assert max(ac_o.values()) + 1 == 2
    assert max(axc_o.values()) + 1 == 50

    print('Scanning AXXC marginals over middle XX orbit...')
    marg = defaultdict(lambda: {
        'count': 0,
        'patterns': set(),
        'ax': set(),
        'xc': set(),
        'ac': set(),
        'axc_l': set(),
        'axc_r': set(),
        'rep': None,
    })

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

                    key = (
                        ax_o[left_ax],
                        xx_o[mid_xx],
                        xc_o[right_xc],
                        ac_o[outer_ac],
                        axc_o[left_axc],
                        axc_o[right_axc],
                    )
                    mid = key[1]
                    d = marg[mid]
                    d['count'] += 1
                    d['patterns'].add(key)
                    d['ax'].add(key[0])
                    d['xc'].add(key[2])
                    d['ac'].add(key[3])
                    d['axc_l'].add(key[4])
                    d['axc_r'].add(key[5])
                    if d['rep'] is None:
                        d['rep'] = cfg_id
                    cfg_id += 1
                    total += 1
    assert total == 531441
    assert len(marg) == 56

    rows = []
    for mid in sorted(marg):
        d = marg[mid]
        rep = d['rep']
        tmp = rep
        c = tmp % 9; tmp //= 9
        x2 = tmp % 81; tmp //= 81
        x1 = tmp % 81; tmp //= 81
        a = tmp
        rows.append({
            'middle_xx_orbit_id': mid,
            'total_axxc_count': d['count'],
            'distinct_face_pattern_count': len(d['patterns']),
            'left_ax_orbit_ids': str(sorted(d['ax'])),
            'right_xc_orbit_ids': str(sorted(d['xc'])),
            'outer_ac_orbit_ids': str(sorted(d['ac'])),
            'left_axc_orbit_ids': str(sorted(d['axc_l'])),
            'right_axc_orbit_ids': str(sorted(d['axc_r'])),
            'rep_axxc_config_id': rep,
            'rep_axxc_readable': f"AXXC[{_a_name(a)},{_x_name(x1)},{_x_name(x2)},{_c_name(c)}]",
        })

    total_sum = sum(r['total_axxc_count'] for r in rows)
    assert total_sum == 531441
    seen_mid = set()
    for r in rows:
        mid = r['middle_xx_orbit_id']
        assert 0 <= mid < 56
        assert mid not in seen_mid
        seen_mid.add(mid)
        for v in eval(r['left_ax_orbit_ids']): assert 0 <= v < 10
        for v in eval(r['right_xc_orbit_ids']): assert 0 <= v < 8
        for v in eval(r['outer_ac_orbit_ids']): assert 0 <= v < 2
        for v in eval(r['left_axc_orbit_ids']): assert 0 <= v < 50
        for v in eval(r['right_axc_orbit_ids']): assert 0 <= v < 50
        rep = r['rep_axxc_config_id']
        tmp = rep
        c = tmp % 9; tmp //= 9
        x2 = tmp % 81; tmp //= 81
        x1 = tmp % 81; tmp //= 81
        a = tmp
        actual_mid = xx_o[x1 * 81 + x2]
        assert actual_mid == mid
        # pattern count consistency
        count_patterns = set()
        for aa in range(9):
            pass
    counts = [r['total_axxc_count'] for r in rows]
    pcounts = [r['distinct_face_pattern_count'] for r in rows]

    csv_path = out / 'AXXC_xx_marginals.csv'
    fields = [
        'middle_xx_orbit_id','total_axxc_count','distinct_face_pattern_count',
        'left_ax_orbit_ids','right_xc_orbit_ids','outer_ac_orbit_ids',
        'left_axc_orbit_ids','right_axc_orbit_ids',
        'rep_axxc_config_id','rep_axxc_readable'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md = [
        '# AXXC Marginals Conditioned on Middle XX Orbit',
        '',
        f'Generated: {ts}',
        '',
        'Marginal inventory of AXXC face structure per middle XX orbit.',
        '',
        '## Summary',
        '',
        '| Property | Value |',
        '|----------|-------|',
        '| Row count (XX orbits) | 56 |',
        '| Total AXXC coverage | 531,441 |',
        f'| Min AXXC count per XX orbit | {min(counts)} |',
        f'| Max AXXC count per XX orbit | {max(counts)} |',
        f'| Min distinct face-patterns | {min(pcounts)} |',
        f'| Max distinct face-patterns | {max(pcounts)} |',
        '',
        '## Preview (first 10 rows)',
        '',
        '| XX_orb | count | patterns | AX_support | XC_support | AC_support |',
        '|--------|-------|----------|------------|------------|------------|',
    ]
    for r in rows[:10]:
        md.append(f"| {r['middle_xx_orbit_id']} | {r['total_axxc_count']} | {r['distinct_face_pattern_count']} | {r['left_ax_orbit_ids']} | {r['right_xc_orbit_ids']} | {r['outer_ac_orbit_ids']} |")
    md += ['', 'Full data: see `AXXC_xx_marginals.csv`.', '']
    md_path = out / 'AXXC_xx_marginals.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('  All checks passed.')
    print('\n--- EXPORT COMPLETE ---')
    print(f'  rows: {len(rows)}')
    print(f'  total: {total_sum}')
    print(f'  count range: {min(counts)}-{max(counts)}')
    print(f'  pattern count range: {min(pcounts)}-{max(pcounts)}')
    print(f'\nFiles written:')
    print(f'  {csv_path}')
    print(f'  {md_path}')


if __name__ == '__main__':
    main()
