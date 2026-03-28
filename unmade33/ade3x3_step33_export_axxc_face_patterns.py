"""
ade3x3_step33_export_axxc_face_patterns.py

Aggregate AXXC face inventory by orbit-pattern key and export counts.
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


def _decode_rep(cfg_id):
    c = cfg_id % 9
    cfg_id //= 9
    x2 = cfg_id % 81
    cfg_id //= 81
    x1 = cfg_id % 81
    cfg_id //= 81
    a = cfg_id
    return a, x1, x2, c


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

    print('Scanning AXXC face patterns...')
    patterns = defaultdict(lambda: {'count': 0, 'rep': None})
    cfg_id = 0
    for a in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c in range(9):
                    key = (
                        ax_o[a * 81 + x1],
                        xx_o[x1 * 81 + x2],
                        xc_o[x2 * 9 + c],
                        ac_o[a * 9 + c],
                        axc_o[(a * 81 + x1) * 9 + c],
                        axc_o[(a * 81 + x2) * 9 + c],
                    )
                    p = patterns[key]
                    p['count'] += 1
                    if p['rep'] is None:
                        p['rep'] = cfg_id
                    cfg_id += 1
    assert cfg_id == 531441
    print(f'  Total rows: {cfg_id}')
    print(f'  Distinct patterns: {len(patterns)}')

    sorted_keys = sorted(patterns.keys())
    rows = []
    for pid, key in enumerate(sorted_keys):
        rep = patterns[key]['rep']
        a, x1, x2, c = _decode_rep(rep)
        rows.append({
            'pattern_id': pid,
            'count': patterns[key]['count'],
            'left_ax_orbit_id': key[0],
            'middle_xx_orbit_id': key[1],
            'right_xc_orbit_id': key[2],
            'outer_ac_orbit_id': key[3],
            'left_axc_orbit_id': key[4],
            'right_axc_orbit_id': key[5],
            'rep_axxc_config_id': rep,
            'rep_axxc_readable': f"AXXC[{_a_name(a)},{_x_name(x1)},{_x_name(x2)},{_c_name(c)}]",
        })

    csv_path = out / 'AXXC_face_patterns.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = sum(p['count'] for p in patterns.values())
    assert total == 531441, f'Sum {total} != 531441'
    for key in sorted_keys:
        rep = patterns[key]['rep']
        a, x1, x2, c = _decode_rep(rep)
        actual = (
            ax_o[a * 81 + x1],
            xx_o[x1 * 81 + x2],
            xc_o[x2 * 9 + c],
            ac_o[a * 9 + c],
            axc_o[(a * 81 + x1) * 9 + c],
            axc_o[(a * 81 + x2) * 9 + c],
        )
        assert actual == key, f'rep {rep} mismatch'

    counts = [p['count'] for p in patterns.values()]
    mn, mx = min(counts), max(counts)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md = [
        '# AXXC Face-Pattern Summary',
        '',
        f'Generated: {ts}',
        '',
        'Orbit-level face-pattern summary for AXXC = A × X × X × C.',
        '',
        '## Summary',
        '',
        '| Property | Value |',
        '|----------|-------|',
        '| Total AXXC rows | 531,441 |',
        f'| Distinct face-patterns | {len(patterns)} |',
        f'| Min pattern count | {mn} |',
        f'| Max pattern count | {mx} |',
        '',
        'Patterns are keyed by 6-tuple:',
        '(left AX orbit, middle XX orbit, right XC orbit,',
        'outer AC orbit, left AXC orbit, right AXC orbit).',
        '',
        '## Preview (first 10 patterns)',
        '',
        '| pid | count | AX | XX | XC | AC | AXC_L | AXC_R |',
        '|-----|-------|----|----|----|----|-------|-------|',
    ]
    for r in rows[:10]:
        md.append(
            f"| {r['pattern_id']} | {r['count']} | {r['left_ax_orbit_id']} | {r['middle_xx_orbit_id']} | {r['right_xc_orbit_id']} | {r['outer_ac_orbit_id']} | {r['left_axc_orbit_id']} | {r['right_axc_orbit_id']} |"
        )
    md += ['', 'Full data: see `AXXC_face_patterns.csv`.', '']

    md_path = out / 'AXXC_face_patterns.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('  All checks passed.')
    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  distinct patterns: {len(patterns)}")
    print(f"  count range: {mn}-{mx}")
    print(f"  sum: {total}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == '__main__':
    main()
