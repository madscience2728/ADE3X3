"""
ade3x3_step27_export_cxxc_xx_marginals.py

Export CXXC marginals conditioned on the middle XX orbit.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Group + actions ────────────────────────────────────────────────

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]

def _act_CX(cfg, act):
    c,x=divmod(cfg,81); r,u=divmod(c,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3)
    return (3*act[0][r]+act[2][u])*81+9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x,c=divmod(cfg,9); r,s=divmod(x//9,3); t,u=divmod(x%9,3); r2,u2=divmod(c,3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def _act_CC(cfg, act):
    c1,c2=divmod(cfg,9); r1,u1=divmod(c1,3); r2,u2=divmod(c2,3)
    return 9*(3*act[0][r1]+act[2][u1])+(3*act[0][r2]+act[2][u2])

def _act_XX(cfg, act):
    x1,x2=divmod(cfg,81)
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return (9*(3*act[0][r1]+act[1][s1])+(3*act[1][t1]+act[2][u1]))*81+(9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2]))

def _act_CXC(cfg, act):
    c1=cfg//(81*9); rem=cfg%(81*9); x=rem//9; c2=rem%9
    r1,u1=divmod(c1,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3); r3,u3=divmod(c2,3)
    return ((3*act[0][r1]+act[2][u1])*81+9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2]))*9+(3*act[0][r3]+act[2][u3])

def _orbit_map(n, fn):
    mr={}
    for c in range(n):
        m=c
        for a in ACTIONS:
            v=fn(c,a)
            if v<m: m=v
        mr[c]=m
    g=defaultdict(list)
    for c in range(n): g[mr[c]].append(c)
    om={}
    for oid,(_,ms) in enumerate(sorted(g.items())):
        for m in ms: om[m]=oid
    return om

def _c_name(c):
    r,u=divmod(c,3); return f"C[{r},{u}]"

def _x_name(x):
    r,s=divmod(x//9,3); t,u=divmod(x%9,3); return f"X[{r},{s}|{t},{u}]"

def _decode(cfg):
    c2=cfg%9; cfg//=9; x2=cfg%81; cfg//=81; x1=cfg%81; c1=cfg//81
    return c1,x1,x2,c2

# ── Main ───────────────────────────────────────────────────────────

def main():
    out=Path('exports'); out.mkdir(exist_ok=True)

    print("Computing orbit maps...")
    cx_o=_orbit_map(9*81,_act_CX)
    xc_o=_orbit_map(81*9,_act_XC)
    cc_o=_orbit_map(81,_act_CC)
    xx_o=_orbit_map(81*81,_act_XX)
    cxc_o=_orbit_map(9*81*9,_act_CXC)

    print("Scanning CXXC marginals...")
    # Per-XX-orbit accumulator
    acc = defaultdict(lambda:{
        'total':0,
        'patterns':defaultdict(int),
        'cx':set(),'xc':set(),'cc':set(),'cxc_l':set(),'cxc_r':set(),
        'rep':None
    })

    cfg_id=0
    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    xx=xx_o[x1*81+x2]
                    a=acc[xx]
                    a['total']+=1
                    pat=(cx_o[c1*81+x1], xc_o[x2*9+c2], cc_o[c1*9+c2],
                         cxc_o[(c1*81+x1)*9+c2], cxc_o[(c1*81+x2)*9+c2])
                    a['patterns'][pat]+=1
                    a['cx'].add(cx_o[c1*81+x1])
                    a['xc'].add(xc_o[x2*9+c2])
                    a['cc'].add(cc_o[c1*9+c2])
                    a['cxc_l'].add(cxc_o[(c1*81+x1)*9+c2])
                    a['cxc_r'].add(cxc_o[(c1*81+x2)*9+c2])
                    if a['rep'] is None: a['rep']=cfg_id
                    cfg_id+=1

    assert cfg_id==531441

    # Build rows sorted by XX orbit id
    rows=[]
    for xx_orb in sorted(acc):
        a=acc[xx_orb]
        rep=a['rep']; c1,x1,x2,c2=_decode(rep)
        rows.append({
            'middle_xx_orbit_id':xx_orb,
            'total_cxxc_count':a['total'],
            'distinct_face_pattern_count':len(a['patterns']),
            'left_cx_orbit_ids':str(sorted(a['cx'])),
            'right_xc_orbit_ids':str(sorted(a['xc'])),
            'outer_cc_orbit_ids':str(sorted(a['cc'])),
            'left_cxc_orbit_ids':str(sorted(a['cxc_l'])),
            'right_cxc_orbit_ids':str(sorted(a['cxc_r'])),
            'rep_cxxc_config_id':rep,
            'rep_cxxc_readable':f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
        })

    # Sanity checks
    assert len(rows)==56
    total_sum=sum(r['total_cxxc_count'] for r in rows)
    assert total_sum==531441

    for r in rows:
        # Rep validity
        c1,x1,x2,c2=_decode(r['rep_cxxc_config_id'])
        assert xx_o[x1*81+x2]==r['middle_xx_orbit_id']
        # Support validity
        for v in eval(r['left_cx_orbit_ids']): assert 0<=v<8
        for v in eval(r['right_xc_orbit_ids']): assert 0<=v<8
        for v in eval(r['outer_cc_orbit_ids']): assert 0<=v<4
        for v in eval(r['left_cxc_orbit_ids']): assert 0<=v<50
        for v in eval(r['right_cxc_orbit_ids']): assert 0<=v<50

    counts=[r['total_cxxc_count'] for r in rows]
    pcounts=[r['distinct_face_pattern_count'] for r in rows]
    print("  All checks passed.")

    # Write CSV
    csv_path=out/'CXXC_xx_marginals.csv'
    FIELDS=['middle_xx_orbit_id','total_cxxc_count','distinct_face_pattern_count',
            'left_cx_orbit_ids','right_xc_orbit_ids','outer_cc_orbit_ids',
            'left_cxc_orbit_ids','right_cxc_orbit_ids',
            'rep_cxxc_config_id','rep_cxxc_readable']
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        w.writerows(rows)

    # Markdown
    ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md=[
        "# CXXC Marginals Conditioned on Middle XX Orbit",
        "",
        f"Generated: {ts}",
        "",
        "Marginal inventory of CXXC face structure per middle XX orbit.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Row count (XX orbits) | 56 |",
        f"| Total CXXC coverage | 531,441 |",
        f"| Min CXXC count per XX orbit | {min(counts)} |",
        f"| Max CXXC count per XX orbit | {max(counts)} |",
        f"| Min distinct face-patterns | {min(pcounts)} |",
        f"| Max distinct face-patterns | {max(pcounts)} |",
        "",
        "## Preview (first 10 rows)",
        "",
        "| XX_orb | count | patterns | CX_support | XC_support | CC_support |",
        "|--------|-------|----------|------------|------------|------------|",
    ]
    for r in rows[:10]:
        md.append(f"| {r['middle_xx_orbit_id']} | {r['total_cxxc_count']} "
                  f"| {r['distinct_face_pattern_count']} "
                  f"| {r['left_cx_orbit_ids']} "
                  f"| {r['right_xc_orbit_ids']} "
                  f"| {r['outer_cc_orbit_ids']} |")
    md+=["", "Full data: see `CXXC_xx_marginals.csv`.",""]

    md_path=out/'CXXC_xx_marginals.md'
    with open(md_path,'w',encoding='utf-8') as f: f.write("\n".join(md))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  rows: {len(rows)}")
    print(f"  total: {total_sum}")
    print(f"  count range: {min(counts)}-{max(counts)}")
    print(f"  pattern count range: {min(pcounts)}-{max(pcounts)}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__=='__main__':
    main()
