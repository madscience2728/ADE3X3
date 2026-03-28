"""
ade3x3_step28_export_cxxc_cxc_marginals.py

Export CXXC marginals conditioned on left and right CXC faces.
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

def _build_rows(acc, key_name):
    rows=[]
    for orb in sorted(acc):
        a=acc[orb]
        rep=a['rep']; c1,x1,x2,c2=_decode(rep)
        rows.append({
            f'{key_name}_orbit_id':orb,
            'total_cxxc_count':a['total'],
            'distinct_face_pattern_count':len(a['patterns']),
            'left_cx_orbit_ids':str(sorted(a['cx'])),
            'middle_xx_orbit_ids':str(sorted(a['xx'])),
            'right_xc_orbit_ids':str(sorted(a['xc'])),
            'outer_cc_orbit_ids':str(sorted(a['cc'])),
            'rep_cxxc_config_id':rep,
            'rep_cxxc_readable':f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
        })
    return rows

def _validate(rows, key_name, cxc_o, cx_o, xx_o, xc_o, cc_o):
    for r in rows:
        c1,x1,x2,c2=_decode(r['rep_cxxc_config_id'])
        if key_name=='left_cxc':
            assert cxc_o[(c1*81+x1)*9+c2]==r[f'{key_name}_orbit_id']
        else:
            assert cxc_o[(c1*81+x2)*9+c2]==r[f'{key_name}_orbit_id']
        for v in eval(r['left_cx_orbit_ids']): assert 0<=v<8
        for v in eval(r['middle_xx_orbit_ids']): assert 0<=v<56
        for v in eval(r['right_xc_orbit_ids']): assert 0<=v<8
        for v in eval(r['outer_cc_orbit_ids']): assert 0<=v<4

def _write_csv(rows, key_name, path):
    FIELDS=[f'{key_name}_orbit_id','total_cxxc_count','distinct_face_pattern_count',
            'left_cx_orbit_ids','middle_xx_orbit_ids','right_xc_orbit_ids',
            'outer_cc_orbit_ids','rep_cxxc_config_id','rep_cxxc_readable']
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        w.writerows(rows)

# ── Main ───────────────────────────────────────────────────────────

def main():
    out=Path('exports'); out.mkdir(exist_ok=True)

    print("Computing orbit maps...")
    cx_o=_orbit_map(9*81,_act_CX)
    xc_o=_orbit_map(81*9,_act_XC)
    cc_o=_orbit_map(81,_act_CC)
    xx_o=_orbit_map(81*81,_act_XX)
    cxc_o=_orbit_map(9*81*9,_act_CXC)

    print("Scanning CXXC CXC marginals...")
    left_acc=defaultdict(lambda:{'total':0,'patterns':defaultdict(int),
        'cx':set(),'xx':set(),'xc':set(),'cc':set(),'rep':None})
    right_acc=defaultdict(lambda:{'total':0,'patterns':defaultdict(int),
        'cx':set(),'xx':set(),'xc':set(),'cc':set(),'rep':None})

    cfg_id=0
    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    cx= cx_o[c1*81+x1]; xx=xx_o[x1*81+x2]; xc=xc_o[x2*9+c2]
                    cc= cc_o[c1*9+c2]; l_cxc=cxc_o[(c1*81+x1)*9+c2]; r_cxc=cxc_o[(c1*81+x2)*9+c2]
                    pat=(cx,xx,xc,cc,r_cxc)

                    la=left_acc[l_cxc]
                    la['total']+=1; la['patterns'][pat]+=1
                    la['cx'].add(cx); la['xx'].add(xx); la['xc'].add(xc); la['cc'].add(cc)
                    if la['rep'] is None: la['rep']=cfg_id

                    ra=right_acc[r_cxc]
                    ra['total']+=1; ra['patterns'][pat]+=1
                    ra['cx'].add(cx); ra['xx'].add(xx); ra['xc'].add(xc); ra['cc'].add(cc)
                    if ra['rep'] is None: ra['rep']=cfg_id

                    cfg_id+=1

    assert cfg_id==531441

    left_rows=_build_rows(left_acc,'left_cxc')
    right_rows=_build_rows(right_acc,'right_cxc')

    # Checks
    assert sum(r['total_cxxc_count'] for r in left_rows)==531441
    assert sum(r['total_cxxc_count'] for r in right_rows)==531441
    _validate(left_rows,'left_cxc',cxc_o,cx_o,xx_o,xc_o,cc_o)
    _validate(right_rows,'right_cxc',cxc_o,cx_o,xx_o,xc_o,cc_o)
    print("  All checks passed.")

    left_csv=out/'CXXC_left_CXC_marginals.csv'
    right_csv=out/'CXXC_right_CXC_marginals.csv'
    _write_csv(left_rows,'left_cxc',left_csv)
    _write_csv(right_rows,'right_cxc',right_csv)

    lc=[r['total_cxxc_count'] for r in left_rows]
    rc=[r['total_cxxc_count'] for r in right_rows]
    lp=[r['distinct_face_pattern_count'] for r in left_rows]
    rp=[r['distinct_face_pattern_count'] for r in right_rows]

    # Markdown
    ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md=[
        "# CXXC Marginals Conditioned on Left and Right CXC Faces",
        "",
        f"Generated: {ts}",
        "",
        "Marginal inventory of CXXC face structure per left and right CXC orbit.",
        "",
        "## Summary",
        "",
        "| Property | Left CXC | Right CXC |",
        "|----------|----------|-----------|",
        f"| Row count | {len(left_rows)} | {len(right_rows)} |",
        f"| Total coverage | 531,441 | 531,441 |",
        f"| Min count | {min(lc)} | {min(rc)} |",
        f"| Max count | {max(lc)} | {max(rc)} |",
        f"| Min distinct patterns | {min(lp)} | {min(rp)} |",
        f"| Max distinct patterns | {max(lp)} | {max(rp)} |",
        "",
        "## Left CXC Preview (first 10 rows)",
        "",
        "| CXC_orb | count | patterns | CX_support | XX_support | XC_support | CC_support |",
        "|---------|-------|----------|------------|------------|------------|------------|",
    ]
    for r in left_rows[:10]:
        md.append(f"| {r['left_cxc_orbit_id']} | {r['total_cxxc_count']} "
                  f"| {r['distinct_face_pattern_count']} "
                  f"| {r['left_cx_orbit_ids']} | {r['middle_xx_orbit_ids']} "
                  f"| {r['right_xc_orbit_ids']} | {r['outer_cc_orbit_ids']} |")
    md+=["",
        "## Right CXC Preview (first 10 rows)",
        "",
        "| CXC_orb | count | patterns | CX_support | XX_support | XC_support | CC_support |",
        "|---------|-------|----------|------------|------------|------------|------------|",
    ]
    for r in right_rows[:10]:
        md.append(f"| {r['right_cxc_orbit_id']} | {r['total_cxxc_count']} "
                  f"| {r['distinct_face_pattern_count']} "
                  f"| {r['left_cx_orbit_ids']} | {r['middle_xx_orbit_ids']} "
                  f"| {r['right_xc_orbit_ids']} | {r['outer_cc_orbit_ids']} |")
    md+=["","Full data: see the CSVs.",""]

    md_path=out/'CXXC_cxc_marginals.md'
    with open(md_path,'w',encoding='utf-8') as f: f.write("\n".join(md))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  left rows: {len(left_rows)}, count {min(lc)}-{max(lc)}, patterns {min(lp)}-{max(lp)}")
    print(f"  right rows: {len(right_rows)}, count {min(rc)}-{max(rc)}, patterns {min(rp)}-{max(rp)}")
    print(f"\nFiles written:")
    print(f"  {left_csv}")
    print(f"  {right_csv}")
    print(f"  {md_path}")


if __name__=='__main__':
    main()
