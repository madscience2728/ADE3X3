"""
ade3x3_step29_export_cxxc_joint_cxc_xx_cxc.py

Export joint co-occurrence of left CXC, middle XX, right CXC orbits in CXXC.
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

    print("Scanning CXXC joint keys...")
    joint=defaultdict(lambda:{'count':0,'rep':None,'cc':set(),'cx':set(),'xc':set()})

    cfg_id=0
    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    l_cxc=cxc_o[(c1*81+x1)*9+c2]
                    xx=xx_o[x1*81+x2]
                    r_cxc=cxc_o[(c1*81+x2)*9+c2]
                    key=(l_cxc,xx,r_cxc)
                    j=joint[key]
                    j['count']+=1
                    j['cc'].add(cc_o[c1*9+c2])
                    j['cx'].add(cx_o[c1*81+x1])
                    j['xc'].add(xc_o[x2*9+c2])
                    if j['rep'] is None: j['rep']=cfg_id
                    cfg_id+=1

    assert cfg_id==531441

    sorted_keys=sorted(joint.keys())
    rows=[]
    for jid,key in enumerate(sorted_keys):
        j=joint[key]
        rep=j['rep']; c1,x1,x2,c2=_decode(rep)
        rows.append({
            'joint_id':jid,'count':j['count'],
            'left_cxc_orbit_id':key[0],'middle_xx_orbit_id':key[1],'right_cxc_orbit_id':key[2],
            'outer_cc_orbit_ids':str(sorted(j['cc'])),
            'left_cx_orbit_ids':str(sorted(j['cx'])),
            'right_xc_orbit_ids':str(sorted(j['xc'])),
            'rep_cxxc_config_id':rep,
            'rep_cxxc_readable':f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
        })

    total=sum(r['count'] for r in rows)
    assert total==531441

    for r in rows:
        c1,x1,x2,c2=_decode(r['rep_cxxc_config_id'])
        assert cxc_o[(c1*81+x1)*9+c2]==r['left_cxc_orbit_id']
        assert xx_o[x1*81+x2]==r['middle_xx_orbit_id']
        assert cxc_o[(c1*81+x2)*9+c2]==r['right_cxc_orbit_id']
        assert 0<=r['left_cxc_orbit_id']<50
        assert 0<=r['middle_xx_orbit_id']<56
        assert 0<=r['right_cxc_orbit_id']<50

    counts=[r['count'] for r in rows]
    mn,mx=min(counts),max(counts)
    print("  All checks passed.")

    csv_path=out/'CXXC_joint_CXC_XX_CXC.csv'
    FIELDS=['joint_id','count','left_cxc_orbit_id','middle_xx_orbit_id','right_cxc_orbit_id',
            'outer_cc_orbit_ids','left_cx_orbit_ids','right_xc_orbit_ids',
            'rep_cxxc_config_id','rep_cxxc_readable']
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        w.writerows(rows)

    ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md=[
        "# CXXC Joint Co-occurrence: Left CXC × Middle XX × Right CXC",
        "",
        f"Generated: {ts}",
        "",
        "Joint co-occurrence inventory for the three key interior face orbits of CXXC.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Total CXXC rows | 531,441 |",
        f"| Distinct joint keys | {len(rows)} |",
        f"| Min joint count | {mn} |",
        f"| Max joint count | {mx} |",
        "",
        "## Preview (first 10 rows)",
        "",
        "| jid | count | L_CXC | XX | R_CXC | CC_support | CX_support | XC_support |",
        "|-----|-------|-------|----|-------|------------|------------|------------|",
    ]
    for r in rows[:10]:
        md.append(f"| {r['joint_id']} | {r['count']} "
                  f"| {r['left_cxc_orbit_id']} | {r['middle_xx_orbit_id']} | {r['right_cxc_orbit_id']} "
                  f"| {r['outer_cc_orbit_ids']} | {r['left_cx_orbit_ids']} | {r['right_xc_orbit_ids']} |")
    md+=["","Full data: see `CXXC_joint_CXC_XX_CXC.csv`.",""]

    md_path=out/'CXXC_joint_CXC_XX_CXC.md'
    with open(md_path,'w',encoding='utf-8') as f: f.write("\n".join(md))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  distinct joint keys: {len(rows)}")
    print(f"  count range: {mn}-{mx}")
    print(f"  sum: {total}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__=='__main__':
    main()
