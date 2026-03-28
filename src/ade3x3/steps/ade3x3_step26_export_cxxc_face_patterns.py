"""
ade3x3_step26_export_cxxc_face_patterns.py

Aggregate CXXC face inventory by orbit-pattern key and export counts.
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

# ── Main ───────────────────────────────────────────────────────────

def main():
    out=Path('exports'); out.mkdir(exist_ok=True)

    print("Computing orbit maps...")
    cx_o=_orbit_map(9*81,_act_CX)
    xc_o=_orbit_map(81*9,_act_XC)
    cc_o=_orbit_map(81,_act_CC)
    xx_o=_orbit_map(81*81,_act_XX)
    cxc_o=_orbit_map(9*81*9,_act_CXC)

    print("Scanning CXXC face patterns...")
    # pattern_key -> {'count': int, 'rep': cxxc_cfg}
    patterns=defaultdict(lambda:{'count':0,'rep':None})

    cfg_id=0
    for c1 in range(9):
        for x1 in range(81):
            for x2 in range(81):
                for c2 in range(9):
                    key=(cx_o[c1*81+x1], xx_o[x1*81+x2], xc_o[x2*9+c2],
                         cc_o[c1*9+c2], cxc_o[(c1*81+x1)*9+c2], cxc_o[(c1*81+x2)*9+c2])
                    p=patterns[key]
                    p['count']+=1
                    if p['rep'] is None: p['rep']=cfg_id
                    cfg_id+=1

    assert cfg_id==531441
    print(f"  Total rows: {cfg_id}")
    print(f"  Distinct patterns: {len(patterns)}")

    # Sort keys for deterministic pattern_id
    sorted_keys=sorted(patterns.keys())

    # Write CSV
    csv_path=out/'CXXC_face_patterns.csv'
    FIELDS=['pattern_id','count',
            'left_cx_orbit_id','middle_xx_orbit_id','right_xc_orbit_id',
            'outer_cc_orbit_id','left_cxc_orbit_id','right_cxc_orbit_id',
            'rep_cxxc_config_id','rep_cxxc_readable']
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        rows=[]
        for pid,key in enumerate(sorted_keys):
            entry=patterns[key]
            rep=entry['rep']
            c1,x1,x2,c2=_decode_rep(rep)
            rows.append({
                'pattern_id':pid,'count':entry['count'],
                'left_cx_orbit_id':key[0],'middle_xx_orbit_id':key[1],
                'right_xc_orbit_id':key[2],'outer_cc_orbit_id':key[3],
                'left_cxc_orbit_id':key[4],'right_cxc_orbit_id':key[5],
                'rep_cxxc_config_id':rep,
                'rep_cxxc_readable':f"CXXC[{_c_name(c1)},{_x_name(x1)},{_x_name(x2)},{_c_name(c2)}]",
            })
        w.writerows(rows)

    # Sanity checks
    total=sum(e['count'] for e in patterns.values())
    assert total==531441, f"Sum {total} != 531441"

    # Verify representatives
    for pid,key in enumerate(sorted_keys):
        rep=patterns[key]['rep']
        c1,x1,x2,c2=_decode_rep(rep)
        actual_key=(cx_o[c1*81+x1], xx_o[x1*81+x2], xc_o[x2*9+c2],
                    cc_o[c1*9+c2], cxc_o[(c1*81+x1)*9+c2], cxc_o[(c1*81+x2)*9+c2])
        assert actual_key==key, f"Rep {rep} key mismatch"

    counts=[e['count'] for e in patterns.values()]
    mn,mx=min(counts),max(counts)

    print("  All checks passed.")

    # Markdown
    ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md=[
        "# CXXC Face-Pattern Summary",
        "",
        f"Generated: {ts}",
        "",
        "Orbit-level face-pattern summary for CXXC = C × X × X × C.",
        "",
        "## Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Total CXXC rows | 531,441 |",
        f"| Distinct face-patterns | {len(patterns)} |",
        f"| Min pattern count | {mn} |",
        f"| Max pattern count | {mx} |",
        "",
        "Patterns are keyed by 6-tuple:",
        "(left CX orbit, middle XX orbit, right XC orbit,",
        "outer CC orbit, left CXC orbit, right CXC orbit).",
        "",
        "## Preview (first 10 patterns)",
        "",
        "| pid | count | CX | XX | XC | CC | CXC_L | CXC_R |",
        "|-----|-------|----|----|----|----|----|-----|",
    ]
    for r in rows[:10]:
        md.append(f"| {r['pattern_id']} | {r['count']} | {r['left_cx_orbit_id']} "
                  f"| {r['middle_xx_orbit_id']} | {r['right_xc_orbit_id']} "
                  f"| {r['outer_cc_orbit_id']} | {r['left_cxc_orbit_id']} "
                  f"| {r['right_cxc_orbit_id']} |")
    md+=["", "Full data: see `CXXC_face_patterns.csv`.",""]

    md_path=out/'CXXC_face_patterns.md'
    with open(md_path,'w',encoding='utf-8') as f: f.write("\n".join(md))

    print(f"\n--- EXPORT COMPLETE ---")
    print(f"  distinct patterns: {len(patterns)}")
    print(f"  count range: {mn}-{mx}")
    print(f"  sum: {total}")
    print(f"\nFiles written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")


def _decode_rep(cfg_id):
    """Decode CXXC config_id back to (c1,x1,x2,c2)."""
    c2=cfg_id%9; cfg_id//=9
    x2=cfg_id%81; cfg_id//=81
    x1=cfg_id%81; cfg_id//=81
    c1=cfg_id
    return c1,x1,x2,c2


if __name__=='__main__':
    main()
