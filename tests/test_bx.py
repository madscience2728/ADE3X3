import csv, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def s3(): return [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def canon_pat(v):
    m, n = {}, 0
    for x in v:
        if x not in m: m[x] = n; n += 1
    return tuple(m.values())

def BX_sig(cfg):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3)
    t2, u2 = divmod(x % 9, 3)
    live = s2 == t2
    return (t==t2, u==u2, live)

def BX_act(cfg, act):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3)
    t2, u2 = divmod(x % 9, 3)
    b_new = 3*act[1][t] + act[2][u]
    x_new = 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t2]+act[2][u2])
    return b_new*81 + x_new

print("BX Sig Test:", BX_sig(0))

acts = [(p1,p2,p3) for p1 in s3() for p2 in s3() for p3 in s3()]
print("acts", len(acts))

# orbit rep search 
can = {}
for cfg in range(9*81):
    m = cfg
    for a in acts:
        img = BX_act(cfg, a)
        if img < m: m = img
    can[cfg] = m

groups = defaultdict(list)
for c in range(9*81): groups[can[c]].append(c)

print("BX orbits:", len(groups))
for k,v in sorted(groups.items())[:3]:
    print(k, len(v))