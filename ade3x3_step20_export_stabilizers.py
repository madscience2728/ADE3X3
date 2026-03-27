"""
ade3x3_step20_export_stabilizers.py

Export explicit stabilizer action ID lists for every orbit representative.
Raw-data export only — no analysis, no redesign.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Group of size 216 ──────────────────────────────────────────────

S3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
ACTIONS = [(p1, p2, p3) for p1 in S3 for p2 in S3 for p3 in S3]

# ── Schema action functions ────────────────────────────────────────

def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return (9*(3*act[0][r1]+act[1][s1])+(3*act[1][t1]+act[2][u1]))*81 + \
           (9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2]))

def _act_CX(cfg, act):
    c,x = divmod(cfg,81); r,u=divmod(c,3)
    r2,s=divmod(x//9,3); t,u2=divmod(x%9,3)
    return (3*act[0][r]+act[2][u])*81 + 9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x,c = divmod(cfg,9); r,s=divmod(x//9,3); t,u=divmod(x%9,3)
    r2,u2=divmod(c,3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def _act_CC(cfg, act):
    c1,c2=divmod(cfg,9); r1,u1=divmod(c1,3); r2,u2=divmod(c2,3)
    return 9*(3*act[0][r1]+act[2][u1])+(3*act[0][r2]+act[2][u2])

def _act_AX(cfg, act):
    a,x=divmod(cfg,81); r,s=divmod(a,3)
    r2,s2=divmod(x//9,3); t,u=divmod(x%9,3)
    return (3*act[0][r]+act[1][s])*81+9*(3*act[0][r2]+act[1][s2])+(3*act[1][t]+act[2][u])

def _act_BX(cfg, act):
    b,x=divmod(cfg,81); t,u=divmod(b,3)
    r2,s2=divmod(x//9,3); t2,u2=divmod(x%9,3)
    return (3*act[1][t]+act[2][u])*81+9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2])

def _act_CXC(cfg, act):
    c1=cfg//(81*9); rem=cfg%(81*9); x=rem//9; c2=rem%9
    r1,u1=divmod(c1,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3); r3,u3=divmod(c2,3)
    return ((3*act[0][r1]+act[2][u1])*81+9*(3*act[0][r2]+act[1][s])+
            (3*act[1][t]+act[2][u2]))*9+(3*act[0][r3]+act[2][u3])

def _name_XX(cfg):
    x1,x2=divmod(cfg,81)
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return f"XX[X[{r1},{s1}|{t1},{u1}],X[{r2},{s2}|{t2},{u2}]]"

def _name_CX(cfg):
    c,x=divmod(cfg,81); r,u=divmod(c,3)
    r2,s=divmod(x//9,3); t,u2=divmod(x%9,3)
    return f"CX[C[{r},{u}],X[{r2},{s}|{t},{u2}]]"

def _name_XC(cfg):
    x,c=divmod(cfg,9); r,s=divmod(x//9,3); t,u=divmod(x%9,3)
    r2,u2=divmod(c,3)
    return f"XC[X[{r},{s}|{t},{u}],C[{r2},{u2}]]"

def _name_CC(cfg):
    c1,c2=divmod(cfg,9); r1,u1=divmod(c1,3); r2,u2=divmod(c2,3)
    return f"CC[C[{r1},{u1}],C[{r2},{u2}]]"

def _name_AX(cfg):
    a,x=divmod(cfg,81); r,s=divmod(a,3)
    r2,s2=divmod(x//9,3); t,u=divmod(x%9,3)
    return f"AX[A[{r},{s}],X[{r2},{s2}|{t},{u}]]"

def _name_BX(cfg):
    b,x=divmod(cfg,81); t,u=divmod(b,3)
    r2,s2=divmod(x//9,3); t2,u2=divmod(x%9,3)
    return f"BX[B[{t},{u}],X[{r2},{s2}|{t2},{u2}]]"

def _name_CXC(cfg):
    c1=cfg//(81*9); rem=cfg%(81*9); x=rem//9; c2=rem%9
    r1,u1=divmod(c1,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3); r3,u3=divmod(c2,3)
    return f"CXC[C[{r1},{u1}],X[{r2},{s}|{t},{u2}],C[{r3},{u3}]]"

SCHEMA = {
    'XX':  (81*81,  _name_XX,  _act_XX),
    'CX':  (9*81,   _name_CX,  _act_CX),
    'XC':  (81*9,   _name_XC,  _act_XC),
    'CC':  (81,     _name_CC,  _act_CC),
    'AX':  (9*81,   _name_AX,  _act_AX),
    'BX':  (9*81,   _name_BX,  _act_BX),
    'CXC': (9*81*9, _name_CXC, _act_CXC),
}

EXPECTED_ORBITS = {'XX':56,'CX':8,'XC':8,'CC':4,'AX':10,'BX':10,'CXC':50}
SCHEMAS = ['XX','CX','XC','CC','AX','BX','CXC']

# ── Orbit computation + stabilizer enumeration ─────────────────────

def build_orbits_with_stabilizers(schema):
    n, name_fn, act_fn = SCHEMA[schema]
    # Canonical representatives
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

    orbits = []
    for oid, (rep, members) in enumerate(sorted(groups.items())):
        size = len(members)
        # Enumerate actual stabilizer action IDs
        stab_ids = sorted(aid for aid, a in enumerate(ACTIONS) if act_fn(rep, a) == rep)
        orbits.append({
            'orbit_id': oid,
            'rep_config_id': rep,
            'rep_readable': name_fn(rep),
            'orbit_size': size,
            'stabilizer_size': len(stab_ids),
            'stabilizer_action_ids': stab_ids,
        })
    return orbits

# ── CSV / Markdown writers ─────────────────────────────────────────

CSV_FIELDS = ['schema','orbit_id','rep_config_id','rep_readable',
              'orbit_size','stabilizer_size','stabilizer_action_ids']

def write_csv(orbits, schema, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for o in orbits:
            w.writerow({
                'schema': schema,
                'orbit_id': o['orbit_id'],
                'rep_config_id': o['rep_config_id'],
                'rep_readable': o['rep_readable'],
                'orbit_size': o['orbit_size'],
                'stabilizer_size': o['stabilizer_size'],
                'stabilizer_action_ids': json.dumps(o['stabilizer_action_ids']),
            })

def write_summary(results, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# Stabilizer Element Export Summary",
        "",
        f"Generated: {ts}",
        "",
        "Explicit stabilizer action ID lists for every orbit representative.",
        "",
        "## Summary Table",
        "",
        "| Schema | Orbit Count | Min Stab | Max Stab |",
        "|--------|-------------|----------|----------|",
    ]
    for sch, orbits in results:
        sizes = [o['stabilizer_size'] for o in orbits]
        lines.append(f"| {sch} | {len(orbits)} | {min(sizes)} | {max(sizes)} |")
    lines += ["", "## Exported CSV Files", ""]
    for sch, _ in results:
        lines.append(f"- `stabilizers_{sch}.csv`")
    lines += [
        "",
        "Stabilizer element sets are now explicitly recorded as",
        "object-level warehouse data.",
        "",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Computing orbits and stabilizer elements...\n")
    results = []

    for sch in SCHEMAS:
        orbits = build_orbits_with_stabilizers(sch)
        n_orb = len(orbits)
        exp = EXPECTED_ORBITS[sch]
        assert n_orb == exp, f"{sch}: {n_orb} orbits != expected {exp}"

        # Sanity checks per row
        for o in orbits:
            assert len(o['stabilizer_action_ids']) == o['stabilizer_size'], \
                f"{sch} orbit {o['orbit_id']}: stab list size mismatch"
            assert o['orbit_size'] * o['stabilizer_size'] == 216, \
                f"{sch} orbit {o['orbit_id']}: {o['orbit_size']}*{o['stabilizer_size']} != 216"
            assert 0 in o['stabilizer_action_ids'], \
                f"{sch} orbit {o['orbit_id']}: identity action 0 missing"
            _, _, act_fn = SCHEMA[sch]
            for aid in o['stabilizer_action_ids']:
                assert act_fn(o['rep_config_id'], ACTIONS[aid]) == o['rep_config_id'], \
                    f"{sch} orbit {o['orbit_id']}: action {aid} does not fix rep"

        write_csv(orbits, sch, out / f'stabilizers_{sch}.csv')
        results.append((sch, orbits))

        sizes = [o['stabilizer_size'] for o in orbits]
        print(f"  {sch}: {n_orb} orbits, stab range {min(sizes)}-{max(sizes)}")

    print("\nAll checks passed.")

    md_path = out / 'stabilizers_summary.md'
    write_summary(results, md_path)

    print(f"\n--- EXPORT COMPLETE ---")
    for sch, orbits in results:
        sizes = [o['stabilizer_size'] for o in orbits]
        print(f"  {sch}: {len(orbits)} rows, stab {min(sizes)}-{max(sizes)}")
    print(f"\nFiles written to {out}/")
    for sch, _ in results:
        print(f"  stabilizers_{sch}.csv")
    print(f"  stabilizers_summary.md")


if __name__ == '__main__':
    main()
