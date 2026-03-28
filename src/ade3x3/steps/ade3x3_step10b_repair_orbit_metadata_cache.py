"""
ade3x3_step10b_repair_orbit_metadata_cache.py

Repair the orbit metadata/signature cache.
Bug: step10 used orbit_id as config_id index, producing wrong representatives.
Fix: recompute from actual canonical orbit representatives.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Group of size 216 ──────────────────────────────────────────────

S3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
ACTIONS = [(p1, p2, p3) for p1 in S3 for p2 in S3 for p3 in S3]

# ── Schema action functions (step18 encoding, correct B-action) ────

def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3); t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3); t2, u2 = divmod(x2 % 9, 3)
    x1n = 9*(3*act[0][r1]+act[1][s1]) + (3*act[1][t1]+act[2][u1])
    x2n = 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t2]+act[2][u2])
    return x1n * 81 + x2n

def _act_CX(cfg, act):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    return (3*act[0][r]+act[2][u]) * 81 + 9*(3*act[0][r2]+act[1][s]) + (3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r2, u2 = divmod(c, 3)
    return (9*(3*act[0][r]+act[1][s]) + (3*act[1][t]+act[2][u])) * 9 + (3*act[0][r2]+act[2][u2])

def _act_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return 9*(3*act[0][r1]+act[2][u1]) + (3*act[0][r2]+act[2][u2])

def _act_AX(cfg, act):
    a, x = divmod(cfg, 81)
    r, s = divmod(a, 3)
    r2, s2 = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    return (3*act[0][r]+act[1][s]) * 81 + 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t]+act[2][u])

def _act_BX(cfg, act):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3); t2, u2 = divmod(x % 9, 3)
    return (3*act[1][t]+act[2][u]) * 81 + 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t2]+act[2][u2])

def _act_CXC(cfg, act):
    c1 = cfg // (81*9); rem = cfg % (81*9); x = rem // 9; c2 = rem % 9
    r1, u1 = divmod(c1, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    r3, u3 = divmod(c2, 3)
    return ((3*act[0][r1]+act[2][u1])*81 + 9*(3*act[0][r2]+act[1][s]) + (3*act[1][t]+act[2][u2]))*9 + (3*act[0][r3]+act[2][u3])

def _name_XX(cfg):
    x1, x2 = divmod(cfg, 81)
    r1,s1 = divmod(x1//9,3); t1,u1 = divmod(x1%9,3)
    r2,s2 = divmod(x2//9,3); t2,u2 = divmod(x2%9,3)
    return f"XX[X[{r1},{s1}|{t1},{u1}],X[{r2},{s2}|{t2},{u2}]]"

def _name_CX(cfg):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x//9, 3); t, u2 = divmod(x%9, 3)
    return f"CX[C[{r},{u}],X[{r2},{s}|{t},{u2}]]"

def _name_XC(cfg):
    x, c = divmod(cfg, 9)
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    r2, u2 = divmod(c, 3)
    return f"XC[X[{r},{s}|{t},{u}],C[{r2},{u2}]]"

def _name_CC(cfg):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return f"CC[C[{r1},{u1}],C[{r2},{u2}]]"

def _name_AX(cfg):
    a, x = divmod(cfg, 81)
    r, s = divmod(a, 3)
    r2, s2 = divmod(x//9, 3); t, u = divmod(x%9, 3)
    return f"AX[A[{r},{s}],X[{r2},{s2}|{t},{u}]]"

def _name_BX(cfg):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x//9, 3); t2, u2 = divmod(x%9, 3)
    return f"BX[B[{t},{u}],X[{r2},{s2}|{t2},{u2}]]"

def _name_CXC(cfg):
    c1 = cfg//(81*9); rem = cfg%(81*9); x = rem//9; c2 = rem%9
    r1,u1 = divmod(c1,3)
    r2,s = divmod(x//9,3); t,u2 = divmod(x%9,3)
    r3,u3 = divmod(c2,3)
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

# ── Slot decode + signature functions (step10, verbatim) ───────────

def _slots(cfg, schema):
    if schema == 'XX':
        return divmod(cfg, 81)
    elif schema in ('CX', 'AX', 'BX'):
        return divmod(cfg, 81)
    elif schema == 'XC':
        return divmod(cfg, 9)
    elif schema == 'CC':
        return divmod(cfg, 9)
    elif schema == 'CXC':
        c1 = cfg//(81*9); rem = cfg%(81*9)
        return (c1, rem//9, rem%9)

def _sig_XX(s):
    x1,x2 = s
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return (s1==t1, s2==t2, r1==r2, s1==s2, t1==t2, u1==u2,
            r1==r2 and s1==s2, t1==t2 and u1==u2)

def _sig_CX(s):
    c,x = s; ru = divmod(c,3)
    r,ss=divmod(x//9,3); t,u=divmod(x%9,3)
    live = ss==t
    return (live, live and c==3*r+u, ru[0]==r, ru[1]==u)

def _sig_XC(s):
    x,c = s
    r,ss=divmod(x//9,3); t,u=divmod(x%9,3)
    ru = divmod(c,3); live = ss==t
    return (live, live and c==3*r+u, r==ru[0], u==ru[1])

def _sig_CC(s):
    c1,c2 = s
    r1,u1=divmod(c1,3); r2,u2=divmod(c2,3)
    return (c1==c2, r1==r2, u1==u2)

def _sig_AX(s):
    a,x = s; ra,sa=divmod(a,3)
    r,ss=divmod(x//9,3); t,u=divmod(x%9,3)
    return (ra==r, sa==ss, ss==t)

def _sig_BX(s):
    b,x = s; tb,ub=divmod(b,3)
    r,ss=divmod(x//9,3); t,u=divmod(x%9,3)
    return (tb==t, ub==u, ss==t)

def _sig_CXC(s):
    c1,x,c2 = s
    r1,u1=divmod(c1,3)
    r,ss=divmod(x//9,3); t,u=divmod(x%9,3)
    r2,u2=divmod(c2,3); live=ss==t
    return (live, c1==c2, live and c1==3*r+u, live and c2==3*r+u,
            (r1,r,r2), (u1,u,u2))

SIG = {'XX':_sig_XX,'CX':_sig_CX,'XC':_sig_XC,'CC':_sig_CC,
       'AX':_sig_AX,'BX':_sig_BX,'CXC':_sig_CXC}

# ── Orbit computation ──────────────────────────────────────────────

def build_orbits(schema):
    n, name_fn, act_fn = SCHEMA[schema]
    sig_fn = SIG[schema]
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
        stab = sum(1 for a in ACTIONS if act_fn(rep, a) == rep)
        sig = sig_fn(_slots(rep, schema))
        orbits.append({
            'orbit_id': oid,
            'rep_config_id': rep,
            'rep_readable': name_fn(rep),
            'orbit_size': len(members),
            'stabilizer_size': stab,
            'signature_key': sig,
            'members': members,
        })
    return orbits

# ── Corrected truths ───────────────────────────────────────────────

EXPECTED_ORBITS = {'XX':56,'CX':8,'XC':8,'CC':4,'AX':10,'BX':10,'CXC':50}
EXPECTED_SIGS   = {'XX':48,'CX':8,'XC':8,'CC':4,'AX':8,'BX':8,'CXC':50}
OLD_SIGS        = {'XX':20,'CX':4,'XC':4,'CC':3,'AX':3,'BX':8,'CXC':50}

SCHEMAS = ['XX','CX','XC','CC','AX','BX','CXC']

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    print("Rebuilding orbit metadata from actual representatives...\n")
    all_orbits = {}

    for sch in SCHEMAS:
        orbits = build_orbits(sch)
        all_orbits[sch] = orbits

        # Verify rep is an actual orbit member
        for o in orbits:
            assert o['rep_config_id'] in o['members'], \
                f"{sch} orbit {o['orbit_id']}: rep {o['rep_config_id']} not in members"
            assert o['orbit_size'] * o['stabilizer_size'] == 216, \
                f"{sch} orbit {o['orbit_id']}: {o['orbit_size']}*{o['stabilizer_size']} != 216"

        n_orb = len(orbits)
        n_sig = len(set(repr(o['signature_key']) for o in orbits))
        exp_o = EXPECTED_ORBITS[sch]
        exp_s = EXPECTED_SIGS[sch]
        assert n_orb == exp_o, f"{sch}: orbits {n_orb} != {exp_o}"
        assert n_sig == exp_s, f"{sch}: sigs {n_sig} != {exp_s}"
        print(f"  {sch}: {n_orb} orbits, {n_sig} sigs OK")

    print("\nAll verifications passed.\n")

    # ── Comparison table ──
    rows = []
    for sch in SCHEMAS:
        o = all_orbits[sch]
        n_orb = len(o)
        n_sig = len(set(repr(r['signature_key']) for r in o))
        old = OLD_SIGS[sch]
        changed = (n_sig != old)
        complete = (n_sig == n_orb)
        rows.append((sch, n_orb, old, n_sig, changed, complete))

    # ── Console summary ──
    print(f"{'schema':<6} {'orbits':>6} {'old':>5} {'new':>5} {'changed':>8} {'complete':>9}")
    print("-" * 45)
    for sch, n_orb, old, new, ch, comp in rows:
        print(f"{sch:<6} {n_orb:>6} {old:>5} {new:>5} {'yes' if ch else 'no':>8} {'yes' if comp else 'no':>9}")

    # ── Write repaired summary CSV ──
    csv_path = out / 'orbit_metadata_repaired_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['schema','orbit_count','old_signature_count',
                     'repaired_signature_count','changed','orbit_complete'])
        for sch, n_orb, old, new, ch, comp in rows:
            w.writerow([sch, n_orb, old, new,
                        'yes' if ch else 'no',
                        'yes' if comp else 'no'])
    print(f"\nWrote {csv_path}")

    # ── Write repair report ──
    md_path = out / 'orbit_metadata_repair_report.md'
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# Orbit Metadata Cache Repair Report",
        "",
        f"Generated: {ts}",
        "",
        "## Bug Description",
        "",
        "Step10 orbit_metadata keyed records by orbit_id (sequential 0..N-1)",
        "and used that orbit_id to index into the configs list.",
        "Since orbit_id != canonical representative config_id (except by",
        "coincidence for the first few orbits), every signature was computed",
        "from the wrong representative config.",
        "",
        "## Schemas Affected",
        "",
        "All schemas were affected, but only some showed changed signature",
        "counts: XX, CX, XC, CC, AX. BX and CXC happened to match because",
        "the first N configs aligned with orbit representatives.",
        "",
        "## Comparison Table",
        "",
        "| Schema | Orbits | Old Sigs | Repaired Sigs | Changed | Orbit Complete |",
        "|--------|--------|----------|---------------|---------|----------------|",
    ]
    for sch, n_orb, old, new, ch, comp in rows:
        lines.append(
            f"| {sch} | {n_orb} | {old} | {new} "
            f"| {'yes' if ch else 'no'} | {'yes' if comp else 'no'} |"
        )
    lines += [
        "",
        "## Repair Confirmation",
        "",
        "Repaired metadata now uses actual canonical orbit representatives",
        "(lexicographic minimum over group images) for every orbit record.",
        "Signatures are computed from the correct representative config.",
        "",
        "## Verdict",
        "",
        "Old cached signature counts are superseded.",
        "Repaired counts should now be treated as current warehouse truth.",
        "",
    ]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}")

    changed_schemas = [sch for sch, _, _, _, ch, _ in rows if ch]
    print(f"\nSchemas with changed signature counts: {changed_schemas}")


if __name__ == '__main__':
    main()
