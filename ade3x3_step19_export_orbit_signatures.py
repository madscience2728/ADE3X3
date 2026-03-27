"""
ade3x3_step19_export_orbit_signatures.py

Export explicit orbit-signature tables for all indexed schemas.
Uses step18's orbit computation (correct B-action) and step10's signature functions.
One narrow export job only — no redesign, no generalization.

Signature keys are the exact step10 signature functions applied to the
correct canonical orbit representatives (not step10's orbit_id-as-config_id
indexing, which was a storage bug in orbit_metadata).
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Group of size 216 ──────────────────────────────────────────────

S3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

def _build_actions():
    return [(pi_rA, pi_shared, pi_cB)
            for pi_rA in S3 for pi_shared in S3 for pi_cB in S3]

ACTIONS = _build_actions()

# ── Schema definitions (step18 encoding) ───────────────────────────

def _get_name_XX(cfg):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3); t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3); t2, u2 = divmod(x2 % 9, 3)
    return f"XX[X[{r1},{s1}|{t1},{u1}],X[{r2},{s2}|{t2},{u2}]]"

def _act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3); t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3); t2, u2 = divmod(x2 % 9, 3)
    x1n = 9*(3*act[0][r1]+act[1][s1]) + (3*act[1][t1]+act[2][u1])
    x2n = 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t2]+act[2][u2])
    return x1n * 81 + x2n

def _get_name_CX(cfg):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    return f"CX[C[{r},{u}],X[{r2},{s}|{t},{u2}]]"

def _act_CX(cfg, act):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    cn = 3*act[0][r] + act[2][u]
    xn = 9*(3*act[0][r2]+act[1][s]) + (3*act[1][t]+act[2][u2])
    return cn * 81 + xn

def _get_name_XC(cfg):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r2, u2 = divmod(c, 3)
    return f"XC[X[{r},{s}|{t},{u}],C[{r2},{u2}]]"

def _act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r2, u2 = divmod(c, 3)
    xn = 9*(3*act[0][r]+act[1][s]) + (3*act[1][t]+act[2][u])
    cn = 3*act[0][r2] + act[2][u2]
    return xn * 9 + cn

def _get_name_CC(cfg):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return f"CC[C[{r1},{u1}],C[{r2},{u2}]]"

def _act_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    c1n = 3*act[0][r1] + act[2][u1]
    c2n = 3*act[0][r2] + act[2][u2]
    return 9 * c1n + c2n

def _get_name_AX(cfg):
    a, x = divmod(cfg, 81)
    r, s = divmod(a, 3)
    r2, s2 = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    return f"AX[A[{r},{s}],X[{r2},{s2}|{t},{u}]]"

def _act_AX(cfg, act):
    a, x = divmod(cfg, 81)
    r, s = divmod(a, 3)
    r2, s2 = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    an = 3*act[0][r] + act[1][s]
    xn = 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t]+act[2][u])
    return an * 81 + xn

def _get_name_BX(cfg):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3); t2, u2 = divmod(x % 9, 3)
    return f"BX[B[{t},{u}],X[{r2},{s2}|{t2},{u2}]]"

def _act_BX(cfg, act):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3); t2, u2 = divmod(x % 9, 3)
    bn = 3*act[1][t] + act[2][u]
    xn = 9*(3*act[0][r2]+act[1][s2]) + (3*act[1][t2]+act[2][u2])
    return bn * 81 + xn

def _get_name_CXC(cfg):
    c1 = cfg // (81*9); rem = cfg % (81*9); x = rem // 9; c2 = rem % 9
    r1, u1 = divmod(c1, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    r3, u3 = divmod(c2, 3)
    return f"CXC[C[{r1},{u1}],X[{r2},{s}|{t},{u2}],C[{r3},{u3}]]"

def _act_CXC(cfg, act):
    c1 = cfg // (81*9); rem = cfg % (81*9); x = rem // 9; c2 = rem % 9
    r1, u1 = divmod(c1, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    r3, u3 = divmod(c2, 3)
    c1n = 3*act[0][r1] + act[2][u1]
    xn  = 9*(3*act[0][r2]+act[1][s]) + (3*act[1][t]+act[2][u2])
    c2n = 3*act[0][r3] + act[2][u3]
    return (c1n*81 + xn)*9 + c2n

SCHEMA_META = {
    'XX':  (81*81,   _get_name_XX,  _act_XX),
    'CX':  (9*81,    _get_name_CX,  _act_CX),
    'XC':  (81*9,    _get_name_XC,  _act_XC),
    'CC':  (81,      _get_name_CC,  _act_CC),
    'AX':  (9*81,    _get_name_AX,  _act_AX),
    'BX':  (9*81,    _get_name_BX,  _act_BX),
    'CXC': (9*81*9,  _get_name_CXC, _act_CXC),
}

# ── Signature functions (from step10, verbatim) ───────────────────

def _decode_slots(cfg, schema):
    if schema == 'XX':
        x1, x2 = divmod(cfg, 81)
        return (x1, x2)
    elif schema == 'CX':
        c, x = divmod(cfg, 81)
        return (c, x)
    elif schema == 'XC':
        x, c = divmod(cfg, 9)
        return (x, c)
    elif schema == 'CC':
        c1, c2 = divmod(cfg, 9)
        return (c1, c2)
    elif schema == 'AX':
        a, x = divmod(cfg, 81)
        return (a, x)
    elif schema == 'BX':
        b, x = divmod(cfg, 81)
        return (b, x)
    elif schema == 'CXC':
        c1 = cfg // (81*9); rem = cfg % (81*9); x = rem // 9; c2 = rem % 9
        return (c1, x, c2)
    return ()

def _sig_XX(slots):
    x1, x2 = slots
    r1, s1 = divmod(x1 // 9, 3); t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3); t2, u2 = divmod(x2 % 9, 3)
    live1 = (s1 == t1); live2 = (s2 == t2)
    return (live1, live2, r1==r2, s1==s2, t1==t2, u1==u2,
            r1==r2 and s1==s2, t1==t2 and u1==u2)

def _sig_CX(slots):
    c, x = slots
    r_u = divmod(c, 3)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    live = (s == t)
    return (live, live and c == 3*r+u, r_u[0]==r, r_u[1]==u)

def _sig_XC(slots):
    x, c = slots
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r_u = divmod(c, 3)
    live = (s == t)
    return (live, live and c == 3*r+u, r==r_u[0], u==r_u[1])

def _sig_CC(slots):
    c1, c2 = slots
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return (c1==c2, r1==r2, u1==u2)

def _sig_AX(slots):
    a, x = slots
    r_a, s_a = divmod(a, 3)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    live = (s == t)
    return (r_a==r, s_a==s, live)

def _sig_BX(slots):
    b, x = slots
    t_b, u_b = divmod(b, 3)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    live = (s == t)
    return (t_b==t, u_b==u, live)

def _sig_CXC(slots):
    c1, x, c2 = slots
    r1, u1 = divmod(c1, 3)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r2, u2 = divmod(c2, 3)
    live = (s == t)
    return (live, c1==c2,
            live and c1 == 3*r+u, live and c2 == 3*r+u,
            (r1, r, r2), (u1, u, u2))

SIG_FUNC = {
    'XX': _sig_XX, 'CX': _sig_CX, 'XC': _sig_XC, 'CC': _sig_CC,
    'AX': _sig_AX, 'BX': _sig_BX, 'CXC': _sig_CXC,
}

# ── Orbit computation (step18 logic, correct B-action) ────────────

def compute_orbits(schema):
    """Return list of orbit dicts with correct canonical representatives
    and their step10 signatures."""
    n_configs, get_name, act_func = SCHEMA_META[schema]

    # Canonical = lexicographic min over group images
    min_repr = {}
    for cfg in range(n_configs):
        m = cfg
        for act in ACTIONS:
            img = act_func(cfg, act)
            if img < m:
                m = img
        min_repr[cfg] = m

    groups = defaultdict(list)
    for cfg in range(n_configs):
        groups[min_repr[cfg]].append(cfg)

    sig_fn = SIG_FUNC[schema]
    orbit_data = []
    for oid, (rep, members) in enumerate(sorted(groups.items())):
        size = len(members)
        stab = sum(1 for a in ACTIONS if act_func(rep, a) == rep)
        slots = _decode_slots(rep, schema)
        sig = sig_fn(slots)
        orbit_data.append({
            'orbit_id': oid,
            'rep_config_id': rep,
            'rep_readable': get_name(rep),
            'orbit_size': size,
            'stabilizer_size': stab,
            'signature_key': sig,
        })
    return orbit_data

# ── CSV / Markdown writers ─────────────────────────────────────────

CSV_FIELDS = ['schema', 'orbit_id', 'rep_config_id', 'rep_readable',
              'signature_key', 'orbit_size', 'stabilizer_size']

def write_csv(orbit_data, schema, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in orbit_data:
            w.writerow({
                'schema': schema,
                'orbit_id': row['orbit_id'],
                'rep_config_id': row['rep_config_id'],
                'rep_readable': row['rep_readable'],
                'signature_key': repr(row['signature_key']),
                'orbit_size': row['orbit_size'],
                'stabilizer_size': row['stabilizer_size'],
            })

def write_summary_md(results, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# Orbit-Signature Export Summary",
        "",
        f"Generated: {ts}",
        "",
        "Explicit orbit-signature assignments for the ADE3x3 warehouse.",
        "",
        "Signature keys are step10 signature functions applied to the correct",
        "canonical orbit representatives (step18 orbit computation with proper",
        "B-action). Previous step10 orbit_metadata used orbit_id as config_id",
        "index, which was a storage bug producing incorrect representatives.",
        "",
        "## Summary Table",
        "",
        "| Schema | Orbit Count | Distinct Signatures | Orbit Complete |",
        "|--------|-------------|---------------------|----------------|",
    ]
    for schema, data in results:
        orbits = len(data)
        n_sigs = len(set(repr(r['signature_key']) for r in data))
        complete = "yes" if n_sigs == orbits else "no"
        lines.append(f"| {schema} | {orbits} | {n_sigs} | {complete} |")

    lines += ["", "## Exported CSV Files", ""]
    for schema, _ in results:
        lines.append(f"- `signatures_{schema}.csv`")

    lines += [
        "",
        "Signature keys are now explicitly recorded as warehouse artifacts.",
        "",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ── Expected counts ────────────────────────────────────────────────

EXPECTED_ORBITS = {'XX': 56, 'CX': 8, 'XC': 8, 'CC': 4, 'AX': 10, 'BX': 10, 'CXC': 50}
# Orbit-complete schemas: signature fully separates orbits
ORBIT_COMPLETE = {'CXC'}

# ── Main ───────────────────────────────────────────────────────────

def main():
    out = Path('exports')
    out.mkdir(exist_ok=True)

    schemas = ['XX', 'CX', 'XC', 'CC', 'AX', 'BX', 'CXC']
    results = []

    print("Computing orbits and signatures...")
    for schema in schemas:
        data = compute_orbits(schema)
        csv_path = out / f'signatures_{schema}.csv'
        write_csv(data, schema, csv_path)
        results.append((schema, data))
        print(f"  {schema}: {len(data)} orbits -> {csv_path.name}")

    # ── Sanity checks ──
    print("\n--- SANITY CHECKS ---")
    all_ok = True
    summary_rows = []

    for schema, data in results:
        n_orbits = len(data)
        n_sigs = len(set(repr(r['signature_key']) for r in data))
        exp_o = EXPECTED_ORBITS[schema]
        orbit_ok = (n_orbits == exp_o)
        complete = (n_sigs == n_orbits)

        if not orbit_ok:
            print(f"  {schema}: orbit count {n_orbits} != expected {exp_o} [FAIL]")
            all_ok = False
        else:
            print(f"  {schema}: {n_orbits} orbits OK")

        # Verify orbit-stabilizer product
        for row in data:
            assert row['orbit_size'] * row['stabilizer_size'] == 216, \
                f"{schema} orbit {row['orbit_id']}: {row['orbit_size']}*{row['stabilizer_size']} != 216"

        # Verify orbit-complete schemas
        if schema in ORBIT_COMPLETE:
            assert complete, \
                f"{schema}: expected orbit-complete but {n_sigs} != {n_orbits}"
            print(f"  {schema}: orbit-complete verified ({n_sigs} sigs)")

        summary_rows.append((schema, n_orbits, n_sigs, complete))

    print("  All orbit*stabilizer == 216: OK")

    if not all_ok:
        raise RuntimeError("Orbit count sanity checks failed — see above.")

    # ── Markdown summary ──
    md_path = out / 'signature_summary.md'
    write_summary_md(results, md_path)

    # ── Console summary ──
    print(f"\n--- EXPORT COMPLETE ---")
    print(f"{'schema':<6} {'orbits':>6} {'sigs':>5} {'complete':>9}")
    for schema, n_o, n_s, comp in summary_rows:
        print(f"{schema:<6} {n_o:>6} {n_s:>5} {'yes' if comp else 'no':>9}")
    print(f"\nFiles written to {out}/")
    for schema, _ in results:
        print(f"  signatures_{schema}.csv")
    print(f"  signature_summary.md")


if __name__ == '__main__':
    main()
