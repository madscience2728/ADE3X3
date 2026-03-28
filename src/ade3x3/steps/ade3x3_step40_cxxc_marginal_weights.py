"""
ade3x3_step40_cxxc_marginal_weights.py

CXXC Marginal Weight Profile: for each of the 7 marginal projections,
compute the fiber-size histogram — how many CXXC configurations project
onto each lower-arity orbit, and how many CXXC orbits.

The 7 projections (from Section 20 of the canon dossier):
  CXC_left  (C1, X1, C2) → CXC orbit  [50 possible]
  CXC_right (C1, X2, C2) → CXC orbit  [50 possible]
  XX        (X1, X2)     → XX  orbit  [56 possible]
  CX_left   (C1, X1)     → CX  orbit  [8  possible]
  CX_right  (C1, X2)     → CX  orbit  [8  possible]
  XC_left   (X1, C2)     → XC  orbit  [8  possible]
  XC_right  (X2, C2)     → XC  orbit  [8  possible]

Key observation: projection commutes with group action (both are equivariant).
Therefore, all 531,441 configs in a CXXC orbit project to the *same* lower-arity
orbit. So we only need to:
  1. Read orbits_CXXC.csv (2744 rows: rep_config_id, orbit_size)
  2. Compute lower-arity orbit maps (fast, small spaces)
  3. Decode each rep and compute 7 projection orbit_ids
  4. Accumulate: target_orbit → cxxc_orbit_count, cxxc_config_count (=sum orbit_sizes)

No full 531,441-row scan needed. No parallel computation needed.

Output:
  - cxxc_marginal_weight_profile.csv: all 7 projections combined
  - cxxc_marginal_weight_profile.md: analysis and histograms

Provenance: [EXACT_DERIVED]
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

EXPORTS = Path('outputs/exports')

# ──────────────────────────────────────────────────────────────────────────────
# Group
# ──────────────────────────────────────────────────────────────────────────────

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]

# ──────────────────────────────────────────────────────────────────────────────
# Action functions
# ──────────────────────────────────────────────────────────────────────────────

def _act_CX(cfg, act):
    c,x=divmod(cfg,81); r,u=divmod(c,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3)
    return (3*act[0][r]+act[2][u])*81+9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def _act_XC(cfg, act):
    x,c=divmod(cfg,9); r,s=divmod(x//9,3); t,u=divmod(x%9,3); r2,u2=divmod(c,3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def _act_XX(cfg, act):
    x1,x2=divmod(cfg,81)
    r1,s1=divmod(x1//9,3); t1,u1=divmod(x1%9,3)
    r2,s2=divmod(x2//9,3); t2,u2=divmod(x2%9,3)
    return (9*(3*act[0][r1]+act[1][s1])+(3*act[1][t1]+act[2][u1]))*81 + \
           (9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2]))

def _act_CXC(cfg, act):
    c1=cfg//(81*9); rem=cfg%(81*9); x=rem//9; c2=rem%9
    r1,u1=divmod(c1,3); r2,s=divmod(x//9,3); t,u2=divmod(x%9,3); r3,u3=divmod(c2,3)
    return ((3*act[0][r1]+act[2][u1])*81+9*(3*act[0][r2]+act[1][s]) +
            (3*act[1][t]+act[2][u2]))*9+(3*act[0][r3]+act[2][u3])

# ──────────────────────────────────────────────────────────────────────────────
# Orbit map builder
# ──────────────────────────────────────────────────────────────────────────────

def _orbit_map(n: int, act_fn) -> list[int]:
    """Return list oid_of[cfg] using canonical representative method."""
    min_r = [0] * n
    for cfg in range(n):
        m = cfg
        for a in ACTIONS:
            v = act_fn(cfg, a)
            if v < m:
                m = v
        min_r[cfg] = m
    # Assign orbit IDs in sorted order of representatives
    reps_sorted = sorted(set(min_r))
    rep_to_oid = {rep: oid for oid, rep in enumerate(reps_sorted)}
    return [rep_to_oid[min_r[cfg]] for cfg in range(n)]

# ──────────────────────────────────────────────────────────────────────────────
# CXXC decode
# ──────────────────────────────────────────────────────────────────────────────

def decode_CXXC(cfg: int) -> tuple[int,int,int,int]:
    """Return (c1, x1, x2, c2)."""
    c2 = cfg % 9
    rem = cfg // 9
    x2 = rem % 81
    rem //= 81
    x1 = rem % 81
    c1 = rem // 81
    return c1, x1, x2, c2

# ──────────────────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────────────────

PROJECTIONS = [
    ('CXC_left',  'CXC', 50),
    ('CXC_right', 'CXC', 50),
    ('XX',        'XX',  56),
    ('CX_left',   'CX',  8),
    ('CX_right',  'CX',  8),
    ('XC_left',   'XC',  8),
    ('XC_right',  'XC',  8),
]

TOTAL_CXXC_CONFIGS = 531_441


def build_lower_orbit_maps() -> dict[str, list[int]]:
    print("  Building CXC orbit map (6,561 configs × 216)...")
    cxc_oid = _orbit_map(9*81*9, _act_CXC)
    print("  Building XX  orbit map (6,561 configs × 216)...")
    xx_oid  = _orbit_map(81*81,  _act_XX)
    print("  Building CX  orbit map (729 configs × 216)...")
    cx_oid  = _orbit_map(9*81,   _act_CX)
    print("  Building XC  orbit map (729 configs × 216)...")
    xc_oid  = _orbit_map(81*9,   _act_XC)
    return {'CXC': cxc_oid, 'XX': xx_oid, 'CX': cx_oid, 'XC': xc_oid}


def compute_projection_for_rep(rep: int, oid_maps: dict) -> dict[str, int]:
    """Given a CXXC canonical rep, return {projection_name: target_orbit_id}."""
    c1, x1, x2, c2 = decode_CXXC(rep)
    cxc = oid_maps['CXC']
    xx  = oid_maps['XX']
    cx  = oid_maps['CX']
    xc  = oid_maps['XC']
    return {
        'CXC_left':  cxc[(c1*81 + x1)*9 + c2],
        'CXC_right': cxc[(c1*81 + x2)*9 + c2],
        'XX':        xx[x1*81 + x2],
        'CX_left':   cx[c1*81 + x1],
        'CX_right':  cx[c1*81 + x2],
        'XC_left':   xc[x1*9  + c2],
        'XC_right':  xc[x2*9  + c2],
    }


def load_cxxc_orbits() -> list[dict]:
    path = EXPORTS / 'orbits_CXXC.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ade3x3_step38_export_arity4_orbit_parity.py first."
        )
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['schema'] == 'CXXC':
                rows.append({
                    'orbit_id': int(row['orbit_id']),
                    'rep_config_id': int(row['rep_config_id']),
                    'orbit_size': int(row['orbit_size']),
                    'stabilizer_size': int(row['stabilizer_size']),
                })
    return rows


def build_weight_profile(
    cxxc_orbits: list[dict],
    oid_maps: dict,
) -> dict[str, dict[int, dict]]:
    """
    Returns: {projection_name: {target_orbit_id: {cxxc_orbit_count, cxxc_config_count}}}
    """
    profile = {}
    for proj_name, _, _ in PROJECTIONS:
        profile[proj_name] = defaultdict(lambda: {'cxxc_orbit_count': 0, 'cxxc_config_count': 0})

    for orb in cxxc_orbits:
        rep = orb['rep_config_id']
        orbit_size = orb['orbit_size']
        proj_targets = compute_projection_for_rep(rep, oid_maps)
        for proj_name, _, _ in PROJECTIONS:
            target = proj_targets[proj_name]
            profile[proj_name][target]['cxxc_orbit_count'] += 1
            profile[proj_name][target]['cxxc_config_count'] += orbit_size

    return profile


def verify_profile(profile: dict, n_cxxc_orbits: int):
    for proj_name, schema, expected_target_count in PROJECTIONS:
        data = profile[proj_name]
        total_orbits = sum(v['cxxc_orbit_count'] for v in data.values())
        total_configs = sum(v['cxxc_config_count'] for v in data.values())
        realized = len(data)
        assert total_orbits == n_cxxc_orbits, \
            f"{proj_name}: orbit count {total_orbits} != {n_cxxc_orbits}"
        assert total_configs == TOTAL_CXXC_CONFIGS, \
            f"{proj_name}: config count {total_configs} != {TOTAL_CXXC_CONFIGS}"
        assert realized == expected_target_count, \
            f"{proj_name}: realized targets {realized} != expected {expected_target_count}"
        print(f"  [{proj_name}] orbits={total_orbits}, configs={total_configs}, "
              f"targets_realized={realized}/{expected_target_count} ✓")

# ──────────────────────────────────────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(profile: dict, path: Path):
    fields = ['projection', 'target_schema', 'target_orbit_id',
              'cxxc_orbit_count', 'cxxc_config_count']
    rows = []
    for proj_name, schema, _ in PROJECTIONS:
        for target_oid in sorted(profile[proj_name].keys()):
            v = profile[proj_name][target_oid]
            rows.append({
                'projection': proj_name,
                'target_schema': schema,
                'target_orbit_id': target_oid,
                'cxxc_orbit_count': v['cxxc_orbit_count'],
                'cxxc_config_count': v['cxxc_config_count'],
            })
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")
    return rows


def write_markdown(profile: dict, path: Path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# CXXC Marginal Weight Profile",
        "",
        f"Generated: {ts}",
        "",
        "[EXACT_DERIVED]",
        "",
        "## Overview",
        "",
        "For each of the 7 CXXC marginal projections (Section 20 of the canon dossier),",
        "this section records the fiber-size profile: how many CXXC orbits and",
        "configurations project onto each lower-arity orbit.",
        "",
        "**Key structural property:** Projection commutes with the group action.",
        "Every configuration in a given CXXC orbit projects to the *same* lower-arity orbit.",
        "Therefore, CXXC orbits partition cleanly across target orbits with no mixing.",
        "",
        "**Verification:** For every projection, the total config count = 531,441 (all CXXC configs),",
        "and the total orbit count = 2,744 (all CXXC orbits). All 7 projections are confirmed",
        "to achieve 100% target-orbit coverage.",
        "",
        "## Summary Table",
        "",
        "| Projection | Target Schema | Targets Realized | Min Fiber Orbits | Max Fiber Orbits "
        "| Min Fiber Configs | Max Fiber Configs |",
        "|------------|---------------|-----------------|------------------|---"
        "---------------|------------------|------------------|",
    ]
    for proj_name, schema, expected in PROJECTIONS:
        data = profile[proj_name]
        orbit_counts = [v['cxxc_orbit_count'] for v in data.values()]
        config_counts = [v['cxxc_config_count'] for v in data.values()]
        lines.append(
            f"| {proj_name} | {schema} | {len(data)}/{expected} "
            f"| {min(orbit_counts)} | {max(orbit_counts)} "
            f"| {min(config_counts)} | {max(config_counts)} |"
        )

    for proj_name, schema, expected in PROJECTIONS:
        data = profile[proj_name]
        lines += [
            "",
            f"## {proj_name} → {schema} Fiber Profile",
            "",
            f"| {schema} Orbit | CXXC Orbit Count | CXXC Config Count | Avg Orbit Size |",
            f"|{'-'*13}|{'-'*19}|{'-'*20}|{'-'*17}|",
        ]
        for target_oid in sorted(data.keys()):
            v = data[target_oid]
            n_orb = v['cxxc_orbit_count']
            n_cfg = v['cxxc_config_count']
            avg = n_cfg / n_orb if n_orb > 0 else 0
            lines.append(
                f"| {target_oid} | {n_orb} | {n_cfg} | {avg:.1f} |"
            )

    lines += ["", "---", "*Generated by ade3x3_step40_cxxc_marginal_weights.py*", ""]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Wrote markdown → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=== CXXC Marginal Weight Profile ===")
    print()

    print("Building lower-arity orbit maps...")
    oid_maps = build_lower_orbit_maps()
    # Verify orbit counts
    assert len(set(oid_maps['CXC'])) == 50,  f"CXC: {len(set(oid_maps['CXC']))} orbits"
    assert len(set(oid_maps['XX']))  == 56,  f"XX: {len(set(oid_maps['XX']))} orbits"
    assert len(set(oid_maps['CX']))  == 8,   f"CX: {len(set(oid_maps['CX']))} orbits"
    assert len(set(oid_maps['XC']))  == 8,   f"XC: {len(set(oid_maps['XC']))} orbits"
    print(f"  CXC={len(set(oid_maps['CXC']))}, XX={len(set(oid_maps['XX']))}, "
          f"CX={len(set(oid_maps['CX']))}, XC={len(set(oid_maps['XC']))} orbits verified")
    print()

    print("Loading CXXC orbit data...")
    cxxc_orbits = load_cxxc_orbits()
    n_orbits = len(cxxc_orbits)
    total_configs = sum(o['orbit_size'] for o in cxxc_orbits)
    assert n_orbits == 2744, f"CXXC orbits: {n_orbits}"
    assert total_configs == TOTAL_CXXC_CONFIGS, f"CXXC configs: {total_configs}"
    print(f"  {n_orbits} CXXC orbits, {total_configs} total configs ✓")
    print()

    print("Computing weight profile (2744 orbit reps × 7 projections)...")
    profile = build_weight_profile(cxxc_orbits, oid_maps)
    print()

    print("Verifying...")
    verify_profile(profile, n_orbits)
    print()

    print("Writing outputs...")
    csv_path = EXPORTS / 'cxxc_marginal_weight_profile.csv'
    md_path  = EXPORTS / 'cxxc_marginal_weight_profile.md'
    write_csv(profile, csv_path)
    write_markdown(profile, md_path)

    print()
    print("=== Done ===")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).resolve().parents[3])
    main()
