"""
ade3x3_step35_analyze_cxxc_marginals.py

Analyze CXXC marginal projections onto lower-arity schemas.

CXXC = C × X × X × C has natural projections:
- π_CXC_left: CXXC → CXC (keep C1, X1, C2)  [drop X2]
- π_CXC_right: CXXC → CXC (keep C1, X2, C2) [drop X1]
- π_XX: CXXC → XX (keep X1, X2)             [drop both C]
- π_CX_left: CXXC → CX (keep C1, X1)        [drop X2, C2]
- π_CX_right: CXXC → CX (keep C1, X2)       [drop X1, C2]
- π_XC_left: CXXC → XC (keep X1, C2)        [drop C1, X2]
- π_XC_right: CXXC → XC (keep X2, C2)       [drop C1, X1]
- π_CC: CXXC → CC (keep C1, C2)             [drop both X]

This analysis reveals which orbit combinations are realized in CXXC.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Load orbit data ────────────────────────────────────────────────

def load_orbit_mapping(csv_path, schema_name):
    """Load orbit IDs from CSV: config_id -> orbit_id."""
    oid_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # For each orbit, we only need the representative
            # But we need to reconstruct the full orbit from the group action
            # For now, let's just use the CSV data directly
            pass

    # Instead, let's recompute orbits on the fly
    return None

# ─── Schema action functions ────────────────────────────────────────

S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
ACTIONS = [(p1,p2,p3) for p1 in S3 for p2 in S3 for p3 in S3]

def act_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return 9*(3*act[0][r1]+act[2][u1])+(3*act[0][r2]+act[2][u2])

def act_CX(cfg, act):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x//9, 3); t, u2 = divmod(x%9, 3)
    return (3*act[0][r]+act[2][u])*81 + 9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])

def act_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x//9, 3); t, u = divmod(x%9, 3)
    r2, u2 = divmod(c, 3)
    return (9*(3*act[0][r]+act[1][s])+(3*act[1][t]+act[2][u]))*9+(3*act[0][r2]+act[2][u2])

def act_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1//9, 3); t1, u1 = divmod(x1%9, 3)
    r2, s2 = divmod(x2//9, 3); t2, u2 = divmod(x2%9, 3)
    x1n = 9*(3*act[0][r1]+act[1][s1])+(3*act[1][t1]+act[2][u1])
    x2n = 9*(3*act[0][r2]+act[1][s2])+(3*act[1][t2]+act[2][u2])
    return x1n*81+x2n

def act_CXC(cfg, act):
    c1 = cfg//(81*9); rem = cfg%(81*9); x = rem//9; c2 = rem%9
    r1, u1 = divmod(c1, 3)
    r2, s = divmod(x//9, 3); t, u2 = divmod(x%9, 3)
    r3, u3 = divmod(c2, 3)
    c1n = 3*act[0][r1]+act[2][u1]
    xn  = 9*(3*act[0][r2]+act[1][s])+(3*act[1][t]+act[2][u2])
    c2n = 3*act[0][r3]+act[2][u3]
    return (c1n*81+xn)*9+c2n

def compute_orbit_map(n, act_fn):
    """Compute config_id -> orbit_id mapping."""
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

    oid_of = {}
    for oid, (rep, members) in enumerate(sorted(groups.items())):
        for m in members:
            oid_of[m] = oid

    return oid_of

# ─── CXXC projection functions ──────────────────────────────────────

def decode_CXXC(cfg):
    """Decode CXXC config into (c1, x1, x2, c2)."""
    c2 = cfg % 9
    rem = cfg // 9
    x2 = rem % 81
    rem = rem // 81
    x1 = rem % 81
    c1 = rem // 81
    return c1, x1, x2, c2

def project_CXXC_to_CXC_left(cxxc_cfg):
    """Project CXXC → CXC (keep C1, X1, C2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return (c1 * 81 + x1) * 9 + c2

def project_CXXC_to_CXC_right(cxxc_cfg):
    """Project CXXC → CXC (keep C1, X2, C2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return (c1 * 81 + x2) * 9 + c2

def project_CXXC_to_XX(cxxc_cfg):
    """Project CXXC → XX (keep X1, X2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return x1 * 81 + x2

def project_CXXC_to_CX_left(cxxc_cfg):
    """Project CXXC → CX (keep C1, X1)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return c1 * 81 + x1

def project_CXXC_to_CX_right(cxxc_cfg):
    """Project CXXC → CX (keep C1, X2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return c1 * 81 + x2

def project_CXXC_to_XC_left(cxxc_cfg):
    """Project CXXC → XC (keep X1, C2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return x1 * 9 + c2

def project_CXXC_to_XC_right(cxxc_cfg):
    """Project CXXC → XC (keep X2, C2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return x2 * 9 + c2

def project_CXXC_to_CC(cxxc_cfg):
    """Project CXXC → CC (keep C1, C2)."""
    c1, x1, x2, c2 = decode_CXXC(cxxc_cfg)
    return c1 * 9 + c2

# ─── Main analysis ──────────────────────────────────────────────────

def main():
    exports = Path('outputs/exports')
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("="*70)
    print("CXXC Marginal Projection Analysis")
    print("="*70)
    print(f"Generated: {ts}\n")

    # Load CXXC orbits
    print("[1/9] Loading CXXC orbit data...")
    cxxc_csv = exports / 'signatures_CXXC.csv'
    cxxc_reps = {}
    with open(cxxc_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = int(row['orbit_id'])
            rep = int(row['rep_config_id'])
            cxxc_reps[oid] = rep
    print(f"  Loaded {len(cxxc_reps)} CXXC orbits")

    # Compute orbit maps for target schemas
    print("\n[2/9] Computing orbit maps for target schemas...")
    print("  CC...", end='', flush=True)
    cc_oid = compute_orbit_map(81, act_CC)
    print(f" {max(cc_oid.values())+1} orbits")

    print("  CX...", end='', flush=True)
    cx_oid = compute_orbit_map(9*81, act_CX)
    print(f" {max(cx_oid.values())+1} orbits")

    print("  XC...", end='', flush=True)
    xc_oid = compute_orbit_map(81*9, act_XC)
    print(f" {max(xc_oid.values())+1} orbits")

    print("  XX...", end='', flush=True)
    xx_oid = compute_orbit_map(81*81, act_XX)
    print(f" {max(xx_oid.values())+1} orbits")

    print("  CXC...", end='', flush=True)
    cxc_oid = compute_orbit_map(9*81*9, act_CXC)
    print(f" {max(cxc_oid.values())+1} orbits")

    # Analyze projections
    print("\n[3/9] Analyzing CXC_left projection (C1,X1,C2)...")
    cxc_left_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        cxc_cfg = project_CXXC_to_CXC_left(cxxc_rep)
        cxc_o = cxc_oid[cxc_cfg]
        cxc_left_pairs[cxc_o] += 1
    print(f"  Realized CXC orbits: {len(cxc_left_pairs)} / 50")

    print("\n[4/9] Analyzing CXC_right projection (C1,X2,C2)...")
    cxc_right_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        cxc_cfg = project_CXXC_to_CXC_right(cxxc_rep)
        cxc_o = cxc_oid[cxc_cfg]
        cxc_right_pairs[cxc_o] += 1
    print(f"  Realized CXC orbits: {len(cxc_right_pairs)} / 50")

    print("\n[5/9] Analyzing XX projection (X1,X2)...")
    xx_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        xx_cfg = project_CXXC_to_XX(cxxc_rep)
        xx_o = xx_oid[xx_cfg]
        xx_pairs[xx_o] += 1
    print(f"  Realized XX orbits: {len(xx_pairs)} / 56")

    print("\n[6/9] Analyzing CX_left projection (C1,X1)...")
    cx_left_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        cx_cfg = project_CXXC_to_CX_left(cxxc_rep)
        cx_o = cx_oid[cx_cfg]
        cx_left_pairs[cx_o] += 1
    print(f"  Realized CX orbits: {len(cx_left_pairs)} / 8")

    print("\n[7/9] Analyzing CX_right projection (C1,X2)...")
    cx_right_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        cx_cfg = project_CXXC_to_CX_right(cxxc_rep)
        cx_o = cx_oid[cx_cfg]
        cx_right_pairs[cx_o] += 1
    print(f"  Realized CX orbits: {len(cx_right_pairs)} / 8")

    print("\n[8/9] Analyzing XC_left projection (X1,C2)...")
    xc_left_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        xc_cfg = project_CXXC_to_XC_left(cxxc_rep)
        xc_o = xc_oid[xc_cfg]
        xc_left_pairs[xc_o] += 1
    print(f"  Realized XC orbits: {len(xc_left_pairs)} / 8")

    print("\n[9/9] Analyzing XC_right projection (X2,C2)...")
    xc_right_pairs = defaultdict(int)
    for cxxc_rep in cxxc_reps.values():
        xc_cfg = project_CXXC_to_XC_right(cxxc_rep)
        xc_o = xc_oid[xc_cfg]
        xc_right_pairs[xc_o] += 1
    print(f"  Realized XC orbits: {len(xc_right_pairs)} / 8")

    # Write summary
    summary_lines = [
        "# CXXC Marginal Projection Analysis",
        "",
        f"**Generated:** {ts}",
        "**Source:** CXXC orbit data (2744 orbits)",
        "",
        "## Marginal Projections",
        "",
        "| Projection | Target Schema | Realized Orbits | Total Orbits | Coverage |",
        "|------------|---------------|-----------------|--------------|----------|",
        f"| CXC_left (C1,X1,C2) | CXC | {len(cxc_left_pairs)} | 50 | {100*len(cxc_left_pairs)/50:.1f}% |",
        f"| CXC_right (C1,X2,C2) | CXC | {len(cxc_right_pairs)} | 50 | {100*len(cxc_right_pairs)/50:.1f}% |",
        f"| XX (X1,X2) | XX | {len(xx_pairs)} | 56 | {100*len(xx_pairs)/56:.1f}% |",
        f"| CX_left (C1,X1) | CX | {len(cx_left_pairs)} | 8 | {100*len(cx_left_pairs)/8:.1f}% |",
        f"| CX_right (C1,X2) | CX | {len(cx_right_pairs)} | 8 | {100*len(cx_right_pairs)/8:.1f}% |",
        f"| XC_left (X1,C2) | XC | {len(xc_left_pairs)} | 8 | {100*len(xc_left_pairs)/8:.1f}% |",
        f"| XC_right (X2,C2) | XC | {len(xc_right_pairs)} | 8 | {100*len(xc_right_pairs)/8:.1f}% |",
        "",
        "## Interpretation",
        "",
        "This analysis shows which lower-arity orbit combinations are realized",
        "when CXXC configurations are projected onto their natural marginal schemas.",
        "",
        "Full coverage (100%) indicates that every orbit of the target schema",
        "appears in at least one CXXC configuration under the projection.",
        "",
    ]

    md_path = exports / 'cxxc_marginal_analysis.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print(f"\n\nWritten: {md_path}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
