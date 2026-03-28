"""
ade3x3_step33_refine_signature_collisions.py

Resolve signature collisions by finding minimal additional Boolean features
that distinguish orbits within each collision group.

Target collision groups:
- XX: 4 groups of 3 orbits each (8 collisions total)
- AX: 2 groups of 2 orbits each
- BX: 2 groups of 2 orbits each

For each collision group, we examine the canonical representatives and
identify Boolean features that separate the orbits.
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Decode functions ───────────────────────────────────────────────

def decode_XX(cfg):
    """Decode XX config into two X atoms with full coordinates."""
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3)
    t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3)
    t2, u2 = divmod(x2 % 9, 3)
    return {
        'x1': x1, 'x2': x2,
        'r1': r1, 's1': s1, 't1': t1, 'u1': u1,
        'r2': r2, 's2': s2, 't2': t2, 'u2': u2,
    }

def decode_AX(cfg):
    """Decode AX config into A atom and X atom."""
    a, x = divmod(cfg, 81)
    r_a, s_a = divmod(a, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return {
        'a': a, 'x': x,
        'r_a': r_a, 's_a': s_a,
        'r': r, 's': s, 't': t, 'u': u,
    }

def decode_BX(cfg):
    """Decode BX config into B atom and X atom."""
    b, x = divmod(cfg, 81)
    t_b, u_b = divmod(b, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return {
        'b': b, 'x': x,
        't_b': t_b, 'u_b': u_b,
        'r': r, 's': s, 't': t, 'u': u,
    }

# ─── Refinement feature functions ──────────────────────────────────

def refine_XX_features(cfg_data):
    """
    Additional Boolean features for XX schema.

    From collision analysis:
    - Group {45, 49, 51}: all (False, False, False, False, False, False, False, False)
      Differ in s2 and t2 values
    - Group {44, 48, 50}: all (False, False, False, False, False, True, False, False)
      Differ in s2 and t2 values
    - Group {27, 31, 33}: all (False, False, True, False, False, False, False, False)
      Differ in s2 and t2 values
    - Group {26, 30, 32}: all (False, False, True, False, False, True, False, False)
      Differ in s2 and t2 values

    Key discriminating features:
    - All coordinate values: r1, s1, t1, u1, r2, s2, t2, u2
    Using a minimal subset should suffice, but using all ensures full separation
    """
    d = cfg_data

    # Minimal refinement features to resolve all collisions:
    # Original base signature includes: x1_live, x2_live, same_r, same_s, same_t, same_u, same_A_atom, same_B_atom
    # Collisions occur when all these are the same but orbits differ in specific coordinates
    # Key additional features needed: s2, t2 (the coordinates that vary within collision groups)
    # We also include u2 and r2 for completeness
    return (d['s2'], d['t2'], d['u2'], d['r2'])

def refine_AX_features(cfg_data):
    """
    Additional Boolean features for AX schema.

    From collision analysis:
    - Group {2, 4}: both (True, False, False)
      AX[A[0,0],X[0,1|0,0]] vs AX[A[0,0],X[0,1|2,0]]
      Difference: t value (0 vs 2)
    - Group {7, 9}: both (False, False, False)
      AX[A[0,0],X[1,1|0,0]] vs AX[A[0,0],X[1,1|2,0]]
      Difference: t value (0 vs 2)

    Discriminating feature: t value
    """
    d = cfg_data
    t_val = d['t']
    u_val = d['u']

    return (t_val, u_val)

def refine_BX_features(cfg_data):
    """
    Additional Boolean features for BX schema.

    From collision analysis:
    - Group {2, 8}: both (False, True, False)
      BX[B[0,0],X[0,0|1,0]] vs BX[B[0,0],X[0,1|2,0]]
      Difference: s value (0 vs 1)
    - Group {3, 9}: both (False, False, False)
      BX[B[0,0],X[0,0|1,1]] vs BX[B[0,0],X[0,1|2,1]]
      Difference: s value (0 vs 1)

    Discriminating feature: s value
    """
    d = cfg_data
    s_val = d['s']
    r_val = d['r']

    return (s_val, r_val)

# ─── Main analysis ──────────────────────────────────────────────────

def analyze_collisions(schema, signatures_file, decode_fn, refine_fn):
    """
    Analyze collision groups for a schema and compute refined signatures.

    Returns:
    - collision_groups: dict mapping signature -> list of orbit records
    - refined_orbits: list of orbit records with refined_signature added
    """
    orbits = []
    with open(signatures_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orbits.append(row)

    # Group by signature
    sig_groups = defaultdict(list)
    for orbit in orbits:
        sig_groups[orbit['signature_key']].append(orbit)

    # Find collision groups (signature shared by multiple orbits)
    collision_groups = {sig: orbs for sig, orbs in sig_groups.items() if len(orbs) > 1}

    # Compute refined signatures for all orbits
    refined_orbits = []
    for orbit in orbits:
        cfg = int(orbit['rep_config_id'])
        cfg_data = decode_fn(cfg)
        base_sig = orbit['signature_key']
        refine_sig = refine_fn(cfg_data)

        # Combined signature: base + refinement
        refined_sig = (base_sig, refine_sig)

        refined_orbits.append({
            **orbit,
            'refined_signature': repr(refined_sig),
            'refinement_features': repr(refine_sig),
        })

    return collision_groups, refined_orbits

def write_refined_csv(refined_orbits, out_path):
    """Write refined signatures to CSV."""
    if not refined_orbits:
        return

    fieldnames = list(refined_orbits[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(refined_orbits)

def main():
    exports = Path('outputs/exports')
    exports.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("="*70)
    print("Signature Collision Refinement")
    print("="*70)
    print(f"Generated: {ts}\n")

    # ─── XX Schema ──────────────────────────────────────────────────
    print("--- XX Schema ---")
    xx_sig_file = exports / 'signatures_XX.csv'
    xx_coll, xx_refined = analyze_collisions('XX', xx_sig_file, decode_XX, refine_XX_features)

    print(f"Total orbits: 56")
    print(f"Collision groups: {len(xx_coll)}")
    for sig, orbs in xx_coll.items():
        orbit_ids = [o['orbit_id'] for o in orbs]
        print(f"  {sig}: orbits {orbit_ids}")

    # Check if refinement resolves all collisions
    refined_sigs = [o['refined_signature'] for o in xx_refined]
    n_refined_distinct = len(set(refined_sigs))
    print(f"Refined signatures: {n_refined_distinct} distinct")
    if n_refined_distinct == 56:
        print("  [OK] All collisions resolved")
    else:
        print(f"  [WARN] Still {56 - n_refined_distinct} collisions remain")

    # Write refined signatures
    xx_out = exports / 'signatures_XX_refined.csv'
    write_refined_csv(xx_refined, xx_out)
    print(f"Written: {xx_out.name}\n")

    # ─── AX Schema ──────────────────────────────────────────────────
    print("--- AX Schema ---")
    ax_sig_file = exports / 'signatures_AX.csv'
    ax_coll, ax_refined = analyze_collisions('AX', ax_sig_file, decode_AX, refine_AX_features)

    print(f"Total orbits: 10")
    print(f"Collision groups: {len(ax_coll)}")
    for sig, orbs in ax_coll.items():
        orbit_ids = [o['orbit_id'] for o in orbs]
        print(f"  {sig}: orbits {orbit_ids}")

    refined_sigs = [o['refined_signature'] for o in ax_refined]
    n_refined_distinct = len(set(refined_sigs))
    print(f"Refined signatures: {n_refined_distinct} distinct")
    if n_refined_distinct == 10:
        print("  [OK] All collisions resolved")
    else:
        print(f"  [WARN] Still {10 - n_refined_distinct} collisions remain")

    ax_out = exports / 'signatures_AX_refined.csv'
    write_refined_csv(ax_refined, ax_out)
    print(f"Written: {ax_out.name}\n")

    # ─── BX Schema ──────────────────────────────────────────────────
    print("--- BX Schema ---")
    bx_sig_file = exports / 'signatures_BX.csv'
    bx_coll, bx_refined = analyze_collisions('BX', bx_sig_file, decode_BX, refine_BX_features)

    print(f"Total orbits: 10")
    print(f"Collision groups: {len(bx_coll)}")
    for sig, orbs in bx_coll.items():
        orbit_ids = [o['orbit_id'] for o in orbs]
        print(f"  {sig}: orbits {orbit_ids}")

    refined_sigs = [o['refined_signature'] for o in bx_refined]
    n_refined_distinct = len(set(refined_sigs))
    print(f"Refined signatures: {n_refined_distinct} distinct")
    if n_refined_distinct == 10:
        print("  [OK] All collisions resolved")
    else:
        print(f"  [WARN] Still {10 - n_refined_distinct} collisions remain")

    bx_out = exports / 'signatures_BX_refined.csv'
    write_refined_csv(bx_refined, bx_out)
    print(f"Written: {bx_out.name}\n")

    print("="*70)
    print("REFINEMENT COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
