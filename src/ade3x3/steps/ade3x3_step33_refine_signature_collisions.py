"""
ade3x3_step33_refine_signature_collisions.py

Resolve signature collisions using workbench-minimal refinement features.

Target schemas:
- XX: 4 groups of 3 orbits each (12 collisions)  — refiner (s2, t2)
- AX: 2 groups of 2 orbits each (4 collisions)    — refiner (t,)
- BX: 2 groups of 2 orbits each (4 collisions)    — refiner (s,)
- CXXC: 784 groups (2744 collisions)               — refiner (s4, t4)
- CCXX: 784 groups (2744 collisions)               — refiner (s4, t4)

Refiners are the minimal appendages discovered by the workbench (step 36).
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
    Minimal refinement features for XX schema.

    Workbench-discovered: (x2_s2, x2_t2) is the smallest order-2
    appendage that fully resolves all 12 collisions.
    Historical refiner (s2, t2, u2, r2) over-provisioned by 2 features.
    """
    d = cfg_data
    return (d['s2'], d['t2'])

def refine_AX_features(cfg_data):
    """
    Minimal refinement features for AX schema.

    Workbench-discovered: x_t is the smallest order-1 appendage
    that fully resolves all 4 collisions.
    Historical refiner (t, u) over-provisioned by 1 feature.
    """
    d = cfg_data
    return (d['t'],)

def refine_BX_features(cfg_data):
    """
    Minimal refinement features for BX schema.

    Workbench-discovered: x_s is the smallest order-1 appendage
    that fully resolves all 4 collisions.
    Historical refiner (s, r) over-provisioned by 1 feature.
    """
    d = cfg_data
    return (d['s'],)

def decode_CXXC(cfg):
    """Decode CXXC config into (c1, x1, x2, c2) components."""
    c2 = cfg % 9
    rem = cfg // 9
    x2 = rem % 81
    rem = rem // 81
    x1 = rem % 81
    c1 = rem // 81
    r1, u1 = divmod(c1, 3)
    r2, s2 = divmod(x1 // 9, 3)
    t2, u2 = divmod(x1 % 9, 3)
    r4, s4 = divmod(x2 // 9, 3)
    t4, u4 = divmod(x2 % 9, 3)
    r3, u3 = divmod(c2, 3)
    return {
        'c1': c1, 'x1': x1, 'x2': x2, 'c2': c2,
        'r1': r1, 'u1': u1,
        'r2': r2, 's2': s2, 't2': t2, 'u2': u2,
        'r4': r4, 's4': s4, 't4': t4, 'u4': u4,
        'r3': r3, 'u3': u3,
    }

def refine_CXXC_features(cfg_data):
    """
    Minimal refinement features for CXXC schema.

    Workbench-discovered: (x2_s4, x2_t4) is the smallest order-2
    appendage that fully resolves all 2744 collisions.
    """
    d = cfg_data
    return (d['s4'], d['t4'])

def decode_CCXX(cfg):
    """Decode CCXX config into (c1, c2, x1, x2) components."""
    x2 = cfg % 81
    rem = cfg // 81
    x1 = rem % 81
    rem = rem // 81
    c2 = rem % 9
    c1 = rem // 9
    r1, u1 = divmod(c1, 3)
    r2, u2 = divmod(c2, 3)
    r3, s3 = divmod(x1 // 9, 3)
    t3, u3 = divmod(x1 % 9, 3)
    r4, s4 = divmod(x2 // 9, 3)
    t4, u4 = divmod(x2 % 9, 3)
    return {
        'c1': c1, 'c2': c2, 'x1': x1, 'x2': x2,
        'r1': r1, 'u1': u1,
        'r2': r2, 'u2': u2,
        'r3': r3, 's3': s3, 't3': t3, 'u3': u3,
        'r4': r4, 's4': s4, 't4': t4, 'u4': u4,
    }

def refine_CCXX_features(cfg_data):
    """
    Minimal refinement features for CCXX schema.

    Workbench-discovered: (x2_s4, x2_t4) is the smallest order-2
    appendage that fully resolves all 2744 collisions.
    """
    d = cfg_data
    return (d['s4'], d['t4'])

# ─── Refiner metadata ───────────────────────────────────────────────

REFINERS = {
    'XX':   {'feature_names': '(s2, t2)',        'size': 2,
             'historical': '(s2, t2, u2, r2)',    'hist_size': 4},
    'AX':   {'feature_names': '(t,)',            'size': 1,
             'historical': '(t, u)',              'hist_size': 2},
    'BX':   {'feature_names': '(s,)',            'size': 1,
             'historical': '(s, r)',              'hist_size': 2},
    'CXXC': {'feature_names': '(s4, t4)',        'size': 2,
             'historical': '(none)',              'hist_size': 0},
    'CCXX': {'feature_names': '(s4, t4)',        'size': 2,
             'historical': '(none)',              'hist_size': 0},
}

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
    print("Signature Collision Refinement (workbench-minimal refiners)")
    print("="*70)
    print(f"Generated: {ts}\n")

    summary_rows = []

    def run_schema(name, total_orbits, sig_file, decode_fn, refine_fn):
        print(f"--- {name} Schema ---")
        coll, refined = analyze_collisions(name, sig_file, decode_fn, refine_fn)
        n_coll = sum(len(v) for v in coll.values())

        refined_sigs = [o['refined_signature'] for o in refined]
        n_distinct = len(set(refined_sigs))
        resolved = n_distinct == total_orbits

        print(f"  Total orbits: {total_orbits}")
        print(f"  Collision groups: {len(coll)}")
        print(f"  Collisions: {n_coll}")
        print(f"  Refined signatures: {n_distinct} distinct")
        if resolved:
            print(f"  [OK] All collisions resolved")
        else:
            print(f"  [WARN] Still {total_orbits - n_distinct} collisions remain")

        out = exports / f'signatures_{name}_refined.csv'
        write_refined_csv(refined, out)
        print(f"  Written: {out.name}\n")

        meta = REFINERS[name]
        summary_rows.append({
            'schema': name,
            'total_orbits': total_orbits,
            'base_collisions': n_coll,
            'historical_refiner': meta['historical'],
            'historical_size': meta['hist_size'],
            'workbench_refiner': meta['feature_names'],
            'workbench_size': meta['size'],
            'refined_distinct': n_distinct,
            'fully_resolved': resolved,
        })

        return refined

    # ─── XX ───────────────────────────────────────────────────────
    run_schema('XX', 56, exports / 'signatures_XX.csv',
               decode_XX, refine_XX_features)

    # ─── AX ───────────────────────────────────────────────────────
    run_schema('AX', 10, exports / 'signatures_AX.csv',
               decode_AX, refine_AX_features)

    # ─── BX ───────────────────────────────────────────────────────
    run_schema('BX', 10, exports / 'signatures_BX.csv',
               decode_BX, refine_BX_features)

    # ─── CXXC ─────────────────────────────────────────────────────
    run_schema('CXXC', 2744, exports / 'signatures_CXXC.csv',
               decode_CXXC, refine_CXXC_features)

    # ─── CCXX ─────────────────────────────────────────────────────
    run_schema('CCXX', 2744, exports / 'signatures_CCXX.csv',
               decode_CCXX, refine_CCXX_features)

    # ─── Summary report ───────────────────────────────────────────
    print("="*70)
    print("REFINEMENT SUMMARY")
    print("="*70)
    print()
    print(f"{'Schema':<8} {'Hist':>6} {'WB':>4} {'Hist refiner':<22} "
          f"{'WB refiner':<16} {'Distinct':>8} {'Resolved':>8}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['schema']:<8} {r['historical_size']:>6} {r['workbench_size']:>4} "
              f"{r['historical_refiner']:<22} {r['workbench_refiner']:<16} "
              f"{r['refined_distinct']:>8} {'YES' if r['fully_resolved'] else 'NO':>8}")

    # write summary CSV
    summary_path = exports / 'refinement_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWritten: {summary_path.name}")

    print()
    print("="*70)
    print("REFINEMENT COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
