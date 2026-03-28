"""
ade3x3_step34_compute_cxxc_orbits.py

Compute orbits and signatures for the CXXC schema.

CXXC = C × X × X × C
- Typed arity: 4
- Raw arity: 6  (C, A_X1, B_X1, A_X2, B_X2, C)
- Total configs: 9 × 81 × 81 × 9 = 531,441

This is the first arity-4 schema to be fully cataloged.
The bridge is already defined in the canonical dossier Section 7.

Hardware: Ryzen 5900X (12 cores), 80 GB RAM
Expected: ~5-10 minutes for orbit computation
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Group of size 216 ──────────────────────────────────────────────

S3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

def _build_actions():
    return [(pi_rA, pi_shared, pi_cB)
            for pi_rA in S3 for pi_shared in S3 for pi_cB in S3]

ACTIONS = _build_actions()
print(f"Group size: {len(ACTIONS)}")

# ─── CXXC encoding/decoding ─────────────────────────────────────────

def decode_CXXC(cfg):
    """
    Decode CXXC config into (c1, x1, x2, c2) components.

    Encoding: cfg = ((c1 * 81 + x1) * 81 + x2) * 9 + c2
    """
    c2 = cfg % 9
    rem = cfg // 9
    x2 = rem % 81
    rem = rem // 81
    x1 = rem % 81
    c1 = rem // 81

    # Decode C atoms
    r1, u1 = divmod(c1, 3)
    r3, u3 = divmod(c2, 3)

    # Decode X atoms
    r2, s2 = divmod(x1 // 9, 3)
    t2, u2 = divmod(x1 % 9, 3)

    r4, s4 = divmod(x2 // 9, 3)
    t4, u4 = divmod(x2 % 9, 3)

    return {
        'c1': c1, 'x1': x1, 'x2': x2, 'c2': c2,
        'r1': r1, 'u1': u1,
        'r2': r2, 's2': s2, 't2': t2, 'u2': u2,
        'r4': r4, 's4': s4, 't4': t4, 'u4': u4,
        'r3': r3, 'u3': u3,
    }

def get_name_CXXC(cfg):
    """Generate human-readable name for CXXC config."""
    d = decode_CXXC(cfg)
    return (f"CXXC[C[{d['r1']},{d['u1']}],"
            f"X[{d['r2']},{d['s2']}|{d['t2']},{d['u2']}],"
            f"X[{d['r4']},{d['s4']}|{d['t4']},{d['u4']}],"
            f"C[{d['r3']},{d['u3']}]]")

def act_CXXC(cfg, act):
    """Apply group action to CXXC config."""
    d = decode_CXXC(cfg)

    # Apply action to each component
    c1n = 3*act[0][d['r1']] + act[2][d['u1']]
    x1n = 9*(3*act[0][d['r2']]+act[1][d['s2']]) + (3*act[1][d['t2']]+act[2][d['u2']])
    x2n = 9*(3*act[0][d['r4']]+act[1][d['s4']]) + (3*act[1][d['t4']]+act[2][d['u4']])
    c2n = 3*act[0][d['r3']] + act[2][d['u3']]

    # Re-encode
    return ((c1n * 81 + x1n) * 81 + x2n) * 9 + c2n

# ─── Signature function ─────────────────────────────────────────────

def sig_CXXC(cfg):
    """
    Compute signature for CXXC configuration.

    Signature fields (following CXC pattern, extended):
    - x1_live: s2 == t2
    - x2_live: s4 == t4
    - c1_equals_c2: c1 == c2
    - c1_is_target_of_x1_if_live: x1_live and c1 == 3*r2+u2
    - c1_is_target_of_x2_if_live: x2_live and c1 == 3*r4+u4
    - c2_is_target_of_x1_if_live: x1_live and c2 == 3*r2+u2
    - c2_is_target_of_x2_if_live: x2_live and c2 == 3*r4+u4
    - row_quad(c1,x1,x2,c2): equality partition of (r1, r2, r4, r3)
    - col_quad(c1,x1,x2,c2): equality partition of (u1, u2, u4, u3)

    Note: row_quad and col_quad encode the equality partition as a canonical tuple.
    For example:
    - (0,0,0,0) = all four equal
    - (0,0,0,1) = first three equal, fourth distinct
    - (0,1,2,3) = all four distinct
    etc.
    """
    d = decode_CXXC(cfg)

    x1_live = (d['s2'] == d['t2'])
    x2_live = (d['s4'] == d['t4'])

    # Compute row and column "quads" - canonical encoding of equality partitions
    row_vals = (d['r1'], d['r2'], d['r4'], d['r3'])
    col_vals = (d['u1'], d['u2'], d['u4'], d['u3'])

    row_quad = _canonical_partition(row_vals)
    col_quad = _canonical_partition(col_vals)

    return (
        x1_live,
        x2_live,
        d['c1'] == d['c2'],
        x1_live and d['c1'] == 3*d['r2']+d['u2'],
        x2_live and d['c1'] == 3*d['r4']+d['u4'],
        x1_live and d['c2'] == 3*d['r2']+d['u2'],
        x2_live and d['c2'] == 3*d['r4']+d['u4'],
        row_quad,
        col_quad,
    )

def _canonical_partition(vals):
    """
    Encode equality partition of a tuple as canonical representative.

    Example:
    - (1, 1, 1, 1) → (0, 0, 0, 0)  [all equal]
    - (1, 1, 2, 2) → (0, 0, 1, 1)  [first two equal, last two equal]
    - (1, 2, 3, 4) → (0, 1, 2, 3)  [all distinct]
    - (2, 1, 2, 1) → (0, 1, 0, 1)  [1st & 3rd equal, 2nd & 4th equal]
    """
    mapping = {}
    next_id = 0
    result = []
    for v in vals:
        if v not in mapping:
            mapping[v] = next_id
            next_id += 1
        result.append(mapping[v])
    return tuple(result)

# ─── Orbit computation ──────────────────────────────────────────────

def compute_orbits():
    """
    Compute all orbits for CXXC schema.

    Returns: list of orbit dicts with fields:
    - orbit_id
    - rep_config_id
    - rep_readable
    - orbit_size
    - stabilizer_size
    - signature_key
    """
    n_configs = 9 * 81 * 81 * 9
    print(f"\nComputing orbits for CXXC ({n_configs:,} configs)...")
    print("This may take a few minutes...\n")

    # Compute canonical representative for each config
    print("[1/3] Computing canonical representatives...")
    min_repr = {}
    progress_step = n_configs // 20  # 5% increments

    for cfg in range(n_configs):
        if cfg % progress_step == 0 and cfg > 0:
            pct = 100 * cfg / n_configs
            print(f"  {pct:5.1f}% ({cfg:,} / {n_configs:,})")

        m = cfg
        for act in ACTIONS:
            img = act_CXXC(cfg, act)
            if img < m:
                m = img
        min_repr[cfg] = m

    print(f"  100.0% ({n_configs:,} / {n_configs:,})")

    # Group configs by canonical representative
    print("\n[2/3] Grouping orbits...")
    groups = defaultdict(list)
    for cfg in range(n_configs):
        groups[min_repr[cfg]].append(cfg)

    print(f"  Found {len(groups)} orbits")

    # Compute orbit metadata and signatures
    print("\n[3/3] Computing signatures and stabilizers...")
    orbit_data = []
    for oid, (rep, members) in enumerate(sorted(groups.items())):
        size = len(members)
        stab = sum(1 for a in ACTIONS if act_CXXC(rep, a) == rep)
        sig = sig_CXXC(rep)

        orbit_data.append({
            'orbit_id': oid,
            'rep_config_id': rep,
            'rep_readable': get_name_CXXC(rep),
            'orbit_size': size,
            'stabilizer_size': stab,
            'signature_key': sig,
        })

        if (oid + 1) % 50 == 0:
            print(f"  {oid+1} orbits processed...")

    print(f"  {len(orbit_data)} orbits complete\n")

    # Verify orbit-stabilizer theorem
    print("Verifying orbit-stabilizer theorem...")
    for o in orbit_data:
        assert o['orbit_size'] * o['stabilizer_size'] == 216, \
            f"Orbit {o['orbit_id']}: {o['orbit_size']}*{o['stabilizer_size']} != 216"
    print("  [OK] All orbits satisfy |orbit| x |stabilizer| = 216\n")

    return orbit_data

# ─── CSV export ─────────────────────────────────────────────────────

def write_csv(orbit_data, path):
    """Write orbit data to CSV."""
    fieldnames = ['schema', 'orbit_id', 'rep_config_id', 'rep_readable',
                  'signature_key', 'orbit_size', 'stabilizer_size']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in orbit_data:
            w.writerow({
                'schema': 'CXXC',
                'orbit_id': row['orbit_id'],
                'rep_config_id': row['rep_config_id'],
                'rep_readable': row['rep_readable'],
                'signature_key': repr(row['signature_key']),
                'orbit_size': row['orbit_size'],
                'stabilizer_size': row['stabilizer_size'],
            })

def write_summary_md(orbit_data, path):
    """Write summary markdown."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    n_orbits = len(orbit_data)
    n_sigs = len(set(repr(o['signature_key']) for o in orbit_data))
    total_configs = sum(o['orbit_size'] for o in orbit_data)

    lines = [
        "# CXXC Orbit Computation Summary",
        "",
        f"**Generated:** {ts}",
        "**Schema:** CXXC (C × X × X × C)",
        "**Typed Arity:** 4",
        "**Raw Arity:** 6",
        "",
        "## Results",
        "",
        f"- Total configurations: {total_configs:,}",
        f"- Total orbits: {n_orbits}",
        f"- Distinct signatures: {n_sigs}",
        f"- Orbit-complete: {'yes' if n_sigs == n_orbits else 'no'}",
        "",
        "## Verification",
        "",
        "- All orbits satisfy |orbit| × |stabilizer| = 216 ✓",
        f"- Sum of orbit sizes = {total_configs:,} ✓",
        "",
        "## Signature Format",
        "",
        "Signature tuple fields:",
        "```",
        "(x1_live, x2_live, c1_equals_c2,",
        " c1_is_target_of_x1_if_live, c1_is_target_of_x2_if_live,",
        " c2_is_target_of_x1_if_live, c2_is_target_of_x2_if_live,",
        " row_quad, col_quad)",
        "```",
        "",
        "Where:",
        "- `x1_live`, `x2_live`: Boolean liveness of each X atom",
        "- `c1_equals_c2`: Whether first and last C atoms are equal",
        "- Target flags: Whether C atoms are targets of live X atoms",
        "- `row_quad`, `col_quad`: Canonical encoding of equality partitions",
        "",
        "## Exported Files",
        "",
        "- `signatures_CXXC.csv`: Complete orbit roster with signatures",
        "- `cxxc_computation_summary.md`: This file",
        "",
    ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

# ─── Main ───────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("CXXC Orbit Computation")
    print("="*70)

    # Compute orbits
    import time
    start = time.time()
    orbit_data = compute_orbits()
    elapsed = time.time() - start

    # Export results
    exports = Path('outputs/exports')
    exports.mkdir(parents=True, exist_ok=True)

    csv_path = exports / 'signatures_CXXC.csv'
    write_csv(orbit_data, csv_path)
    print(f"Written: {csv_path}")

    md_path = exports / 'cxxc_computation_summary.md'
    write_summary_md(orbit_data, md_path)
    print(f"Written: {md_path}")

    # Summary
    n_orbits = len(orbit_data)
    n_sigs = len(set(repr(o['signature_key']) for o in orbit_data))

    print("\n" + "="*70)
    print("COMPUTATION COMPLETE")
    print("="*70)
    print(f"Orbits: {n_orbits}")
    print(f"Distinct signatures: {n_sigs}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("="*70)

if __name__ == '__main__':
    main()
