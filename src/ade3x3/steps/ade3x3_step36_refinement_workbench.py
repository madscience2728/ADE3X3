"""
ade3x3_step36_refinement_workbench.py

Generic refinement workbench for signature collisions.

The workbench is schema-agnostic in its core operations:
  1. Load base signatures from CSV
  2. Detect collision groups
  3. Evaluate candidate feature appendages (single + pair)
  4. Rank by resolving power
  5. Export machine-readable CSV + human-readable markdown

CXXC is the first stress test.  To adapt to a new schema, provide:
  - CSV path
  - decode function
  - feature registry (name -> callable)
"""

import csv
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Schema plug-in: CXXC
# ═══════════════════════════════════════════════════════════════════════

def decode_cxxc(cfg):
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


def build_cxxc_feature_registry():
    """Return {name: callable(decode_dict) -> hashable} for every candidate
    raw-coordinate feature extractable from CXXC atoms."""

    def _r(name, key):
        return name, lambda d, k=key: d[k]

    features = {}

    # --- per-atom coordinates ---
    for label, prefix, coords in [
            ('C1', 'c1', ['r1', 'u1']),
            ('X1', 'x1', ['r2', 's2', 't2', 'u2']),
            ('X2', 'x2', ['r4', 's4', 't4', 'u4']),
            ('C2', 'c2', ['r3', 'u3']),
    ]:
        for c in coords:
            name, fn = _r(f'{prefix}_{c}', c)
            features[name] = fn

    # --- atom-identity features ---
    features['c1_equals_c2'] = lambda d: d['c1'] == d['c2']
    features['x1_equals_x2'] = lambda d: d['x1'] == d['x2']
    features['x1_live'] = lambda d: d['s2'] == d['t2']
    features['x2_live'] = lambda d: d['s4'] == d['t4']

    # --- cross-atom coordinate equalities ---
    features['r1_eq_r3'] = lambda d: d['r1'] == d['r3']
    features['u1_eq_u3'] = lambda d: d['u1'] == d['u3']

    return features


# ═══════════════════════════════════════════════════════════════════════
# Generic core
# ═══════════════════════════════════════════════════════════════════════

def load_signatures(csv_path):
    """Return list of orbit dicts from a signatures CSV."""
    orbits = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            orbits.append(row)
    return orbits


def detect_collisions(orbits, sig_key='signature_key'):
    """Group orbits by base signature, return only groups with >1 orbit."""
    groups = defaultdict(list)
    for o in orbits:
        groups[o[sig_key]].append(o)
    return {s: mems for s, mems in groups.items() if len(mems) > 1}


def count_collisions(groups):
    """Total orbits caught in a collision (= total_orbits - distinct_sigs)."""
    return sum(len(m) for m in groups.values())


# ═══════════════════════════════════════════════════════════════════════
# Feature evaluation engine
# ═══════════════════════════════════════════════════════════════════════

def evaluate_candidates(orbits, collisions, decode_fn, feature_registry):
    """Evaluate every single-feature and pair-feature appendage.

    Returns list of result dicts sorted by (remaining_collisions asc,
    collision_reduction desc, feature_count asc)."""

    results = []
    total = count_collisions(collisions)
    names = sorted(feature_registry.keys())

    def _eval(feature_names):
        sub = {n: feature_registry[n] for n in feature_names}
        sigs = set()
        for o in orbits:
            cfg = int(o['rep_config_id'])
            d = decode_fn(cfg)
            fv = tuple(sub[n](d) for n in feature_names)
            sigs.add((o['signature_key'], fv))
        remaining = len(orbits) - len(sigs)
        return remaining, total - remaining

    # --- singles ---
    for fn in names:
        rem, red = _eval([fn])
        results.append({
            'feature_count': 1,
            'feature_names': fn,
            'distinct_signatures': len(orbits) - rem,
            'remaining_collisions': rem,
            'collision_reduction': red,
        })

    # --- pairs ---
    for f1, f2 in combinations(names, 2):
        rem, red = _eval([f1, f2])
        results.append({
            'feature_count': 2,
            'feature_names': f'{f1}+{f2}',
            'distinct_signatures': len(orbits) - rem,
            'remaining_collisions': rem,
            'collision_reduction': red,
        })

    results.sort(key=lambda r: (r['remaining_collisions'],
                                 -r['collision_reduction'],
                                 r['feature_count']))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def write_csv(results, path, top_n=50):
    fieldnames = ['rank', 'feature_count', 'feature_names',
                  'distinct_signatures', 'remaining_collisions',
                  'collision_reduction']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(results[:top_n], 1):
            w.writerow({'rank': i, **r})


def write_report(results, collisions, total_orbits, n_sigs, n_features, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    base_total = count_collisions(collisions)
    top = results[:20]

    L = []
    def w(s=""): L.append(s)

    w("# CXXC Refinement Workbench Report")
    w()
    w(f"**Generated:** {ts}")
    w(f"**Source:** `signatures_CXXC.csv`")
    w()

    w("## 1. Configuration")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| Schema | CXXC |")
    w(f"| Decode function | `decode_cxxc` |")
    w(f"| Total orbits | {total_orbits} |")
    w(f"| Base distinct signatures | {n_sigs} |")
    w(f"| Base collisions | {base_total} |")
    w(f"| Collision groups | {len(collisions)} |")
    w(f"| Candidate features | {n_features} |")
    w(f"| Candidates evaluated | {len(results)} |")
    w()

    w("## 2. Feature Registry")
    w()
    w("| Feature | Source | Coordinate | Description |")
    w("|---------|--------|------------|-------------|")
    meta = [
        ('c1_r', 'C1', 'row', 'row index of C1'),
        ('c1_u', 'C1', 'col', 'column index of C1'),
        ('x1_r2', 'X1', 'row-left', 'left row of X1'),
        ('x1_s2', 'X1', 'col-left', 'left col of X1'),
        ('x1_t2', 'X1', 'row-right', 'right row of X1'),
        ('x1_u2', 'X1', 'col-right', 'right col of X1'),
        ('x2_r4', 'X2', 'row-left', 'left row of X2'),
        ('x2_s4', 'X2', 'col-left', 'left col of X2'),
        ('x2_t4', 'X2', 'row-right', 'right row of X2'),
        ('x2_u4', 'X2', 'col-right', 'right col of X2'),
        ('c2_r3', 'C2', 'row', 'row index of C2'),
        ('c2_u3', 'C2', 'col', 'column index of C2'),
        ('c1_equals_c2', 'C1,C2', 'identity', 'C1 == C2'),
        ('x1_equals_x2', 'X1,X2', 'identity', 'X1 == X2'),
        ('x1_live', 'X1', 'liveness', 's2 == t2'),
        ('x2_live', 'X2', 'liveness', 's4 == t4'),
        ('r1_eq_r3', 'C1,C2', 'equality', 'r1 == r3'),
        ('u1_eq_u3', 'C1,C2', 'equality', 'u1 == u3'),
    ]
    for feat, src, coord, desc in meta:
        w(f"| {feat} | {src} | {coord} | {desc} |")
    w()

    w("## 3. Collision Structure")
    w()
    sizes = sorted(set(len(v) for v in collisions.values()), reverse=True)
    for sz in sizes:
        n = sum(1 for v in collisions.values() if len(v) == sz)
        w(f"- {n} group(s) of size {sz}")
    w()
    w(f"**Total collisions:** {base_total} (sum of group sizes)")
    w()

    w("## 4. Top Candidates by Resolving Power")
    w()
    w("| Rank | Features | Count | Distinct | Remaining | Reduction |")
    w("|------|----------|-------|----------|-----------|-----------|")
    for i, r in enumerate(top, 1):
        w(f"| {i} | {r['feature_names']} | {r['feature_count']} "
          f"| {r['distinct_signatures']} | {r['remaining_collisions']} "
          f"| {r['collision_reduction']} |")
    w()

    if top:
        pct = 100 * top[0]['collision_reduction'] / base_total if base_total else 0
        w(f"**Best candidate** resolves {pct:.1f}% of collisions "
          f"({top[0]['collision_reduction']} / {base_total}).")
    w()

    fully = [r for r in results if r['remaining_collisions'] == 0]
    w("## 5. Full Resolution")
    w()
    if fully:
        w(f"**{len(fully)} feature set(s) fully resolve all collisions.**")
        w()
        w("| Features | Count |")
        w("|----------|-------|")
        for r in fully[:20]:
            w(f"| {r['feature_names']} | {r['feature_count']} |")
        singles = [r for r in fully if r['feature_count'] == 1]
        pairs = [r for r in fully if r['feature_count'] == 2]
        w()
        w(f"- Single-feature full resolvers: {len(singles)}")
        w(f"- Pair-feature full resolvers: {len(pairs)}")
    else:
        w("**No evaluated feature set fully resolves all collisions.**")
        w("Higher-order combinations or domain-derived features may be needed.")
    w()

    w("## 6. Summary")
    w()
    w(f"- {len(results)} candidate feature sets evaluated "
      f"({n_features} singles + {n_features*(n_features-1)//2} pairs)")
    if fully:
        w(f"- {len(fully)} feature set(s) achieve full resolution")
    w(f"- Best single-feature: {top[0]['feature_names'] if top else 'n/a'} "
      f"({top[0]['collision_reduction'] if top else 0} reduction)")
    best_pair = next((r for r in results if r['feature_count'] == 2), None)
    w(f"- Best pair-feature: {best_pair['feature_names'] if best_pair else 'n/a'} "
      f"({best_pair['collision_reduction'] if best_pair else 0} reduction)")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # console summary
    print(f"  Base collisions: {base_total}")
    print(f"  Candidates evaluated: {len(results)}")
    if top:
        print(f"  Best: {top[0]['feature_names']} "
              f"(remaining={top[0]['remaining_collisions']}, "
              f"reduction={top[0]['collision_reduction']})")
    print(f"  Full resolvers: {len(fully)}")


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════

def main():
    exports = Path('outputs/exports')
    exports.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Refinement Workbench - CXXC Stress Test")
    print("=" * 70)

    # 1. Load
    orbits = load_signatures(exports / 'signatures_CXXC.csv')
    print(f"\nLoaded {len(orbits)} CXXC orbits")

    # 2. Detect collisions
    collisions = detect_collisions(orbits)
    print(f"Collision groups: {len(collisions)}")
    print(f"Collisions: {count_collisions(collisions)}")

    # 3. Build feature registry
    registry = build_cxxc_feature_registry()
    print(f"Candidate features: {len(registry)}")

    # 4. Evaluate
    print("\nEvaluating candidates...")
    results = evaluate_candidates(orbits, collisions, decode_cxxc, registry)
    print(f"Evaluated {len(results)} feature sets")

    # 5. Export
    n_sigs = len(set(o['signature_key'] for o in orbits))
    write_csv(results, exports / 'cxxc_refinement_workbench.csv')
    write_report(results, collisions, len(orbits), n_sigs, len(registry),
                 exports / 'cxxc_refinement_workbench.md')

    print("\nWritten: cxxc_refinement_workbench.csv")
    print("Written: cxxc_refinement_workbench.md")
    print("=" * 70)
    print("WORKBENCH COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
