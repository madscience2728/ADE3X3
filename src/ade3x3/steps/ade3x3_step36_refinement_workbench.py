"""
ade3x3_step36_refinement_workbench.py

Generic refinement workbench for signature collisions.

The workbench is schema-agnostic.  A schema plug-in supplies:
  - schema_name     (str)
  - csv_path        (Path to signatures CSV)
  - decode_fn       (config_id -> dict of coordinate values)
  - feature_registry (name -> callable(dict) -> hashable)

The engine then:
  1. Loads base signatures
  2. Detects collision groups
  3. Evaluates candidate feature appendages up to order K
  4. Ranks by resolving power
  5. Exports machine-readable CSV + markdown
"""

import csv
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# Generic engine
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
    """Total orbits caught in a collision."""
    return sum(len(m) for m in groups.values())


def evaluate_candidates(orbits, collisions, decode_fn, feature_registry,
                        max_order=2):
    """Evaluate every feature combination up to order *max_order*.

    Single code path: for k in 1..max_order, enumerate C(n_features, k).

    Returns list of result dicts sorted by (remaining_collisions asc,
    collision_reduction desc, feature_count asc).
    """
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

    for k in range(1, max_order + 1):
        for combo in combinations(names, k):
            combo = list(combo)
            rem, red = _eval(combo)
            results.append({
                'feature_count': k,
                'feature_names': '+'.join(combo),
                'distinct_signatures': len(orbits) - rem,
                'remaining_collisions': rem,
                'collision_reduction': red,
            })

    results.sort(key=lambda r: (r['remaining_collisions'],
                                 -r['collision_reduction'],
                                 r['feature_count']))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting (generic — takes schema_name, no hard-coded strings)
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


def write_report(schema_name, results, collisions, total_orbits,
                 n_sigs, n_features, max_order, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    base_total = count_collisions(collisions)
    top = results[:20]

    L = []
    def w(s=""): L.append(s)

    w(f"# {schema_name} Refinement Workbench Report")
    w()
    w(f"**Generated:** {ts}")
    w()

    w("## 1. Configuration")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| Schema | {schema_name} |")
    w(f"| Total orbits | {total_orbits} |")
    w(f"| Base distinct signatures | {n_sigs} |")
    w(f"| Base collisions | {base_total} |")
    w(f"| Collision groups | {len(collisions)} |")
    w(f"| Candidate features | {n_features} |")
    w(f"| Max feature order | {max_order} |")
    w(f"| Candidates evaluated | {len(results)} |")
    w()

    w("## 2. Collision Structure")
    w()
    sizes = sorted(set(len(v) for v in collisions.values()), reverse=True)
    for sz in sizes:
        n = sum(1 for v in collisions.values() if len(v) == sz)
        w(f"- {n} group(s) of size {sz}")
    w()
    w(f"**Total collisions:** {base_total} (sum of group sizes)")
    w()

    w("## 3. Top Candidates by Resolving Power")
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
    w("## 4. Full Resolution")
    w()
    if fully:
        w(f"**{len(fully)} feature set(s) fully resolve all collisions.**")
        w()
        w("| Features | Count |")
        w("|----------|-------|")
        for r in fully[:30]:
            w(f"| {r['feature_names']} | {r['feature_count']} |")
        w()
        for k in range(1, max_order + 1):
            n_k = sum(1 for r in fully if r['feature_count'] == k)
            w(f"- Order-{k} full resolvers: {n_k}")
    else:
        w("**No evaluated feature set fully resolves all collisions.**")
        w("Higher-order combinations or domain-derived features may be needed.")
    w()

    w("## 5. Summary")
    w()
    # count per-order
    order_counts = defaultdict(int)
    for r in results:
        order_counts[r['feature_count']] += 1
    parts = [f"{order_counts[k]} order-{k}" for k in sorted(order_counts)]
    w(f"- {len(results)} candidate feature sets evaluated ({', '.join(parts)})")
    if fully:
        w(f"- {len(fully)} feature set(s) achieve full resolution")
    if top:
        w(f"- Best overall: {top[0]['feature_names']} "
          f"(order {top[0]['feature_count']}, "
          f"{top[0]['collision_reduction']} reduction)")

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
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════

def run_workbench(schema_name, csv_path, decode_fn, feature_registry,
                  output_dir=None, output_prefix=None, max_order=2):
    """Run the full refinement workbench for one schema.

    Parameters
    ----------
    schema_name : str
        Human-readable schema label (e.g. 'CXXC').
    csv_path : str or Path
        Path to signatures CSV (must contain 'signature_key' and
        'rep_config_id' columns).
    decode_fn : callable(int) -> dict
        Decodes a config_id into a dict of coordinate values.
    feature_registry : dict[str, callable(dict) -> hashable]
        Maps feature names to extractors.
    output_dir : str or Path, optional
        Directory for outputs.  Defaults to 'outputs/exports'.
    output_prefix : str, optional
        Filename prefix for outputs.  Defaults to schema_name lowered.
    max_order : int
        Maximum feature combination order (1 = singles only,
        2 = singles + pairs, 3 = + triples, etc.).

    Returns
    -------
    results : list[dict]
        Ranked candidate evaluations.
    """
    if output_dir is None:
        output_dir = Path('outputs/exports')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = schema_name.lower()

    csv_path = Path(csv_path)
    print("=" * 70)
    print(f"Refinement Workbench — {schema_name}")
    print("=" * 70)

    # 1. Load
    orbits = load_signatures(csv_path)
    print(f"\nLoaded {len(orbits)} {schema_name} orbits")

    # 2. Detect collisions
    collisions = detect_collisions(orbits)
    print(f"Collision groups: {len(collisions)}")
    print(f"Collisions: {count_collisions(collisions)}")

    # 3. Feature registry info
    n_features = len(feature_registry)
    print(f"Candidate features: {n_features}")

    # 4. Evaluate
    print(f"\nEvaluating candidates (max_order={max_order})...")
    results = evaluate_candidates(orbits, collisions, decode_fn,
                                  feature_registry, max_order=max_order)
    print(f"Evaluated {len(results)} feature sets")

    # 5. Export
    n_sigs = len(set(o['signature_key'] for o in orbits))
    csv_out = output_dir / f'{output_prefix}_refinement_workbench.csv'
    md_out = output_dir / f'{output_prefix}_refinement_workbench.md'

    write_csv(results, csv_out)
    write_report(schema_name, results, collisions, len(orbits),
                 n_sigs, n_features, max_order, md_out)

    print(f"\nWritten: {csv_out.name}")
    print(f"Written: {md_out.name}")
    print("=" * 70)
    print("WORKBENCH COMPLETE")
    print("=" * 70)

    return results


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
# Schema plug-in: XX
# ═══════════════════════════════════════════════════════════════════════

def decode_xx(cfg):
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


def build_xx_feature_registry():
    features = {}

    for prefix, atom_coords in [
            ('x1', ['r1', 's1', 't1', 'u1']),
            ('x2', ['r2', 's2', 't2', 'u2']),
    ]:
        for c in atom_coords:
            features[f'{prefix}_{c}'] = lambda d, k=c: d[k]

    features['x1_live'] = lambda d: d['s1'] == d['t1']
    features['x2_live'] = lambda d: d['s2'] == d['t2']
    features['x1_equals_x2'] = lambda d: d['x1'] == d['x2']

    features['same_r'] = lambda d: d['r1'] == d['r2']
    features['same_s'] = lambda d: d['s1'] == d['s2']
    features['same_t'] = lambda d: d['t1'] == d['t2']
    features['same_u'] = lambda d: d['u1'] == d['u2']
    features['same_A_atom'] = lambda d: (d['r1'] == d['r2']) and (d['s1'] == d['s2'])
    features['same_B_atom'] = lambda d: (d['t1'] == d['t2']) and (d['u1'] == d['u2'])

    return features


# ═══════════════════════════════════════════════════════════════════════
# Schema plug-in: AX
# ═══════════════════════════════════════════════════════════════════════

def decode_ax(cfg):
    a, x = divmod(cfg, 81)
    r_a, s_a = divmod(a, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return {
        'a': a, 'x': x,
        'r_a': r_a, 's_a': s_a,
        'r': r, 's': s, 't': t, 'u': u,
    }


def build_ax_feature_registry():
    features = {}

    for c in ['r_a', 's_a']:
        features[f'a_{c}'] = lambda d, k=c: d[k]
    for c in ['r', 's', 't', 'u']:
        features[f'x_{c}'] = lambda d, k=c: d[k]

    features['x_live'] = lambda d: d['s'] == d['t']
    features['a_row_eq_x_r'] = lambda d: d['r_a'] == d['r']
    features['a_col_eq_x_s'] = lambda d: d['s_a'] == d['s']

    return features


# ═══════════════════════════════════════════════════════════════════════
# Schema plug-in: BX
# ═══════════════════════════════════════════════════════════════════════

def decode_bx(cfg):
    b, x = divmod(cfg, 81)
    t_b, u_b = divmod(b, 3)
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return {
        'b': b, 'x': x,
        't_b': t_b, 'u_b': u_b,
        'r': r, 's': s, 't': t, 'u': u,
    }


def build_bx_feature_registry():
    features = {}

    for c in ['t_b', 'u_b']:
        features[f'b_{c}'] = lambda d, k=c: d[k]
    for c in ['r', 's', 't', 'u']:
        features[f'x_{c}'] = lambda d, k=c: d[k]

    features['x_live'] = lambda d: d['s'] == d['t']
    features['b_row_eq_x_t'] = lambda d: d['t_b'] == d['t']
    features['b_col_eq_x_u'] = lambda d: d['u_b'] == d['u']

    return features


# ═══════════════════════════════════════════════════════════════════════
# Schema registry
# ═══════════════════════════════════════════════════════════════════════

SCHEMAS = {
    'CXXC': dict(
        csv_path='outputs/exports/signatures_CXXC.csv',
        decode_fn=decode_cxxc,
        feature_registry_fn=build_cxxc_feature_registry,
        output_prefix='cxxc',
    ),
    'XX': dict(
        csv_path='outputs/exports/signatures_XX.csv',
        decode_fn=decode_xx,
        feature_registry_fn=build_xx_feature_registry,
        output_prefix='xx',
    ),
    'AX': dict(
        csv_path='outputs/exports/signatures_AX.csv',
        decode_fn=decode_ax,
        feature_registry_fn=build_ax_feature_registry,
        output_prefix='ax',
    ),
    'BX': dict(
        csv_path='outputs/exports/signatures_BX.csv',
        decode_fn=decode_bx,
        feature_registry_fn=build_bx_feature_registry,
        output_prefix='bx',
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════

def main():
    import sys
    args = sys.argv[1:]

    if not args:
        # run all registered schemas
        targets = list(SCHEMAS.keys())
    else:
        targets = [a.upper() for a in args]
        for t in targets:
            if t not in SCHEMAS:
                print(f"Unknown schema: {t}")
                print(f"Available: {', '.join(SCHEMAS.keys())}")
                sys.exit(1)

    for name in targets:
        cfg = SCHEMAS[name]
        run_workbench(
            schema_name=name,
            csv_path=cfg['csv_path'],
            decode_fn=cfg['decode_fn'],
            feature_registry=cfg['feature_registry_fn'](),
            output_prefix=cfg['output_prefix'],
            max_order=2,
        )
        print()


if __name__ == '__main__':
    main()
