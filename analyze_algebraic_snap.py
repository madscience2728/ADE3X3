"""Build a comprehensive lookup table of algebraic numbers and compare baseline coefficients."""
import json, numpy as np
from collections import defaultdict

# Load baseline
with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    baseline = json.load(f)

# ── Build lookup table: x/y, sqrt(x/y), cbrt(x/y) for x,y in 1..10 ──
table = {}  # name -> positive value

for x in range(1, 11):
    for y in range(1, 11):
        v = x / y
        # rational x/y
        key = f"{x}/{y}"
        if v not in table.values() or key not in table:
            table[key] = v
        # sqrt(x/y)
        sv = np.sqrt(v)
        skey = f"sqrt({x}/{y})" if y > 1 else f"sqrt({x})"
        table[skey] = sv
        # cbrt(x/y)
        cv = v ** (1/3)
        ckey = f"cbrt({x}/{y})" if y > 1 else f"cbrt({x})"
        table[ckey] = cv

# Also add 0
table["0"] = 0.0

# Deduplicate: keep shortest name for each unique value (to 10dp)
deduped = {}
for name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
    rounded = round(val, 10)
    if rounded not in deduped:
        deduped[rounded] = (name, val)
    else:
        # keep shorter name
        existing_name = deduped[rounded][0]
        if len(name) < len(existing_name):
            deduped[rounded] = (name, val)

lookup = sorted(deduped.values(), key=lambda x: x[1])
print(f"Lookup table: {len(lookup)} unique algebraic values from 0 to {lookup[-1][1]:.4f}")
print()

# ── Extract all coefficients with their term/axis/position ──
coeffs = []
for ti, t in enumerate(baseline['terms']):
    for axis, skey, vkey in [('alpha','alpha_support','alpha_values'),
                              ('beta','beta_support','beta_values'),
                              ('gamma','gamma_support','gamma_values')]:
        for pos, (idx, val) in enumerate(zip(t[skey], t[vkey])):
            coeffs.append({
                'term': ti+1, 'axis': axis, 'index': idx,
                'value': val, 'absval': abs(val)
            })

print(f"Total coefficients: {len(coeffs)}")
nontrivial = [c for c in coeffs if c['absval'] > 0.01]
print(f"Non-trivial (|v| > 0.01): {len(nontrivial)}")
print()

# ── For each non-trivial coefficient, find best match in lookup ──
def find_best_match(absval):
    best_name, best_val, best_dist = None, None, 999
    for name, lv in lookup:
        d = abs(absval - lv)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_val = lv
    return best_name, best_val, best_dist

print("=" * 100)
print(f"{'term':>4} {'axis':>6} {'idx':>3} {'value':>12} {'|value|':>10} {'nearest':>16} "
      f"{'algebraic':>12} {'error':>12} {'rel_err%':>9}")
print("-" * 100)

matches = []
for c in sorted(nontrivial, key=lambda x: x['absval']):
    name, aval, dist = find_best_match(c['absval'])
    rel_err = dist / c['absval'] * 100 if c['absval'] > 0 else 0
    sign = "+" if c['value'] >= 0 else "-"
    flag = " ***" if dist < 0.002 else " **" if dist < 0.005 else " *" if dist < 0.01 else ""
    matches.append((c, name, aval, dist, rel_err))
    print(f"{c['term']:4d} {c['axis']:>6} {c['index']:3d} {c['value']:+12.6f} {c['absval']:10.6f} "
          f"{sign}{name:>15} {aval:12.6f} {dist:12.6f} {rel_err:8.3f}%{flag}")

# ── Summary statistics ──
print()
print("=" * 100)
print("MATCH QUALITY SUMMARY")
print("=" * 100)
dists = [m[3] for m in matches]
rel_errs = [m[4] for m in matches]
print(f"  Total non-trivial: {len(matches)}")
print(f"  Within 0.001: {sum(1 for d in dists if d < 0.001)}")
print(f"  Within 0.002: {sum(1 for d in dists if d < 0.002)}")
print(f"  Within 0.005: {sum(1 for d in dists if d < 0.005)}")
print(f"  Within 0.01:  {sum(1 for d in dists if d < 0.01)}")
print(f"  Within 0.02:  {sum(1 for d in dists if d < 0.02)}")
print(f"  Within 0.05:  {sum(1 for d in dists if d < 0.05)}")
print(f"  Mean distance: {np.mean(dists):.6f}")
print(f"  Median distance: {np.median(dists):.6f}")
print(f"  Max distance: {np.max(dists):.6f}")
print(f"  Mean rel error: {np.mean(rel_errs):.3f}%")

# ── Which algebraic values appear most often? ──
print()
print("=" * 100)
print("MOST COMMON NEAREST ALGEBRAIC VALUES")
print("=" * 100)
freq = defaultdict(list)
for c, name, aval, dist, rel_err in matches:
    if dist < 0.02:  # close matches only
        freq[name].append((c['term'], c['axis'], dist))

for name, hits in sorted(freq.items(), key=lambda x: -len(x[1])):
    if len(hits) >= 2:
        avg_dist = np.mean([h[2] for h in hits])
        terms = [f"t{h[0]}.{h[1]}" for h in hits]
        print(f"  {name:>16} ({len(hits):2d}x, avg_dist={avg_dist:.6f}): {', '.join(terms)}")

# ── Do the near-zero values also match a pattern? ──
print()
print("=" * 100)
print("NEAR-ZERO COEFFICIENTS (|v| < 0.01)")
print("=" * 100)
trivial = [c for c in coeffs if c['absval'] <= 0.01 and c['absval'] > 0.0001]
print(f"Count: {len(trivial)}")
# Are they all ≈ the same value?
if trivial:
    tvs = [c['absval'] for c in trivial]
    print(f"  min: {min(tvs):.6f}")
    print(f"  max: {max(tvs):.6f}")
    print(f"  mean: {np.mean(tvs):.6f}")
    print(f"  std: {np.std(tvs):.6f}")

# ── Per-term: what's the "type" of each term? ──
print()
print("=" * 100)
print("PER-TERM ALGEBRAIC SIGNATURE (close matches only, dist < 0.01)")
print("=" * 100)
for ti in range(1, 20):
    term_matches = [(c, name, aval, dist) for c, name, aval, dist, _ in matches
                    if c['term'] == ti and dist < 0.01]
    if term_matches:
        sig = []
        for c, name, aval, dist in term_matches:
            sign = "+" if c['value'] > 0 else "-"
            sig.append(f"{sign}{name}")
        print(f"  term {ti:2d}: {', '.join(sig)}")
