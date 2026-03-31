"""Extended algebraic snap analysis — push coverage as far as possible."""
import json, numpy as np
from collections import defaultdict
from itertools import product as iprod

with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    baseline = json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build a very large lookup table
# ══════════════════════════════════════════════════════════════════════════════
table = {}  # name -> value (positive only; we handle signs separately)

# ── Rationals x/y, x in 0..20, y in 1..20 ──
for x in range(0, 21):
    for y in range(1, 21):
        v = x / y
        if v > 3.0:
            continue  # coefficients don't go past ~1.1
        key = f"{x}/{y}"
        table[key] = v

# ── sqrt(x/y) for x in 1..20, y in 1..20 ──
for x in range(1, 21):
    for y in range(1, 21):
        v = np.sqrt(x / y)
        if v > 3.0:
            continue
        key = f"√({x}/{y})" if y > 1 else f"√{x}"
        table[key] = v

# ── cbrt(x/y) for x in 1..20, y in 1..20 ──
for x in range(1, 21):
    for y in range(1, 21):
        v = (x / y) ** (1/3)
        if v > 3.0:
            continue
        key = f"∛({x}/{y})" if y > 1 else f"∛{x}"
        table[key] = v

# ── 4th root: (x/y)^(1/4) ──
for x in range(1, 21):
    for y in range(1, 21):
        v = (x / y) ** 0.25
        if v > 3.0:
            continue
        key = f"⁴√({x}/{y})"
        table[key] = v

# ── 6th root: (x/y)^(1/6) — natural for trilinear+bilinear combined ──
for x in range(1, 11):
    for y in range(1, 11):
        v = (x / y) ** (1/6)
        if v > 3.0:
            continue
        key = f"⁶√({x}/{y})"
        table[key] = v

# ── Products: sqrt(a/b) * cbrt(c/d) for small a,b,c,d ──
for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 6):
            for d in range(1, 6):
                v = np.sqrt(a/b) * ((c/d)**(1/3))
                if 0.05 < v < 2.5:
                    key = f"√({a}/{b})·∛({c}/{d})"
                    table[key] = v

# ── sqrt(a/b) * c/d for small values ──
for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 4):
            for d in range(1, 4):
                v = np.sqrt(a/b) * (c/d)
                if 0.05 < v < 2.5:
                    key = f"{c}/{d}·√({a}/{b})"
                    table[key] = v

# ── cbrt(a/b) * c/d ──
for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 4):
            for d in range(1, 4):
                v = ((a/b)**(1/3)) * (c/d)
                if 0.05 < v < 2.5:
                    key = f"{c}/{d}·∛({a}/{b})"
                    table[key] = v

# ── Special: (2^a * 3^b)^(1/c) for small exponents ──
for a in range(-3, 4):
    for b in range(-3, 4):
        for c in [2, 3, 6]:
            v = abs((2**a) * (3**b)) ** (1/c)
            if 0.05 < v < 2.5:
                key = f"(2^{a}·3^{b})^(1/{c})"
                table[key] = v

table["0"] = 0.0

# Deduplicate: keep shortest name per unique value (8dp)
deduped = {}
for name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
    rounded = round(val, 8)
    if rounded not in deduped:
        deduped[rounded] = (name, val)
    else:
        existing = deduped[rounded][0]
        if len(name) < len(existing):
            deduped[rounded] = (name, val)

lookup = sorted(deduped.values(), key=lambda x: x[1])
print(f"Lookup table: {len(lookup)} unique algebraic candidates")
print(f"Range: [{lookup[0][1]:.6f}, {lookup[-1][1]:.6f}]")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Extract all coefficients
# ══════════════════════════════════════════════════════════════════════════════
coeffs = []
for ti, t in enumerate(baseline['terms']):
    for axis, skey, vkey in [('α','alpha_support','alpha_values'),
                              ('β','beta_support','beta_values'),
                              ('γ','gamma_support','gamma_values')]:
        for pos, (idx, val) in enumerate(zip(t[skey], t[vkey])):
            coeffs.append({
                'term': ti+1, 'axis': axis, 'index': idx,
                'value': val, 'absval': abs(val)
            })

nontrivial = [c for c in coeffs if c['absval'] > 0.01]
nearzero = [c for c in coeffs if c['absval'] <= 0.01]
print(f"Total coefficients: {len(coeffs)}")
print(f"Non-trivial (|v| > 0.01): {len(nontrivial)}")
print(f"Near-zero (|v| <= 0.01): {len(nearzero)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Match each non-trivial coefficient
# ══════════════════════════════════════════════════════════════════════════════
# Use binary search for speed
lookup_vals = np.array([v for _, v in lookup])
lookup_names = [n for n, _ in lookup]

def find_best_match(absval):
    idx = np.searchsorted(lookup_vals, absval)
    best_name, best_val, best_dist = None, None, 999
    for i in [idx-1, idx, idx+1]:
        if 0 <= i < len(lookup_vals):
            d = abs(absval - lookup_vals[i])
            if d < best_dist:
                best_dist = d
                best_name = lookup_names[i]
                best_val = lookup_vals[i]
    return best_name, best_val, best_dist

# Match everything
matches = []
for c in nontrivial:
    name, aval, dist = find_best_match(c['absval'])
    rel_err = dist / c['absval'] * 100
    matches.append((c, name, aval, dist, rel_err))

# Sort by distance (best matches first)
matches.sort(key=lambda x: x[3])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Coverage bands
# ══════════════════════════════════════════════════════════════════════════════
bands = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
print("=" * 90)
print("COVERAGE BANDS — how many coefficients fall within each distance threshold")
print("=" * 90)
for thresh in bands:
    n = sum(1 for _,_,_,d,_ in matches if d <= thresh)
    pct = n / len(matches) * 100
    bar = "█" * int(pct / 2)
    print(f"  ≤ {thresh:.4f}: {n:3d}/{len(matches)} ({pct:5.1f}%) {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: All matches sorted by quality
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 110)
print("ALL NON-TRIVIAL COEFFICIENTS — sorted by match quality (best first)")
print("=" * 110)
print(f"{'term':>4} {'ax':>2} {'i':>1} {'value':>11} {'|val|':>9}  {'→':>1} {'algebraic name':>22} {'exact':>11} {'dist':>10} {'status'}")
print("-" * 110)

EXPLAINED = 0
PLAUSIBLE = 0
UNCLEAR = 0

for c, name, aval, dist, rel_err in matches:
    sign = "+" if c['value'] >= 0 else "-"
    
    if dist < 0.002:
        status = "EXPLAINED"
        EXPLAINED += 1
    elif dist < 0.008:
        status = "plausible"
        PLAUSIBLE += 1
    else:
        status = "UNCLEAR"
        UNCLEAR += 1
    
    print(f"{c['term']:4d} {c['axis']:>2} {c['index']:1d} {c['value']:+11.6f} {c['absval']:9.6f}  {sign} {name:>22} {aval:11.6f} {dist:10.6f}  {status}")

print()
print(f"EXPLAINED (dist<0.002): {EXPLAINED}")
print(f"plausible (dist<0.008): {PLAUSIBLE}")
print(f"UNCLEAR   (dist>=0.008): {UNCLEAR}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Focus on the UNCLEAR ones — what ARE they?
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 110)
print("UNCLEAR COEFFICIENTS — deep dive")
print("=" * 110)

unclear = [(c, name, aval, dist, rel_err) for c, name, aval, dist, rel_err in matches if dist >= 0.008]
unclear.sort(key=lambda x: x[0]['term'])

for c, name, aval, dist, rel_err in unclear:
    print(f"\n  term {c['term']:2d} {c['axis']} [{c['index']}] = {c['value']:+.6f}  (|v| = {c['absval']:.6f})")
    print(f"    Best match: {name} = {aval:.6f} (dist = {dist:.6f})")
    # Show top-5 closest matches
    dists_all = []
    for lname, lval in lookup:
        d = abs(c['absval'] - lval)
        dists_all.append((d, lname, lval))
    dists_all.sort()
    print(f"    Top-5 candidates:")
    for d, n, v in dists_all[:5]:
        print(f"      {n:>25} = {v:.8f}  (dist = {d:.6f})")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Per-term summary with term "type" classification
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 110)
print("PER-TERM CLASSIFICATION")
print("=" * 110)

for ti in range(1, 20):
    term_m = [(c, name, aval, dist) for c, name, aval, dist, _ in matches if c['term'] == ti]
    
    # Classify term type
    n_one = sum(1 for c,n,a,d in term_m if abs(a - 1.0) < 0.01 and d < 0.01)
    n_sqrt2 = sum(1 for c,n,a,d in term_m if abs(a - 0.707107) < 0.01 and d < 0.01)
    n_cbrt = sum(1 for c,n,a,d in term_m if '∛' in n and d < 0.01)
    n_explained = sum(1 for c,n,a,d in term_m if d < 0.002)
    n_plausible = sum(1 for c,n,a,d in term_m if 0.002 <= d < 0.008)
    n_unclear = sum(1 for c,n,a,d in term_m if d >= 0.008)
    
    if n_one >= 2 and n_sqrt2 == 0 and n_cbrt == 0:
        ttype = "UNIT"
    elif n_sqrt2 >= 2:
        ttype = "HADAMARD"
    elif n_cbrt >= 2:
        ttype = "CUBIC"
    else:
        ttype = "MIXED"
    
    sig_parts = []
    for c, name, aval, dist in sorted(term_m, key=lambda x: x[0]['axis']):
        if dist < 0.008:
            sign = "+" if c['value'] > 0 else "-"
            sig_parts.append(f"{sign}{name}")
        else:
            sig_parts.append(f"?{c['value']:+.3f}")
    
    status = f"✓{n_explained} ~{n_plausible} ?{n_unclear}"
    print(f"  term {ti:2d} [{ttype:>8}] {status:>12}  {', '.join(sig_parts)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8: Check if near-zero values have a pattern too
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 110)
print("NEAR-ZERO COEFFICIENT ANALYSIS (153 values with |v| <= 0.01)")
print("=" * 110)
nz_vals = sorted([c['absval'] for c in nearzero if c['absval'] > 0.00005])
if nz_vals:
    print(f"  Count with |v| > 0.00005: {len(nz_vals)}")
    print(f"  Min: {min(nz_vals):.6f}")
    print(f"  Max: {max(nz_vals):.6f}")
    print(f"  Mean: {np.mean(nz_vals):.6f}")
    
    # Are any of them exactly some pattern like 1/100, 1/1000?
    bins = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
    hist, _ = np.histogram(nz_vals, bins=bins)
    for i, count in enumerate(hist):
        if count > 0:
            print(f"    [{bins[i]:.4f}, {bins[i+1]:.4f}): {count}")
    
    # Check if they cluster at the EA's mutation floor
    nz_all = sorted([c['absval'] for c in nearzero])
    # How many are essentially the same value?
    from collections import Counter
    rounded = Counter(np.round(nz_all, 4))
    common = [(v, n) for v, n in rounded.items() if n >= 5 and v > 0]
    if common:
        print(f"\n  Common near-zero values:")
        for v, n in sorted(common):
            print(f"    {v:.4f}: appears {n}x")
