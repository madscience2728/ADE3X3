"""Combinatorial snapping: snap baseline coefficients to nearest algebraic values,
evaluate residual at various aggressiveness levels."""
import json, numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# Load baseline
# ══════════════════════════════════════════════════════════════════════════════
with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    baseline = json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# Build lookup table (same as extended analysis)
# ══════════════════════════════════════════════════════════════════════════════
table = {}

for x in range(0, 21):
    for y in range(1, 21):
        v = x / y
        if v > 3.0: continue
        table[f"{x}/{y}"] = v

for x in range(1, 21):
    for y in range(1, 21):
        v = np.sqrt(x / y)
        if v > 3.0: continue
        table[f"sqrt({x}/{y})"] = v

for x in range(1, 21):
    for y in range(1, 21):
        v = (x / y) ** (1/3)
        if v > 3.0: continue
        table[f"cbrt({x}/{y})"] = v

for x in range(1, 21):
    for y in range(1, 21):
        v = (x / y) ** 0.25
        if v > 3.0: continue
        table[f"4rt({x}/{y})"] = v

for x in range(1, 11):
    for y in range(1, 11):
        v = (x / y) ** (1/6)
        if v > 3.0: continue
        table[f"6rt({x}/{y})"] = v

for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 6):
            for d in range(1, 6):
                v = np.sqrt(a/b) * ((c/d)**(1/3))
                if 0.01 < v < 2.5:
                    table[f"sqrt({a}/{b})*cbrt({c}/{d})"] = v

for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 4):
            for d in range(1, 4):
                v = np.sqrt(a/b) * (c/d)
                if 0.01 < v < 2.5:
                    table[f"{c}/{d}*sqrt({a}/{b})"] = v

for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 4):
            for d in range(1, 4):
                v = ((a/b)**(1/3)) * (c/d)
                if 0.01 < v < 2.5:
                    table[f"{c}/{d}*cbrt({a}/{b})"] = v

for a in range(-3, 4):
    for b in range(-3, 4):
        for c in [2, 3, 6]:
            v = abs((2**a) * (3**b)) ** (1/c)
            if 0.01 < v < 2.5:
                table[f"(2^{a}*3^{b})^(1/{c})"] = v

table["0"] = 0.0

# Deduplicate
deduped = {}
for name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
    rounded = round(val, 8)
    if rounded not in deduped:
        deduped[rounded] = (name, val)
    elif len(name) < len(deduped[rounded][0]):
        deduped[rounded] = (name, val)

lookup = sorted(deduped.values(), key=lambda x: x[1])
lookup_vals = np.array([v for _, v in lookup])
lookup_names = [n for n, _ in lookup]

print(f"Lookup table: {len(lookup)} entries\n")

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

def reconstruct(data):
    candidate = np.zeros((9,9,9))
    for term in data['terms']:
        a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
        for i, v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
        for i, v in zip(term['beta_support'], term['beta_values']): b[i] = v
        for i, v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
        candidate += np.einsum('c,a,b->cab', g, a, b)
    return candidate

def evaluate(data):
    T_hat = reconstruct(data)
    R = T_hat - T
    return np.max(np.abs(R)), np.linalg.norm(R)

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

# ══════════════════════════════════════════════════════════════════════════════
# Baseline evaluation
# ══════════════════════════════════════════════════════════════════════════════
mx0, fro0 = evaluate(baseline)
print(f"BASELINE:  max_abs = {mx0:.8f}   fro = {fro0:.8f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Snap experiments at various thresholds
# ══════════════════════════════════════════════════════════════════════════════
import copy

def snap_candidate(data, snap_thresh, zero_thresh):
    """Snap coefficients: 
    - |v| < zero_thresh → force to 0
    - dist_to_nearest_algebraic < snap_thresh → snap to algebraic
    - otherwise → leave as-is
    Returns (snapped_data, stats_dict)
    """
    snapped = copy.deepcopy(data)
    stats = {'zeroed': 0, 'snapped': 0, 'kept': 0, 'total': 0}
    
    for term in snapped['terms']:
        for vkey in ['alpha_values', 'beta_values', 'gamma_values']:
            new_vals = []
            for v in term[vkey]:
                stats['total'] += 1
                if abs(v) < zero_thresh:
                    new_vals.append(0.0)
                    stats['zeroed'] += 1
                else:
                    sign = 1.0 if v >= 0 else -1.0
                    name, aval, dist = find_best_match(abs(v))
                    if dist < snap_thresh:
                        new_vals.append(sign * aval)
                        stats['snapped'] += 1
                    else:
                        new_vals.append(v)
                        stats['kept'] += 1
            term[vkey] = new_vals
    return snapped, stats

# ── Experiment matrix ──
print("=" * 100)
print(f"{'Experiment':<40} {'zero_th':>8} {'snap_th':>8} {'zeroed':>7} {'snapped':>8} "
      f"{'kept':>5} {'max_abs':>12} {'fro':>12} {'Δmax':>10}")
print("-" * 100)

experiments = [
    # (name, zero_thresh, snap_thresh)
    ("Baseline (no change)", 0, 0),
    ("Zero leak only (<0.01)", 0.01, 0),
    ("Zero leak only (<0.04)", 0.04, 0),
    ("Zero leak + tight snap (0.0005)", 0.04, 0.0005),
    ("Zero leak + snap (0.001)", 0.04, 0.001),
    ("Zero leak + snap (0.002)", 0.04, 0.002),
    ("Zero leak + snap (0.005)", 0.04, 0.005),
    ("Zero leak + snap (0.01)", 0.04, 0.01),
    ("Zero leak + snap (0.02)", 0.04, 0.02),
    ("Zero leak + snap (0.05)", 0.04, 0.05),
    ("Aggressive: zero<0.04, snap<0.1", 0.04, 0.1),
    ("Nuclear: snap everything", 0.04, 999),
]

results = []
for name, zero_th, snap_th in experiments:
    if zero_th == 0 and snap_th == 0:
        # baseline
        mx, fro = mx0, fro0
        stats = {'zeroed': 0, 'snapped': 0, 'kept': 275, 'total': 275}
        snapped = baseline
    else:
        snapped, stats = snap_candidate(baseline, snap_th, zero_th)
        mx, fro = evaluate(snapped)
    
    delta = mx - mx0
    flag = " <<<" if mx < mx0 else ""
    print(f"{name:<40} {zero_th:8.4f} {snap_th:8.4f} {stats['zeroed']:7d} {stats['snapped']:8d} "
          f"{stats['kept']:5d} {mx:12.8f} {fro:12.8f} {delta:+10.6f}{flag}")
    results.append((name, snapped, mx, fro, stats))

# ══════════════════════════════════════════════════════════════════════════════
# Best result deep-dive
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("BEST RESULT ANALYSIS")
print("=" * 100)

best = min(results[1:], key=lambda x: x[2])  # skip baseline
name, snapped, mx, fro, stats = best
print(f"\nBest: {name}")
print(f"  max_abs = {mx:.8f}  (baseline: {mx0:.8f}, delta = {mx - mx0:+.8f})")
print(f"  fro     = {fro:.8f}  (baseline: {fro0:.8f}, delta = {fro - fro0:+.8f})")
print(f"  zeroed={stats['zeroed']}  snapped={stats['snapped']}  kept={stats['kept']}")

# Show what changed
print(f"\n  Coefficient changes:")
for ti, (bt, st) in enumerate(zip(baseline['terms'], snapped['terms'])):
    for axis, vkey in [('α','alpha_values'), ('β','beta_values'), ('γ','gamma_values')]:
        for pi, (bv, sv) in enumerate(zip(bt[vkey], st[vkey])):
            if bv != sv:
                delta_v = sv - bv
                print(f"    term {ti+1:2d} {axis}[{pi}]: {bv:+.6f} → {sv:+.6f}  (Δ={delta_v:+.6f})")

# ══════════════════════════════════════════════════════════════════════════════
# Additional: Greedy per-coefficient snapping
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("GREEDY PER-COEFFICIENT SNAPPING")
print("=" * 100)
print("Start from baseline, snap one coefficient at a time, keep if max_abs improves.")

greedy = copy.deepcopy(baseline)
# First zero out all leak values
for term in greedy['terms']:
    for vkey in ['alpha_values', 'beta_values', 'gamma_values']:
        term[vkey] = [0.0 if abs(v) < 0.04 else v for v in term[vkey]]

greedy_mx, greedy_fro = evaluate(greedy)
print(f"\nAfter zeroing leaks: max_abs = {greedy_mx:.8f}")

# Build list of (term_idx, axis, pos, current_val, snap_name, snap_val, dist)
snap_candidates = []
for ti, term in enumerate(greedy['terms']):
    for vkey, skey in [('alpha_values','alpha_support'), ('beta_values','beta_support'), 
                       ('gamma_values','gamma_support')]:
        for pi, v in enumerate(term[vkey]):
            if abs(v) > 0.04:  # non-zero
                sign = 1.0 if v >= 0 else -1.0
                name, aval, dist = find_best_match(abs(v))
                if dist > 0.0001 and dist < 0.1:  # only if not already snapped and close enough
                    snap_candidates.append((ti, vkey, pi, v, name, sign * aval, dist))

# Sort by distance (snap closest first)
snap_candidates.sort(key=lambda x: x[6])
print(f"Candidates to snap: {len(snap_candidates)}")

accepted = 0
rejected = 0
for ti, vkey, pi, old_v, name, snap_v, dist in snap_candidates:
    # Try snapping
    prev = greedy['terms'][ti][vkey][pi]
    greedy['terms'][ti][vkey][pi] = snap_v
    new_mx, new_fro = evaluate(greedy)
    
    if new_mx <= greedy_mx + 1e-8:  # accept if no worse (with tiny tolerance)
        accepted += 1
        greedy_mx = new_mx
        improved = " IMPROVED!" if new_mx < greedy_mx - 1e-6 else ""
        print(f"  ACCEPT t{ti+1:2d} {vkey[0]}[{pi}]: {old_v:+.8f} → {snap_v:+.8f} ({name})  "
              f"max_abs={new_mx:.8f}{improved}")
    else:
        # revert
        greedy['terms'][ti][vkey][pi] = prev
        rejected += 1

print(f"\nGreedy result: accepted={accepted}, rejected={rejected}")
print(f"Final: max_abs = {greedy_mx:.8f}  (baseline: {mx0:.8f})")

# Save the best greedy-snapped candidate
with open('snapped_candidate.json', 'w') as f:
    json.dump(greedy, f, indent=2)
print(f"\nSaved to snapped_candidate.json")

# Final: evaluate the fully snapped version
full_mx, full_fro = evaluate(greedy)
print(f"\n{'='*60}")
print(f"FINAL SNAPPED CANDIDATE")
print(f"  max_abs = {full_mx:.8f}")
print(f"  fro     = {full_fro:.8f}")
print(f"  vs baseline max_abs delta = {full_mx - mx0:+.8f}")
