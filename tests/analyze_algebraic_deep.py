"""
Deep algebraic analysis: for every coefficient, find the best rational/algebraic explanation.
Also check multiple copies to see which algebraic assignments are stable.
"""
import json, numpy as np, glob, os
from collections import defaultdict
from itertools import product as iprod

# ── Build extended lookup table ──
# Level 1: x/y for x,y in 1..10
# Level 2: sqrt(x/y) for x,y in 1..10
# Level 3: cbrt(x/y) for x,y in 1..10
# Level 4: sqrt(x)/y and x/sqrt(y) — mixed forms
# Level 5: common products like sqrt(2)*x/y, sqrt(3)*x/y

table = {}  # name -> value (positive only, we handle signs separately)

# Basic rationals
for x in range(0, 11):
    for y in range(1, 11):
        v = x / y
        name = f"{x}/{y}" if y > 1 else str(x)
        if v <= 3.0:  # reasonable range
            table[name] = v

# sqrt(x/y)
for x in range(1, 11):
    for y in range(1, 11):
        v = np.sqrt(x / y)
        name = f"√({x}/{y})" if y > 1 else f"√{x}"
        if v <= 3.0:
            table[name] = v

# cbrt(x/y)
for x in range(1, 11):
    for y in range(1, 11):
        v = (x / y) ** (1/3)
        name = f"∛({x}/{y})" if y > 1 else f"∛{x}"
        if v <= 3.0:
            table[name] = v

# 4th roots — why not
for x in range(1, 11):
    for y in range(1, 11):
        v = (x / y) ** 0.25
        name = f"⁴√({x}/{y})" if y > 1 else f"⁴√{x}"
        if v <= 3.0:
            table[name] = v

# Products: √2 * (x/y), √3 * (x/y) for small x/y
for base_name, base_val in [("√2", np.sqrt(2)), ("√3", np.sqrt(3)), ("∛2", 2**(1/3)), ("∛3", 3**(1/3))]:
    for x in range(1, 7):
        for y in range(1, 7):
            v = base_val * x / y
            name = f"{base_name}·{x}/{y}" if y > 1 else f"{base_name}·{x}" if x > 1 else base_name
            if v <= 3.0:
                table[name] = v

# Also: x / (y * √2), etc.
for base_name, base_val in [("√2", np.sqrt(2)), ("√3", np.sqrt(3))]:
    for x in range(1, 7):
        for y in range(1, 7):
            v = x / (y * base_val)
            name = f"{x}/({y}·{base_name})" if y > 1 else f"{x}/{base_name}"
            if 0 < v <= 3.0:
                table[name] = v

# Deduplicate by value (keep shortest name)
deduped = {}
for name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
    key = round(val, 12)
    if key not in deduped:
        deduped[key] = (name, val)
    else:
        existing = deduped[key][0]
        if len(name) < len(existing):
            deduped[key] = (name, val)

lookup = sorted(deduped.values(), key=lambda x: x[1])
print(f"Extended lookup table: {len(lookup)} unique values")

def find_best(absval, max_dist=0.05):
    """Find best algebraic match."""
    best_name, best_val, best_dist = "???", absval, 999
    for name, lv in lookup:
        d = abs(absval - lv)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_val = lv
    if best_dist > max_dist:
        return "???", absval, best_dist
    return best_name, best_val, best_dist

# ── Load ALL converged copies ──
files = sorted(glob.glob('outputs/exports/step84_batches/run_20260331_111612_906/copy_*/step84_best_individual.json'))

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

# Reference: copy_008
with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    ref = json.load(f)

# ── Phase 1: Detailed per-coefficient analysis of copy_008 ──
print("\n" + "=" * 120)
print("PHASE 1: COMPLETE COEFFICIENT MAP (copy_008)")
print("=" * 120)

# Categorize confidence levels
confident = []    # dist < 0.002
likely = []       # dist < 0.005
plausible = []    # dist < 0.01
unclear = []      # dist < 0.02
mystery = []      # dist >= 0.02

for ti, t in enumerate(ref['terms']):
    for axis, skey, vkey in [('α','alpha_support','alpha_values'),
                              ('β','beta_support','beta_values'),
                              ('γ','gamma_support','gamma_values')]:
        for pos, (idx, val) in enumerate(zip(t[skey], t[vkey])):
            absval = abs(val)
            if absval < 0.0005:  # effectively zero
                continue
            
            sign = "+" if val > 0 else "-"
            name, aval, dist = find_best(absval)
            
            entry = {
                'term': ti+1, 'axis': axis, 'index': idx,
                'value': val, 'absval': absval, 'sign': sign,
                'match_name': name, 'match_val': aval, 'dist': dist,
                'rel_err': dist/absval*100 if absval > 0 else 0
            }
            
            if absval < 0.01:
                # Near-zero: is it a recognizable small fraction?
                name2, aval2, dist2 = find_best(absval)
                entry['category'] = 'near-zero'
                confident.append(entry) if dist2 < 0.002 else mystery.append(entry)
            elif dist < 0.002:
                entry['category'] = 'confident'
                confident.append(entry)
            elif dist < 0.005:
                entry['category'] = 'likely'
                likely.append(entry)
            elif dist < 0.01:
                entry['category'] = 'plausible'
                plausible.append(entry)
            elif dist < 0.02:
                entry['category'] = 'unclear'
                unclear.append(entry)
            else:
                entry['category'] = 'mystery'
                mystery.append(entry)

total = len(confident) + len(likely) + len(plausible) + len(unclear) + len(mystery)
print(f"\nTotal non-zero coefficients: {total}")
print(f"  CONFIDENT (dist < 0.002): {len(confident):3d}  ({100*len(confident)/total:.0f}%)")
print(f"  LIKELY    (dist < 0.005): {len(likely):3d}  ({100*len(likely)/total:.0f}%)")
print(f"  PLAUSIBLE (dist < 0.01):  {len(plausible):3d}  ({100*len(plausible)/total:.0f}%)")
print(f"  UNCLEAR   (dist < 0.02):  {len(unclear):3d}  ({100*len(unclear)/total:.0f}%)")
print(f"  MYSTERY   (dist >= 0.02): {len(mystery):3d}  ({100*len(mystery)/total:.0f}%)")

for cat_name, cat_list in [("CONFIDENT", confident), ("LIKELY", likely), 
                            ("PLAUSIBLE", plausible), ("UNCLEAR", unclear), ("MYSTERY", mystery)]:
    if not cat_list:
        continue
    print(f"\n{'─'*120}")
    print(f"  {cat_name}")
    print(f"{'─'*120}")
    print(f"  {'term':>4} {'axis':>2} {'[i]':>3} {'value':>12} {'→':>2} {'match':>18} {'= ':>2}{'algebraic':>12} {'dist':>10} {'rel%':>6}")
    for e in sorted(cat_list, key=lambda x: (x['term'], x['axis'], x['index'])):
        print(f"  {e['term']:4d} {e['axis']:>2} [{e['index']}] {e['value']:+12.6f}  → {e['sign']}{e['match_name']:>17} = {e['match_val']:12.6f} {e['dist']:10.6f} {e['rel_err']:5.2f}%")

# ── Phase 2: Per-term algebraic formula ──
print("\n" + "=" * 120)
print("PHASE 2: PER-TERM ALGEBRAIC FORMULA (confident + likely only)")
print("=" * 120)

all_entries = confident + likely + plausible + unclear + mystery
for ti in range(1, 20):
    term_entries = sorted([e for e in all_entries if e['term'] == ti], 
                          key=lambda x: ('αβγ'.index(x['axis']), x['index']))
    
    print(f"\n  Term {ti:2d}:")
    for axis in ['α', 'β', 'γ']:
        axis_entries = [e for e in term_entries if e['axis'] == axis]
        if not axis_entries:
            continue
        parts = []
        has_mystery = False
        for e in axis_entries:
            if e['category'] in ('confident', 'likely'):
                tag = ""
            elif e['category'] == 'plausible':
                tag = "?"
            elif e['category'] == 'unclear':
                tag = "??"
            else:
                tag = "!!!"
                has_mystery = True
            
            if e['absval'] < 0.01:
                parts.append(f"~0")
            else:
                parts.append(f"{e['sign']}{e['match_name']}{tag}")
        
        vec = ", ".join(parts)
        print(f"    {axis} = [{vec}]")

# ── Phase 3: Cross-copy stability ──
print("\n" + "=" * 120)
print("PHASE 3: CROSS-COPY STABILITY (do the same algebraic values appear in all copies?)")
print("=" * 120)

# For the "clean" terms (7-10, 12-16, 13-14, 18-19), check if the same
# algebraic assignment works across all converged copies
clean_terms = [7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19]

# First, establish the reference assignment from copy_008
ref_assignments = {}  # (term, axis, position) -> algebraic_name
for e in all_entries:
    if e['term'] in clean_terms and e['category'] in ('confident', 'likely'):
        key = (e['term'], e['axis'], e['index'])
        ref_assignments[key] = (e['match_name'], e['sign'])

# Now check each copy
copy_results = {}
for fp in files:
    copy_name = os.path.basename(os.path.dirname(fp))
    with open(fp) as f:
        data = json.load(f)
    
    # Check fitness first
    candidate = np.zeros((9,9,9))
    for term in data['terms']:
        a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
        for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
        for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
        for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
        candidate += np.einsum('c,a,b->cab', g, a, b)
    R = candidate - T
    fitness = np.max(np.abs(R))
    if fitness > 0.55:
        continue
    
    # For each reference assignment, check if this copy has the same value
    dists = []
    for (ti_key, axis_key, idx_key), (ref_name, ref_sign) in ref_assignments.items():
        ti_idx = ti_key - 1
        axis_map = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma'}
        axis_full = axis_map[axis_key]
        
        term = data['terms'][ti_idx]
        support = term[f'{axis_full}_support']
        values = term[f'{axis_full}_values']
        
        # Find this index in the support
        val = None
        for si, sv in zip(support, values):
            if si == idx_key:
                val = sv
                break
        
        if val is not None:
            # What does this copy have?
            name, aval, dist = find_best(abs(val))
            sign = "+" if val > 0 else "-"
            same_name = (name == ref_name)
            same_sign = (sign == ref_sign)
            dists.append(dist)
    
    if dists:
        copy_results[copy_name] = {
            'fitness': fitness,
            'mean_dist': np.mean(dists),
            'max_dist': np.max(dists),
            'within_002': sum(1 for d in dists if d < 0.002),
            'total': len(dists)
        }

print(f"\n  {'copy':>10} {'fitness':>8} {'mean_d':>8} {'max_d':>8} {'<0.002':>8} / {'total':>5}")
for name, r in sorted(copy_results.items()):
    print(f"  {name:>10} {r['fitness']:8.4f} {r['mean_dist']:8.5f} {r['max_dist']:8.5f} "
          f"{r['within_002']:8d} / {r['total']:5d}")

# ── Phase 4: The mystery coefficients — what ARE they? ──
print("\n" + "=" * 120)
print("PHASE 4: THE MYSTERY/UNCLEAR COEFFICIENTS — DEEP DIVE")
print("=" * 120)

# For unclear and mystery entries, try more exotic combinations
# Try: a*sqrt(b/c) where a is a small integer/fraction
exotic_table = {}
for a_num in range(1, 6):
    for a_den in range(1, 6):
        a = a_num / a_den
        for b in range(1, 11):
            for c in range(1, 11):
                v = a * np.sqrt(b / c)
                name = f"({a_num}/{a_den})·√({b}/{c})" if a_den > 1 else f"{a_num}·√({b}/{c})" if a_num > 1 else f"√({b}/{c})"
                if 0.01 < v < 2.0:
                    exotic_table[name] = v
                
                # a * cbrt(b/c)
                v2 = a * (b/c)**(1/3)
                name2 = f"({a_num}/{a_den})·∛({b}/{c})" if a_den > 1 else f"{a_num}·∛({b}/{c})" if a_num > 1 else f"∛({b}/{c})"
                if 0.01 < v2 < 2.0:
                    exotic_table[name2] = v2

# Dedupe exotic
exotic_deduped = {}
for name, val in sorted(exotic_table.items(), key=lambda x: (len(x[0]), x[0])):
    key = round(val, 10)
    if key not in exotic_deduped:
        exotic_deduped[key] = (name, val)
    elif len(name) < len(exotic_deduped[key][0]):
        exotic_deduped[key] = (name, val)

exotic_lookup = sorted(exotic_deduped.values(), key=lambda x: x[1])
print(f"  Exotic table: {len(exotic_lookup)} values")

def find_exotic(absval):
    best_name, best_val, best_dist = "???", absval, 999
    for name, lv in exotic_lookup:
        d = abs(absval - lv)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_val = lv
    return best_name, best_val, best_dist

problematic = [e for e in all_entries if e['category'] in ('unclear', 'mystery', 'plausible')]
if problematic:
    print(f"\n  {'term':>4} {'axis':>2} {'[i]':>3} {'value':>12} {'basic match':>18} {'b_dist':>8} {'exotic match':>24} {'e_dist':>8}")
    for e in sorted(problematic, key=lambda x: -x['dist']):
        ename, eval_, edist = find_exotic(e['absval'])
        improved = "  ✓" if edist < e['dist'] * 0.5 else ""
        print(f"  {e['term']:4d} {e['axis']:>2} [{e['index']}] {e['value']:+12.6f} "
              f"{e['sign']}{e['match_name']:>17} {e['dist']:8.5f} "
              f"{e['sign']}{ename:>23} {edist:8.5f}{improved}")

# ── Phase 5: If we snap confident+likely values, what's the residual? ──
print("\n" + "=" * 120)
print("PHASE 5: HYPOTHETICAL SNAP — what if we round confident+likely to exact algebraic values?")
print("=" * 120)

import copy as copymod
snapped = copymod.deepcopy(ref)
snap_count = 0
snap_log = []

for ti, t in enumerate(snapped['terms']):
    for axis_full, skey, vkey in [('alpha','alpha_support','alpha_values'),
                                   ('beta','beta_support','beta_values'),
                                   ('gamma','gamma_support','gamma_values')]:
        axis_short = {'alpha':'α','beta':'β','gamma':'γ'}[axis_full]
        for pos in range(len(t[vkey])):
            val = t[vkey][pos]
            absval = abs(val)
            sign = 1 if val >= 0 else -1
            
            if absval < 0.0005:
                t[vkey][pos] = 0.0
                continue
            
            name, aval, dist = find_best(absval)
            
            if dist < 0.005:  # snap confident + likely
                old = t[vkey][pos]
                t[vkey][pos] = sign * aval
                snap_count += 1
                snap_log.append(f"  t{ti+1}.{axis_short}[{t[skey][pos]}]: {old:+.6f} → {t[vkey][pos]:+.6f} ({name}, Δ={dist:.6f})")

print(f"  Snapped {snap_count} coefficients")

# Evaluate snapped version
candidate_snap = np.zeros((9,9,9))
for term in snapped['terms']:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate_snap += np.einsum('c,a,b->cab', g, a, b)

R_snap = candidate_snap - T
print(f"  Snapped max_abs: {np.max(np.abs(R_snap)):.6f}  (was 0.499981)")
print(f"  Snapped fro:     {np.linalg.norm(R_snap):.6f}  (was 2.829120)")

# Also try snapping only the very confident ones
snapped2 = copymod.deepcopy(ref)
snap_count2 = 0
for ti, t in enumerate(snapped2['terms']):
    for axis_full, skey, vkey in [('alpha','alpha_support','alpha_values'),
                                   ('beta','beta_support','beta_values'),
                                   ('gamma','gamma_support','gamma_values')]:
        for pos in range(len(t[vkey])):
            val = t[vkey][pos]
            absval = abs(val)
            sign = 1 if val >= 0 else -1
            if absval < 0.0005:
                t[vkey][pos] = 0.0
                continue
            name, aval, dist = find_best(absval)
            if dist < 0.002:  # only most confident
                t[vkey][pos] = sign * aval
                snap_count2 += 1

candidate_snap2 = np.zeros((9,9,9))
for term in snapped2['terms']:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate_snap2 += np.einsum('c,a,b->cab', g, a, b)

R_snap2 = candidate_snap2 - T
print(f"\n  Conservative snap ({snap_count2} coefficients, dist < 0.002):")
print(f"  max_abs: {np.max(np.abs(R_snap2)):.6f}")
print(f"  fro:     {np.linalg.norm(R_snap2):.6f}")

# Try even setting the near-zeros to exactly 0
snapped3 = copymod.deepcopy(snapped2)
for ti, t in enumerate(snapped3['terms']):
    for vkey in ['alpha_values', 'beta_values', 'gamma_values']:
        for pos in range(len(t[vkey])):
            if abs(t[vkey][pos]) < 0.01:
                t[vkey][pos] = 0.0

candidate_snap3 = np.zeros((9,9,9))
for term in snapped3['terms']:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate_snap3 += np.einsum('c,a,b->cab', g, a, b)

R_snap3 = candidate_snap3 - T
print(f"\n  Conservative snap + zero out tiny (< 0.01):")
print(f"  max_abs: {np.max(np.abs(R_snap3)):.6f}")
print(f"  fro:     {np.linalg.norm(R_snap3):.6f}")
