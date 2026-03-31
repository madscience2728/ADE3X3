"""Analyze coefficient value spectrum — are they roots/ratios of small integers?"""
import json, numpy as np
from collections import Counter

# Load both candidates
with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    baseline = json.load(f)
with open('chat_gpt.json') as f:
    chatgpt = json.load(f)

def extract_all_values(data):
    vals = []
    for t in data['terms']:
        vals.extend(t['alpha_values'])
        vals.extend(t['beta_values'])
        vals.extend(t['gamma_values'])
    return np.array(vals)

def extract_nonzero_values(data, thresh=0.01):
    vals = []
    for t in data['terms']:
        vals.extend(t['alpha_values'])
        vals.extend(t['beta_values'])
        vals.extend(t['gamma_values'])
    vals = np.array(vals)
    return vals[np.abs(vals) > thresh]

# Known "nice" values to test against
nice_vals = {
    '0': 0.0,
    '1/4': 0.25,
    '1/3': 1/3,
    '1/√8': 1/np.sqrt(8),
    '1/√6': 1/np.sqrt(6),
    '1/√5': 1/np.sqrt(5),
    '1/2': 0.5,
    '1/√3': 1/np.sqrt(3),
    '√(2/5)': np.sqrt(2/5),
    '1/√2': 1/np.sqrt(2),
    '√(3/5)': np.sqrt(3/5),
    '√(2/3)': np.sqrt(2/3),
    '3/4': 0.75,
    '√(3)/2': np.sqrt(3)/2,
    '1': 1.0,
    '√(4/3)': np.sqrt(4/3),
    '√(3/2)': np.sqrt(3/2),
    '√2': np.sqrt(2),
    '√3': np.sqrt(3),
    '2': 2.0,
    '3': 3.0,
}

# Add n/m for small integers
for n in range(1, 6):
    for m in range(1, 6):
        key = f"{n}/{m}"
        if key not in nice_vals:
            nice_vals[key] = n/m

# Add sqrt(n/m) for small integers
for n in range(1, 10):
    for m in range(1, 10):
        key = f"√({n}/{m})" if m > 1 else f"√{n}"
        if key not in nice_vals:
            nice_vals[key] = np.sqrt(n/m)

# Add cbrt(n) for small integers  
for n in range(1, 10):
    nice_vals[f"∛{n}"] = n**(1/3)

nice_list = sorted(nice_vals.items(), key=lambda x: x[1])

def find_nearest_nice(v):
    """Find the closest 'nice' value to |v|."""
    av = abs(v)
    best_name, best_val, best_dist = None, None, 999
    for name, nv in nice_list:
        d = abs(av - nv)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_val = nv
    return best_name, best_val, best_dist

print("=" * 80)
print("CHATGPT CANDIDATE — Non-trivial coefficient values")
print("=" * 80)
cg_vals = extract_nonzero_values(chatgpt)
print(f"Total non-trivial coefficients: {len(cg_vals)}\n")

# Sort by absolute value
sorted_abs = sorted(set(np.round(np.abs(cg_vals), 6)))
print(f"Unique |values| (rounded to 6dp): {len(sorted_abs)}\n")

print(f"{'|value|':>10}  {'nearest nice':>12}  {'nice val':>10}  {'error':>10}")
print("-" * 50)
for av in sorted_abs:
    name, nv, dist = find_nearest_nice(av)
    flag = " ***" if dist < 0.005 else ""
    print(f"{av:10.6f}  {name:>12}  {nv:10.6f}  {dist:10.6f}{flag}")

print("\n" + "=" * 80)
print("BASELINE (EA) CANDIDATE — Non-trivial coefficient values")
print("=" * 80)
bl_vals = extract_nonzero_values(baseline)
print(f"Total non-trivial coefficients: {len(bl_vals)}\n")

sorted_abs_bl = sorted(set(np.round(np.abs(bl_vals), 6)))
print(f"Unique |values| (rounded to 6dp): {len(sorted_abs_bl)}\n")
print(f"{'|value|':>10}  {'nearest nice':>12}  {'nice val':>10}  {'error':>10}")
print("-" * 50)
for av in sorted_abs_bl:
    name, nv, dist = find_nearest_nice(av)
    flag = " ***" if dist < 0.005 else ""
    print(f"{av:10.6f}  {name:>12}  {nv:10.6f}  {dist:10.6f}{flag}")

# Histogram of ChatGPT values
print("\n" + "=" * 80)
print("CHATGPT — Histogram of |coefficient| values")
print("=" * 80)
bins = np.arange(0, 1.2, 0.05)
hist, edges = np.histogram(np.abs(cg_vals), bins=bins)
for i, count in enumerate(hist):
    if count > 0:
        bar = "#" * count
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f}): {count:3d}  {bar}")

# Check specific recognizable values
print("\n" + "=" * 80)
print("CHATGPT — Exact value frequency (rounded to 3dp)")
print("=" * 80)
rounded = np.round(np.abs(cg_vals), 3)
counts = Counter(rounded)
for val, cnt in sorted(counts.items()):
    if cnt >= 2:
        name, nv, dist = find_nearest_nice(val)
        match = f"  ≈ {name}" if dist < 0.01 else ""
        print(f"  |{val:.3f}|: appears {cnt}x{match}")

# What about the product structure? In exact decompositions,
# each rank-1 term has α⊗β⊗γ where the norms relate to cube roots
print("\n" + "=" * 80)
print("CHATGPT — Per-term norms and products")
print("=" * 80)
for i, t in enumerate(chatgpt['terms']):
    a = np.array(t['alpha_values'])
    b = np.array(t['beta_values'])
    g = np.array(t['gamma_values'])
    na, nb, ng = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(g)
    prod = na * nb * ng
    print(f"  term {i+1:2d}: ||α||={na:.4f}  ||β||={nb:.4f}  ||γ||={ng:.4f}  product={prod:.4f}")

print("\n" + "=" * 80)
print("BASELINE — Per-term norms and products")
print("=" * 80)
for i, t in enumerate(baseline['terms']):
    a = np.array(t['alpha_values'])
    b = np.array(t['beta_values'])
    g = np.array(t['gamma_values'])
    na, nb, ng = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(g)
    prod = na * nb * ng
    print(f"  term {i+1:2d}: ||α||={na:.4f}  ||β||={nb:.4f}  ||γ||={ng:.4f}  product={prod:.4f}")
