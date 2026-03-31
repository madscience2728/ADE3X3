"""Analyze the residual basin structure of step84 metaheuristic results."""
import json
import glob
import numpy as np
from collections import Counter

def matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                c = r * n + u
                a = r * n + s
                b = s * n + u
                tensor[c, a, b] = 1.0
    return tensor

T = matrix_multiplication_tensor(3)
print(f"Target tensor shape: {T.shape}")
print(f"Target Frobenius norm: {np.linalg.norm(T):.6f} = sqrt({np.linalg.norm(T)**2:.1f})")
print(f"Nonzero entries: {np.count_nonzero(T)} (all equal 1.0)")
print()

# Build live/dead mask
live_mask = (T != 0)
dead_mask = ~live_mask
print(f"Live entries: {live_mask.sum()}, Dead entries: {dead_mask.sum()}")
print()

# Now reconstruct residuals from saved best individuals
def reconstruct_residual(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    
    R = len(data["terms"])
    alpha = np.zeros((R, 9))
    beta = np.zeros((R, 9))
    gamma = np.zeros((R, 9))
    
    for i, term in enumerate(data["terms"]):
        for idx, val in zip(term["alpha_support"], term["alpha_values"]):
            alpha[i, idx] = val
        for idx, val in zip(term["beta_support"], term["beta_values"]):
            beta[i, idx] = val
        for idx, val in zip(term["gamma_support"], term["gamma_values"]):
            gamma[i, idx] = val
    
    # Reconstruct: sum_i gamma_i (x) alpha_i (x) beta_i  -> T[c,a,b]
    # Actually need to check the axis ordering
    candidate = np.zeros((9, 9, 9))
    for i in range(R):
        candidate += np.einsum('c,a,b->cab', gamma[i], alpha[i], beta[i])
    
    residual = candidate - T
    return data, residual, candidate

# Process all copies
all_json = sorted(glob.glob("outputs/exports/step84_batches/*/copy_*/step84_best_individual.json"))
print(f"Found {len(all_json)} result files")
print()

# Group by approximate fitness basin
basins = {}
for jp in all_json:
    data, residual, candidate = reconstruct_residual(jp)
    max_abs = float(np.max(np.abs(residual)))
    fro = float(np.linalg.norm(residual))
    reported_fitness = data["fitness"]
    reported_fro = data["fro_residual"]
    
    # Check if our reconstruction matches
    match_ok = abs(max_abs - reported_fitness) < 0.01
    
    # Classify into basin
    if max_abs > 0.9:
        basin = "1.0"
    elif max_abs > 0.6:
        basin = "~0.67"
    elif max_abs > 0.45:
        basin = "~0.5"
    elif max_abs > 0.2:
        basin = "~0.25"
    else:
        basin = f"<0.2 ({max_abs:.4f})"
    
    if basin not in basins:
        basins[basin] = []
    basins[basin].append({
        "path": jp,
        "max_abs": max_abs,
        "fro": fro,
        "reported_fitness": reported_fitness,
        "reported_fro": reported_fro,
        "match": match_ok,
        "residual": residual,
    })

for basin_name in sorted(basins.keys()):
    entries = basins[basin_name]
    print(f"{'='*70}")
    print(f"BASIN: {basin_name}  ({len(entries)} copies)")
    print(f"{'='*70}")
    
    # Pick the best example
    best = min(entries, key=lambda e: e["max_abs"])
    res = best["residual"]
    
    print(f"  max_abs: {best['max_abs']:.8f}  (reported: {best['reported_fitness']:.8f})")
    print(f"  fro:     {best['fro']:.8f}  (reported: {best['reported_fro']:.8f})")
    print(f"  fro²:    {best['fro']**2:.8f}")
    print(f"  match:   {best['match']}")
    print()
    
    # Analyze residual structure
    abs_res = np.abs(res)
    live_res = res[live_mask]
    dead_res = res[dead_mask]
    
    print(f"  Live residual:  max={np.max(np.abs(live_res)):.6f}, fro²={np.sum(live_res**2):.6f}, nonzero(>1e-6)={np.sum(np.abs(live_res)>1e-6)}")
    print(f"  Dead residual:  max={np.max(np.abs(dead_res)):.6f}, fro²={np.sum(dead_res**2):.6f}, nonzero(>1e-6)={np.sum(np.abs(dead_res)>1e-6)}")
    print()
    
    # Count residual values by magnitude
    thresholds = [0.001, 0.01, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5]
    for th in thresholds:
        count = np.sum(abs_res > th - 0.001)
        if count > 0:
            print(f"  |res| > {th-0.001:.3f}: {count} entries")
    
    # Histogram of live residual values  
    print(f"\n  Live residual values (27 entries):")
    live_vals = np.sort(live_res)
    for i, v in enumerate(live_vals):
        fiber_info = ""
        print(f"    [{i:2d}] {v:+.6f}")
    
    # Check: are live residuals at specific fractions?
    unique_abs = np.unique(np.round(np.abs(live_res), 3))
    print(f"\n  Unique |live residual| values (rounded to 0.001): {unique_abs}")
    
    # Check fiber structure of residual
    print(f"\n  Residual by output fiber C[r,u]:")
    for r in range(3):
        for u in range(3):
            c_idx = r * 3 + u
            fiber_residuals = []
            for s in range(3):
                a_idx = r * 3 + s
                b_idx = s * 3 + u
                fiber_residuals.append(res[c_idx, a_idx, b_idx])
            fiber_max = max(abs(v) for v in fiber_residuals)
            fiber_str = ", ".join(f"{v:+.4f}" for v in fiber_residuals)
            print(f"    C[{r},{u}] (c={c_idx}): [{fiber_str}]  max={fiber_max:.4f}")
    
    print()

# Theoretical analysis
print("=" * 70)
print("THEORETICAL ANALYSIS")
print("=" * 70)
print()
print("The 3x3 matrix multiplication tensor T has:")
print(f"  - Shape: 9 x 9 x 9 = 729 entries")
print(f"  - 27 live entries (all = 1.0)")
print(f"  - 702 dead entries (all = 0.0)")
print(f"  - ||T||_F = sqrt(27) = {np.sqrt(27):.6f}")
print()
print("Observed: ||residual||_F² ≈ 8 consistently")
print(f"  This means residual captures {8/27*100:.1f}% of target energy")
print(f"  The decomposition captures {(27-8)/27*100:.1f}% = {27-8}/27 of the target")
print()
print("Basin prediction from fiber structure:")
print("  Each C[r,u] has fiber of 3 live X atoms.")
print("  If the residual is concentrated on k complete fibers:")
for k in range(10):
    fro2 = 3 * k  # each fiber contributes 3 unit residuals
    print(f"    k={k} fibers: fro²={fro2}, max_abs=1.0")
print()
print("  But fro²=8 doesn't match any integer k (8/3 = 2.667)")
print("  So the residual is NOT organized by complete fibers.")
print()
print("  If residual has n entries each with value v:")
print("    n * v² = 8 (fro² constraint)")
print("    max |v| = basin_level")
for level in [1.0, 0.5, 0.25, 0.125]:
    n_needed = int(round(8 / level**2))
    print(f"    basin={level}: n={n_needed} entries of value ±{level}")
print()
print("  Pattern: 8 → 32 → 128 → 512 entries")
print("  These are 8·4^k = 8·(2²)^k entries of value 1/2^k")
