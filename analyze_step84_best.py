#!/usr/bin/env python3
"""Analyze the step84 best decomposition to understand what makes it tick."""

import json
import numpy as np
from pathlib import Path

# Load the best individual
best_path = Path("outputs/exports/step84_best_individual.json")
with open(best_path) as f:
    best = json.load(f)

print(f"=== STEP84 BEST DECOMPOSITION ANALYSIS ===")
print(f"Fitness (max abs residual): {best['fitness']:.6f}")
print(f"Frobenius residual: {best['fro_residual']:.6f}")
print(f"Support signature: {best['support_signature']}")
print(f"Number of terms: {len(best['terms'])}")
print(f"Total variables: {best['variable_count']}")
print(f"Found at generation: {best['generation_found']}")
print()

# Reconstruct the tensor approximation
approx = np.zeros((9, 9, 9))

for i, term in enumerate(best['terms'], 1):
    alpha_idx = term['alpha_support']
    beta_idx = term['beta_support']
    gamma_idx = term['gamma_support']
    
    alpha = np.array(term['alpha_values'])
    beta = np.array(term['beta_values'])
    gamma = np.array(term['gamma_values'])
    
    # Add outer product to approximation
    for ai, av in zip(alpha_idx, alpha):
        for bi, bv in zip(beta_idx, beta):
            for gi, gv in zip(gamma_idx, gamma):
                approx[ai, bi, gi] += av * bv * gv
    
    sparsity = len(alpha_idx) * len(beta_idx) * len(gamma_idx)
    print(f"Term {i:2d}: {term['support_signature']:8s} "
          f"=> {sparsity:4d} vars, "
          f"max|α|={max(abs(alpha)):.3f}, "
          f"max|β|={max(abs(beta)):.3f}, "
          f"max|γ|={max(abs(gamma)):.3f}")

# Load the true tensor
import sys
sys.path.insert(0, str(Path(__file__).parent))
from outputs.ade3x3_attack.attack_common import TARGET_TENSOR

true_tensor = TARGET_TENSOR
residual = true_tensor - approx

print()
print("=== RESIDUAL ANALYSIS ===")
print(f"Max absolute error: {np.max(np.abs(residual)):.6f}")
print(f"Frobenius norm error: {np.linalg.norm(residual):.6f}")
print(f"Mean absolute error: {np.mean(np.abs(residual)):.6f}")
print(f"Median absolute error: {np.median(np.abs(residual)):.6f}")

# Find worst approximated entries
flat_residuals = np.abs(residual).flatten()
worst_indices = np.argsort(flat_residuals)[-10:][::-1]

print()
print("=== TOP 10 WORST ENTRIES ===")
for idx in worst_indices:
    i, j, k = np.unravel_index(idx, (9, 9, 9))
    true_val = true_tensor[i, j, k]
    approx_val = approx[i, j, k]
    err = residual[i, j, k]
    print(f"  [{i},{j},{k}]: true={true_val:6.3f}, approx={approx_val:7.3f}, "
          f"error={err:7.3f} (|err|={abs(err):.3f})")

# Analyze which entries are exactly zero in the true tensor
true_nonzero = (true_tensor != 0)
true_zero = ~true_nonzero

print()
print("=== SUPPORT ANALYSIS ===")
print(f"True tensor nonzeros: {np.sum(true_nonzero)} / 729")
print(f"Approximation puts mass on true zeros:")
print(f"  Mean |approx| on true zeros: {np.mean(np.abs(approx[true_zero])):.6f}")
print(f"  Max |approx| on true zeros: {np.max(np.abs(approx[true_zero])):.6f}")
print()
print(f"Approximation on true nonzeros:")
print(f"  Mean |error| on nonzeros: {np.mean(np.abs(residual[true_nonzero])):.6f}")
print(f"  Max |error| on nonzeros: {np.max(np.abs(residual[true_nonzero])):.6f}")

# Analyze term overlap - which terms share support?
print()
print("=== TERM OVERLAP ANALYSIS ===")
support_sets = []
for i, term in enumerate(best['terms'], 1):
    support = set()
    for ai in term['alpha_support']:
        for bi in term['beta_support']:
            for gi in term['gamma_support']:
                support.add((ai, bi, gi))
    support_sets.append(support)

# Find pairs with high overlap
overlaps = []
for i in range(len(support_sets)):
    for j in range(i+1, len(support_sets)):
        overlap = len(support_sets[i] & support_sets[j])
        union = len(support_sets[i] | support_sets[j])
        if overlap > 0:
            overlaps.append((i+1, j+1, overlap, union, overlap/union))

overlaps.sort(key=lambda x: x[4], reverse=True)
print("Top 5 overlapping term pairs (Jaccard similarity):")
for t1, t2, overlap, union, jaccard in overlaps[:5]:
    print(f"  Terms {t1:2d} & {t2:2d}: {overlap:4d} shared / {union:4d} union = {jaccard:.3f}")
