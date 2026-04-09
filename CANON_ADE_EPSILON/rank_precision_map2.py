import numpy as np
import json
from pathlib import Path

# Build T_matmul as flat 729-vector
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_vec = T.ravel()  # 729

# Standard 27 rank-1 tensors
triples = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]

def standard_tensor(r,s,u):
    a = np.zeros(9); a[r*3+s] = 1.0
    b = np.zeros(9); b[s*3+u] = 1.0
    g = np.zeros(9); g[r*3+u] = 1.0
    return np.kron(np.kron(a, b), g)  # 729

M_full = np.stack([standard_tensor(r,s,u) for r,s,u in triples], axis=1)  # 729 x 27

# Verify orthonormality
gram = M_full.T @ M_full
print("Gram matrix diag (should be 1s):", np.diag(gram)[:5], "...")
print("Gram matrix off-diag max:", np.max(np.abs(gram - np.diag(np.diag(gram)))))
print()

print(f"{'R':>3}  {'eps_std':>10}  {'sqrt(27-R)':>12}  {'b*_std':>8}  {'b*_bound':>10}")
print("-"*55)

for R in range(1, 28):
    analytic = float(np.sqrt(27 - R))
    b_analytic = float(np.log2(3*R / analytic)) if analytic > 1e-12 else 0.0
    if R < 27:
        # Standard basis residual: keep first R terms (any R suffices by orthogonality)
        kept = list(range(R))
        M_kept = M_full[:, kept]  # 729 x R
        # Optimal coefficients (orthonormal => c_t = M_t^T @ T_vec = 1 for all)
        c = M_kept.T @ T_vec
        residual = float(np.linalg.norm(T_vec - M_kept @ c))
        b_std = float(np.log2(3*R / residual)) if residual > 1e-12 else 0.0
    else:
        residual = 0.0
        b_std = 0.0
    print(f"{R:3d}  {residual:10.6f}  {analytic:12.6f}  {b_std:8.3f}  {b_analytic:10.3f}")

# Now load actual best-known residuals per R from candidate files
print()
print("=== Comparison: standard basis vs optimizer-found candidates ===")
print(f"{'R':>3}  {'eps_std=sqrt(27-R)':>20}  {'best_optimizer':>16}  {'improvement':>12}  {'bits_saved':>10}")
print("-"*68)

# Known best residuals from candidates found
best_known = {
    19: 0.074,   # slp_best_at_0.074.json
    22: None,    # 10_border23_0000 — load from file
}

# Load phase4 best
phase4_dir = Path("outputs/ade3x3_attack/phase4_gradient_search/best_decompositions")
r22_residuals = []
if phase4_dir.exists():
    for f in phase4_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if 'residual' in d:
                r22_residuals.append(d['residual'])
            elif 'fitness' in d:
                r22_residuals.append(d['fitness'])
        except:
            pass
if r22_residuals:
    best_known[22] = min(r22_residuals)

for R in sorted(best_known):
    eps_std = float(np.sqrt(27 - R))
    best_opt = best_known[R]
    if best_opt is None:
        print(f"{R:3d}  {eps_std:20.6f}  {'unknown':>16}")
        continue
    improvement = eps_std / best_opt
    b_std   = float(np.log2(3*R / eps_std))   if eps_std  > 1e-12 else 0.0
    b_opt   = float(np.log2(3*R / best_opt))  if best_opt > 1e-12 else 0.0
    saved   = b_opt - b_std
    print(f"{R:3d}  {eps_std:20.6f}  {best_opt:16.6f}  {improvement:12.2f}x  {saved:10.3f} bits")
    print(f"      b*(standard basis) = {b_std:.3f},  b*(optimizer) = {b_opt:.3f}")
