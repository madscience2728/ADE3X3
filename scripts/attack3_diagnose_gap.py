#!/usr/bin/env python3
"""Diagnose WHY rank([H|Δ]) = rank(H) + 1 for all rank-19 candidates.

Key question: H has 18 columns but lives in ker(Γ) which is 10-dim.
Generically 18 vectors would span all 10 dims. But rank(H in ker(Γ)) = 9, not 10.
Why? What structural constraint kills one dimension?

Hypothesis: H = [Eta1 | Eta2] has a built-in linear dependency because
  Eta1[k,c] = a[r,0]*b[0,u] - a[r,1]*b[1,u]
  Eta2[k,c] = a[r,1]*b[1,u] - a[r,2]*b[2,u]
  => Eta1 + Eta2 = a[r,0]*b[0,u] - a[r,2]*b[2,u]  (another "difference" mode)
  
But these are all (R,9) matrices, so H is (R,18). We need to understand
what subspace of R^R the columns of H span.
"""

import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_optimizer.tensor import fiber_mode_decomposition


def load_candidate(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["alpha"]), np.array(d["beta"]), np.array(d["gamma"])


def diagnose(name, alpha, beta, gamma):
    R = alpha.shape[0]
    fm = fiber_mode_decomposition(alpha, beta, gamma)
    
    Sigma = fm["Sigma"]  # (R, 9)
    Eta1 = fm["Eta1"]    # (R, 9)  
    Eta2 = fm["Eta2"]    # (R, 9)
    Delta = fm["Delta"]  # (R, 54)
    Gam = fm["Gamma"]    # (9, R)
    
    H = np.hstack([Eta1, Eta2])  # (R, 18)
    
    # Project everything onto ker(Gamma)
    # ker(Gamma) = null space of Gam (9 x R), so find V s.t. Gam @ V = 0
    U, s, Vt = np.linalg.svd(Gam, full_matrices=True)
    # ker(Gamma) basis = last R-9 rows of Vt, transposed → (R, R-9)
    ker_basis = Vt[9:].T  # (R, R-9) = (R, 10) for R=19
    
    # Project H columns onto ker(Gamma)
    H_proj = ker_basis.T @ H  # (10, 18) — H expressed in ker(Gamma) coords
    Delta_proj = ker_basis.T @ Delta  # (10, 54)
    Sigma_proj = ker_basis.T @ Sigma  # (10, 9) — should be ~0 if Sigma ⊂ row(Gamma)
    
    print(f"\n{'='*60}")
    print(f"  {name} (R={R})")
    print(f"{'='*60}")
    
    print(f"  ||Sigma projected onto ker(Γ)||  = {np.linalg.norm(Sigma_proj):.6e}")
    print(f"  rank(H_proj)     = {np.linalg.matrix_rank(H_proj, tol=1e-8)}")
    print(f"  rank(Delta_proj) = {np.linalg.matrix_rank(Delta_proj, tol=1e-8)}")
    print(f"  rank([H|Δ]_proj) = {np.linalg.matrix_rank(np.hstack([H_proj, Delta_proj]), tol=1e-8)}")
    print(f"  ker(Γ) dim       = {ker_basis.shape[1]}")
    
    # SVD of H_proj to see the gap
    sv_H = np.linalg.svd(H_proj, compute_uv=False)
    print(f"  Singular values of H_proj (ker(Γ) coords):")
    for i, s in enumerate(sv_H):
        print(f"    σ_{i} = {s:.6e}")
    
    # What direction in ker(Gamma) is NOT covered by H?
    # It's the left singular vector of H_proj corresponding to σ ≈ 0
    U_h, s_h, Vt_h = np.linalg.svd(H_proj, full_matrices=True)
    missing_dir = U_h[:, -1]  # last left singular vector (smallest σ)
    
    # Project Delta onto this missing direction
    delta_in_missing = missing_dir @ Delta_proj  # (54,)
    print(f"  ||Delta along missing direction|| = {np.linalg.norm(delta_in_missing):.6e}")
    print(f"  ||Delta||                         = {np.linalg.norm(Delta_proj):.6e}")
    print(f"  Fraction in missing dir           = {np.linalg.norm(delta_in_missing)/max(np.linalg.norm(Delta_proj),1e-30):.6e}")
    
    # Now check: is Sigma in the image of Gamma^T?
    # Sigma should satisfy Gamma @ Sigma = 3I, so Sigma rows are NOT in ker(Gamma)
    # Sigma columns are in R^R. Gamma @ Sigma = 3I means Sigma^T @ Gamma^T = 3I
    # i.e. the 9 Sigma columns map to independent directions under Gamma
    
    # Key structural analysis: 
    # For each term k, define the "products" p_k[r,s,u] = alpha_k[r,s] * beta_k[s,u]
    # Then:
    #   Sigma_k[r,u] = p[r,0,u] + p[r,1,u] + p[r,2,u]
    #   Eta1_k[r,u]  = p[r,0,u] - p[r,1,u]
    #   Eta2_k[r,u]  = p[r,1,u] - p[r,2,u]
    # So: p[r,0,u] = (2*Sigma + 2*Eta1 + Eta2) / 3  ... actually let's solve:
    #   Sigma = p0+p1+p2, Eta1=p0-p1, Eta2=p1-p2
    #   => p0 = (Sigma + 2*Eta1 + Eta2)/3
    #      p1 = (Sigma - Eta1 + Eta2)/3  ... hmm let me recheck
    #   p0 = Sigma/3 + (2*Eta1+Eta2)/3
    #   Just trust the algebra. The point is: the 27 "live-fiber products" p[r,s,u] 
    #   span the same space as [Sigma|Eta1|Eta2], which is (R, 27).
    
    # The DEAD products: Delta[k, j] = alpha_k[r,s] * beta_k[t,u] for s≠t
    # These are products of DIFFERENT indices. 
    
    # Structural constraint on H:
    # Eta1 and Eta2 are DIFFERENCES of fiber products, so sum_c Eta1_k[c] = sum_{r,u} (p0-p1) 
    # Actually let's check: does H have a null direction related to "sum over c"?
    
    # Check if Gamma @ H ≈ 0 (H should be in ker(Gamma) column-wise)
    GH = Gam @ H  # (9, 18)
    print(f"  ||Γ @ H||        = {np.linalg.norm(GH):.6e}")
    print(f"  ||Γ @ Delta||    = {np.linalg.norm(Gam @ Delta):.6e}")
    
    # AH-HA: if Γ @ Delta ≠ 0, then Delta is NOT in ker(Gamma)!
    # That would explain why the containment fails.
    gd_norm = np.linalg.norm(Gam @ Delta)
    gh_norm = np.linalg.norm(GH)
    if gd_norm > 1e-6:
        print(f"\n  *** KEY: Γ @ Δ ≠ 0 ! Delta is NOT fully in ker(Γ)! ***")
        print(f"  This means Delta has a component in im(Γ^T) that H doesn't cover.")
        GD = Gam @ Delta
        print(f"  rank(Γ@Δ) = {np.linalg.matrix_rank(GD, tol=1e-8)}")


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    for name, fname in [("slp_turbo_best", "slp_turbo_best.json"),
                         ("slp_best@0.074", "slp_best_at_0.074.json")]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            a, b, g = load_candidate(path)
            diagnose(name, a, b, g)


if __name__ == "__main__":
    main()
