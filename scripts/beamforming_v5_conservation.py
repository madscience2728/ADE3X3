#!/usr/bin/env python3
"""
Beamforming v5: Conservation-law-guided initialization.

THE KEY INSIGHT FROM THEORY:
  For exact R=19, conservation law demands η_nullity = 8, rank(H_proj) = 2.
  This means the anisotropy vectors Σ_k = α_k⊗α_k / ||α_k||² must span
  at most a 2-dimensional subspace (after projection off the target).

  The current best has rank(H_proj) = 10 (full!) — the search found a basin
  with MAXIMUM anisotropy diversity, which is exactly wrong.

WHAT WE NEED:
  α vectors should be drawn from a LOW-DIMENSIONAL subspace of R^9.
  If α_k ∈ span{v₁, v₂, v₃} (3D subspace of R^9), then the anisotropy
  Σ_k = α_k⊗α_k lives in at most a 6-dim subspace of Sym²(R^9)=R^45,
  which after projection could give low rank(H_proj).

  Similarly for β. The extreme case: all α_k proportional to the same vector
  gives rank(H_proj)=0 — but then rank(A^T)=1 and we can't fit the tensor.

  The sweet spot: α_k from a 3D subspace, β_k also from a low-D subspace.
  This constrains the search to the "right kind" of basin.

STRASSEN ANALOGY:
  For ⟨2,2,2⟩ (Strassen), all 7 α vectors live in {±1}^4, effectively
  a 4D discrete set in R^4. The anisotropy Σ_k = α_k⊗α_k always has
  all diagonal entries = 1, so rank(H_proj) = 3 = dim(ker(Γ)).
  Conservation: R=7, η_nullity=1, R+η_nullity=8=n³. ✓

FOR ⟨3,3,3⟩:
  n³=27, target R=19 → need η_nullity=8 → rank(H_proj)=2
  This means the 19 anisotropy vectors must be VERY similar
  (projecting onto only 2 independent directions beyond the target).

STRATEGY:
  1. Constrain α_k = U_α @ z_k^α where U_α is (9, d_α) with d_α small (2-4)
  2. Same for β_k = U_β @ z_k^β
  3. Optimize over (U_α, U_β, z^α, z^β) — much smaller search space
  4. Compute γ via LP as before
  
  Then feed the best into slp_turbo for full refinement.
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog, minimize, differential_evolution

np.set_printoptions(precision=6, suppress=True, linewidth=120)
ROOT = Path(__file__).parent.parent


def build_matmul_tensor(n=3):
    T = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[n*i+j, n*j+k, n*i+k] = 1.0
    return T


def linf_optimal_gamma(alpha, beta, T_target, n2=9):
    R = alpha.shape[0]
    n4 = n2 * n2
    A = np.zeros((R, n4))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    
    gamma = np.zeros((R, n2))
    worst = 0.0
    for c in range(n2):
        target_col = T_target[:, :, c].ravel()
        c_obj = np.zeros(R + 1)
        c_obj[-1] = 1.0
        A_ub = np.zeros((2 * n4, R + 1))
        b_ub = np.zeros(2 * n4)
        A_ub[:n4, :R] = A.T
        A_ub[:n4, R] = -1.0
        b_ub[:n4] = target_col
        A_ub[n4:, :R] = -A.T
        A_ub[n4:, R] = -1.0
        b_ub[n4:] = -target_col
        bounds = [(None, None)] * R + [(0, None)]
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            gamma[:, c] = result.x[:R]
            worst = max(worst, result.x[R])
        else:
            gamma[:, c] = np.linalg.lstsq(A.T, target_col, rcond=None)[0]
            worst = max(worst, np.max(np.abs(A.T @ gamma[:, c] - target_col)))
    return gamma, worst


def compute_hproj_rank(alpha):
    """Compute rank(H_proj) — the anisotropy rank that conservation law constrains."""
    R, n2 = alpha.shape
    # Anisotropy: Σ_k = α_k ⊗ α_k / ||α_k||²
    # H_proj = projection of span(Σ_k) onto complement of identity
    sigmas = []
    for k in range(R):
        nrm = np.linalg.norm(alpha[k])
        if nrm < 1e-10:
            continue
        s = np.outer(alpha[k], alpha[k]) / nrm**2  # (9,9)
        sigmas.append(s.ravel())  # (81,)
    
    if not sigmas:
        return 0
    
    S = np.array(sigmas)  # (R, 81)
    # Center: subtract mean (≈ identity direction)
    S_centered = S - S.mean(axis=0, keepdims=True)
    
    sv = np.linalg.svd(S_centered, compute_uv=False)
    rank = np.sum(sv > 1e-6)
    return rank


def subspace_parametrize(params, R, d_alpha, d_beta, n2=9):
    """
    Decode parameters into (α, β).
    
    params layout:
      U_α: (n2, d_alpha) = 9*d_alpha params
      U_β: (n2, d_beta) = 9*d_beta params
      z_α: (R, d_alpha) = R*d_alpha params
      z_β: (R, d_beta) = R*d_beta params
    
    Total: 9*(d_alpha + d_beta) + R*(d_alpha + d_beta)
         = (9 + R) * (d_alpha + d_beta)
    """
    idx = 0
    U_a = params[idx:idx + n2 * d_alpha].reshape(n2, d_alpha)
    idx += n2 * d_alpha
    U_b = params[idx:idx + n2 * d_beta].reshape(n2, d_beta)
    idx += n2 * d_beta
    z_a = params[idx:idx + R * d_alpha].reshape(R, d_alpha)
    idx += R * d_alpha
    z_b = params[idx:idx + R * d_beta].reshape(R, d_beta)
    
    alpha = z_a @ U_a.T  # (R, 9)
    beta = z_b @ U_b.T   # (R, 9)
    
    return alpha, beta


def objective(params, R, d_alpha, d_beta, T_target, n2=9):
    """Evaluate L∞ fitness for subspace-parametrized (α, β)."""
    alpha, beta = subspace_parametrize(params, R, d_alpha, d_beta, n2)
    _, fitness = linf_optimal_gamma(alpha, beta, T_target, n2)
    return fitness


def run_subspace_search(R, d_alpha, d_beta, T_target, n2=9,
                        n_restarts=10, max_evals_per=500):
    """
    Search over low-dimensional (α, β) subspaces.
    """
    n_params = (n2 + R) * (d_alpha + d_beta)
    print(f"\n  R={R}, d_α={d_alpha}, d_β={d_beta}: {n_params} params "
          f"(vs {3*R*n2}={3*R*n2} full)")
    
    best_fitness = float('inf')
    best_params = None
    t0 = time.time()
    
    for restart in range(n_restarts):
        rng = np.random.default_rng(restart * 137 + R * 17 + d_alpha)
        x0 = rng.standard_normal(n_params) * 0.5
        
        result = minimize(
            objective, x0,
            args=(R, d_alpha, d_beta, T_target, n2),
            method='nelder-mead',
            options={'maxfev': max_evals_per, 'adaptive': True,
                     'xatol': 1e-5, 'fatol': 1e-8}
        )
        
        elapsed = time.time() - t0
        fit = result.fun
        tag = " ***" if fit < best_fitness else ""
        
        alpha_r, beta_r = subspace_parametrize(result.x, R, d_alpha, d_beta, n2)
        hrank = compute_hproj_rank(alpha_r)
        
        print(f"    restart {restart:2d} ({elapsed:6.1f}s, {result.nfev} evals): "
              f"fitness={fit:.6f}  H_rank={hrank}{tag}")
        
        if fit < best_fitness:
            best_fitness = fit
            best_params = result.x.copy()
    
    alpha_best, beta_best = subspace_parametrize(best_params, R, d_alpha, d_beta, n2)
    gamma_best, _ = linf_optimal_gamma(alpha_best, beta_best, T_target, n2)
    hrank = compute_hproj_rank(alpha_best)
    
    elapsed = time.time() - t0
    print(f"\n  Best: fitness={best_fitness:.6f}, H_rank={hrank} ({elapsed:.1f}s)")
    
    return alpha_best, beta_best, gamma_best, best_fitness


def main():
    print("=" * 70)
    print("  BEAMFORMING v5: CONSERVATION-LAW CONSTRAINED SEARCH")
    print("=" * 70)
    
    T = build_matmul_tensor(3)
    n2 = 9
    
    data = json.loads((ROOT / "slp_turbo_best.json").read_text())
    ref_fitness = data["fitness"]
    ref_alpha = np.array(data["alpha"])
    ref_hrank = compute_hproj_rank(ref_alpha)
    print(f"Reference: R={len(data['alpha'])}, fitness={ref_fitness:.8f}, H_rank={ref_hrank}")
    
    # ── Sweep over subspace dimensions ──
    results = {}
    
    for R in [19, 20]:
        print(f"\n{'='*70}")
        print(f"  R = {R}")
        print(f"{'='*70}")
        
        for d_alpha, d_beta in [(3, 3), (4, 4), (5, 5), (3, 5), (5, 3), (6, 6)]:
            a, b, g, f = run_subspace_search(
                R, d_alpha, d_beta, T, n2,
                n_restarts=8, max_evals_per=400
            )
            results[(R, d_alpha, d_beta)] = (a, b, g, f)
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Reference: {ref_fitness:.8f} (H_rank={ref_hrank})")
    print(f"  {'R':>3s} {'d_α':>3s} {'d_β':>3s} {'fitness':>10s} {'H_rank':>7s} {'params':>7s}")
    
    best_overall = float('inf')
    best_key = None
    
    for (R, da, db), (a, b, g, f) in sorted(results.items()):
        hrank = compute_hproj_rank(a)
        n_params = (n2 + R) * (da + db)
        print(f"  {R:3d} {da:3d} {db:3d} {f:10.6f} {hrank:7d} {n_params:7d}")
        if f < best_overall:
            best_overall = f
            best_key = (R, da, db)
    
    if best_key:
        R, da, db = best_key
        a, b, g, f = results[best_key]
        print(f"\n  Best: R={R}, d_α={da}, d_β={db}, fitness={f:.6f}")
        
        # Save as slp_turbo seed
        out = {
            "alpha": a.tolist(), "beta": b.tolist(), "gamma": g.tolist(),
            "fitness": float(f), "rank": R, "dim": 9,
            "method": f"conservation_law_d{da}_{db}"
        }
        path = ROOT / f"conservation_init_R{R}_d{da}_{db}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"  Saved to {path.name} — feed to slp_turbo for refinement")


if __name__ == "__main__":
    main()
