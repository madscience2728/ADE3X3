#!/usr/bin/env python3
"""
Beamforming v4: Theory-guided structured initialization.

Conservation law demands rank(H_proj)=2 for exact R=19.
The current best has rank(H_proj)=10 (full) — wrong basin entirely.

Strategy:
  1. Analyze the TARGET subspace: what does the matmul tensor's
     column space look like in the (a,b) pattern space?
  2. Construct (α,β) pairs whose steering matrix row space
     CONTAINS the target subspace by design.
  3. Use remaining degrees of freedom for dead-entry suppression.
  4. Optimize from this structured start via L∞ + coordinate descent.

The 9 target columns (one per output c) each have exactly 3 nonzeros.
They span a subspace of R^81. We need the row space of A (R×81) to
contain this subspace. Then γ has enough freedom to hit the live entries,
and the residual depends only on dead-entry leakage.
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog, minimize
from itertools import product

np.set_printoptions(precision=6, suppress=True, linewidth=120)
ROOT = Path(__file__).parent.parent


def build_matmul_tensor(n=3):
    T = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[n*i+j, n*j+k, n*i+k] = 1.0
    return T


def build_steering_matrix(alpha, beta, n2=9):
    R = alpha.shape[0]
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    return A


def linf_optimal_gamma(A, T_target, n2=9):
    R = A.shape[0]
    n4 = n2 * n2
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


def uncontrollable_energy(A, T_target, n2=9):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank_A = np.sum(S > 1e-8)
    V_full = np.linalg.svd(A.T, full_matrices=True)[0]
    V_perp = V_full[:, rank_A:]
    total_unc = 0.0
    total_norm = 0.0
    for c in range(n2):
        t_c = T_target[:, :, c].ravel()
        proj = V_perp @ (V_perp.T @ t_c)
        total_unc += np.linalg.norm(proj)**2
        total_norm += np.linalg.norm(t_c)**2
    return np.sqrt(total_unc / total_norm)


def analyze_target_subspace():
    """Analyze the structure of the 9 target columns in R^81."""
    T = build_matmul_tensor(3)
    n2 = 9
    
    print("  Target column structure (each column c of the matmul tensor):")
    target_vecs = []
    for c in range(n2):
        col = T[:, :, c].ravel()  # (81,)
        nz = np.where(col != 0)[0]
        target_vecs.append(col)
        r, u = divmod(c, 3)
        # Live entries: (a,b) = (3r+j, 3j+u) for j=0,1,2
        live_pairs = [(3*r+j, 3*j+u) for j in range(3)]
        live_flat = [a*9+b for a,b in live_pairs]
        print(f"    c={c} (r={r},u={u}): live at (a,b)={live_pairs}  flat={live_flat}")
    
    V = np.array(target_vecs).T  # (81, 9)
    rank_V = np.linalg.matrix_rank(V)
    print(f"\n  Target matrix V: (81, 9), rank = {rank_V}")
    print(f"  The 9 target columns span a {rank_V}-dim subspace of R^81.")
    
    # SVD of target subspace
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    print(f"  Singular values: {S}")
    
    return V, U[:, :rank_V]  # V matrix and orthonormal basis for target subspace


def construct_structured_ab(R, n=3, method='kronecker'):
    """
    Construct (α,β) pairs whose outer products span a space
    containing the target subspace.
    
    Key insight: The target columns have structure
      T[:,;,c=(r,u)] has nonzeros at (3r+j, 3j+u) for j=0,1,2.
    
    This means α_k[a] · β_k[b] needs to be able to produce
    weight at (3r+j, 3j+u). If α_k has weight on indices
    {3r+0, 3r+1, 3r+2} (row r block) and β_k on {0+u, 3+u, 6+u}
    (column u block), then the outer product covers the right entries.
    
    For 3×3 matmul: there are 3 row-blocks × 3 col-blocks = 9 
    natural (α,β) structures. With R=19, we have 9 "structural" 
    terms + 10 "suppression" terms.
    """
    n2 = n * n
    
    if method == 'kronecker':
        # Method 1: Kronecker-structured initialization
        # 9 terms: one per (row-block, col-block) pair
        # Each targets a specific (r,u) output entry
        alpha_list = []
        beta_list = []
        
        for r in range(n):
            for u in range(n):
                a = np.zeros(n2)
                b = np.zeros(n2)
                # α has support on row-block r: indices 3r, 3r+1, 3r+2
                for j in range(n):
                    a[n*r + j] = 1.0
                # β has support on col-block u: indices u, 3+u, 6+u
                for j in range(n):
                    b[n*j + u] = 1.0
                alpha_list.append(a)
                beta_list.append(b)
        
        # Remaining R-9 terms: random for dead-entry suppression
        rng = np.random.default_rng(2026)
        for _ in range(R - 9):
            alpha_list.append(rng.standard_normal(n2) * 0.3)
            beta_list.append(rng.standard_normal(n2) * 0.3)
        
        return np.array(alpha_list), np.array(beta_list)
    
    elif method == 'strassen_like':
        # Method 2: Strassen-inspired — use ±1 entries with overlapping support
        # This creates more "interference" patterns for dead-entry cancellation
        alpha_list = []
        beta_list = []
        rng = np.random.default_rng(2026)
        
        # Generate R terms with structured ±1 entries
        for k in range(R):
            a = np.zeros(n2)
            b = np.zeros(n2)
            # Random sparse ±1 pattern
            n_nonzero = rng.integers(2, 6)
            idx_a = rng.choice(n2, size=n_nonzero, replace=False)
            idx_b = rng.choice(n2, size=n_nonzero, replace=False)
            a[idx_a] = rng.choice([-1.0, 1.0], size=n_nonzero)
            b[idx_b] = rng.choice([-1.0, 1.0], size=n_nonzero)
            alpha_list.append(a)
            beta_list.append(b)
        
        return np.array(alpha_list), np.array(beta_list)
    
    elif method == 'svd_seed':
        # Method 3: Initialize from SVD of target subspace
        # Make the first 9 rows of A be exactly the target columns
        T = build_matmul_tensor(n)
        V_targets = []
        for c in range(n2):
            V_targets.append(T[:, :, c].ravel())
        V = np.array(V_targets)  # (9, 81) — each row is a target column
        
        # Factor each target row as α⊗β via best rank-1 approx
        alpha_list = []
        beta_list = []
        for c in range(n2):
            mat = V[c].reshape(n2, n2)  # interpret as 9×9
            U, S, Vt = np.linalg.svd(mat)
            # Best rank-1: σ_1 · u_1 ⊗ v_1
            a = U[:, 0] * np.sqrt(S[0])
            b = Vt[0, :] * np.sqrt(S[0])
            alpha_list.append(a)
            beta_list.append(b)
        
        # Remaining terms: random
        rng = np.random.default_rng(2026)
        for _ in range(R - n2):
            alpha_list.append(rng.standard_normal(n2) * 0.3)
            beta_list.append(rng.standard_normal(n2) * 0.3)
        
        return np.array(alpha_list), np.array(beta_list)


def optimize_single_term(k, alpha, beta, T_target, n2=9, max_evals=200):
    """Optimize term k via Nelder-Mead with L∞ inner loop."""
    def obj(x):
        a = alpha.copy()
        b = beta.copy()
        a[k] = x[:n2]
        b[k] = x[n2:]
        A = build_steering_matrix(a, b, n2)
        _, fit = linf_optimal_gamma(A, T_target, n2)
        return fit
    
    x0 = np.concatenate([alpha[k], beta[k]])
    result = minimize(obj, x0, method='nelder-mead',
                      options={'maxfev': max_evals, 'xatol': 1e-6,
                               'fatol': 1e-9, 'adaptive': True})
    
    alpha_new = alpha.copy()
    beta_new = beta.copy()
    alpha_new[k] = result.x[:n2]
    beta_new[k] = result.x[n2:]
    return alpha_new, beta_new, result.fun, result.nfev


def coordinate_descent(alpha, beta, T_target, n_rounds=3, max_evals=200, label=""):
    """Cycle through all terms, optimizing each one."""
    R = alpha.shape[0]
    A = build_steering_matrix(alpha, beta)
    _, best_fitness = linf_optimal_gamma(A, T_target)
    best_alpha, best_beta = alpha.copy(), beta.copy()
    
    print(f"  Initial fitness: {best_fitness:.8f}")
    unc = uncontrollable_energy(A, T_target)
    print(f"  Uncontrollable energy: {unc:.6f}")
    
    t0 = time.time()
    total_evals = 0
    
    for rnd in range(n_rounds):
        improved_any = False
        order = np.random.permutation(R)
        
        for idx, k in enumerate(order):
            a_new, b_new, fit_new, nfev = optimize_single_term(
                k, best_alpha, best_beta, T_target, max_evals=max_evals)
            total_evals += nfev
            elapsed = time.time() - t0
            
            if fit_new < best_fitness - 1e-10:
                delta = best_fitness - fit_new
                print(f"  R{rnd+1} k={k:2d} ({elapsed:6.1f}s): "
                      f"{best_fitness:.8f} → {fit_new:.8f}  Δ={delta:.2e} *")
                best_fitness = fit_new
                best_alpha, best_beta = a_new, b_new
                improved_any = True
            elif (idx + 1) % 5 == 0:
                print(f"  R{rnd+1} k={k:2d} ({elapsed:6.1f}s): no impr ({fit_new:.8f})")
        
        elapsed = time.time() - t0
        A_best = build_steering_matrix(best_alpha, best_beta)
        unc = uncontrollable_energy(A_best, T_target)
        print(f"  --- Round {rnd+1} ({elapsed:.1f}s): best={best_fitness:.8f} unc={unc:.6f} ---")
        
        if not improved_any:
            print(f"  No improvement, stopping.")
            break
    
    A_best = build_steering_matrix(best_alpha, best_beta)
    best_gamma, _ = linf_optimal_gamma(A_best, T_target)
    
    return best_alpha, best_beta, best_gamma, best_fitness


def multi_start_search(R, T_target, n_starts=20, n2=9):
    """
    Launch many structured random initializations, evaluate L∞-optimal γ,
    then refine the best ones.
    """
    print(f"\n  Multi-start screening: {n_starts} initializations for R={R}")
    
    results = []
    t0 = time.time()
    
    methods = ['kronecker', 'strassen_like', 'svd_seed']
    
    for i in range(n_starts):
        # Pick method
        if i < 3:
            method = methods[i]
            a, b = construct_structured_ab(R, method=method)
        else:
            # Pure random with varying sparsity and scale
            rng = np.random.default_rng(i * 137 + 42)
            sparsity = rng.uniform(0.3, 1.0)
            scale = rng.uniform(0.1, 2.0)
            a = rng.standard_normal((R, n2)) * scale
            b = rng.standard_normal((R, n2)) * scale
            # Sparsify
            mask_a = rng.random((R, n2)) < sparsity
            mask_b = rng.random((R, n2)) < sparsity
            a *= mask_a
            b *= mask_b
        
        A = build_steering_matrix(a, b, n2)
        gamma, fitness = linf_optimal_gamma(A, T_target, n2)
        unc = uncontrollable_energy(A, T_target, n2)
        
        elapsed = time.time() - t0
        tag = f"({methods[i]})" if i < 3 else f"(random seed {i})"
        print(f"    init {i:3d} {tag:25s}: fitness={fitness:.6f}  unc={unc:.4f}  ({elapsed:.1f}s)")
        
        results.append((fitness, unc, a.copy(), b.copy(), gamma.copy()))
    
    # Sort by fitness
    results.sort(key=lambda x: x[0])
    
    print(f"\n  Top 5 initializations:")
    for rank_i, (fit, unc, _, _, _) in enumerate(results[:5]):
        print(f"    #{rank_i+1}: fitness={fit:.6f}  unc={unc:.4f}")
    
    return results


def main():
    print("=" * 70)
    print("  BEAMFORMING v4: THEORY-GUIDED STRUCTURED INIT")
    print("=" * 70)
    
    T = build_matmul_tensor(3)
    n2 = 9
    
    # Load reference
    data = json.loads((ROOT / "slp_turbo_best.json").read_text())
    ref_fitness = data["fitness"]
    print(f"Reference: rank-{len(data['alpha'])} at fitness {ref_fitness:.8f}")
    
    # ── Analyze target subspace ──
    print(f"\n{'─'*70}")
    print(f"  TARGET SUBSPACE ANALYSIS")
    print(f"{'─'*70}")
    V, basis = analyze_target_subspace()
    
    # ── Test structured initializations ──
    for R in [19, 20]:
        print(f"\n{'='*70}")
        print(f"  R = {R}")
        print(f"{'='*70}")
        
        # Phase 1: Screen many initializations
        print(f"\n{'─'*70}")
        print(f"  PHASE 1: MULTI-START SCREENING")
        print(f"{'─'*70}")
        
        results = multi_start_search(R, T, n_starts=30)
        
        # Phase 2: Refine top 3
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: REFINE TOP 3")
        print(f"{'─'*70}")
        
        best_overall = float('inf')
        best_soln = None
        
        for rank_i in range(min(3, len(results))):
            fit0, unc0, a0, b0, g0 = results[rank_i]
            print(f"\n  Refining #{rank_i+1} (fitness={fit0:.6f}, unc={unc0:.4f})...")
            
            a_opt, b_opt, g_opt, f_opt = coordinate_descent(
                a0, b0, T, n_rounds=3, max_evals=200,
                label=f"R={R} #{rank_i+1}")
            
            if f_opt < best_overall:
                best_overall = f_opt
                best_soln = (a_opt, b_opt, g_opt)
        
        print(f"\n  R={R} BEST: {best_overall:.8f}  (reference: {ref_fitness:.8f})")
        
        if best_overall < ref_fitness - 1e-8:
            a_best, b_best, g_best = best_soln
            out = {
                "fitness": float(best_overall),
                "alpha": a_best.tolist(),
                "beta": b_best.tolist(),
                "gamma": g_best.tolist(),
                "method": f"beamforming_v4_structured_R{R}"
            }
            (ROOT / f"beamforming_v4_R{R}_best.json").write_text(json.dumps(out, indent=2))
            print(f"  *** NEW BEST! Saved to beamforming_v4_R{R}_best.json ***")
        else:
            print(f"  Did not beat reference.")


if __name__ == "__main__":
    main()
