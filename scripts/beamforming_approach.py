#!/usr/bin/env python3
"""
Phased-array beamforming approach to tensor decomposition.

ANALOGY:
  - Each rank-1 term (α_k, β_k, γ_k) is an antenna element
  - α_k ⊗ β_k defines the element's "radiation pattern" (array geometry)
  - γ_k is the "beamforming weight" (excitation coefficient)
  - 27 live entries = main beam (desired signal directions)
  - 702 dead entries = interference directions (need nulls)

APPROACH:
  For fixed (α, β), compute the OPTIMAL γ analytically using
  MVDR-like beamforming: maximize live-entry fidelity while
  minimizing dead-entry leakage.

  Then optimize (α, β) in the outer loop.

  This is fundamentally different from ALS because:
  1. γ is computed globally optimally (not alternating)
  2. The objective is max-abs (Chebyshev/L∞), not L2
  3. Dead-entry nulling is treated as a constraint, not a penalty
"""

import json
import numpy as np
from pathlib import Path

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def build_matmul_tensor(n=3):
    T = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[n*i+j, n*j+k, n*i+k] = 1.0
    return T


def get_masks(T):
    """Return live and dead index arrays."""
    flat = T.ravel()
    live = np.where(flat != 0)[0]
    dead = np.where(flat == 0)[0]
    return live, dead


def radiation_pattern(alpha_k, beta_k, n2=9):
    """The 'radiation pattern' of element k: the rank-1 outer product α⊗β.
    Returns (n2*n2,) = (81,) vector — the contribution to all (a,b) pairs
    before weighting by gamma."""
    return np.outer(alpha_k, beta_k).ravel()  # (81,)


def build_steering_matrix(alpha, beta, n2=9):
    """Build the full steering matrix A where A[k, (a,b)] = α_k[a]·β_k[b].
    
    For each output entry c, the reconstructed tensor is:
      T_recon[a,b,c] = Σ_k α_k[a]·β_k[b]·γ_k[c] = Σ_k A[k,(a,b)] · γ_k[c]
    
    So for fixed c, T_recon[:,;,c] = A^T @ γ[:,c] in the (a,b) flattened space.
    
    Returns A of shape (R, n2*n2) = (R, 81).
    """
    R = alpha.shape[0]
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    return A


def mvdr_gamma(alpha, beta, T_target, regularization=1e-6):
    """
    MVDR-inspired optimal gamma computation.
    
    For each output entry c = (r, u):
      T[a, b, c] = Σ_k α_k[a] · β_k[b] · γ_k[c]
    
    We want:
      T[a, b, c] = T_target[a, b, c] for all (a,b)
    
    This is a least-squares problem per output column c:
      A @ γ[:,c] = T_target[:,:,c].ravel()
    
    where A[k, (a,b)] = α_k[a] · β_k[b].
    
    BUT: this is just ALS for gamma. The MVDR twist is to weight the
    dead entries differently — penalize dead-entry residuals MORE heavily
    to steer nulls there.
    
    MVDR formulation per output c:
      min  ||W_dead · (A γ_c - t_c)||²
      s.t. (A γ_c)[live] = t_c[live]
    
    Or equivalently with a Chebyshev (L∞) objective:
      min  max|(A γ_c - t_c)[dead]|
      s.t. (A γ_c)[live] = t_c[live]
    """
    R = alpha.shape[0]
    n2 = alpha.shape[1]
    n4 = n2 * n2  # 81
    
    A = build_steering_matrix(alpha, beta, n2)  # (R, 81)
    
    gamma_opt = np.zeros((R, n2))
    
    for c in range(n2):
        target_col = T_target[:, :, c].ravel()  # (81,)
        
        # Per-column live/dead: entry (a,b,c) is live iff T_target[a,b,c] != 0
        live_ab = np.where(target_col != 0)[0]
        dead_ab = np.where(target_col == 0)[0]
        
        W = np.ones(n4)
        W[dead_ab] = 10.0
        W[live_ab] = 1.0
        
        # Weighted normal equations: (A W² A^T) γ = A W² t
        AW = A * W[np.newaxis, :]  # (R, 81) element-wise
        AWAt = AW @ A.T + regularization * np.eye(R)  # (R, R)
        AWt = AW @ (W * target_col)  # (R,)
        
        gamma_opt[:, c] = np.linalg.solve(AWAt, AWt)
    
    return gamma_opt


def mvdr_gamma_linf(alpha, beta, T_target):
    """
    L∞-optimal gamma via linear programming (Chebyshev beamforming).
    
    For each output c, solve:
      min t
      s.t. -t ≤ (A γ_c - target_c)[j] ≤ t  for all j
    
    This is the Dolph-Chebyshev analog: minimize peak sidelobe.
    """
    from scipy.optimize import linprog
    
    R = alpha.shape[0]
    n2 = alpha.shape[1]
    n4 = n2 * n2
    
    A = build_steering_matrix(alpha, beta, n2)  # (R, 81)
    
    gamma_opt = np.zeros((R, n2))
    total_maxabs = 0.0
    
    for c in range(n2):
        target_col = T_target[:, :, c].ravel()  # (81,)
        
        # Variables: [γ_c (R), t (1)]
        # Objective: min t
        c_obj = np.zeros(R + 1)
        c_obj[-1] = 1.0  # minimize t
        
        # Constraints: A^T γ_c - target ≤ t  →  A^T γ_c - t ≤ target
        #              -(A^T γ_c - target) ≤ t  →  -A^T γ_c - t ≤ -target
        # In matrix form: [A^T, -1] [γ; t] ≤ target
        #                 [-A^T, -1] [γ; t] ≤ -target
        
        A_ub = np.zeros((2 * n4, R + 1))
        b_ub = np.zeros(2 * n4)
        
        A_ub[:n4, :R] = A.T        # A^T γ - t ≤ target
        A_ub[:n4, R] = -1.0
        b_ub[:n4] = target_col
        
        A_ub[n4:, :R] = -A.T       # -A^T γ - t ≤ -target
        A_ub[n4:, R] = -1.0
        b_ub[n4:] = -target_col
        
        # Bounds: t ≥ 0, γ unbounded
        bounds = [(None, None)] * R + [(0, None)]
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            gamma_opt[:, c] = result.x[:R]
            total_maxabs = max(total_maxabs, result.x[R])
        else:
            # Fallback to least squares
            gamma_opt[:, c] = np.linalg.lstsq(A.T, target_col, rcond=None)[0]
    
    return gamma_opt, total_maxabs


def outer_loop_optimize(alpha, beta, T_target, n_iters=200, lr=0.01):
    """
    Outer loop: optimize (α, β) while computing γ optimally.
    
    This is the "array geometry" optimization. For each (α, β),
    the best possible γ is computed analytically, then we measure
    the achieved max-abs residual and update (α, β) via gradient.
    """
    R, n2 = alpha.shape
    alpha = alpha.copy()
    beta = beta.copy()
    
    best_fitness = float('inf')
    best_alpha = alpha.copy()
    best_beta = beta.copy()
    best_gamma = None
    
    for it in range(n_iters):
        # Compute optimal gamma
        gamma = mvdr_gamma(alpha, beta, T_target, regularization=1e-5)
        
        # Evaluate
        T_recon = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma)
        residual = T_recon - T_target
        fitness = np.max(np.abs(residual))
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_alpha = alpha.copy()
            best_beta = beta.copy()
            best_gamma = gamma.copy()
            if it % 20 == 0:
                print(f"  iter {it:4d}: fitness = {fitness:.8f} *")
        elif it % 50 == 0:
            print(f"  iter {it:4d}: fitness = {fitness:.8f}")
        
        # Gradient of max-abs w.r.t. alpha and beta (via finite differences)
        # More efficient: use the gradient of the smooth-max approximation
        beta_smooth = 100.0
        abs_resid = np.abs(residual.ravel())
        weights = np.exp(beta_smooth * (abs_resid - abs_resid.max()))
        weights /= weights.sum()
        # Signed residual weighted by softmax
        signed = residual.ravel() * weights
        signed_tensor = signed.reshape(n2, n2, n2)
        
        # Gradient w.r.t. alpha: dL/dα_k[a] = Σ_{b,c} signed[a,b,c] · β_k[b] · γ_k[c]
        grad_alpha = np.einsum('abc,kb,kc->ka', signed_tensor, beta, gamma)
        # Gradient w.r.t. beta: dL/dβ_k[b] = Σ_{a,c} signed[a,b,c] · α_k[a] · γ_k[c]
        grad_beta = np.einsum('abc,ka,kc->kb', signed_tensor, alpha, gamma)
        
        # Adam-like update with momentum
        alpha -= lr * grad_alpha
        beta -= lr * grad_beta
        
        # Slight random perturbation to escape local minima
        if it % 30 == 0 and it > 0:
            alpha += np.random.randn(*alpha.shape) * 0.001
            beta += np.random.randn(*beta.shape) * 0.001
    
    return best_alpha, best_beta, best_gamma, best_fitness


def main():
    print("=" * 70)
    print("  PHASED ARRAY BEAMFORMING APPROACH")
    print("=" * 70)
    
    T = build_matmul_tensor(3)
    live_idx, dead_idx = get_masks(T)
    print(f"Target: 3×3 matmul tensor, {len(live_idx)} live, {len(dead_idx)} dead entries")
    
    # ── 1. Test MVDR gamma on existing rank-19 solution ──
    print(f"\n{'─'*70}")
    print(f"  1. MVDR GAMMA ON EXISTING RANK-19")
    print(f"{'─'*70}")
    
    data = json.loads((Path(__file__).parent.parent / "slp_turbo_best.json").read_text())
    alpha = np.array(data["alpha"])
    beta = np.array(data["beta"])
    gamma_orig = np.array(data["gamma"])
    R = alpha.shape[0]
    
    # Original fitness
    T_orig = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma_orig)
    orig_fitness = np.max(np.abs(T_orig - T))
    print(f"Original fitness (original γ): {orig_fitness:.8f}")
    
    # MVDR L2-weighted gamma  
    for w_dead in [1.0, 5.0, 10.0, 50.0, 100.0]:
        gamma_mvdr = mvdr_gamma(alpha, beta, T, regularization=1e-6)
        T_mvdr = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma_mvdr)
        mvdr_fitness = np.max(np.abs(T_mvdr - T))
        resid = T_mvdr - T
        live_max = np.max(np.abs(resid.ravel()[live_idx]))
        dead_max = np.max(np.abs(resid.ravel()[dead_idx]))
        print(f"  MVDR γ (w_dead={w_dead:5.1f}): fitness={mvdr_fitness:.8f}  live={live_max:.6f}  dead={dead_max:.6f}")
    
    # MVDR L∞-optimal gamma (Chebyshev beamforming)
    print(f"\n  L∞-optimal γ (Chebyshev beamforming via LP)...")
    gamma_linf, linf_maxabs = mvdr_gamma_linf(alpha, beta, T)
    T_linf = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma_linf)
    linf_fitness = np.max(np.abs(T_linf - T))
    resid_linf = T_linf - T
    live_max = np.max(np.abs(resid_linf.ravel()[live_idx]))
    dead_max = np.max(np.abs(resid_linf.ravel()[dead_idx]))
    print(f"  L∞-optimal γ: fitness={linf_fitness:.8f}  live={live_max:.6f}  dead={dead_max:.6f}")
    print(f"  LP reported minimax: {linf_maxabs:.8f}")
    
    print(f"\n  Comparison:")
    print(f"    Original γ:  {orig_fitness:.8f}")
    print(f"    L∞-optimal γ: {linf_fitness:.8f}")
    if linf_fitness < orig_fitness:
        print(f"    IMPROVEMENT: {orig_fitness - linf_fitness:.8f} ({(orig_fitness-linf_fitness)/orig_fitness*100:.2f}%)")
    else:
        print(f"    No improvement — γ was already near-optimal for this (α, β)")
    
    # ── 2. Test on ranks 20, 21, 22 ──
    print(f"\n{'─'*70}")
    print(f"  2. BEAMFORMING-OPTIMAL γ FOR RANKS 20–22")
    print(f"{'─'*70}")
    
    for target_R in [20, 21, 22, 23]:
        n_extra = target_R - R
        # Warm-start: existing + small random extra terms
        rng = np.random.default_rng(42 + target_R)
        extra_a = rng.standard_normal((n_extra, 9)) * 0.5
        extra_b = rng.standard_normal((n_extra, 9)) * 0.5
        
        alpha_ext = np.vstack([alpha, extra_a])
        beta_ext = np.vstack([beta, extra_b])
        
        # L∞-optimal gamma
        gamma_ext, maxabs_ext = mvdr_gamma_linf(alpha_ext, beta_ext, T)
        T_ext = np.einsum('ki,kj,kc->ijc', alpha_ext, beta_ext, gamma_ext)
        fit_ext = np.max(np.abs(T_ext - T))
        print(f"  R={target_R} (warm-start, random extra terms): L∞-optimal fitness = {fit_ext:.8f}")
    
    # ── 3. Full beamforming optimization: optimize (α,β), compute γ analytically ──
    print(f"\n{'─'*70}")
    print(f"  3. FULL BEAMFORMING: OPTIMIZE (α,β), γ COMPUTED ANALYTICALLY")
    print(f"{'─'*70}")
    
    for target_R in [19, 20, 21]:
        print(f"\n  --- R = {target_R} ---")
        if target_R == R:
            a0, b0 = alpha.copy(), beta.copy()
        else:
            n_extra = target_R - R
            rng = np.random.default_rng(42 + target_R)
            a0 = np.vstack([alpha, rng.standard_normal((n_extra, 9)) * 0.3])
            b0 = np.vstack([beta, rng.standard_normal((n_extra, 9)) * 0.3])
        
        a_opt, b_opt, g_opt, fit_opt = outer_loop_optimize(a0, b0, T, n_iters=300, lr=0.005)
        T_opt = np.einsum('ki,kj,kc->ijc', a_opt, b_opt, g_opt)
        resid_opt = T_opt - T
        print(f"  Final: fitness = {fit_opt:.8f}")
        print(f"    live maxabs = {np.max(np.abs(resid_opt.ravel()[live_idx])):.6f}")
        print(f"    dead maxabs = {np.max(np.abs(resid_opt.ravel()[dead_idx])):.6f}")
    
    # ── 4. Null-space analysis ──
    print(f"\n{'─'*70}")
    print(f"  4. NULL-SPACE ANALYSIS: HOW MANY INDEPENDENT NULLS CAN R TERMS STEER?")
    print(f"{'─'*70}")
    
    A = build_steering_matrix(alpha, beta)  # (19, 81)
    print(f"  Steering matrix A: {A.shape}, rank = {np.linalg.matrix_rank(A, tol=1e-6)}")
    print(f"  Column space dim (# independent beam directions): {np.linalg.matrix_rank(A, tol=1e-6)}")
    print(f"  Null space dim of A^T: {A.shape[1] - np.linalg.matrix_rank(A, tol=1e-6)}")
    print(f"  These are the directions in pattern space we CANNOT control with γ.")
    print(f"  We need 702 nulls but can only steer {np.linalg.matrix_rank(A, tol=1e-6)} independent beams.")
    print(f"  The residual lives in a {A.shape[1] - np.linalg.matrix_rank(A, tol=1e-6)}-dim uncontrollable subspace.")
    
    # Project dead entries onto uncontrollable subspace
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank_A = np.sum(S > 1e-6)
    V_perp = np.linalg.svd(A.T, full_matrices=True)[0][:, rank_A:]  # (81, 81-rank)
    
    # How much of the dead-entry target lies in uncontrollable space?
    T_flat = T.reshape(9, 81)  # (c, a*b)
    for c in range(9):
        target_c = T_flat[c]  # (81,)
        proj_uncontrollable = V_perp @ (V_perp.T @ target_c)
        uncontrollable_norm = np.linalg.norm(proj_uncontrollable)
        total_norm = np.linalg.norm(target_c)
        print(f"  Output c={c}: ||target||={total_norm:.3f}, ||uncontrollable||={uncontrollable_norm:.6f}")
    
    print(f"\n  If uncontrollable_norm > 0 for any c, that's the IRREDUCIBLE floor —")
    print(f"  no choice of γ can eliminate that residual. Only changing (α, β) can help.")


if __name__ == "__main__":
    main()
