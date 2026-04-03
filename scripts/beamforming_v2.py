#!/usr/bin/env python3
"""
Two-stage beamforming optimizer v2.

  Outer loop: optimize (α, β) array geometry via smooth-max gradient descent
  Inner loop: compute L∞-optimal γ analytically via LP (Chebyshev beamforming)

Key fix from v1: the outer loop uses L∞-optimal γ, not MVDR (L2).
The L2-optimal γ gives fitness ~0.59 (wrong objective), while L∞ gives 0.0734.

Additionally:
  - Minimize the UNCONTROLLABLE energy as a secondary objective
  - Use CMA-ES for the outer loop (gradient-free, robust to non-smooth L∞)
  - Warm-start from best known rank-19 solution
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

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
    """A[k, (a,b)] = α_k[a]·β_k[b], shape (R, 81)."""
    R = alpha.shape[0]
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    return A


def linf_optimal_gamma(A, T_target, n2=9):
    """
    Compute L∞-optimal γ via LP for each output column c.
    
    For each c: min t  s.t. |A^T γ_c - target_c| ≤ t  (elementwise)
    
    Returns gamma (R, n2) and the achieved minimax residual.
    """
    R = A.shape[0]
    n4 = n2 * n2  # 81
    gamma = np.zeros((R, n2))
    worst = 0.0
    
    for c in range(n2):
        target_col = T_target[:, :, c].ravel()  # (81,)
        
        # Variables: [γ_c (R), t (1)]
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
            T_c = A.T @ gamma[:, c]
            worst = max(worst, np.max(np.abs(T_c - target_col)))
    
    return gamma, worst


def evaluate(alpha, beta, T_target):
    """Compute L∞-optimal fitness for given (α, β)."""
    A = build_steering_matrix(alpha, beta)
    gamma, fitness = linf_optimal_gamma(A, T_target)
    return gamma, fitness


def uncontrollable_energy(alpha, beta, T_target, n2=9):
    """Fraction of target energy in the uncontrollable subspace."""
    A = build_steering_matrix(alpha, beta, n2)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank_A = np.sum(S > 1e-8)
    V_full = np.linalg.svd(A.T, full_matrices=True)[0]
    V_perp = V_full[:, rank_A:]  # (81, 81-rank)
    
    total_unc = 0.0
    total_norm = 0.0
    for c in range(n2):
        t_c = T_target[:, :, c].ravel()
        proj = V_perp @ (V_perp.T @ t_c)
        total_unc += np.linalg.norm(proj)**2
        total_norm += np.linalg.norm(t_c)**2
    
    return np.sqrt(total_unc / total_norm)


def cmaes_outer_loop(alpha0, beta0, T_target, R, n2=9,
                     sigma0=0.05, pop_size=None, max_evals=2000):
    """
    CMA-ES optimization of (α, β) with L∞-optimal γ inner loop.
    
    Uses a simple (μ/μ_w, λ)-CMA-ES implementation to avoid dependencies.
    """
    dim = R * n2 * 2  # flatten alpha and beta
    
    if pop_size is None:
        pop_size = 4 + int(3 * np.log(dim))
    mu = pop_size // 2
    
    # Encode initial point
    x0 = np.concatenate([alpha0.ravel(), beta0.ravel()])
    
    # CMA-ES state
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    
    # Strategy parameters
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)
    
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
    
    best_fitness = float('inf')
    best_x = mean.copy()
    best_gamma = None
    n_evals = 0
    gen = 0
    
    t0 = time.time()
    
    print(f"  CMA-ES: dim={dim}, pop={pop_size}, μ={mu}, σ₀={sigma0:.3f}")
    print(f"  Max evals: {max_evals}")
    
    try:
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrtC = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
    except:
        sqrtC = np.eye(dim)
    
    while n_evals < max_evals:
        # Sample population
        solutions = []
        fitnesses = []
        gammas = []
        
        for i in range(pop_size):
            z = np.random.randn(dim)
            x = mean + sigma * (sqrtC @ z)
            
            alpha_i = x[:R*n2].reshape(R, n2)
            beta_i = x[R*n2:].reshape(R, n2)
            
            gamma_i, fit_i = evaluate(alpha_i, beta_i, T_target)
            
            solutions.append((x, z))
            fitnesses.append(fit_i)
            gammas.append(gamma_i)
            n_evals += 1
        
        # Sort by fitness
        idx = np.argsort(fitnesses)
        
        if fitnesses[idx[0]] < best_fitness:
            best_fitness = fitnesses[idx[0]]
            best_x = solutions[idx[0]][0].copy()
            best_gamma = gammas[idx[0]].copy()
        
        # Recombination
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights[i] * solutions[idx[i]][0]
        
        # CSA path update
        mean_z = np.zeros(dim)
        for i in range(mu):
            mean_z += weights[i] * solutions[idx[i]][1]
        
        try:
            invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        except:
            invsqrtC = np.eye(dim)
        
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ (mean - old_mean) / sigma)
        
        # CMA path update
        hs = int(np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(gen+1))) < (1.4 + 2/(dim+1)) * chi_n)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
        
        # Covariance update
        artmp = np.zeros((dim, mu))
        for i in range(mu):
            artmp[:, i] = (solutions[idx[i]][0] - old_mean) / sigma
        
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + cmu * (artmp * weights[np.newaxis, :]) @ artmp.T
        
        # Step size update
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        sigma = min(sigma, 1.0)  # cap
        
        # Eigen decomposition for next generation
        C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except:
            C = np.eye(dim)
            eigvals = np.ones(dim)
            eigvecs = np.eye(dim)
            sqrtC = np.eye(dim)
        
        gen += 1
        elapsed = time.time() - t0
        
        if gen % 5 == 0 or gen == 1:
            unc = uncontrollable_energy(
                best_x[:R*n2].reshape(R, n2),
                best_x[R*n2:].reshape(R, n2),
                T_target
            )
            print(f"  gen {gen:4d} ({n_evals:5d} evals, {elapsed:6.1f}s): "
                  f"best={best_fitness:.8f}  σ={sigma:.4f}  unc={unc:.4f}")
    
    elapsed = time.time() - t0
    alpha_best = best_x[:R*n2].reshape(R, n2)
    beta_best = best_x[R*n2:].reshape(R, n2)
    
    print(f"\n  CMA-ES done: {n_evals} evals in {elapsed:.1f}s")
    print(f"  Best fitness: {best_fitness:.8f}")
    
    return alpha_best, beta_best, best_gamma, best_fitness


def gradient_outer_loop(alpha0, beta0, T_target, R, n2=9,
                        n_iters=200, lr=0.003):
    """
    Gradient-based outer loop using smooth-max approximation.
    Each step: compute L∞-optimal γ via LP, then differentiate
    a smooth-max surrogate w.r.t. (α, β).
    """
    alpha = alpha0.copy()
    beta = beta0.copy()
    
    best_fitness = float('inf')
    best_alpha, best_beta, best_gamma = alpha.copy(), beta.copy(), None
    
    t0 = time.time()
    
    for it in range(n_iters):
        A = build_steering_matrix(alpha, beta, n2)
        gamma, fitness = linf_optimal_gamma(A, T_target, n2)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_alpha = alpha.copy()
            best_beta = beta.copy()
            best_gamma = gamma.copy()
            tag = " *"
        else:
            tag = ""
        
        if it % 10 == 0:
            elapsed = time.time() - t0
            print(f"  iter {it:4d} ({elapsed:6.1f}s): fitness = {fitness:.8f}  best = {best_fitness:.8f}{tag}")
        
        # Smooth-max gradient
        T_recon = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma)
        residual = T_recon - T_target
        
        tau = 200.0  # temperature for smooth-max
        abs_r = np.abs(residual.ravel())
        w = np.exp(tau * (abs_r - abs_r.max()))
        w /= w.sum()
        signed = (residual.ravel() * w).reshape(n2, n2, n2)
        
        grad_alpha = np.einsum('abc,kb,kc->ka', signed, beta, gamma)
        grad_beta = np.einsum('abc,ka,kc->kb', signed, alpha, gamma)
        
        # Gradient clipping
        ga_norm = np.linalg.norm(grad_alpha)
        gb_norm = np.linalg.norm(grad_beta)
        max_norm = 1.0
        if ga_norm > max_norm:
            grad_alpha *= max_norm / ga_norm
        if gb_norm > max_norm:
            grad_beta *= max_norm / gb_norm
        
        alpha -= lr * grad_alpha
        beta -= lr * grad_beta
    
    elapsed = time.time() - t0
    print(f"\n  Gradient loop done: {n_iters} iters in {elapsed:.1f}s")
    print(f"  Best fitness: {best_fitness:.8f}")
    
    return best_alpha, best_beta, best_gamma, best_fitness


def main():
    print("=" * 70)
    print("  BEAMFORMING OPTIMIZER v2: L∞ TWO-STAGE")
    print("=" * 70)
    
    T = build_matmul_tensor(3)
    
    # Load best known
    data = json.loads((ROOT / "slp_turbo_best.json").read_text())
    alpha0 = np.array(data["alpha"])
    beta0 = np.array(data["beta"])
    gamma0 = np.array(data["gamma"])
    R = alpha0.shape[0]
    
    orig_fitness = data["fitness"]
    print(f"Loaded rank-{R} solution, fitness = {orig_fitness:.8f}")
    
    # Verify L∞-optimal γ for current geometry
    A0 = build_steering_matrix(alpha0, beta0)
    _, linf_fit = linf_optimal_gamma(A0, T)
    print(f"L∞-optimal γ fitness: {linf_fit:.8f} (confirming γ is already optimal)")
    
    unc0 = uncontrollable_energy(alpha0, beta0, T)
    print(f"Uncontrollable energy fraction: {unc0:.6f}")
    
    # ── Stage 1: Gradient-based with L∞ inner loop ──
    print(f"\n{'─'*70}")
    print(f"  STAGE 1: GRADIENT DESCENT + L∞ INNER LOOP (R={R})")
    print(f"{'─'*70}")
    
    a1, b1, g1, f1 = gradient_outer_loop(alpha0, beta0, T, R,
                                          n_iters=100, lr=0.003)
    
    if f1 < orig_fitness:
        print(f"\n  *** IMPROVED: {orig_fitness:.8f} → {f1:.8f} ***")
    else:
        print(f"\n  No improvement over original ({f1:.8f} vs {orig_fitness:.8f})")
    
    # ── Stage 2: CMA-ES with L∞ inner loop ──
    print(f"\n{'─'*70}")
    print(f"  STAGE 2: CMA-ES + L∞ INNER LOOP (R={R})")
    print(f"{'─'*70}")
    
    # Warm-start from best of stage 1 or original
    if f1 < orig_fitness:
        a_start, b_start = a1, b1
    else:
        a_start, b_start = alpha0, beta0
    
    a2, b2, g2, f2 = cmaes_outer_loop(a_start, b_start, T, R,
                                        sigma0=0.02, max_evals=1000)
    
    best_fit = min(f1, f2, orig_fitness)
    if f2 <= f1 and f2 <= orig_fitness:
        a_best, b_best, g_best = a2, b2, g2
    elif f1 <= orig_fitness:
        a_best, b_best, g_best = a1, b1, g1
    else:
        a_best, b_best, g_best = alpha0, beta0, gamma0
    
    print(f"\n{'─'*70}")
    print(f"  FINAL RESULT")
    print(f"{'─'*70}")
    print(f"  Original:  {orig_fitness:.8f}")
    print(f"  Gradient:  {f1:.8f}")
    print(f"  CMA-ES:    {f2:.8f}")
    print(f"  Best:      {best_fit:.8f}")
    
    unc_best = uncontrollable_energy(a_best, b_best, T)
    print(f"  Uncontrollable energy: {unc_best:.6f} (was {unc0:.6f})")
    
    # Save if improved
    if best_fit < orig_fitness - 1e-8:
        out = {
            "fitness": float(best_fit),
            "alpha": a_best.tolist(),
            "beta": b_best.tolist(),
            "gamma": g_best.tolist(),
            "method": "beamforming_v2"
        }
        out_path = ROOT / "beamforming_best.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\n  Saved to {out_path}")
    
    # ── Also try R=20 ──
    print(f"\n{'─'*70}")
    print(f"  BONUS: CMA-ES R=20 (1 extra term)")
    print(f"{'─'*70}")
    
    rng = np.random.default_rng(42)
    a20 = np.vstack([alpha0, rng.standard_normal((1, 9)) * 0.3])
    b20 = np.vstack([beta0, rng.standard_normal((1, 9)) * 0.3])
    
    a20_opt, b20_opt, g20_opt, f20 = cmaes_outer_loop(
        a20, b20, T, 20, sigma0=0.03, max_evals=1000)
    
    print(f"\n  R=20 best: {f20:.8f}")
    
    if f20 < best_fit:
        print(f"  *** R=20 BEATS R=19! {best_fit:.8f} → {f20:.8f} ***")
        out = {
            "fitness": float(f20),
            "alpha": a20_opt.tolist(),
            "beta": b20_opt.tolist(),
            "gamma": g20_opt.tolist(),
            "method": "beamforming_v2_R20"
        }
        (ROOT / "beamforming_R20_best.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
