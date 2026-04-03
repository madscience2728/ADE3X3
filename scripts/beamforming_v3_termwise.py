#!/usr/bin/env python3
"""
Beamforming v3: Term-wise coordinate descent.

Instead of perturbing all 342 dims at once (which destroys the basin),
optimize ONE term (α_k, β_k) at a time — only 18 dims — while holding
the rest fixed. γ is recomputed optimally via LP each step.

This is like adjusting one antenna element at a time in a phased array.
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.optimize import linprog, minimize

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


def fitness_for_term(x, k, alpha, beta, T_target, n2=9):
    """Evaluate fitness when term k has (α_k, β_k) = x reshaped."""
    alpha_mod = alpha.copy()
    beta_mod = beta.copy()
    alpha_mod[k] = x[:n2]
    beta_mod[k] = x[n2:]
    A = build_steering_matrix(alpha_mod, beta_mod, n2)
    _, fit = linf_optimal_gamma(A, T_target, n2)
    return fit


def optimize_single_term(k, alpha, beta, T_target, n2=9,
                         method='nelder-mead', max_evals=200):
    """Optimize term k while holding others fixed."""
    x0 = np.concatenate([alpha[k], beta[k]])
    
    result = minimize(
        fitness_for_term, x0,
        args=(k, alpha, beta, T_target, n2),
        method=method,
        options={'maxfev': max_evals, 'xatol': 1e-6, 'fatol': 1e-9,
                 'adaptive': True}
    )
    
    alpha_new = alpha.copy()
    beta_new = beta.copy()
    alpha_new[k] = result.x[:n2]
    beta_new[k] = result.x[n2:]
    
    return alpha_new, beta_new, result.fun, result.nfev


def run_coordinate_descent(alpha, beta, T_target, R, n_rounds=5,
                           max_evals_per_term=150, label=""):
    """Full coordinate descent: cycle through all R terms multiple times."""
    best_fitness = None
    best_alpha = alpha.copy()
    best_beta = beta.copy()
    
    # Initial fitness
    A = build_steering_matrix(alpha, beta)
    _, init_fit = linf_optimal_gamma(A, T_target)
    best_fitness = init_fit
    print(f"  Initial fitness: {init_fit:.8f}")
    
    t0 = time.time()
    total_evals = 0
    
    for rnd in range(n_rounds):
        improved_any = False
        
        # Randomize term order each round
        order = np.random.permutation(R)
        
        for idx, k in enumerate(order):
            a_new, b_new, fit_new, nfev = optimize_single_term(
                k, best_alpha, best_beta, T_target,
                max_evals=max_evals_per_term
            )
            total_evals += nfev
            elapsed = time.time() - t0
            
            if fit_new < best_fitness - 1e-10:
                delta = best_fitness - fit_new
                print(f"  round {rnd+1} term {k:2d} ({nfev:3d} evals, {elapsed:6.1f}s): "
                      f"{best_fitness:.8f} → {fit_new:.8f}  Δ={delta:.2e} *")
                best_fitness = fit_new
                best_alpha = a_new
                best_beta = b_new
                improved_any = True
            elif (idx + 1) % 5 == 0:
                print(f"  round {rnd+1} term {k:2d} ({nfev:3d} evals, {elapsed:6.1f}s): "
                      f"no improvement (tried {fit_new:.8f})")
        
        elapsed = time.time() - t0
        print(f"  --- Round {rnd+1} done ({elapsed:.1f}s, {total_evals} evals): "
              f"best = {best_fitness:.8f} ---")
        
        if not improved_any:
            print(f"  No term improved in round {rnd+1}, stopping early.")
            break
    
    # Get final gamma
    A = build_steering_matrix(best_alpha, best_beta)
    best_gamma, _ = linf_optimal_gamma(A, T_target)
    
    elapsed = time.time() - t0
    print(f"\n  {label} done: {total_evals} evals in {elapsed:.1f}s")
    print(f"  Best fitness: {best_fitness:.8f}")
    
    return best_alpha, best_beta, best_gamma, best_fitness


def main():
    print("=" * 70)
    print("  BEAMFORMING v3: TERM-WISE COORDINATE DESCENT")
    print("=" * 70)
    
    T = build_matmul_tensor(3)
    
    data = json.loads((ROOT / "slp_turbo_best.json").read_text())
    alpha0 = np.array(data["alpha"])
    beta0 = np.array(data["beta"])
    R = alpha0.shape[0]
    
    print(f"Loaded rank-{R} solution, fitness = {data['fitness']:.8f}")
    
    # ── R=19 ──
    print(f"\n{'─'*70}")
    print(f"  R=19: TERM-WISE OPTIMIZATION")
    print(f"{'─'*70}")
    
    a19, b19, g19, f19 = run_coordinate_descent(
        alpha0, beta0, T, R,
        n_rounds=3, max_evals_per_term=150, label="R=19"
    )
    
    if f19 < data["fitness"] - 1e-8:
        out = {
            "fitness": float(f19),
            "alpha": a19.tolist(),
            "beta": b19.tolist(),
            "gamma": g19.tolist(),
            "method": "beamforming_v3_termwise"
        }
        (ROOT / "beamforming_v3_best.json").write_text(json.dumps(out, indent=2))
        print(f"  Saved to beamforming_v3_best.json")
    
    # ── R=20 ──
    print(f"\n{'─'*70}")
    print(f"  R=20: TERM-WISE OPTIMIZATION (warm-start + 1 extra)")
    print(f"{'─'*70}")
    
    # Start with best R=19 geometry + one random extra term
    rng = np.random.default_rng(123)
    a20 = np.vstack([a19, rng.standard_normal((1, 9)) * 0.3])
    b20 = np.vstack([b19, rng.standard_normal((1, 9)) * 0.3])
    
    # First: aggressively optimize just the new 20th term
    print(f"\n  Phase A: Optimize ONLY term 19 (the new one)...")
    a20_a, b20_a, fit20_a, nfev_a = optimize_single_term(
        19, a20, b20, T, max_evals=500
    )
    A_tmp = build_steering_matrix(a20_a, b20_a)
    _, fit_check = linf_optimal_gamma(A_tmp, T)
    print(f"  After optimizing new term: fitness = {fit_check:.8f} ({nfev_a} evals)")
    
    # Try several random seeds for the extra term
    print(f"\n  Phase B: Try 10 random seeds for term 19...")
    best_seed_fit = fit_check
    best_a20, best_b20 = a20_a, b20_a
    
    for seed in range(10):
        rng_s = np.random.default_rng(seed * 137 + 7)
        a20_try = np.vstack([a19, rng_s.standard_normal((1, 9)) * 0.5])
        b20_try = np.vstack([b19, rng_s.standard_normal((1, 9)) * 0.5])
        
        a20_s, b20_s, fit_s, nfev_s = optimize_single_term(
            19, a20_try, b20_try, T, max_evals=300
        )
        A_s = build_steering_matrix(a20_s, b20_s)
        _, fit_s_check = linf_optimal_gamma(A_s, T)
        
        tag = " ***" if fit_s_check < best_seed_fit else ""
        print(f"    seed {seed}: fitness = {fit_s_check:.8f}{tag}")
        
        if fit_s_check < best_seed_fit:
            best_seed_fit = fit_s_check
            best_a20 = a20_s
            best_b20 = b20_s
    
    print(f"\n  Best seed fitness: {best_seed_fit:.8f}")
    
    # Phase C: Full coordinate descent on best R=20
    print(f"\n  Phase C: Full coordinate descent on R=20...")
    a20_final, b20_final, g20_final, f20_final = run_coordinate_descent(
        best_a20, best_b20, T, 20,
        n_rounds=3, max_evals_per_term=150, label="R=20"
    )
    
    print(f"\n{'─'*70}")
    print(f"  SUMMARY")
    print(f"{'─'*70}")
    print(f"  Original R=19: {data['fitness']:.8f}")
    print(f"  Optimized R=19: {f19:.8f}")
    print(f"  Optimized R=20: {f20_final:.8f}")
    
    if f20_final < f19:
        print(f"  R=20 BEATS R=19 by {f19 - f20_final:.2e}")
    
    if f20_final < data["fitness"] - 1e-8:
        out = {
            "fitness": float(f20_final),
            "alpha": a20_final.tolist(),
            "beta": b20_final.tolist(),
            "gamma": g20_final.tolist(),
            "method": "beamforming_v3_termwise_R20"
        }
        (ROOT / "beamforming_v3_R20_best.json").write_text(json.dumps(out, indent=2))
        print(f"  Saved to beamforming_v3_R20_best.json")


if __name__ == "__main__":
    main()
