#!/usr/bin/env python3
"""Phase 1: Constrain rank(H) via L-BFGS-B on composite objective.

Strategy: Smooth optimization over all 513 = 19*27 variables at once.
  L(α,β,γ) = ||R||²_F  +  λ · Σ_{i>10} σᵢ(H)²

where R = Σ_k α_k⊗β_k⊗γ_k - T  and  H = [Eta1|Eta2](α,β).

Gradients are analytic:
  ∂||R||²_F/∂α, ∂β, ∂γ via einsum
  ∂(rank proxy)/∂α, ∂β via H bilinear chain rule

Multi-start: 24 workers sweeping λ and initialization.
"""

import json
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.optimize import minimize

from alternating_linf.solver import (
    alternating_linf, als_init, compute_fitness,
    _solve_gamma_col, T,
)


# ── H construction and rank proxy ────────────────────────────────────

def compute_H(alpha, beta):
    """Build H = [Eta1 | Eta2], shape (R, 18), bilinear in (α, β)."""
    R = alpha.shape[0]
    n = 3
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    for k in range(R):
        for r in range(n):
            for u in range(n):
                idx = n * r + u
                s0 = alpha[k, 3 * r + 0] * beta[k, 0 + u]
                s1 = alpha[k, 3 * r + 1] * beta[k, 3 + u]
                s2 = alpha[k, 3 * r + 2] * beta[k, 6 + u]
                Eta1[k, idx] = s0 - s1
                Eta2[k, idx] = s1 - s2
    return np.hstack([Eta1, Eta2])


def rank_tail_energy(H, target_rank=10):
    """Σ_{i > target_rank} σᵢ²  — energy in the tail singular values."""
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    if len(S) <= target_rank:
        return 0.0, U, S, Vt
    tail_energy = float(np.sum(S[target_rank:] ** 2))
    return tail_energy, U, S, Vt


def grad_rank_penalty(alpha, beta, H_tail):
    """Gradient of rank_tail_energy w.r.t. (α, β) via chain rule.

    ∂f/∂H = 2 * H_tail  (the rank-10 truncated residual of H).
    H_tail columns 0:9 = dEta1, columns 9:18 = dEta2.

    Returns (grad_alpha, grad_beta) each shape (R, 9).
    """
    R = alpha.shape[0]
    n = 3
    dEta1 = 2.0 * H_tail[:, :9]   # (R, 9)
    dEta2 = 2.0 * H_tail[:, 9:]   # (R, 9)

    grad_a = np.zeros_like(alpha)
    grad_b = np.zeros_like(beta)

    for k in range(R):
        for r in range(n):
            for u in range(n):
                idx = n * r + u
                d1 = dEta1[k, idx]
                d2 = dEta2[k, idx]

                # Eta1[k,idx] = a[k,3r+0]*b[k,u] - a[k,3r+1]*b[k,3+u]
                # ∂Eta1/∂a[k,3r+0] = b[k,u]
                grad_a[k, 3 * r + 0] += d1 * beta[k, u]
                # ∂Eta1/∂a[k,3r+1] = -b[k,3+u]
                grad_a[k, 3 * r + 1] += d1 * (-beta[k, 3 + u])
                # ∂Eta1/∂b[k,u] = a[k,3r+0]
                grad_b[k, u] += d1 * alpha[k, 3 * r + 0]
                # ∂Eta1/∂b[k,3+u] = -a[k,3r+1]
                grad_b[k, 3 + u] += d1 * (-alpha[k, 3 * r + 1])

                # Eta2[k,idx] = a[k,3r+1]*b[k,3+u] - a[k,3r+2]*b[k,6+u]
                grad_a[k, 3 * r + 1] += d2 * beta[k, 3 + u]
                grad_a[k, 3 * r + 2] += d2 * (-beta[k, 6 + u])
                grad_b[k, 3 + u] += d2 * alpha[k, 3 * r + 1]
                grad_b[k, 6 + u] += d2 * (-alpha[k, 3 * r + 2])

    return grad_a, grad_b


# ── Penalized alternating solver ─────────────────────────────────────

def _fitness_grad_ab(alpha, beta, gamma):
    """Analytic gradient of ||R||²_F w.r.t. (α, β, γ)."""
    residual = np.einsum('ka,kb,kc->abc', alpha, beta, gamma) - T
    ga = 2.0 * np.einsum('abc,kb,kc->ka', residual, beta, gamma)
    gb = 2.0 * np.einsum('abc,ka,kc->kb', residual, alpha, gamma)
    gc = 2.0 * np.einsum('abc,ka,kb->kc', residual, alpha, beta)
    return ga, gb, gc


def composite_objective_and_grad(x, R, lam, target_rank):
    """L-BFGS-B objective: ||R_tensor||²_F + λ * Σ_{i>target_rank} σᵢ(H)².

    x: flat vector of length R*27 = [alpha.ravel(), beta.ravel(), gamma.ravel()]
    Returns (loss, grad_flat).
    """
    alpha = x[:R * 9].reshape(R, 9)
    beta = x[R * 9:R * 18].reshape(R, 9)
    gamma = x[R * 18:].reshape(R, 9)

    # Fitness term: ||R||²_F
    residual = np.einsum('ka,kb,kc->abc', alpha, beta, gamma) - T
    fro2 = float(np.sum(residual ** 2))

    ga_fit = 2.0 * np.einsum('abc,kb,kc->ka', residual, beta, gamma)
    gb_fit = 2.0 * np.einsum('abc,ka,kc->kb', residual, alpha, gamma)
    gc_fit = 2.0 * np.einsum('abc,ka,kb->kc', residual, alpha, beta)

    # Rank term: Σ_{i>target_rank} σᵢ(H)²
    H = compute_H(alpha, beta)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    tail_energy = float(np.sum(S[target_rank:] ** 2)) if len(S) > target_rank else 0.0

    # Gradient of tail energy w.r.t. H
    S_trunc = S.copy()
    S_trunc[:target_rank] = 0.0
    H_tail = U @ np.diag(S_trunc) @ Vt  # (R, 18)

    ga_rank, gb_rank = grad_rank_penalty(alpha, beta, H_tail)

    loss = fro2 + lam * tail_energy
    grad = np.concatenate([
        (ga_fit + lam * ga_rank).ravel(),
        (gb_fit + lam * gb_rank).ravel(),
        gc_fit.ravel(),
    ])
    return loss, grad


def run_lbfgsb(alpha, beta, gamma, lam=10.0, target_rank=10, maxiter=2000):
    """Run L-BFGS-B on the composite objective, then LP-polish γ.

    Returns (alpha, beta, gamma, best_fitness, H_rank, info_dict).
    """
    R = alpha.shape[0]
    x0 = np.concatenate([alpha.ravel(), beta.ravel(), gamma.ravel()])

    result = minimize(
        composite_objective_and_grad,
        x0,
        args=(R, lam, target_rank),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'maxfun': maxiter * 3, 'ftol': 1e-15, 'gtol': 1e-10},
    )

    alpha = result.x[:R * 9].reshape(R, 9)
    beta = result.x[R * 9:R * 18].reshape(R, 9)
    gamma = result.x[R * 18:].reshape(R, 9)

    # LP-polish γ (exact optimal for fixed α,β — doesn't touch H)
    for c in range(9):
        x = _solve_gamma_col(c, alpha, beta)
        if x is not None:
            gamma[:, c] = x

    fit = compute_fitness(alpha, beta, gamma)
    H = compute_H(alpha, beta)
    h_rank = int(np.linalg.matrix_rank(H, tol=1e-10))

    return alpha, beta, gamma, fit, h_rank, {
        "nit": result.nit,
        "nfev": result.nfev,
        "success": result.success,
        "final_loss": float(result.fun),
    }


# ── Load factors ─────────────────────────────────────────────────────

def load_factors(path):
    with open(path) as f:
        d = json.load(f)
    if "alpha" in d:
        return (np.array(d["alpha"], dtype=np.float64),
                np.array(d["beta"], dtype=np.float64),
                np.array(d["gamma"], dtype=np.float64))
    terms = d["terms"]
    R = len(terms)
    alpha = np.zeros((R, 9), dtype=np.float64)
    beta = np.zeros((R, 9), dtype=np.float64)
    gamma = np.zeros((R, 9), dtype=np.float64)
    for i, t in enumerate(terms):
        for idx, val in zip(t["alpha_support"], t["alpha_values"]):
            alpha[i, idx] = val
        for idx, val in zip(t["beta_support"], t["beta_values"]):
            beta[i, idx] = val
        for idx, val in zip(t["gamma_support"], t["gamma_values"]):
            gamma[i, idx] = val
    return alpha, beta, gamma


# ── Worker configs ───────────────────────────────────────────────────
# Sweep λ and initialization strategies across 24 workers:
#   Workers 0-7:   warm start (perturbed), varying λ
#   Workers 8-15:  warm start, varying rank_lr
#   Workers 16-23: cold ALS start, moderate λ

WORKER_CONFIGS = []
# Group A: warm start, sweep λ
for i, lam in enumerate([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]):
    WORKER_CONFIGS.append({"id": i, "init": "warm", "sigma": 0.0, "lam": lam})
# Group B: warm+perturbation, moderate-to-high λ
for i, (sig, lam) in enumerate([
    (0.01, 1.0), (0.01, 10.0), (0.01, 100.0), (0.01, 1000.0),
    (0.1, 1.0), (0.1, 10.0), (0.1, 100.0), (0.1, 1000.0),
]):
    WORKER_CONFIGS.append({"id": 8 + i, "init": "warm", "sigma": sig, "lam": lam})
# Group C: cold ALS, high λ
for i in range(8):
    WORKER_CONFIGS.append({"id": 16 + i, "init": "cold", "sigma": 0.0, "lam": 100.0})


def _worker(cfg_json):
    """Worker function for ProcessPoolExecutor."""
    cfg = json.loads(cfg_json)
    wid = cfg["id"]
    warm_path = str(ROOT / "slp_turbo_best.json")

    if cfg["init"] == "warm":
        a, b, g = load_factors(warm_path)
        if cfg["sigma"] > 0:
            rng = np.random.default_rng(1000 + wid)
            a += rng.normal(0, cfg["sigma"], a.shape)
            b += rng.normal(0, cfg["sigma"], b.shape)
    else:
        rng = np.random.default_rng(2000 + wid)
        a, b, g = als_init(19, rng)

    a, b, g, best_fit, h_rank, info = run_lbfgsb(
        a, b, g, lam=cfg["lam"], maxiter=3000,
    )

    return {
        "worker": wid,
        "init": cfg["init"],
        "sigma": cfg["sigma"],
        "lam": cfg["lam"],
        "fitness": best_fit,
        "H_rank": h_rank,
        "nit": info["nit"],
        "final_loss": info["final_loss"],
        "alpha": a.tolist(),
        "beta": b.tolist(),
        "gamma": g.tolist(),
    }


def main():
    print("=" * 80)
    print("  Phase 1: Penalized Alternating L∞ — rank(H) ≤ 10")
    print("=" * 80)

    N_WORKERS = min(24, os.cpu_count() or 4)
    print(f"\n  {len(WORKER_CONFIGS)} workers across {N_WORKERS} cores")
    print(f"  Group A (0-7):   warm, sweep λ ∈ [0.1, 300]")
    print(f"  Group B (8-15):  warm+noise, sweep σ×λ")
    print(f"  Group C (16-23): cold ALS, λ=100")
    print(f"  Method: L-BFGS-B on ||R||²_F + λ·Σ_{{i>10}} σᵢ(H)², then LP-polish γ")

    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for cfg in WORKER_CONFIGS:
            cfg_json = json.dumps(cfg)
            futures[pool.submit(_worker, cfg_json)] = cfg["id"]

        for fut in as_completed(futures):
            wid = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"    W{r['worker']:02d} ({r['init']:>4s} σ={r['sigma']:<5.3f} λ={r['lam']:<7.1f}): "
                      f"fitness={r['fitness']:.8f}  H_rank={r['H_rank']}  iters={r['nit']}")
            except Exception as e:
                print(f"    W{wid:02d}: FAILED — {e}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Sort by fitness
    results.sort(key=lambda r: r["fitness"])

    print(f"\n{'Worker':>8s} {'Init':>5s} {'σ':>6s} {'λ':>8s} {'Fitness':>12s} {'H_rank':>7s} {'Iters':>6s}")
    print("=" * 60)
    for r in results:
        print(f"  W{r['worker']:02d}    {r['init']:>5s} {r['sigma']:>6.3f} {r['lam']:>8.1f} "
              f"{r['fitness']:>12.8f} {r['H_rank']:>7d} {r['nit']:>6d}")

    # Save best
    best = results[0]
    print(f"\n  Best: W{best['worker']:02d} fitness={best['fitness']:.10f} H_rank={best['H_rank']}")

    out_path = ROOT / "variety_search" / "phase1_results.json"
    # Save summary (without full factor matrices for the table)
    summary = [{k: v for k, v in r.items() if k not in ("alpha", "beta", "gamma")}
               for r in results]
    with open(out_path, "w") as f:
        json.dump({"results": summary, "elapsed_s": elapsed}, f, indent=2)
    print(f"  Saved to {out_path}")

    # Save best factors
    best_path = ROOT / "variety_search" / "phase1_best.json"
    with open(best_path, "w") as f:
        json.dump({
            "alpha": best["alpha"],
            "beta": best["beta"],
            "gamma": best["gamma"],
            "fitness": best["fitness"],
            "H_rank": best["H_rank"],
        }, f)
    print(f"  Best factors saved to {best_path}")


if __name__ == "__main__":
    main()
