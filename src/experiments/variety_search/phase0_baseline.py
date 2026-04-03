#!/usr/bin/env python3
"""Phase 0: Baseline measurement of fiber-mode profiles.

Measures the canon-derived structural quantities for:
  1. slp_turbo_best.json (our 0.073 champion)
  2. All other warm-start candidates
  3. AlphaTensor R=23 (ground truth — known to saturate the bound)
  4. Several cold-start alternating-L∞ fixed points

For each candidate, computes:
  - fitness (L∞)
  - rank(Sigma), rank(Eta1|Eta2) = rank(H), rank(Delta)
  - rank(Nuisance) = rank([Eta1|Eta2|Delta])
  - Gamma nullity
  - Gamma·Sigma residual (should be 3I₉ for exact decomps)
  - Delta ⊂ span(H) residual: max|Delta - H @ lstsq(H, Delta)|
  - Conservation check: R + η_nullity vs n³

Output: variety_search/phase0_results.json + printed table
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db_optimizer.tensor import fiber_mode_decomposition, fitness_maxabs
from alternating_linf.solver import alternating_linf, als_init, compute_fitness


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


def measure_profile(alpha, beta, gamma, label=""):
    """Compute the full fiber-mode profile for a candidate."""
    R = alpha.shape[0]
    n = 3
    fit = compute_fitness(alpha, beta, gamma)
    fm = fiber_mode_decomposition(alpha, beta, gamma)

    # Build H = [Eta1 | Eta2]
    H = np.hstack([fm["Eta1"], fm["Eta2"]])  # (R, 18)
    H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))

    # Delta containment: project Delta onto span(H)
    Delta = fm["Delta"]  # (R, 54)
    if H_rank > 0:
        # lstsq: find X minimizing ||H @ X - Delta||
        X, _, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
        delta_proj_residual = float(np.max(np.abs(Delta - H @ X)))
        delta_fro_residual = float(np.linalg.norm(Delta - H @ X))
    else:
        delta_proj_residual = float(np.max(np.abs(Delta)))
        delta_fro_residual = float(np.linalg.norm(Delta))

    # Conservation law check
    eta_nullity = fm["gamma_nullity"] - H_rank
    # More precise: eta_nullity = dim(ker(Gamma)) - rank(H)
    # But really: eta_nullity = n²(n-1) - rank(H) when rank(H) = R - n²
    # Let's just measure it directly
    gamma_nullity = fm["gamma_nullity"]

    # Direct eta_nullity: dim of { w ∈ ker(Gamma) : w^T H = 0 }
    Gamma = fm["Gamma"]  # (9, R)
    # ker(Gamma): nullspace of Gamma
    U, S, Vt = np.linalg.svd(Gamma, full_matrices=True)
    null_start = int(np.sum(S > 1e-10))
    ker_gamma = Vt[null_start:].T  # (R, nullity) — columns are basis of ker(Gamma)

    if ker_gamma.shape[1] > 0:
        # Project H onto ker(Gamma): H_proj = ker_gamma @ ker_gamma^T @ H
        # eta_nullity = dim(ker(Gamma)) - rank(ker_gamma^T @ H)
        H_in_kernel = ker_gamma.T @ H  # (nullity, 18)
        H_proj_rank = int(np.linalg.matrix_rank(H_in_kernel, tol=1e-10))
        eta_nullity = gamma_nullity - H_proj_rank
    else:
        H_proj_rank = 0
        eta_nullity = 0

    conservation = R + eta_nullity

    return {
        "label": label,
        "R": R,
        "fitness": fit,
        "sigma_rank": fm["sigma_rank"],
        "H_rank": H_rank,
        "H_proj_rank": H_proj_rank,
        "nuisance_rank": fm["nuisance_rank"],
        "gamma_rank": fm["gamma_rank"],
        "gamma_nullity": gamma_nullity,
        "eta_nullity": eta_nullity,
        "conservation": conservation,
        "conservation_target": n ** 3,
        "gs_residual": fm["gs_residual"],
        "delta_containment_maxabs": delta_proj_residual,
        "delta_containment_fro": delta_fro_residual,
        "nuisance_budget_target": R - 9,
        "nuisance_budget_ok": fm["nuisance_rank"] <= (R - 9),
    }


def print_table(profiles):
    """Print a formatted comparison table."""
    header = (
        f"{'Label':<30s} {'R':>3s} {'Fitness':>12s} {'H_rank':>7s} "
        f"{'Nuis_rk':>8s} {'Budget':>7s} {'η_null':>7s} "
        f"{'R+η':>5s} {'Δ⊂H res':>10s} {'GΣ res':>10s}"
    )
    print(header)
    print("=" * len(header))
    for p in profiles:
        budget_str = "✓" if p["nuisance_budget_ok"] else "✗"
        print(
            f"{p['label']:<30s} {p['R']:>3d} {p['fitness']:>12.8f} {p['H_rank']:>7d} "
            f"{p['nuisance_rank']:>8d} {budget_str:>7s} {p['eta_nullity']:>7d} "
            f"{p['conservation']:>5d} {p['delta_containment_maxabs']:>10.6f} "
            f"{p['gs_residual']:>10.6f}"
        )


def _cold_worker(seed_int):
    rng = np.random.default_rng(seed_int)
    a, b, g = als_init(19, rng)
    a, b, g, fit, _ = alternating_linf(a, b, g, max_cycles=200, tol=1e-14)
    return a, b, g, fit, f"cold_ALS_altLinf_{seed_int - 42}"


def _warm_worker(path_str):
    a, b, g = load_factors(path_str)
    a, b, g, fit, _ = alternating_linf(a, b, g, max_cycles=200, tol=1e-15)
    return a, b, g, fit, "warm_altLinf_best"


def main():
    print("=" * 80)
    print("  Phase 0: Baseline Fiber-Mode Profiles")
    print("=" * 80)

    profiles = []

    # 1. Load warm-start candidates
    seed_files = [
        ("slp_turbo_best", ROOT / "slp_turbo_best.json"),
        ("slp_best", ROOT / "slp_best.json"),
        ("slp_turbo_top0", ROOT / "slp_turbo_best_top0.json"),
        ("slp_turbo_top1", ROOT / "slp_turbo_best_top1.json"),
        ("slp_turbo_top2", ROOT / "slp_turbo_best_top2.json"),
        ("chain_best", ROOT / "chain_best.json"),
        ("plaquette_best", ROOT / "plaquette_best.json"),
        ("shotgun_best", ROOT / "shotgun_best.json"),
    ]

    for label, path in seed_files:
        if path.exists():
            try:
                a, b, g = load_factors(path)
                p = measure_profile(a, b, g, label=label)
                profiles.append(p)
            except Exception as e:
                print(f"  Skip {label}: {e}")

    # 2. Cold-start alternating L∞ fixed points + warm start (parallel)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    N_COLD = 16
    N_WORKERS = min(24, os.cpu_count() or 4)
    print(f"\n  Computing {N_COLD} cold + 1 warm alternating-L∞ ({N_WORKERS} workers)...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for i in range(N_COLD):
            futures[pool.submit(_cold_worker, 42 + i)] = f"cold_{i}"
        warm_path = ROOT / "slp_turbo_best.json"
        if warm_path.exists():
            futures[pool.submit(_warm_worker, str(warm_path))] = "warm"

        for future in as_completed(futures):
            tag = futures[future]
            try:
                a, b, g, fit, label = future.result()
                p = measure_profile(a, b, g, label=label)
                profiles.append(p)
                print(f"    {label}: fitness={fit:.8f}")
            except Exception as e:
                print(f"    {tag} FAILED: {e}")

    # Print table
    print("\n")
    print_table(profiles)

    # Print observations
    print("\n" + "=" * 80)
    print("  Key observations:")
    for p in profiles:
        if p["nuisance_budget_ok"]:
            print(f"  ★ {p['label']}: nuisance_rank={p['nuisance_rank']} ≤ {p['nuisance_budget_target']} — ON BUDGET!")

    best = min(profiles, key=lambda x: x["fitness"])
    print(f"\n  Best fitness: {best['label']} at {best['fitness']:.10f}")
    print(f"  Its nuisance rank: {best['nuisance_rank']} (target ≤ {best['nuisance_budget_target']})")
    print(f"  Its H_rank: {best['H_rank']}")
    print(f"  Its Delta containment: {best['delta_containment_maxabs']:.8f}")
    print(f"  Conservation: {best['R']} + {best['eta_nullity']} = {best['conservation']} (target {best['conservation_target']})")

    # Save
    out_path = Path(__file__).resolve().parent / "phase0_results.json"
    with open(out_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
