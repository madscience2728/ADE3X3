#!/usr/bin/env python3
"""Alternating L∞ — 24-core multi-start parallel runner.

Each worker runs the alternating L∞ solver from a different starting point:
  - Worker 0:    warm start from slp_turbo_best.json (the 0.073 champion)
  - Workers 1-7: warm + small perturbation (σ = 0.001 → 0.01)
  - Workers 8-15: warm + medium perturbation (σ = 0.02 → 0.10)
  - Workers 16-23: cold ALS-converged random starts

Outputs:
  alternating_linf/plots/convergence.png   — per-worker fitness curves
  alternating_linf/plots/best_of_all.png   — best-so-far envelope
  alternating_linf/plots/summary.png       — final fitness bar chart
  alternating_linf/best_result.json        — best solution found
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alternating_linf.solver import (
    alternating_linf, als_init, compute_fitness, T,
)

RANK = int(os.environ.get("ADE_RANK", 19))
N_WORKERS = int(os.environ.get("ALT_LINF_WORKERS", min(24, os.cpu_count() or 4)))
MAX_CYCLES = int(os.environ.get("ALT_LINF_CYCLES", 500))
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_json_factors(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    if "alpha" in d:
        return (np.array(d["alpha"], dtype=np.float64),
                np.array(d["beta"], dtype=np.float64),
                np.array(d["gamma"], dtype=np.float64))
    # Sparse format
    terms = d["terms"]
    alpha = np.zeros((len(terms), 9), dtype=np.float64)
    beta = np.zeros((len(terms), 9), dtype=np.float64)
    gamma = np.zeros((len(terms), 9), dtype=np.float64)
    for i, t in enumerate(terms):
        for idx, val in zip(t["alpha_support"], t["alpha_values"]):
            alpha[i, idx] = val
        for idx, val in zip(t["beta_support"], t["beta_values"]):
            beta[i, idx] = val
        for idx, val in zip(t["gamma_support"], t["gamma_values"]):
            gamma[i, idx] = val
    return alpha, beta, gamma


def save_json_factors(path: str | Path, alpha, beta, gamma, fitness, meta=None):
    d = {
        "rank": int(alpha.shape[0]),
        "fitness": float(fitness),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
    }
    if meta:
        d["meta"] = meta
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ── Worker function (runs in subprocess) ─────────────────────────────

def worker_fn(args: tuple) -> dict:
    """Single worker: initialize → run alternating L∞ → return results."""
    worker_id, init_type, seed_data, sigma, seed_int = args

    rng = np.random.default_rng(seed_int)

    if init_type == "warm":
        alpha, beta, gamma = seed_data
        alpha = alpha + rng.normal(0, sigma, alpha.shape)
        beta = beta + rng.normal(0, sigma, beta.shape)
        gamma = gamma + rng.normal(0, sigma, gamma.shape)
    elif init_type == "warm_exact":
        alpha, beta, gamma = seed_data
        alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    else:  # cold
        alpha, beta, gamma = als_init(RANK, rng, max_iters=30)

    init_fit = compute_fitness(alpha, beta, gamma)

    t0 = time.time()
    alpha, beta, gamma, final_fit, history = alternating_linf(
        alpha, beta, gamma,
        max_cycles=MAX_CYCLES,
        tol=1e-15,
        verbose=False,
    )
    wall = time.time() - t0

    return {
        "worker_id": worker_id,
        "init_type": init_type,
        "sigma": sigma,
        "init_fitness": init_fit,
        "final_fitness": final_fit,
        "n_cycles": len(history) - 1,
        "wall_seconds": wall,
        "history": history,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def main():
    print(f"=" * 72)
    print(f"  Alternating L∞ — Multi-Start Parallel Runner")
    print(f"  Rank={RANK}, Workers={N_WORKERS}, Max cycles={MAX_CYCLES}")
    print(f"=" * 72)

    # Load best known solution for warm starts
    seed_files = [
        ROOT / "slp_turbo_best.json",
        ROOT / "slp_best.json",
        ROOT / "slp_best_at_0.074.json",
        ROOT / "slp_turbo_best_top0.json",
        ROOT / "slp_turbo_best_top1.json",
        ROOT / "slp_turbo_best_top2.json",
        ROOT / "chain_best.json",
        ROOT / "plaquette_best.json",
        ROOT / "shotgun_best.json",
    ]

    warm_factors = []
    for sf in seed_files:
        if sf.exists():
            try:
                a, b, g = load_json_factors(sf)
                if a.shape[0] == RANK:
                    fit = compute_fitness(a, b, g)
                    warm_factors.append((a, b, g, fit, sf.name))
                    print(f"  Loaded {sf.name}: fitness = {fit:.10f}")
            except Exception as e:
                print(f"  Skip {sf.name}: {e}")

    if not warm_factors:
        print("  WARNING: No warm-start files found. All workers will cold-start.")

    # Sort by fitness — best first
    warm_factors.sort(key=lambda x: x[3])

    # Build worker assignments
    tasks = []
    base_seed = int(time.time()) % (2**31)

    # Worker 0: exact warm start from best
    if warm_factors:
        best_a, best_b, best_g, best_fit, best_name = warm_factors[0]
        tasks.append((0, "warm_exact", (best_a, best_b, best_g), 0.0, base_seed))
        print(f"\n  Worker  0: warm_exact from {best_name} (fit={best_fit:.10f})")
    else:
        tasks.append((0, "cold", None, 0.0, base_seed))
        print(f"\n  Worker  0: cold ALS start")

    # Workers 1-7: warm + small perturbation
    sigmas_small = np.geomspace(0.001, 0.01, 7)
    for i, sigma in enumerate(sigmas_small, 1):
        if warm_factors:
            # Cycle through available warm starts
            wf = warm_factors[i % len(warm_factors)]
            tasks.append((i, "warm", (wf[0], wf[1], wf[2]), sigma, base_seed + i))
            print(f"  Worker {i:2d}: warm σ={sigma:.4f} from {wf[4]}")
        else:
            tasks.append((i, "cold", None, 0.0, base_seed + i))
            print(f"  Worker {i:2d}: cold ALS start")

    # Workers 8-15: warm + medium perturbation
    sigmas_med = np.geomspace(0.02, 0.10, 8)
    for i, sigma in enumerate(sigmas_med, 8):
        if warm_factors:
            wf = warm_factors[i % len(warm_factors)]
            tasks.append((i, "warm", (wf[0], wf[1], wf[2]), sigma, base_seed + i))
            print(f"  Worker {i:2d}: warm σ={sigma:.4f} from {wf[4]}")
        else:
            tasks.append((i, "cold", None, 0.0, base_seed + i))
            print(f"  Worker {i:2d}: cold ALS start")

    # Workers 16-23: cold ALS starts
    for i in range(16, N_WORKERS):
        tasks.append((i, "cold", None, 0.0, base_seed + i))
        print(f"  Worker {i:2d}: cold ALS start")

    print(f"\n  Launching {len(tasks)} workers...\n")

    # ── Run in parallel ──────────────────────────────────────────────
    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(worker_fn, task): task[0] for task in tasks}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                res = future.result()
                results.append(res)
                tag = f"W{res['worker_id']:2d}"
                print(f"  [{tag}] {res['init_type']:10s} σ={res['sigma']:.4f} "
                      f"| {res['init_fitness']:.8f} → {res['final_fitness']:.14f} "
                      f"| {res['n_cycles']} cycles in {res['wall_seconds']:.1f}s")
            except Exception as e:
                print(f"  [W{wid:2d}] FAILED: {e}")

    total_wall = time.time() - t_start
    print(f"\n  Total wall time: {total_wall:.1f}s")

    if not results:
        print("  No results. Exiting.")
        return

    # ── Find best ────────────────────────────────────────────────────
    results.sort(key=lambda r: r["final_fitness"])
    best = results[0]

    print(f"\n{'=' * 72}")
    print(f"  BEST: Worker {best['worker_id']} ({best['init_type']} σ={best['sigma']:.4f})")
    print(f"  Fitness: {best['init_fitness']:.10f} → {best['final_fitness']:.14f}")
    print(f"  Cycles: {best['n_cycles']}, Wall: {best['wall_seconds']:.1f}s")
    print(f"{'=' * 72}")

    # Save best result
    out_path = Path(__file__).resolve().parent / "best_result.json"
    save_json_factors(out_path, best["alpha"], best["beta"], best["gamma"],
                      best["final_fitness"],
                      meta={
                          "worker_id": best["worker_id"],
                          "init_type": best["init_type"],
                          "sigma": best["sigma"],
                          "n_cycles": best["n_cycles"],
                          "wall_seconds": best["wall_seconds"],
                          "method": "alternating_linf",
                      })
    print(f"  Saved to {out_path}")

    # ── Plots ────────────────────────────────────────────────────────
    print(f"\n  Generating plots...")

    # 1. Per-worker convergence curves
    fig, ax = plt.subplots(figsize=(14, 8))
    for res in results:
        label = f"W{res['worker_id']} ({res['init_type']} σ={res['sigma']:.3f}) → {res['final_fitness']:.6f}"
        ax.semilogy(res["history"], alpha=0.7, linewidth=1.2, label=label)
    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("L∞ Fitness (max |residual|)", fontsize=12)
    ax.set_title(f"Alternating L∞ — Per-Worker Convergence (R={RANK})", fontsize=14)
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "convergence.png", dpi=150)
    plt.close(fig)

    # 2. Best-of-all envelope
    max_len = max(len(r["history"]) for r in results)
    best_envelope = np.full(max_len, np.inf)
    for res in results:
        h = np.array(res["history"])
        best_envelope[:len(h)] = np.minimum(best_envelope[:len(h)], h)
    # Forward-fill
    for i in range(1, max_len):
        best_envelope[i] = min(best_envelope[i], best_envelope[i - 1])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(best_envelope, color='crimson', linewidth=2)
    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Best L∞ Fitness", fontsize=12)
    ax.set_title(f"Alternating L∞ — Best-of-All Envelope (R={RANK})", fontsize=14)
    ax.axhline(y=0.073, color='gray', linestyle='--', alpha=0.5, label="Previous best (0.073)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "best_of_all.png", dpi=150)
    plt.close(fig)

    # 3. Summary bar chart
    results.sort(key=lambda r: r["worker_id"])
    fig, ax = plt.subplots(figsize=(14, 6))
    wids = [r["worker_id"] for r in results]
    finals = [r["final_fitness"] for r in results]
    colors = []
    for r in results:
        if r["init_type"] == "warm_exact":
            colors.append("#2ecc71")
        elif r["init_type"] == "warm":
            colors.append("#3498db")
        else:
            colors.append("#e74c3c")
    ax.bar(range(len(wids)), finals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(wids)))
    ax.set_xticklabels([f"W{w}" for w in wids], fontsize=8, rotation=45)
    ax.set_ylabel("Final L∞ Fitness", fontsize=12)
    ax.set_title(f"Alternating L∞ — Final Fitness per Worker (R={RANK})", fontsize=14)
    ax.axhline(y=0.073, color='gray', linestyle='--', alpha=0.5, label="Previous best (0.073)")
    # Legend for colors
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ecc71", label="Warm (exact)"),
        Patch(color="#3498db", label="Warm (perturbed)"),
        Patch(color="#e74c3c", label="Cold (ALS)"),
        Patch(color="gray", linestyle="--", label="Previous best"),
    ], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {PLOTS_DIR}/")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
