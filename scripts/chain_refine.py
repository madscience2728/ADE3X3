"""Chain-refine: pull the best candidate and hammer it through the CPU pipeline repeatedly.

This mimics the manual "feed output back as input" approach that reached 0.098.
Each iteration runs the full refine pipeline, then perturbs and re-refines to
escape local minima.

Usage:
    python scripts/chain_refine.py                          # from DB best
    python scripts/chain_refine.py --db K:/ade3x3_optimizer/candidates.db
    python scripts/chain_refine.py --rounds 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR
from db_optimizer.cpu_refine import (
    cpu_minimax_refine, cpu_pair_refine, cpu_lbfgs_refine,
    cpu_algebraic_snap, cpu_full_refine,
)
from db_optimizer.blob import blob_to_factors, factors_to_blob


def load_best_from_db(db_path: Path):
    """Load the best candidate from the SQLite DB."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT factors_blob FROM candidates
           WHERE fitness_fp32 IS NOT NULL
           ORDER BY COALESCE(minimax_improved, fitness_fp32) ASC
           LIMIT 1"""
    ).fetchone()
    conn.close()
    if row is None:
        raise RuntimeError("No scored candidates in DB")
    return blob_to_factors(row["factors_blob"])


def load_from_json(path: Path):
    """Load factors from a chain_best.json file."""
    with open(path) as f:
        data = json.load(f)
    return (np.array(data["alpha"]), np.array(data["beta"]), np.array(data["gamma"]))


def fitness(a, b, g):
    R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
    return float(np.max(np.abs(R)))


def perturb(a, b, g, rng, sigma=0.01, n_coeffs=5):
    """Small random perturbation to escape local minimum."""
    a, b, g = a.copy(), b.copy(), g.copy()
    factors = [a, b, g]
    for _ in range(n_coeffs):
        fi = rng.integers(3)
        ri = rng.integers(RANK)
        di = rng.integers(DIM)
        factors[fi][ri, di] += rng.normal(0, sigma)
    return a, b, g


def heavy_refine(a, b, g):
    """One heavy refinement pass — more aggressive than cpu_full_refine."""
    fit = fitness(a, b, g)

    # Stage 1: Long coordinate descent
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=30, fine_range=0.003, patience=10)

    # Stage 2: Pair moves
    a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=4, fine_range=0.009, n_pairs=60)

    # Stage 3: L-BFGS (aggressive)
    a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=500,
                                     beta_schedule=(10, 30, 100, 300, 1000, 3000))

    # Stage 4: Post-L-BFGS coordinate descent
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=15, fine_range=0.001, patience=8)

    # Stage 5: Pair moves again
    a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=3, fine_range=0.005, n_pairs=60)

    # Stage 6: Re-run L-BFGS with tighter beta
    a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=300,
                                     beta_schedule=(100, 300, 1000, 3000, 10000))

    # Stage 7: Final coord descent polish
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=10, fine_range=0.0005, patience=5)

    # Stage 8: Algebraic snap
    a, b, g, fit = cpu_algebraic_snap(a, b, g, k=8)

    return a, b, g, fit


def save_result(a, b, g, fit, path):
    """Save factors as JSON."""
    data = {
        "fitness": fit,
        "rank": RANK,
        "dim": DIM,
        "alpha": a.tolist(),
        "beta": b.tolist(),
        "gamma": g.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Chain-refine best candidate")
    parser.add_argument("--db", type=Path, default=Path("K:/ade3x3_optimizer/candidates.db"))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("chain_best.json"))
    parser.add_argument("--sigma", type=float, default=0.008,
                        help="Perturbation sigma for escape moves")
    parser.add_argument("--n-perturb", type=int, default=5,
                        help="Number of coefficients to perturb per escape attempt")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from --out file instead of DB")
    args = parser.parse_args()

    if args.resume and args.out.exists():
        print(f"Resuming from {args.out} ...")
        a, b, g = load_from_json(args.out)
    else:
        print(f"Loading best candidate from {args.db} ...")
        a, b, g = load_best_from_db(args.db)
    fit = fitness(a, b, g)
    print(f"Starting fitness: {fit:.10f}")

    best_a, best_b, best_g, best_fit = a.copy(), b.copy(), g.copy(), fit
    rng = np.random.default_rng(42)

    # Initial heavy refine of the best
    print(f"\n{'='*60}")
    print(f"Round 0: Initial heavy refine")
    t0 = time.time()
    a, b, g, fit = heavy_refine(a, b, g)
    dt = time.time() - t0
    print(f"  {fit:.10f}  ({dt:.1f}s)")
    if fit < best_fit:
        best_a, best_b, best_g, best_fit = a.copy(), b.copy(), g.copy(), fit
        save_result(best_a, best_b, best_g, best_fit, args.out)
        print(f"  *** NEW BEST: {best_fit:.10f} (saved to {args.out})")

    stale = 0
    strategies = ["gauss", "gauss_wide", "zero_one", "swap"]
    for round_i in range(1, args.rounds + 1):
        t0 = time.time()

        # Adaptive perturbation: increase sigma and n_coeffs when stuck
        sigma = args.sigma * (1.0 + stale * 0.5)
        n_perturb = args.n_perturb + stale * 2

        # Rotate strategies
        strat = strategies[round_i % len(strategies)]
        if strat == "gauss":
            ca, cb, cg = perturb(best_a, best_b, best_g, rng, sigma=sigma, n_coeffs=n_perturb)
        elif strat == "gauss_wide":
            ca, cb, cg = perturb(best_a, best_b, best_g, rng, sigma=sigma * 3, n_coeffs=max(3, n_perturb // 2))
        elif strat == "zero_one":
            ca, cb, cg = best_a.copy(), best_b.copy(), best_g.copy()
            fi, ri, di = rng.integers(3), rng.integers(RANK), rng.integers(DIM)
            [ca, cb, cg][fi][ri, di] = 0.0
        elif strat == "swap":
            ca, cb, cg = best_a.copy(), best_b.copy(), best_g.copy()
            r1, r2 = rng.choice(RANK, size=2, replace=False)
            fi_swap = rng.integers(3)
            [ca, cb, cg][fi_swap][[r1, r2]] = [ca, cb, cg][fi_swap][[r2, r1]]
            ca, cb, cg = perturb(ca, cb, cg, rng, sigma=sigma * 0.5, n_coeffs=3)

        ca, cb, cg, cfit = heavy_refine(ca, cb, cg)
        dt = time.time() - t0

        if cfit < best_fit - 1e-12:
            best_a, best_b, best_g, best_fit = ca.copy(), cb.copy(), cg.copy(), cfit
            save_result(best_a, best_b, best_g, best_fit, args.out)
            stale = 0
            print(f"Round {round_i}: {best_fit:.10f}  *** NEW BEST via {strat} ({dt:.1f}s)")
        else:
            stale += 1
            print(f"Round {round_i}: {cfit:.10f} no improvement (best={best_fit:.10f}, stale={stale}, {strat}, {dt:.1f}s)")

        # Every 10 rounds, also refine best directly (no perturbation)
        if round_i % 10 == 0:
            t0 = time.time()
            a, b, g, fit = heavy_refine(best_a, best_b, best_g)
            dt = time.time() - t0
            if fit < best_fit - 1e-12:
                best_a, best_b, best_g, best_fit = a, b, g, fit
                save_result(best_a, best_b, best_g, best_fit, args.out)
                stale = 0
                print(f"  Direct re-refine: {best_fit:.10f} *** NEW BEST ({dt:.1f}s)")

    print(f"\n{'='*60}")
    print(f"Final best: {best_fit:.10f}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
