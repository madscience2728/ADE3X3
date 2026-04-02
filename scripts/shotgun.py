"""Shotgun search: fire many independent random→ALS→refine trajectories in parallel.

Instead of nursing one candidate, this generates N random starts per round,
runs each through the full CPU pipeline, and keeps the global best.
Capitalises on the observation that one-shots from different basins
sometimes outperform refinement of an existing good candidate.

Usage:
    python scripts/shotgun.py                         # 24 workers, infinite rounds
    python scripts/shotgun.py --workers 16 --rounds 100
    python scripts/shotgun.py --resume                # continue from shotgun_best.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR
from db_optimizer.cpu_refine import (
    cpu_als_init, cpu_minimax_refine, cpu_pair_refine,
    cpu_lbfgs_refine, cpu_algebraic_snap,
)


def fitness(a, b, g):
    R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
    return float(np.max(np.abs(R)))


def random_factors(rng):
    """Random starting point (Gaussian, scale ~0.3)."""
    a = rng.normal(0, 0.3, (RANK, DIM))
    b = rng.normal(0, 0.3, (RANK, DIM))
    g = rng.normal(0, 0.3, (RANK, DIM))
    return a, b, g


def one_shot_refine(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """One complete random→ALS→heavy_refine trajectory. Runs in a worker process."""
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Stage 0: Random init → ALS basin
    a, b, g = random_factors(rng)
    a, b, g, fit = cpu_als_init(a, b, g, max_iters=50)

    # Stage 1: Coordinate descent (long)
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=20, fine_range=0.003, patience=5)

    # Stage 2: Pair moves
    a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=3, fine_range=0.009, n_pairs=60)

    # Stage 3: L-BFGS (aggressive, with high-beta)
    a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=400,
                                     beta_schedule=(10, 30, 100, 300, 1000, 3000))

    # Stage 4: Post-LBFGS coordinate descent
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=10, fine_range=0.001, patience=5)

    # Stage 5: Pair moves again
    a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=2, fine_range=0.005, n_pairs=40)

    # Stage 6: Second L-BFGS pass (tighter betas)
    if fit < 0.15:
        a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=300,
                                         beta_schedule=(100, 300, 1000, 3000, 10000))

    # Stage 7: Final polish
    a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=5, fine_range=0.0005, patience=3)

    # Stage 8: Algebraic snap
    if fit < 0.15:
        a, b, g, fit = cpu_algebraic_snap(a, b, g, k=8)

    dt = time.time() - t0
    return a, b, g, fit, dt


def save_result(a, b, g, fit, path):
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


def load_from_json(path):
    with open(path) as f:
        data = json.load(f)
    return (np.array(data["alpha"]), np.array(data["beta"]), np.array(data["gamma"])), data["fitness"]


def main():
    parser = argparse.ArgumentParser(description="Shotgun multi-start search")
    parser.add_argument("--workers", type=int, default=max(1, __import__('os').cpu_count() - 2),
                        help="Number of parallel workers")
    parser.add_argument("--batch", type=int, default=0,
                        help="Shots per round (default: 2× workers)")
    parser.add_argument("--rounds", type=int, default=999999,
                        help="Number of rounds (-1 = infinite)")
    parser.add_argument("--out", type=Path, default=Path("shotgun_best.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    batch = args.batch if args.batch > 0 else args.workers * 2

    if args.resume and args.out.exists():
        (best_a, best_b, best_g), best_fit = load_from_json(args.out)
        print(f"Resumed from {args.out}: fitness={best_fit:.10f}")
    else:
        best_a = best_b = best_g = None
        best_fit = float('inf')

    total_shots = 0
    master_rng = np.random.default_rng(int(time.time()))

    print(f"Shotgun search: {args.workers} workers, {batch} shots/round")
    print(f"{'='*70}")

    for round_i in range(args.rounds):
        t0 = time.time()
        seeds = [int(master_rng.integers(0, 2**62)) for _ in range(batch)]

        round_best_fit = float('inf')
        round_best = None
        fits = []

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_shot_refine, s): s for s in seeds}
            for fut in as_completed(futures):
                try:
                    a, b, g, fit, dt = fut.result()
                    fits.append(fit)
                    if fit < round_best_fit:
                        round_best_fit = fit
                        round_best = (a, b, g)
                except Exception as e:
                    print(f"  Worker error: {e}")

        total_shots += len(fits)
        dt_round = time.time() - t0

        if fits:
            fits_arr = np.array(fits)
            med = float(np.median(fits_arr))
            pct10 = float(np.percentile(fits_arr, 10))

            improved = round_best_fit < best_fit - 1e-12
            if improved:
                best_a, best_b, best_g = round_best
                best_fit = round_best_fit
                save_result(best_a, best_b, best_g, best_fit, args.out)

            tag = " *** NEW BEST ***" if improved else ""
            print(
                f"Round {round_i}: best={round_best_fit:.8f}  "
                f"med={med:.4f}  p10={pct10:.4f}  "
                f"global={best_fit:.8f}  "
                f"shots={total_shots}  {dt_round:.1f}s{tag}"
            )

    print(f"\n{'='*70}")
    print(f"Final best: {best_fit:.10f} after {total_shots} shots")
    if args.out.exists():
        print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
