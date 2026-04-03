#!/usr/bin/env python3
"""Massive cold-start basin survey: 1000+ ALS→alternating L∞ on 24 cores.

Continuously generates random ALS inits, refines via alternating L∞,
logs every result, and tracks the best. Runs until killed.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alternating_linf.solver import alternating_linf, als_init, compute_fitness

RANK = int(os.environ.get("ADE_RANK", 19))
N_WORKERS = int(os.environ.get("ALT_LINF_WORKERS", min(24, os.cpu_count() or 4)))
MAX_CYCLES = 500
OUT_DIR = Path(__file__).resolve().parent
LOG_FILE = OUT_DIR / "basin_survey.jsonl"
BEST_FILE = OUT_DIR / "survey_best.json"


def worker_fn(args):
    worker_id, seed_int = args
    rng = np.random.default_rng(seed_int)
    alpha, beta, gamma = als_init(RANK, rng, max_iters=30)
    init_fit = compute_fitness(alpha, beta, gamma)

    t0 = time.time()
    alpha, beta, gamma, final_fit, history = alternating_linf(
        alpha, beta, gamma, max_cycles=MAX_CYCLES, tol=1e-15, verbose=False,
    )
    wall = time.time() - t0

    return {
        "id": worker_id,
        "init_fitness": init_fit,
        "final_fitness": final_fit,
        "n_cycles": len(history) - 1,
        "wall": wall,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def save_best(result, total_done):
    d = {
        "rank": RANK,
        "fitness": result["final_fitness"],
        "alpha": result["alpha"].tolist(),
        "beta": result["beta"].tolist(),
        "gamma": result["gamma"].tolist(),
        "meta": {
            "method": "alternating_linf_survey",
            "worker_id": result["id"],
            "init_fitness": result["init_fitness"],
            "n_cycles": result["n_cycles"],
            "total_surveyed": total_done,
        },
    }
    with open(BEST_FILE, "w") as f:
        json.dump(d, f, indent=2)


def main():
    print(f"Basin Survey: R={RANK}, {N_WORKERS} workers, {MAX_CYCLES} cycles/run")
    print(f"Logging to {LOG_FILE}")
    print(f"Running until killed (Ctrl+C).\n")

    best_fit = float("inf")
    total_done = 0
    batch_size = N_WORKERS * 2  # keep all cores busy
    base_seed = int(time.time() * 1000) % (2**31)

    # Fitness histogram bins
    bins = [0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.0, float("inf")]
    bin_counts = Counter()

    t_start = time.time()

    with open(LOG_FILE, "a") as log_f:
        try:
            while True:
                tasks = [(base_seed + total_done + i, base_seed + total_done + i)
                         for i in range(batch_size)]

                with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                    futures = {pool.submit(worker_fn, t): t[0] for t in tasks}
                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            total_done += 1
                            fit = res["final_fitness"]

                            # Bin it
                            for j in range(len(bins) - 1):
                                if fit < bins[j + 1]:
                                    bin_counts[f"{bins[j]:.2f}-{bins[j+1]:.2f}" if bins[j+1] != float("inf") else f"{bins[j]:.2f}+"] += 1
                                    break

                            # Log
                            log_entry = {
                                "id": res["id"],
                                "init_fit": round(res["init_fitness"], 8),
                                "final_fit": round(fit, 14),
                                "cycles": res["n_cycles"],
                                "wall": round(res["wall"], 1),
                            }
                            log_f.write(json.dumps(log_entry) + "\n")
                            log_f.flush()

                            # New best?
                            if fit < best_fit:
                                best_fit = fit
                                save_best(res, total_done)
                                print(f"  *** NEW BEST: {fit:.14f} (run #{total_done}) ***")

                            # Periodic summary
                            if total_done % N_WORKERS == 0:
                                elapsed = time.time() - t_start
                                rate = total_done / elapsed * 60
                                print(f"  [{total_done:5d} done | {elapsed/60:.1f}min | "
                                      f"{rate:.0f}/min] best={best_fit:.10f} "
                                      f"last={fit:.6f} | {dict(sorted(bin_counts.items()))}")

                        except Exception as e:
                            print(f"  Worker error: {e}")
                            total_done += 1

        except KeyboardInterrupt:
            print(f"\n\nStopped after {total_done} runs.")
            print(f"Best fitness: {best_fit:.14f}")
            print(f"Distribution: {dict(sorted(bin_counts.items()))}")


if __name__ == "__main__":
    main()
