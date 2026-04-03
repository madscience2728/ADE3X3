#!/usr/bin/env python3
"""Timed continuation search with Delta pressure.

Implements three follow-up ideas from the phase-2 result:
1. Continuation: lower the allowed rank gradually instead of snapping to rank 10.
2. Delta pressure: score candidates by fitness plus Delta-containment residual.
3. Diverse seeding: cycle through warm seeds, perturbed seeds, and cold ALS starts.

The inner optimizer is the exact LP fixed-V solver from phase2_grassmannian.
Because Delta containment is not linear in the LP variables, the Delta term is used
as a merit function for continuation selection rather than embedded directly into the
LP objective.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alternating_linf.solver import als_init, alternating_linf, compute_fitness
from db_optimizer.tensor import fiber_mode_decomposition
from variety_search.phase2_grassmannian import compute_H_fast, grassmannian_optimize, load_factors


SEED_FILES = [
    "slp_turbo_best.json",
    "slp_best.json",
    "slp_turbo_best_top0.json",
    "slp_turbo_best_top1.json",
    "slp_turbo_best_top2.json",
    "chain_best.json",
    "plaquette_best.json",
    "shotgun_best.json",
    "phase2_sweep_best.json",
]


def list_available_seed_paths():
    paths = []
    for name in SEED_FILES:
        path = ROOT / name if not name.startswith("phase2_") else ROOT / "variety_search" / name
        if path.exists():
            paths.append(path)
    return paths


def measure_delta_containment(alpha, beta, gamma):
    fm = fiber_mode_decomposition(alpha, beta, gamma)
    H = np.hstack([fm["Eta1"], fm["Eta2"]])
    Delta = fm["Delta"]
    H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))
    if H_rank > 0:
        coef, _, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
        resid = Delta - H @ coef
        delta_maxabs = float(np.max(np.abs(resid)))
        delta_fro = float(np.linalg.norm(resid))
    else:
        delta_maxabs = float(np.max(np.abs(Delta)))
        delta_fro = float(np.linalg.norm(Delta))
    return {
        "delta_maxabs": delta_maxabs,
        "delta_fro": delta_fro,
        "H_rank": H_rank,
    }


def merit(fitness, delta_maxabs, delta_weight):
    return float(fitness + delta_weight * delta_maxabs)


def make_seed(seed_kind, rng, available_paths):
    if seed_kind == "cold":
        alpha, beta, gamma = als_init(19, rng)
        alpha, beta, gamma, _, _ = alternating_linf(alpha, beta, gamma, max_cycles=80, tol=1e-12)
        return alpha, beta, gamma, "cold_als"

    path = available_paths[rng.integers(0, len(available_paths))]
    alpha, beta, gamma = load_factors(str(path))
    label = path.name
    if seed_kind == "warm_perturb":
        sigma = float(rng.choice([0.005, 0.01, 0.02, 0.03]))
        alpha = alpha + rng.normal(0.0, sigma, alpha.shape)
        beta = beta + rng.normal(0.0, sigma, beta.shape)
        label = f"{label}:sigma={sigma:.3f}"
    elif seed_kind == "warm_mix" and len(available_paths) >= 2:
        other_path = available_paths[rng.integers(0, len(available_paths))]
        other_a, other_b, other_g = load_factors(str(other_path))
        mix = float(rng.choice([0.15, 0.25, 0.35]))
        alpha = (1.0 - mix) * alpha + mix * other_a
        beta = (1.0 - mix) * beta + mix * other_b
        gamma = (1.0 - mix) * gamma + mix * other_g
        label = f"mix:{path.name}+{other_path.name}:w={mix:.2f}"
    return alpha, beta, gamma, label


def continuation_schedule():
    return [
        {"target_rank": 14, "outer_iters": 4, "damping": 0.05, "delta_weight": 0.05},
        {"target_rank": 12, "outer_iters": 5, "damping": 0.10, "delta_weight": 0.15},
        {"target_rank": 11, "outer_iters": 5, "damping": 0.15, "delta_weight": 0.30},
        {"target_rank": 10, "outer_iters": 6, "damping": 0.20, "delta_weight": 0.60},
    ]


def run_continuation(alpha, beta, gamma, schedule):
    stage_log = []
    for stage_idx, stage in enumerate(schedule):
        alpha, beta, gamma, fit, _ = grassmannian_optimize(
            alpha,
            beta,
            gamma,
            target_rank=stage["target_rank"],
            outer_iters=stage["outer_iters"],
            damping=stage["damping"],
            verbose=False,
        )
        delta_info = measure_delta_containment(alpha, beta, gamma)
        score = merit(fit, delta_info["delta_maxabs"], stage["delta_weight"])
        stage_log.append({
            "stage": stage_idx,
            "target_rank": stage["target_rank"],
            "fitness": float(fit),
            "delta_maxabs": delta_info["delta_maxabs"],
            "delta_fro": delta_info["delta_fro"],
            "H_rank": delta_info["H_rank"],
            "score": score,
            "delta_weight": stage["delta_weight"],
        })
    return alpha, beta, gamma, stage_log


def worker_main(cfg_json):
    cfg = json.loads(cfg_json)
    worker_id = cfg["worker_id"]
    deadline = cfg["deadline"]
    available_paths = [Path(p) for p in cfg["seed_paths"]]
    rng = np.random.default_rng(cfg["seed"])
    schedule = continuation_schedule()

    best = None
    attempts = 0
    seed_kinds = ["cold", "warm", "warm_perturb", "warm_mix"]

    while attempts == 0 or time.time() < deadline:
        seed_kind = seed_kinds[attempts % len(seed_kinds)]
        alpha, beta, gamma, seed_label = make_seed(seed_kind, rng, available_paths)
        alpha, beta, gamma, stage_log = run_continuation(alpha, beta, gamma, schedule)
        final = stage_log[-1]
        candidate = {
            "worker": worker_id,
            "attempt": attempts,
            "seed_kind": seed_kind,
            "seed_label": seed_label,
            "fitness": final["fitness"],
            "delta_maxabs": final["delta_maxabs"],
            "delta_fro": final["delta_fro"],
            "H_rank": final["H_rank"],
            "score": final["score"],
            "stage_log": stage_log,
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
            "gamma": gamma.tolist(),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
        attempts += 1

    best["attempts"] = attempts
    return best


def save_best(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    workers = 24
    duration_seconds = 600.0
    available_paths = list_available_seed_paths()
    if not available_paths:
        raise RuntimeError("No seed files available for continuation search")

    print("=" * 80)
    print("  Phase 2b: Continuation + Delta Pressure Search")
    print("=" * 80)
    print(f"  Workers: {workers}")
    print(f"  Duration: {duration_seconds:.0f}s")
    print(f"  Seeds: {len(available_paths)} files + cold ALS")

    deadline = time.time() + duration_seconds
    configs = []
    for worker_id in range(workers):
        configs.append({
            "worker_id": worker_id,
            "deadline": deadline,
            "seed_paths": [str(path) for path in available_paths],
            "seed": 10000 + worker_id,
        })

    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_main, json.dumps(cfg)): cfg["worker_id"] for cfg in configs}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"  W{worker_id:02d}: score={result['score']:.6f}  fit={result['fitness']:.6f}  "
                    f"delta={result['delta_maxabs']:.6f}  H_rank={result['H_rank']}  attempts={result['attempts']}"
                )
            except Exception as exc:
                print(f"  W{worker_id:02d}: failed - {exc}")

    elapsed = time.time() - started
    results.sort(key=lambda item: item["score"])
    best = results[0]

    out_dir = Path(__file__).resolve().parent
    best_path = out_dir / "phase2_continuation_best.json"
    summary_path = out_dir / "phase2_continuation_results.json"

    save_best(best_path, best)
    summary = []
    for item in results:
        summary.append({
            "worker": item["worker"],
            "attempts": item["attempts"],
            "seed_kind": item["seed_kind"],
            "seed_label": item["seed_label"],
            "score": item["score"],
            "fitness": item["fitness"],
            "delta_maxabs": item["delta_maxabs"],
            "delta_fro": item["delta_fro"],
            "H_rank": item["H_rank"],
        })
    with open(summary_path, "w") as f:
        json.dump({"elapsed_seconds": elapsed, "results": summary}, f, indent=2)

    print("\nBest result")
    print(
        f"  score={best['score']:.6f}  fit={best['fitness']:.6f}  "
        f"delta={best['delta_maxabs']:.6f}  H_rank={best['H_rank']}"
    )
    print(f"  saved: {best_path.name}, {summary_path.name}")


if __name__ == "__main__":
    main()