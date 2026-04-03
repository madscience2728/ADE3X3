#!/usr/bin/env python3
"""
Launch slp_turbo at ranks 20, 21, 22 with WARM START from rank-19 best.
Pads the existing 19-term solution with small random extra terms.
"""

import subprocess, sys, json, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def warm_start_init(rank_19_path, target_rank, out_path, seed):
    """Take the rank-19 best and add (target_rank - 19) small random terms."""
    data = json.loads(Path(rank_19_path).read_text())
    alpha = np.array(data["alpha"])  # (19, 9)
    beta = np.array(data["beta"])
    gamma = np.array(data["gamma"])

    rng = np.random.default_rng(seed + target_rank)
    n_extra = target_rank - 19
    scale = 0.01  # small perturbation — let the optimizer grow them

    extra_a = rng.standard_normal((n_extra, 9)) * scale
    extra_b = rng.standard_normal((n_extra, 9)) * scale
    extra_g = rng.standard_normal((n_extra, 9)) * scale

    new_data = {
        "alpha": np.vstack([alpha, extra_a]).tolist(),
        "beta": np.vstack([beta, extra_b]).tolist(),
        "gamma": np.vstack([gamma, extra_g]).tolist(),
        "fitness": data["fitness"],  # will be recalculated
        "rank": target_rank,
        "dim": 9,
    }
    Path(out_path).write_text(json.dumps(new_data, indent=2))
    print(f"  Created warm-start: {19} existing + {n_extra} new terms -> rank {target_rank}")


def run_rank(rank, seed=42, rounds=10):
    r19_path = ROOT / "slp_turbo_best.json"
    init_file = ROOT / f"slp_turbo_init_r{rank}.json"
    out_file = ROOT / f"slp_turbo_best_r{rank}.json"

    warm_start_init(r19_path, rank, init_file, seed)

    env = os.environ.copy()
    env["ADE_RANK"] = str(rank)

    cmd = [
        sys.executable, "-m", "slp_turbo",
        "--input", str(init_file),
        "--out", str(out_file),
        "--seed", str(seed),
        "--rounds", str(rounds),
        "--workers", "16",
        "--trust-init", "0.01",
        "--trust-max", "1.0",
    ]
    print(f"\n{'='*60}")
    print(f"  RANK {rank}  ({rounds} rounds, warm-start from R=19)")
    print(f"  Output: {out_file.name}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)

    if out_file.exists():
        data = json.loads(out_file.read_text())
        print(f"\n  >>> RANK {rank} RESULT: fitness = {data['fitness']:.10f}")
    return result.returncode


if __name__ == "__main__":
    for rank in [20, 21, 22]:
        run_rank(rank, rounds=10)
