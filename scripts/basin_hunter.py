#!/usr/bin/env python3
"""
Basin Hunter: launch many independent slp_turbo runs from diverse starting points.

Goal: Find deeper basins. We don't care about R=19 existing — we want the
deepest fitness valley, even if it never reaches 0. To machine epsilon.

Strategy:
  - Multiple independent slp_turbo runs, each from a DIFFERENT random init
  - Some warm-started from best known, some cold random starts
  - Vary perturb-scale, trust-init, seed
  - Collect all results, keep the deepest
  - Run sequentially (one GPU) but with enough diversity
"""

import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent


def generate_random_init(R, n2=9, seed=0):
    """Generate a random (α,β,γ) initialization."""
    rng = np.random.default_rng(seed)
    # Scale matters: too large = fitness > 1, too small = stuck
    scale = rng.uniform(0.3, 1.5)
    alpha = rng.standard_normal((R, n2)) * scale
    beta = rng.standard_normal((R, n2)) * scale
    
    # Compute γ via least squares (gives a decent start)
    T = np.zeros((n2, n2, n2))
    n = int(np.sqrt(n2))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[n*i+j, n*j+k, n*i+k] = 1.0
    
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    
    gamma = np.zeros((R, n2))
    for c in range(n2):
        target_col = T[:, :, c].ravel()
        gamma[:, c] = np.linalg.lstsq(A.T, target_col, rcond=None)[0]
    
    T_recon = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma)
    fitness = float(np.max(np.abs(T_recon - T)))
    
    return {
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "fitness": fitness,
        "rank": R,
        "dim": n2
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=19)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--rounds-per-run", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=300)
    args = p.parse_args()
    
    R = args.rank
    out_dir = ROOT / "basin_hunt"
    out_dir.mkdir(exist_ok=True)
    
    # Load current best for warm-start runs
    best_path = ROOT / "slp_turbo_best.json"
    has_best = best_path.exists()
    if has_best:
        best_data = json.loads(best_path.read_text())
        current_best = best_data["fitness"]
        print(f"Current best: {current_best:.10f}")
    else:
        current_best = float('inf')
    
    # Define run configurations
    configs = []
    
    # Type 1: Warm-start from best, varying perturbation and seed
    if has_best:
        for i, (scale, seed) in enumerate([
            (0.001, 100), (0.003, 200), (0.01, 300), (0.03, 400),
            (0.05, 500), (0.1, 600), (0.001, 700), (0.005, 800),
        ]):
            configs.append({
                "name": f"warm_s{scale}_seed{seed}",
                "input": str(best_path),
                "perturb_scale": scale,
                "seed": seed,
                "trust_init": 0.003 if scale < 0.01 else 0.01,
                "type": "warm"
            })
    
    # Type 2: Cold random starts
    for i in range(args.n_runs - len(configs)):
        seed = 1000 + i * 137
        init_data = generate_random_init(R, seed=seed)
        init_path = out_dir / f"cold_init_{i:03d}.json"
        init_path.write_text(json.dumps(init_data, indent=2))
        
        configs.append({
            "name": f"cold_{i:03d}_seed{seed}",
            "input": str(init_path),
            "perturb_scale": 0.01,
            "seed": seed,
            "trust_init": 0.01,
            "type": "cold",
            "init_fitness": init_data["fitness"]
        })
    
    configs = configs[:args.n_runs]
    
    print(f"\n{'='*70}")
    print(f"  BASIN HUNTER: {len(configs)} runs, R={R}")
    print(f"  {sum(1 for c in configs if c['type']=='warm')} warm-start, "
          f"{sum(1 for c in configs if c['type']=='cold')} cold-start")
    print(f"  {args.rounds_per_run} rounds/run, {args.workers} workers, "
          f"{args.max_iters} iters/round")
    print(f"{'='*70}\n")
    
    results = []
    global_best = current_best
    t0 = time.time()
    
    for idx, cfg in enumerate(configs):
        out_path = out_dir / f"run_{idx:03d}_{cfg['name']}.json"
        
        elapsed = time.time() - t0
        print(f"\n[{idx+1}/{len(configs)}, {elapsed:.0f}s] "
              f"{cfg['name']} ({cfg['type']})", end="")
        if cfg['type'] == 'cold':
            print(f" init={cfg['init_fitness']:.4f}", end="")
        print()
        
        cmd = [
            sys.executable, "-m", "slp_turbo",
            "--input", cfg["input"],
            "--out", str(out_path),
            "--perturb-scale", str(cfg["perturb_scale"]),
            "--seed", str(cfg["seed"]),
            "--trust-init", str(cfg["trust_init"]),
            "--rounds", str(args.rounds_per_run),
            "--workers", str(args.workers),
            "--max-iters", str(args.max_iters),
        ]
        
        env = dict(__import__('os').environ)
        env["ADE_RANK"] = str(R)
        
        try:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env,
                capture_output=True, text=True, timeout=600
            )
            
            if out_path.exists():
                result_data = json.loads(out_path.read_text())
                fit = result_data["fitness"]
                tag = ""
                if fit < global_best:
                    global_best = fit
                    tag = " *** NEW GLOBAL BEST ***"
                    # Copy to main best
                    import shutil
                    shutil.copy2(out_path, ROOT / f"basin_hunt_best_R{R}.json")
                
                results.append((fit, cfg['name'], cfg['type']))
                print(f"  -> fitness = {fit:.10f}{tag}")
            else:
                print(f"  -> FAILED (no output)")
                results.append((float('inf'), cfg['name'], cfg['type']))
                
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT")
            if out_path.exists():
                result_data = json.loads(out_path.read_text())
                fit = result_data["fitness"]
                results.append((fit, cfg['name'], cfg['type']))
                print(f"  -> (partial) fitness = {fit:.10f}")
            else:
                results.append((float('inf'), cfg['name'], cfg['type']))
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append((float('inf'), cfg['name'], cfg['type']))
    
    # Summary
    elapsed = time.time() - t0
    results.sort()
    
    print(f"\n{'='*70}")
    print(f"  BASIN HUNT RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Rank':>4s}  {'Fitness':>14s}  {'Type':>5s}  {'Name'}")
    for i, (fit, name, typ) in enumerate(results[:10]):
        print(f"  {i+1:4d}  {fit:14.10f}  {typ:>5s}  {name}")
    
    if len(results) > 10:
        print(f"  ... ({len(results) - 10} more)")
    
    print(f"\n  Global best: {global_best:.10f}")
    print(f"  Reference:   {current_best:.10f}")
    if global_best < current_best:
        print(f"  IMPROVEMENT: {current_best - global_best:.2e}")
    else:
        print(f"  No improvement over reference.")


if __name__ == "__main__":
    main()
