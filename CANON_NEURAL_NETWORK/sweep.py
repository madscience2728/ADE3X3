"""
sweep.py — Multi-seed rank sweep for the Keth-Varai Machine.

Runs N=start_N down to min_N. Seeds at each N are dispatched to a
worker pool — on a 3090/3060 you can run many seeds in parallel since
the model is small and CUDA time-slices gracefully.

Usage:
    python sweep.py                            # defaults: N=27→19, 5 seeds, 4 workers
    python sweep.py --workers 8 --seeds 10     # 3090: go wide
    python sweep.py --workers 4 --seeds 5 --device cuda
    python sweep.py --start_N 25 --min_N 20 --seeds 3 --steps 100000

Worker note:
    --workers controls how many (N, seed) jobs run simultaneously.
    Each worker uses --device. On a single GPU they share it via CUDA
    time-slicing. For CPU-only machines, set --workers to cpu_count//2.
"""

import argparse
import json
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import netrc

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import train

ABORT_THRESHOLD = 5e-3  # if best over all seeds > this, stop sweep


def _ensure_wandb_api_key_in_env() -> None:
    """Propagate wandb API key to WANDB_API_KEY so spawned workers can auth."""
    if os.environ.get("WANDB_API_KEY"):
        return  # already set
    import pathlib
    # Windows stores netrc as _netrc; Python's netrc module looks for .netrc
    for candidate in [pathlib.Path.home() / "_netrc", pathlib.Path.home() / ".netrc"]:
        if candidate.exists():
            try:
                nrc = netrc.netrc(str(candidate))
                auth = nrc.authenticators("api.wandb.ai")
                if auth:
                    os.environ["WANDB_API_KEY"] = auth[2]  # password field
                    return
            except Exception:
                pass


def _worker(kwargs: dict) -> dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    try:
        return train(**kwargs)
    except Exception as e:
        traceback.print_exc()
        return {
            "N": kwargs["N"],
            "seed": kwargs["seed"],
            "best_rel_err": 999.0,
            "success": False,
            "error": str(e),
            "log": [],
            "steps_run": 0,
            "total_params": 0,
            "d": kwargs["d"],
            "encoder_depth": kwargs["encoder_depth"],
            "encoder_width": kwargs["encoder_width"],
            "device": kwargs["device_str"],
        }


def run_sweep(
    start_N: int,
    min_N: int,
    seeds: int,
    workers: int,
    d: int,
    encoder_depth: int,
    encoder_width: int,
    steps: int,
    batch_size: int,
    lr: float,
    device_str: str,
    out_dir: str,
    use_wandb: bool = False,
    wandb_project: str = "keth-varai",
):
    os.makedirs(out_dir, exist_ok=True)

    if use_wandb:
        _ensure_wandb_api_key_in_env()

    device_label = device_str
    if device_str == "auto":
        device_label = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═'*72}")
    print(f"  KETH-VARAI SWEEP  N={start_N}→{min_N}  seeds={seeds}  workers={workers}")
    print(f"  device={device_label}  steps={steps}  d={d}  depth={encoder_depth}  width={encoder_width}")
    print(f"{'═'*72}\n")

    summary = []

    for N in range(start_N, min_N - 1, -1):
        print(f"\n{'─'*72}")
        print(f"  N = {N}  (dispatching {seeds} seeds to {workers} workers)")
        print(f"{'─'*72}")

        jobs = []
        cached_results = []
        for s in range(seeds):
            out_path = os.path.join(out_dir, f"run_N{N}_s{s}.json")
            if os.path.exists(out_path):
                with open(out_path) as f:
                    r = json.load(f)
                print(f"  [cached] N={N} s={s}  best={r['best_rel_err']:.3e}  success={r['success']}")
                cached_results.append(r)
            else:
                jobs.append({
                    "N": N,
                    "seed": s,
                    "d": d,
                    "encoder_depth": encoder_depth,
                    "encoder_width": encoder_width,
                    "lr": lr,
                    "batch_size": batch_size,
                    "max_steps": steps,
                    "log_every": max(steps // 100, 500),
                    "device_str": device_str,
                    "worker_id": s,
                    "use_wandb": use_wandb,
                    "wandb_project": wandb_project,
                })

        results_for_N = list(cached_results)

        if jobs:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as pool:
                futures = {pool.submit(_worker, job): job for job in jobs}
                for fut in as_completed(futures):
                    job = futures[fut]
                    result = fut.result()
                    results_for_N.append(result)

                    out_path = os.path.join(out_dir, f"run_N{N}_s{job['seed']}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)

                    print(
                        f"  [done]   N={N} s={job['seed']}"
                        f"  best={result['best_rel_err']:.3e}"
                        f"  success={result['success']}"
                        f"  steps={result['steps_run']}"
                    )

        best_for_N = min(r["best_rel_err"] for r in results_for_N)
        successes = sum(1 for r in results_for_N if r["success"])
        sorted_errs = sorted(r["best_rel_err"] for r in results_for_N)
        median_err = sorted_errs[len(sorted_errs) // 2]

        row = {
            "N": N,
            "best_err": best_for_N,
            "median_err": median_err,
            "successes": successes,
            "total_seeds": seeds,
        }
        summary.append(row)

        flag = "✓" if successes > 0 else "✗"
        print(f"\n  N={N}  {flag}  best={best_for_N:.3e}  median={median_err:.3e}  {successes}/{seeds} success")

        if best_for_N > ABORT_THRESHOLD:
            print(f"\n  SWEEP ABORT: best={best_for_N:.3e} > {ABORT_THRESHOLD:.0e} at N={N}. Stopping.\n")
            break

    summary_path = os.path.join(out_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'═'*72}")
    print(f"  SWEEP RESULTS")
    print(f"{'═'*72}")
    print(f"  {'N':>4}  {'best_err':>12}  {'median_err':>12}  {'success':>12}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}")
    for row in summary:
        flag = "✓" if row["successes"] > 0 else "✗"
        print(
            f"  {row['N']:>4}  {row['best_err']:>12.3e}  {row['median_err']:>12.3e}"
            f"  {flag} {row['successes']:>2}/{row['total_seeds']:<2}"
        )

    print(f"\n  Summary → {summary_path}")
    print(f"{'═'*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Rank sweep for Keth-Varai Machine")
    parser.add_argument("--start_N", type=int, default=27)
    parser.add_argument("--min_N", type=int, default=19)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers. On 3090: try 6-8. On 3060: try 4.")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--encoder_width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'cpu'. All workers share this device.")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true", help="Log all runs to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="keth-varai")
    args = parser.parse_args()

    run_sweep(
        start_N=args.start_N,
        min_N=args.min_N,
        seeds=args.seeds,
        workers=args.workers,
        d=args.d,
        encoder_depth=args.encoder_depth,
        encoder_width=args.encoder_width,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        out_dir=args.out_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()

