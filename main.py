#!/usr/bin/env python3
"""
main.py — Keth-Varai Machine TUI launcher.

Simple menu to pick a workload tier and run the neural network sweep.
Supports checkpointing: interrupt at any time, resume by selecting the same workload.
"""

import os
import sys
import json
import time
import math
import glob

# Keep BLAS single-threaded — parallelism is at the process level
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS'):
    os.environ[_v] = '1'

if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CANON_NEURAL_NETWORK"))

import torch
from train import train, SUCCESS_THRESHOLD
from data import sample_batch, frobenius_relative_error

# ═══════════════════════════════════════════════════════════════
# WORKLOAD TIERS
# ═══════════════════════════════════════════════════════════════

WORKLOADS = {
    "1": {
        "name": "Sanity Check",
        "desc": "N=27 only, 1 seed, 5k steps. Verify the pipeline runs.",
        "N_range": (27, 27),
        "seeds": 1,
        "steps": 5_000,
        "batch_size": 4096,
        "log_every": 100,
    },
    "2": {
        "name": "Smoke Test",
        "desc": "N=27→23, 2 seeds each, 20k steps. Quick viability check.",
        "N_range": (27, 23),
        "seeds": 2,
        "steps": 20_000,
        "batch_size": 8192,
        "log_every": 500,
    },
    "3": {
        "name": "10-Minute Run",
        "desc": "N=27→21, 3 seeds each, 50k steps.",
        "N_range": (27, 21),
        "seeds": 3,
        "steps": 50_000,
        "batch_size": 8192,
        "log_every": 1000,
    },
    "4": {
        "name": "1-Hour Run",
        "desc": "N=27→19, 5 seeds each, 200k steps.",
        "N_range": (27, 19),
        "seeds": 5,
        "steps": 200_000,
        "batch_size": 8192,
        "log_every": 2000,
    },
    "5": {
        "name": "Overnight (Unlimited)",
        "desc": "N=27→15, 10 seeds each, 500k steps. Full sweep.",
        "N_range": (27, 15),
        "seeds": 10,
        "steps": 500_000,
        "batch_size": 8192,
        "log_every": 5000,
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CANON_NEURAL_NETWORK", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CANON_NEURAL_NETWORK", "checkpoints")
ABORT_THRESHOLD = 5e-3


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT STATUS
# ═══════════════════════════════════════════════════════════════

def scan_checkpoints():
    """Return dict of (N, seed) → {step, best_rel_err} for existing checkpoints."""
    status = {}
    if not os.path.isdir(CHECKPOINT_DIR):
        return status
    for f in glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_N*_s*.pt")):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            base = os.path.basename(f)
            parts = base.replace("ckpt_N", "").replace(".pt", "").split("_s")
            N, seed = int(parts[0]), int(parts[1])
            status[(N, seed)] = {
                "step": ckpt["step"],
                "best_rel_err": ckpt["best_rel_err"],
            }
        except Exception:
            pass
    return status


def scan_results():
    """Return dict of (N, seed) → result dict for completed runs."""
    results = {}
    if not os.path.isdir(RESULTS_DIR):
        return results
    for f in glob.glob(os.path.join(RESULTS_DIR, "run_N*_s*.json")):
        try:
            with open(f) as fh:
                r = json.load(fh)
            results[(r["N"], r["seed"])] = r
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════

def show_menu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU only"

    print(f"\n{'═'*64}")
    print(f"  KETH-VARAI MACHINE  │  {gpu_name}")
    print(f"{'═'*64}")

    ckpts = scan_checkpoints()
    results = scan_results()
    if ckpts or results:
        print(f"\n  Checkpoints: {len(ckpts)} in-progress, {len(results)} completed")
        all_Ns = sorted(set(N for N, _ in list(ckpts.keys()) + list(results.keys())), reverse=True)
        for N in all_Ns:
            n_done = sum(1 for (n, s) in results if n == N)
            n_ckpt = sum(1 for (n, s) in ckpts if n == N and (n, s) not in results)
            best = min(
                [r["best_rel_err"] for (n, s), r in results.items() if n == N] +
                [c["best_rel_err"] for (n, s), c in ckpts.items() if n == N],
                default=math.inf
            )
            flag = "✓" if best < SUCCESS_THRESHOLD else " "
            print(f"    N={N:2d}  {flag}  done={n_done}  running={n_ckpt}  best={best:.3e}")

    print(f"\n  Select workload:\n")
    for key, wl in WORKLOADS.items():
        start_N, end_N = wl["N_range"]
        print(f"    [{key}]  {wl['name']}")
        print(f"         {wl['desc']}")
        print(f"         N={start_N}→{end_N}, {wl['seeds']} seeds, {wl['steps']//1000}k steps\n")
    print(f"    [c]  Clear checkpoints & results")
    print(f"    [q]  Quit\n")


def show_sweep_summary(summary):
    print(f"\n{'═'*64}")
    print(f"  SWEEP RESULTS")
    print(f"{'═'*64}")
    print(f"  {'N':>4}  {'best_err':>12}  {'median_err':>12}  {'success':>12}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}")
    for row in summary:
        flag = "✓" if row["successes"] > 0 else "✗"
        print(
            f"  {row['N']:>4}  {row['best_err']:>12.3e}  {row['median_err']:>12.3e}"
            f"  {flag} {row['successes']:>2}/{row['total_seeds']:<2}"
        )
    print(f"{'═'*64}\n")


# ═══════════════════════════════════════════════════════════════
# SWEEP RUNNER (sequential, with checkpointing)
# ═══════════════════════════════════════════════════════════════

def run_workload(workload):
    start_N, end_N = workload["N_range"]
    seeds = workload["seeds"]
    steps = workload["steps"]
    batch_size = workload["batch_size"]
    log_every = workload["log_every"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'─'*64}")
    print(f"  Workload: {workload['name']}")
    print(f"  N={start_N}→{end_N}  seeds={seeds}  steps={steps//1000}k  device={device_str}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Ctrl+C to pause (resumes from checkpoint)")
    print(f"{'─'*64}\n")

    summary = []
    t0 = time.time()

    for N in range(start_N, end_N - 1, -1):
        print(f"\n{'─'*48}")
        print(f"  N = {N}  ({seeds} seeds)")
        print(f"{'─'*48}")

        results_for_N = []

        for s in range(seeds):
            out_path = os.path.join(RESULTS_DIR, f"run_N{N}_s{s}.json")

            # Skip if already completed
            if os.path.exists(out_path):
                with open(out_path) as f:
                    r = json.load(f)
                print(f"  [cached] N={N} s={s}  best={r['best_rel_err']:.3e}  success={r['success']}")
                results_for_N.append(r)
                continue

            # Run (will resume from checkpoint if available)
            try:
                result = train(
                    N=N,
                    seed=s,
                    d=64,
                    encoder_depth=3,
                    encoder_width=128,
                    lr=3e-4,
                    batch_size=batch_size,
                    max_steps=steps,
                    log_every=log_every,
                    device_str=device_str,
                    worker_id=s,
                    checkpoint_dir=CHECKPOINT_DIR,
                )
            except KeyboardInterrupt:
                print(f"\n  ⏸  Paused. Checkpoint saved. Re-run to resume.")
                return None

            results_for_N.append(result)

            # Save result
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(
                f"  [done] N={N} s={s}  best={result['best_rel_err']:.3e}"
                f"  success={result['success']}  steps={result['steps_run']}"
            )

        if not results_for_N:
            continue

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
        print(f"\n  N={N}  {flag}  best={best_for_N:.3e}  median={median_err:.3e}  {successes}/{seeds}")

        if best_for_N > ABORT_THRESHOLD:
            print(f"\n  SWEEP ABORT: best={best_for_N:.3e} > {ABORT_THRESHOLD:.0e} at N={N}. Stopping.\n")
            break

    elapsed = time.time() - t0

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    show_sweep_summary(summary)
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Summary → {summary_path}\n")
    return summary


def clear_data():
    confirm = input("  Delete all checkpoints and results? [y/N] ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return
    import shutil
    for d in [CHECKPOINT_DIR, RESULTS_DIR]:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  Deleted {d}")
    print("  Done.\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    while True:
        show_menu()
        choice = input("  > ").strip().lower()

        if choice == 'q':
            print("  Goodbye.\n")
            break
        elif choice == 'c':
            clear_data()
            continue

        if choice not in WORKLOADS:
            print(f"  Invalid choice: {choice!r}")
            continue

        run_workload(WORKLOADS[choice])
        input("  Press Enter to return to menu...")


if __name__ == "__main__":
    main()
