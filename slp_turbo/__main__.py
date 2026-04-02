"""SLP Turbo entry point - orchestrates GPU + CPU engines."""

import json
import sys
import time
import threading
from pathlib import Path

import numpy as np
import torch

from .config import R, D, N_PARAMS, T_NP, parse_args
from .candidate_pool import CandidatePool
from .gpu_engine import GPUEngine
from .cpu_engine import CPUEngine, _maxabs
from .display import Display
from .diagnostics import full_diagnostic


def _load_factors(path):
    """Load A, B, G from JSON and return flat x (513,)."""
    data = json.loads(Path(path).read_text())
    a = np.array(data["alpha"], dtype=np.float64)
    b = np.array(data["beta"], dtype=np.float64)
    g = np.array(data["gamma"], dtype=np.float64)
    return np.concatenate([a.ravel(), b.ravel(), g.ravel()])


def _save_factors(x, fit, path):
    a = x[:R*D].reshape(R, D).tolist()
    b = x[R*D:2*R*D].reshape(R, D).tolist()
    g = x[2*R*D:].reshape(R, D).tolist()
    Path(path).write_text(json.dumps({
        "alpha": a, "beta": b, "gamma": g,
        "fitness": fit, "rank": R, "dim": D
    }, indent=2))


def _make_save_callback(out_path, console):
    """Return a callback that saves every new global best to disk."""
    _lock = threading.Lock()
    _counter = [0]

    def _on_new_best(x, fit):
        with _lock:
            _counter[0] += 1
            count = _counter[0]
        _save_factors(x, fit, out_path)
        console.print(f"  [bold green]* NEW BEST {fit:.10f}[/bold green]  (save #{count} -> {out_path})")

    return _on_new_best


def _make_seeds(base_x, n, scale, rng):
    """Create n perturbations of base_x."""
    xs = np.tile(base_x, (n, 1))
    xs += rng.normal(0, scale, xs.shape)
    xs[0] = base_x  # slot 0 = exact copy
    return xs


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    from rich.console import Console
    console = Console()

    # Load base solution
    base_x = _load_factors(args.input)
    base_fit = _maxabs(base_x)
    console.print(f"Loaded {args.input}: fitness = {base_fit:.10f}")

    # Initial diagnostics
    diag = full_diagnostic(base_x)
    console.print(f"  Live maxabs: {diag['live_maxabs']:.10f}  Dead maxabs: {diag['dead_maxabs']:.10f}")
    console.print(f"  Dead cancel: {diag['cancel_ratio']:.4%}  k_ker: {diag['kappa_ker']:.6f}  nuisance_rank: {diag['nuisance_rank']}/{diag['ker_dim']}")
    console.print(f"  Tier weights: live={args.live_weight}x  dead={args.dead_weight}x  sparsity_lam={args.sparsity_lambda}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    for rnd in range(args.rounds):
        if args.rounds > 1:
            console.print(f"\n{'='*60}\n  Round {rnd+1}/{args.rounds}\n{'='*60}")

        # Build save callback
        save_cb = _make_save_callback(args.out, console)

        # Create pool and seed
        pool = CandidatePool(args.workers, trust_init=args.trust_init, on_new_best=save_cb)
        seeds = _make_seeds(base_x, args.workers, args.perturb_scale, rng)
        for i in range(args.workers):
            pool.seed(i, seeds[i], _maxabs(seeds[i]))

        console.print(f"Seeded {args.workers} workers, best = {pool.global_best_fit:.10f}")

        # Create engines
        gpu = GPUEngine(device, pool, args)
        cpu = CPUEngine(pool, args)
        display = Display(pool, gpu, cpu)

        # Start both engines (both are daemon threads)
        gpu.start()
        cpu.start()

        console.print("Both engines running. Press Ctrl+C to stop early.\n")

        try:
            while cpu.iterations < args.max_iters:
                time.sleep(3.0)
                display.print_status()

                # Check if all dead
                if not pool.alive.any():
                    console.print("All workers dead - ending round.")
                    break
        except KeyboardInterrupt:
            console.print("\nStopping...")

        # Stop engines
        gpu.stop()
        cpu.stop()

        # Final save + diagnostics
        best_fit = pool.global_best_fit
        best_x = pool.global_best_x
        _save_factors(best_x, best_fit, args.out)

        diag = full_diagnostic(best_x)
        console.print(f"\nRound {rnd+1} done. Best = {best_fit:.10f}  ->  {args.out}")
        console.print(f"  Live: {diag['live_maxabs']:.10f}  Dead: {diag['dead_maxabs']:.10f}  Cancel: {diag['cancel_ratio']:.4%}")
        console.print(f"  k_ker: {diag['kappa_ker']:.6f}  nuisance_rank: {diag['nuisance_rank']}/{diag['ker_dim']}")

        # Use best as seed for next round
        base_x = best_x

    console.print("Done.")


if __name__ == "__main__":
    main()
