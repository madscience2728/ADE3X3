"""
benchmark.py — Head-to-head bits-recovered comparison: Control vs Full model.

For each rank N, trains both architectures with identical steps/batch/device
and reports mantissa bits recovered = -log2(best_rel_err).

Usage:
    python benchmark.py                          # N=27→20, 50k steps each
    python benchmark.py --steps 200000           # longer runs
    python benchmark.py --min_N 22 --max_N 27    # custom range
    python benchmark.py --seeds 3                # average over seeds
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from data import sample_batch, frobenius_relative_error
from model import KethVaraiMachine
from control import DirectDecomp


def train_one(model, steps, batch_size, lr, device, tag="", ckpt_path=None):
    """Train a model, return best rel_err. Optionally checkpoint best model.
    Saves full training state (optimizer, scheduler, scaler, step, best)
    so training can be seamlessly resumed."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best = math.inf
    start_step = 1

    # Resume from full checkpoint if it exists
    full_ckpt = ckpt_path.replace(".pt", "_full.pt") if ckpt_path else None
    if full_ckpt and os.path.exists(full_ckpt):
        ckpt_data = torch.load(full_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model"])
        optimizer.load_state_dict(ckpt_data["optimizer"])
        scheduler.load_state_dict(ckpt_data["scheduler"])
        scaler.load_state_dict(ckpt_data["scaler"])
        best = ckpt_data["best"]
        start_step = ckpt_data["step"] + 1
        bits = -math.log2(best) if best > 0 else float("inf")
        print(f"    {tag} RESUMED from step {start_step - 1}  best={best:.3e}  bits={bits:.1f}")

    t0 = time.time()

    for step in range(start_step, steps + 1):
        model.train()
        A, B, C = sample_batch(batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            C_hat = model(A, B)
            loss = frobenius_relative_error(C_hat.float(), C.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 500 == 0 or step == 1:
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                Av, Bv, Cv = sample_batch(16_384, device)
                err = frobenius_relative_error(model(Av, Bv).float(), Cv.float()).item()
            improved = err < best
            if improved:
                best = err
                if ckpt_path is not None:
                    torch.save(model.state_dict(), ckpt_path)
            # Always save full training state for resume
            if ckpt_path is not None:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "best": best,
                }, full_ckpt)
            bits = -math.log2(best) if best > 0 else float("inf")
            elapsed = time.time() - t0
            marker = " *" if improved else ""
            print(f"    {tag} step {step:>7d}/{steps}  rel={err:.3e}  bits={bits:.1f}  t={elapsed:.0f}s{marker}")

            if best < 1e-6:
                break

    return best


def main():
    parser = argparse.ArgumentParser(description="Head-to-head bits benchmark")
    parser.add_argument("--min_N", type=int, default=20)
    parser.add_argument("--max_N", type=int, default=27)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=16_384)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    # Full model hyperparams (kept small for fair wall-clock comparison)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--encoder_width", type=int, default=128)
    args = parser.parse_args()

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"\n{'═'*72}")
    print(f"  BENCHMARK: Control vs Full  |  N={args.max_N}→{args.min_N}  steps={args.steps}")
    print(f"  device={device_str}  batch={args.batch_size}  seeds={args.seeds}")
    print(f"{'═'*72}\n")

    ckpt_dir = os.path.join("results", "benchmark_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    results = []

    for N in range(args.max_N, args.min_N - 1, -1):
        ctrl_bests = []
        full_bests = []

        for seed in range(args.seeds):
            torch.manual_seed(seed)

            # --- Control ---
            ctrl = DirectDecomp(N).to(device)
            ctrl_params = sum(p.numel() for p in ctrl.parameters())
            ctrl_ckpt = os.path.join(ckpt_dir, f"ctrl_N{N}_s{seed}.pt")
            t0 = time.time()
            ctrl_err = train_one(ctrl, args.steps, args.batch_size, args.lr, device,
                                 tag=f"CTRL N={N} s={seed}", ckpt_path=ctrl_ckpt)
            ctrl_time = time.time() - t0
            ctrl_bests.append(ctrl_err)
            del ctrl
            torch.cuda.empty_cache() if device.type == "cuda" else None

            # --- Full model ---
            torch.manual_seed(seed)
            full = KethVaraiMachine(
                N=N, latent_dim=args.d,
                encoder_depth=args.encoder_depth,
                encoder_width=args.encoder_width,
            ).to(device)
            full_params = sum(p.numel() for p in full.parameters())
            full_ckpt = os.path.join(ckpt_dir, f"full_N{N}_s{seed}.pt")
            t0 = time.time()
            full_err = train_one(full, args.steps, args.batch_size, args.lr, device,
                                 tag=f"FULL N={N} s={seed}", ckpt_path=full_ckpt)
            full_time = time.time() - t0
            full_bests.append(full_err)
            del full
            torch.cuda.empty_cache() if device.type == "cuda" else None

        ctrl_best = min(ctrl_bests)
        full_best = min(full_bests)
        ctrl_bits = -math.log2(ctrl_best) if ctrl_best > 0 else float("inf")
        full_bits = -math.log2(full_best) if full_best > 0 else float("inf")

        row = {
            "N": N,
            "ctrl_err": ctrl_best, "ctrl_bits": round(ctrl_bits, 2), "ctrl_params": ctrl_params,
            "full_err": full_best, "full_bits": round(full_bits, 2), "full_params": full_params,
            "delta_bits": round(full_bits - ctrl_bits, 2),
        }
        results.append(row)

        winner = "FULL" if full_bits > ctrl_bits else "CTRL"
        print(f"\n  N={N:>2}  CTRL: {ctrl_bits:5.1f} bits ({ctrl_params:>6d} params)"
              f"  |  FULL: {full_bits:5.1f} bits ({full_params:>6d} params)"
              f"  |  Δ={full_bits - ctrl_bits:+.1f}  [{winner}]\n")

    # Final table
    print(f"\n{'═'*72}")
    print(f"  BITS RECOVERED BY RANK")
    print(f"{'═'*72}")
    print(f"  {'N':>3}  {'CTRL bits':>10}  {'FULL bits':>10}  {'Δ bits':>8}  {'winner':>6}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
    for r in results:
        winner = "FULL" if r["delta_bits"] > 0 else "CTRL"
        print(f"  {r['N']:>3}  {r['ctrl_bits']:>10.2f}  {r['full_bits']:>10.2f}  {r['delta_bits']:>+8.2f}  {winner:>6}")

    print(f"\n  ctrl_params={results[0]['ctrl_params']} (N={results[0]['N']})"
          f"  full_params={results[0]['full_params']}")
    print(f"{'═'*72}\n")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → results/benchmark.json")
    print(f"  Checkpoints → {ckpt_dir}/")


if __name__ == "__main__":
    main()
