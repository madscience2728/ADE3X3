"""
explorer.py — CPU Explorer Network for the Keth-Varai Machine.

Runs a second hypernetwork on CPU that is repelled from the GPU model's
current solution. The GPU exploits one basin; the CPU explores for
structurally different decompositions.

Loss = reconstruction_error - λ_repulsion * distance_from_gpu_model

The explorer:
  1. Periodically loads the GPU model's latest checkpoint
  2. Measures functional distance: same (A,B) batch → compare generated U,V,W
  3. Adds repulsion penalty to stay far from GPU's basin
  4. When it finds a good-but-different solution → saves to discoveries/

Usage:
    python explorer.py --N 23 --workers 12
    (typically launched from main.py alongside the GPU sweep)

Each worker gets a unique seed and 2 CPU threads so they don't fight over cores.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import glob

import torch
import torch.nn as nn

from data import sample_batch, frobenius_relative_error
from model import KethVaraiMachine
from train import SUCCESS_THRESHOLD, _save_checkpoint


def functional_distance(model_a: KethVaraiMachine, model_b: KethVaraiMachine,
                        A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Measure how differently two hypernetworks decompose the same inputs.
    Compare the generated U, V matrices (the decomposition coefficients).
    Returns a scalar distance (mean L2 over batch).
    """
    with torch.no_grad():
        lat_b = model_b._encode(torch.cat([A, B], dim=1))
        U_b = model_b.head_U(lat_b).view(-1, model_b.N, 9)
        V_b = model_b.head_V(lat_b).view(-1, model_b.N, 9)

    lat_a = model_a._encode(torch.cat([A, B], dim=1))
    U_a = model_a.head_U(lat_a).view(-1, model_a.N, 9)
    V_a = model_a.head_V(lat_a).view(-1, model_a.N, 9)

    # Normalize rows before comparing (we care about direction, not scale)
    def row_norm(M):
        return M / (M.norm(dim=2, keepdim=True) + 1e-8)

    # Distance: mean L2 between normalized coefficient vectors
    u_dist = (row_norm(U_a) - row_norm(U_b)).pow(2).sum(dim=2).mean()
    v_dist = (row_norm(V_a) - row_norm(V_b)).pow(2).sum(dim=2).mean()
    return (u_dist + v_dist) / 2


def load_gpu_model(ckpt_path: str, N: int, device: torch.device,
                   latent_dim: int = 64, encoder_depth: int = 3,
                   encoder_width: int = 128) -> KethVaraiMachine | None:
    """Load the GPU model's latest checkpoint onto CPU for comparison."""
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = KethVaraiMachine(
            N=N, latent_dim=latent_dim,
            encoder_depth=encoder_depth, encoder_width=encoder_width
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model
    except Exception:
        return None


def find_latest_gpu_checkpoint(checkpoint_dir: str, N: int) -> str | None:
    """Find the most recent GPU checkpoint for a given N."""
    pattern = os.path.join(checkpoint_dir, f"ckpt_N{N}_s*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def explore(
    N: int,
    seed: int = 1000,
    latent_dim: int = 64,
    encoder_depth: int = 3,
    encoder_width: int = 128,
    lr: float = 1e-4,
    batch_size: int = 2048,
    max_steps: int = 500_000,
    log_every: int = 1000,
    gpu_checkpoint_dir: str = "checkpoints",
    discovery_dir: str = "discoveries",
    checkpoint_dir: str = "checkpoints",
    lambda_repulsion: float = 0.1,
    sync_every: int = 5_000,
    discovery_threshold: float = 5e-3,
) -> dict:
    """
    Run the CPU explorer.

    lambda_repulsion: weight on the repulsion term
    sync_every: reload GPU checkpoint every N steps
    discovery_threshold: save discovery if rel_err < this AND distance > 0.5
    """
    device = torch.device("cpu")
    torch.set_num_threads(2)  # each worker gets 2 threads
    torch.manual_seed(seed)

    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim,
        encoder_depth=encoder_depth, encoder_width=encoder_width
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    tag = f"[CPU-X] N={N} s={seed}"
    print(f"  {tag}  params={total_params}  device=cpu  (explorer)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Cosine schedule (gentle — explorer doesn't need aggressive decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-6
    )

    os.makedirs(discovery_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_explorer_N{N}_s{seed}.pt")

    best_rel_err = math.inf
    best_distance = 0.0
    log = []
    t0 = time.time()
    start_step = 1
    gpu_model = None
    last_gpu_mtime = 0.0
    discoveries = 0

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        best_rel_err = ckpt["best_rel_err"]
        log = ckpt.get("log", [])
        t0 = time.time() - ckpt.get("elapsed_s", 0)
        print(f"  {tag}  RESUMED from step {start_step - 1}, best={best_rel_err:.3e}")

    for step in range(start_step, max_steps + 1):
        # Sync GPU model periodically
        if step % sync_every == 1 or gpu_model is None:
            gpu_ckpt_path = find_latest_gpu_checkpoint(gpu_checkpoint_dir, N)
            if gpu_ckpt_path:
                mtime = os.path.getmtime(gpu_ckpt_path)
                if mtime > last_gpu_mtime:
                    new_gpu = load_gpu_model(
                        gpu_ckpt_path, N, device,
                        latent_dim=latent_dim,
                        encoder_depth=encoder_depth,
                        encoder_width=encoder_width
                    )
                    if new_gpu is not None:
                        gpu_model = new_gpu
                        last_gpu_mtime = mtime
                        if step > 1:
                            print(f"  {tag}  synced GPU checkpoint (step {step})")

        model.train()
        A, B, C = sample_batch(batch_size, device)
        C_hat = model(A, B)

        rec_loss = frobenius_relative_error(C_hat, C)

        # Repulsion: push away from GPU model's basin
        repulsion = torch.tensor(0.0)
        if gpu_model is not None:
            dist = functional_distance(model, gpu_model, A, B)
            # Repulsion is 1/distance — closer = stronger push
            repulsion = 1.0 / (dist + 0.1)

        loss = rec_loss + lambda_repulsion * repulsion

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                Av, Bv, Cv = sample_batch(4096, device)
                C_hat_v = model(Av, Bv)
                rel_err = frobenius_relative_error(C_hat_v, Cv).item()

                # Measure distance from GPU model
                distance = 0.0
                if gpu_model is not None:
                    distance = functional_distance(model, gpu_model, Av, Bv).item()

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                best_distance = distance

            elapsed = time.time() - t0
            bits = -math.log2(rel_err) if rel_err > 0 else float('inf')
            entry = {
                "step": step,
                "rel_err": rel_err,
                "best_rel_err": best_rel_err,
                "distance": round(distance, 4),
                "elapsed_s": round(elapsed, 1),
            }
            log.append(entry)

            gpu_tag = f"dist={distance:.3f}" if gpu_model else "no-gpu-sync"
            print(
                f"  {tag}  step {step:7d}/{max_steps}"
                f"  rel={rel_err:.3e}  bits={bits:.1f}  best={best_rel_err:.3e}"
                f"  {gpu_tag}  t={elapsed:.0f}s"
            )

            # Discovery: good error AND far from GPU solution
            if rel_err < discovery_threshold and distance > 0.5:
                discoveries += 1
                disc_path = os.path.join(
                    discovery_dir,
                    f"discovery_N{N}_s{seed}_step{step}.pt"
                )
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "rel_err": rel_err,
                    "distance": distance,
                    "N": N,
                    "seed": seed,
                }, disc_path)
                print(f"  {tag}  ★ DISCOVERY #{discoveries}: err={rel_err:.3e} dist={distance:.3f} → {disc_path}")

        # Checkpoint
        if step % 5_000 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_rel_err": best_rel_err,
                "log": log,
                "elapsed_s": time.time() - t0,
                # no scaler — CPU doesn't use AMP
            }, ckpt_path)

    return {
        "N": N,
        "seed": seed,
        "best_rel_err": best_rel_err,
        "best_distance": best_distance,
        "discoveries": discoveries,
        "steps_run": step,
        "log": log,
    }


def _worker(kwargs):
    """Entry point for each multiprocessing worker."""
    return explore(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="CPU Explorer for Keth-Varai Machine")
    parser.add_argument("--N", type=int, default=23)
    parser.add_argument("--seed", type=int, default=1000,
                        help="Base seed (each worker gets seed+i)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel explorer workers (0=auto: cpu_count//2)")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--lambda_repulsion", type=float, default=0.1)
    parser.add_argument("--sync_every", type=int, default=5000)
    parser.add_argument("--gpu_checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--discovery_dir", type=str, default="discoveries")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() // 2)

    print(f"\n{'='*60}")
    print(f"  Keth-Varai Explorer Fleet (CPU)")
    print(f"  N={args.N}  |  {num_workers} workers  |  seeds {args.seed}..{args.seed + num_workers - 1}")
    print(f"  2 threads/worker → {num_workers * 2} threads total")
    print(f"{'='*60}\n")

    if num_workers == 1:
        # Single worker — run directly (simpler debugging)
        explore(
            N=args.N,
            seed=args.seed,
            latent_dim=args.latent_dim,
            lr=args.lr,
            batch_size=args.batch_size,
            max_steps=args.steps,
            log_every=args.log_every,
            gpu_checkpoint_dir=args.gpu_checkpoint_dir,
            discovery_dir=args.discovery_dir,
            checkpoint_dir=args.checkpoint_dir,
            lambda_repulsion=args.lambda_repulsion,
            sync_every=args.sync_every,
        )
        return

    # Build kwargs for each worker (different seed each)
    worker_args = []
    for i in range(num_workers):
        worker_args.append({
            "N": args.N,
            "seed": args.seed + i,
            "latent_dim": args.latent_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_steps": args.steps,
            "log_every": args.log_every,
            "gpu_checkpoint_dir": args.gpu_checkpoint_dir,
            "discovery_dir": args.discovery_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "lambda_repulsion": args.lambda_repulsion,
            "sync_every": args.sync_every,
        })

    # Use spawn to avoid CUDA fork issues (even though we're CPU-only)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        try:
            results = pool.map(_worker, worker_args)
        except KeyboardInterrupt:
            print("\n  Shutting down explorer fleet...")
            pool.terminate()
            pool.join()
            return

    total_discoveries = sum(r["discoveries"] for r in results)
    best = min(results, key=lambda r: r["best_rel_err"])
    print(f"\n{'='*60}")
    print(f"  Fleet done: {total_discoveries} discoveries across {num_workers} workers")
    print(f"  Best: seed={best['seed']} err={best['best_rel_err']:.3e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
