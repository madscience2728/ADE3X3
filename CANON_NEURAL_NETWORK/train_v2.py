"""
train_v2.py — Training for the Binary Keth-Varai Machine (H-Bomb variant).

Usage:
    python train_v2.py --N 19 --seed 100
    python train_v2.py --N 19 --seed 100 --n_bits 8 --bottleneck_dim 12 --binary_depth 8
    python train_v2.py --N 23 --seed 100 --steps 500000

Writes results to results/run_v2_N{N}_s{seed}.json
Checkpoints to checkpoints/ckpt_v2_N{N}_s{seed}.pt
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from data import sample_batch, frobenius_relative_error
from model_v2 import BinaryKethVarai

SUCCESS_THRESHOLD = 1e-8
CHECKPOINT_EVERY = 1_000


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_checkpoint(path, model, optimizer, scheduler, scaler,
                     step, best_rel_err, last_improvement, log, elapsed_s):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_rel_err": best_rel_err,
        "last_improvement": last_improvement,
        "log": log,
        "elapsed_s": elapsed_s,
    }, path)


def train(
    N: int,
    seed: int,
    latent_dim: int = 128,
    n_bits: int = 8,
    group_width: int = 16,
    bottleneck_dim: int = 12,
    binary_depth: int = 6,
    binary_width: int = 128,
    lr: float = 3e-4,
    batch_size: int = 4096,
    max_steps: int = 200_000,
    log_every: int = 500,
    device_str: str = "auto",
    worker_id: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "keth-varai-v2",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    if device_str == "auto":
        device_str = _default_device()
    device = torch.device(device_str)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = BinaryKethVarai(
        N=N,
        latent_dim=latent_dim,
        n_bits=n_bits,
        group_width=group_width,
        bottleneck_dim=bottleneck_dim,
        binary_depth=binary_depth,
        binary_width=binary_width,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    tag = f"[W{worker_id}] N={N} s={seed} v2"
    print(f"  {tag}  params={total_params}  device={device_str}")
    print(f"  {tag}  n_bits={n_bits}  bottleneck={bottleneck_dim}  "
          f"binary_depth={binary_depth}  binary_width={binary_width}")

    wb_run = None
    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb.login(force=False)
            wb_run = wandb.init(
                project=wandb_project,
                name=f"v2_N{N}_s{seed}",
                group=f"N{N}_v2",
                config=dict(N=N, seed=seed, latent_dim=latent_dim,
                            n_bits=n_bits, group_width=group_width,
                            bottleneck_dim=bottleneck_dim,
                            binary_depth=binary_depth, binary_width=binary_width,
                            lr=lr, batch_size=batch_size,
                            max_steps=max_steps, device=device_str, params=total_params),
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"  {tag}  [wandb] init failed ({e.__class__.__name__}): {e}")

    # Optimizer: separate groups for binary encoder vs real hyper-heads
    # Binary encoder gets higher LR — STE gradients are noisy, need more signal
    encoder_params = (
        list(model.input_encoder.parameters())
        + list(model.local_layer.parameters())
        + list(model.bottleneck.parameters())
        + list(model.binary_trunk.parameters())
        + list(model.to_latent.parameters())
    )
    head_params = (
        list(model.head_U.parameters())
        + list(model.head_V.parameters())
        + list(model.head_W.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr * 0.5, "weight_decay": 0.0},
    ])

    # Schedule: linear warmup → single cosine decay (no restarts — v2 default)
    warmup_steps = 5_000

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = step - warmup_steps
        total_decay = max_steps - warmup_steps
        progress = min(t / total_decay, 1.0)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_rel_err = math.inf
    log: list[dict] = []
    t0 = time.time()
    plateau_patience = 100_000
    last_improvement = 0
    start_step = 1

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_v2_N{N}_s{seed}.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best_rel_err = ckpt["best_rel_err"]
        last_improvement = ckpt["last_improvement"]
        log = ckpt.get("log", [])
        t0 = time.time() - ckpt.get("elapsed_s", 0)
        print(f"  {tag}  RESUMED from step {start_step - 1}, best={best_rel_err:.3e}")

    for step in range(start_step, max_steps + 1):
        model.train()
        A, B, C = sample_batch(batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            C_hat = model(A, B)
            loss = frobenius_relative_error(C_hat.float(), C.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                Av, Bv, Cv = sample_batch(16_384, device)
                C_hat_v = model(Av, Bv)
                rel_err = frobenius_relative_error(C_hat_v.float(), Cv.float()).item()

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                last_improvement = step

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            entry = {
                "step": step,
                "rel_err": rel_err,
                "best_rel_err": best_rel_err,
                "elapsed_s": round(elapsed, 1),
            }
            log.append(entry)

            bits = -math.log2(rel_err) if rel_err > 0 else float('inf')

            # Binary encoder diagnostics
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    latent_diag = model._encode(torch.cat([Av, Bv], dim=1))
                latent_diag = latent_diag.float()
                U_batch = model.head_U(latent_diag).view(-1, model.N, 9)
                V_batch = model.head_V(latent_diag).view(-1, model.N, 9)
                W_batch = model.head_W(latent_diag).view(-1, 9, model.N)
                u_collapse = (U_batch.std(dim=0).mean() / U_batch.abs().mean().clamp(min=1e-12)).item()
                v_collapse = (V_batch.std(dim=0).mean() / V_batch.abs().mean().clamp(min=1e-12)).item()
                w_collapse = (W_batch.std(dim=0).mean() / W_batch.abs().mean().clamp(min=1e-12)).item()

            print(
                f"  {tag}  step {step:7d}/{max_steps}"
                f"  rel={rel_err:.3e}  bits={bits:.1f}  best={best_rel_err:.3e}"
                f"  lr={current_lr:.1e}  t={elapsed:.0f}s"
                f"  U={u_collapse:.3f} V={v_collapse:.3f} W={w_collapse:.3f}"
            )

            if wb_run is not None:
                metrics = {
                    "loss/rel_err": rel_err,
                    "loss/bits": bits,
                    "loss/best_rel_err": best_rel_err,
                    "train/lr": current_lr,
                    "collapse/U_ratio": u_collapse,
                    "collapse/V_ratio": v_collapse,
                    "collapse/W_ratio": w_collapse,
                }
                wb_run.log(metrics, step=step)

            if best_rel_err < SUCCESS_THRESHOLD:
                print(f"  {tag}  ✓ SUCCESS at step {step}: best={best_rel_err:.3e}")
                _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                                 step, best_rel_err, last_improvement, log, time.time() - t0)
                break

            if step - last_improvement > plateau_patience and step > warmup_steps * 2:
                print(f"  {tag}  ✗ PLATEAU at step {step}: no improvement for {plateau_patience} steps")
                break

            if step % CHECKPOINT_EVERY == 0:
                _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                                 step, best_rel_err, last_improvement, log, time.time() - t0)

    if wb_run is not None:
        wb_run.summary["best_rel_err"] = best_rel_err
        wb_run.summary["success"] = best_rel_err < SUCCESS_THRESHOLD
        wb_run.finish()

    return {
        "N": N,
        "seed": seed,
        "latent_dim": latent_dim,
        "n_bits": n_bits,
        "group_width": group_width,
        "bottleneck_dim": bottleneck_dim,
        "binary_depth": binary_depth,
        "binary_width": binary_width,
        "device": device_str,
        "best_rel_err": best_rel_err,
        "success": best_rel_err < SUCCESS_THRESHOLD,
        "steps_run": step,
        "total_params": total_params,
        "log": log,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Binary Keth-Varai Machine (v2)")
    parser.add_argument("--N", type=int, default=19)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--group_width", type=int, default=16)
    parser.add_argument("--bottleneck_dim", type=int, default=12,
                        help="Must be < 18 to prevent float reconstruction")
    parser.add_argument("--binary_depth", type=int, default=6)
    parser.add_argument("--binary_width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="keth-varai-v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Binary Keth-Varai (H-Bomb)  |  N={args.N}  seed={args.seed}")
    print(f"  bottleneck={args.bottleneck_dim} < 18  |  "
          f"n_bits={args.n_bits}  |  binary activations=ON")
    print(f"{'='*60}")

    if args.bottleneck_dim >= 18:
        print(f"  WARNING: bottleneck_dim={args.bottleneck_dim} >= 18 — "
              f"float reconstruction is possible! Set < 18 to enforce anti-cheat.")

    result = train(
        N=args.N,
        seed=args.seed,
        latent_dim=args.latent_dim,
        n_bits=args.n_bits,
        group_width=args.group_width,
        bottleneck_dim=args.bottleneck_dim,
        binary_depth=args.binary_depth,
        binary_width=args.binary_width,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.steps,
        log_every=args.log_every,
        device_str=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
    )

    out_path = os.path.join(args.out_dir, f"run_v2_N{args.N}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "log"}, f, indent=2)
    print(f"\n  Result → {out_path}")
    print(f"  best_rel_err = {result['best_rel_err']:.3e}")
    if result["best_rel_err"] > 0:
        print(f"  bits = {-math.log2(result['best_rel_err']):.1f}")


if __name__ == "__main__":
    main()
