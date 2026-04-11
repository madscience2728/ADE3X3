"""
train.py — Single training run for the Keth-Varai Machine.

Usage:
    python train.py --N 27 --seed 0
    python train.py --N 23 --seed 42 --steps 200000
    python train.py --N 20 --d 64 --device cuda

Writes results to results/run_N{N}_s{seed}.json

GPU notes:
    --device cuda          → use GPU (auto-detects if available)
    --batch_size 16384     → fill a 3090/3060 with big batches
    AMP is enabled automatically when device=cuda
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
from model import KethVaraiMachine

SUCCESS_THRESHOLD = 1e-5  # relative Frobenius error target


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(
    N: int,
    seed: int,
    d: int = 64,
    encoder_depth: int = 3,
    encoder_width: int = 128,
    lr: float = 3e-4,
    batch_size: int = 8192,
    max_steps: int = 200_000,
    log_every: int = 500,
    device_str: str = "auto",
    worker_id: int = 0,        # for parallel runs: prefix log lines
    use_wandb: bool = False,
    wandb_project: str = "keth-varai",
    lambda_entropy: float = 1e-3,  # channel entropy regularization weight
    input_noise_std: float = 0.05,  # Gaussian noise on A,B inputs during training
) -> dict:
    if device_str == "auto":
        device_str = _default_device()
    device = torch.device(device_str)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = KethVaraiMachine(
        N=N, d=d, encoder_depth=encoder_depth, encoder_width=encoder_width
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    tag = f"[W{worker_id}] N={N} s={seed}"
    print(f"  {tag}  params={total_params}  device={device_str}")

    wb_run = None
    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb.login(force=False)
            wb_run = wandb.init(
                project=wandb_project,
                name=f"N{N}_s{seed}",
                group=f"N{N}",
                config=dict(N=N, seed=seed, d=d, encoder_depth=encoder_depth,
                            encoder_width=encoder_width, lr=lr, batch_size=batch_size,
                            max_steps=max_steps, device=device_str, params=total_params),
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"  {tag}  [wandb] init failed ({e.__class__.__name__}): {e} — continuing without wandb")

    # Optimizer: AdamW with weight decay on encoder, bare Adam on bottleneck/decoder
    enc_params = list(model.enc_a.parameters()) + list(model.enc_b.parameters())
    bilin_params = list(model.U.parameters()) + list(model.V.parameters()) + list(model.W.parameters())
    optimizer = torch.optim.AdamW([
        {"params": enc_params,   "lr": lr, "weight_decay": 1e-4},
        {"params": bilin_params, "lr": lr, "weight_decay": 0.0},
    ])

    # Warmup (2k steps) then cosine decay to 1e-6
    warmup_steps = 2_000

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler — only active on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_rel_err = math.inf
    log: list[dict] = []
    t0 = time.time()
    plateau_patience = 20_000   # steps without improvement before we give up
    last_improvement = 0

    for step in range(1, max_steps + 1):
        model.train()
        A, B, C = sample_batch(batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Input noise: annealed from input_noise_std → 0 over training
            # Prevents encoder from locking into a stable-but-wrong basis early
            noise_scale = input_noise_std * max(0.0, 1.0 - step / max_steps)
            A_noisy = A + noise_scale * torch.randn_like(A)
            B_noisy = B + noise_scale * torch.randn_like(B)
            C_hat = model(A_noisy, B_noisy)
            # Relative Frobenius loss: mean_i(||C_hat_i - C_i||_F / ||C_i||_F)
            # Matches eval metric; prevents bias toward high-magnitude pairs
            rec_loss = frobenius_relative_error(C_hat.float(), C.float())
            # Channel entropy regularization: penalize load concentration
            # -H = sum(p * log(p)), we SUBTRACT it to MAXIMIZE entropy (add to loss)
            u_norms = model.U.weight.norm(dim=1)
            v_norms = model.V.weight.norm(dim=1)
            w_norms = model.W.weight.norm(dim=0)
            act = (u_norms * v_norms * w_norms).clamp(min=1e-12)
            p_act = act / act.sum()
            neg_entropy = (p_act * p_act.log()).sum()  # negative entropy (minimizing this = maximizing entropy)
            loss = rec_loss + lambda_entropy * neg_entropy

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(
                f"  {tag}  step {step:7d}/{max_steps}"
                f"  rel={rel_err:.3e}  bits={bits:.1f}  best={best_rel_err:.3e}"
                f"  lr={current_lr:.1e}  t={elapsed:.0f}s"
            )

            if wb_run is not None:
                metrics = {
                    "loss/rel_err": rel_err,
                    "loss/bits": bits,
                    "loss/best_rel_err": best_rel_err,
                    "loss/noise_scale": input_noise_std * max(0.0, 1.0 - step / max_steps),
                    "train/lr": current_lr,
                }

                # Grad norms per parameter group
                enc_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in list(model.enc_a.parameters()) + list(model.enc_b.parameters())
                    if p.grad is not None
                ) ** 0.5
                bilin_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in list(model.U.parameters()) + list(model.V.parameters()) + list(model.W.parameters())
                    if p.grad is not None
                ) ** 0.5
                metrics["grad/encoder_norm"] = enc_grad
                metrics["grad/bottleneck_norm"] = bilin_grad

                # Channel activity: per-channel |u_k| * |v_k| (proxy for effective rank)
                with torch.no_grad():
                    u_norms = model.U.weight.norm(dim=1)   # (N,)
                    v_norms = model.V.weight.norm(dim=1)   # (N,)
                    w_norms = model.W.weight.norm(dim=0)   # (N,)
                    channel_activity = (u_norms * v_norms * w_norms).cpu()
                    sorted_act, _ = channel_activity.sort(descending=True)
                    metrics["channels/active"] = (channel_activity > channel_activity.max() * 0.01).sum().item()
                    metrics["channels/top1_frac"] = (sorted_act[0] / sorted_act.sum()).item()
                    metrics["channels/top5_frac"] = (sorted_act[:5].sum() / sorted_act.sum()).item()
                    metrics["channels/entropy"] = (
                        -(channel_activity / channel_activity.sum() + 1e-10).log()
                        * (channel_activity / channel_activity.sum() + 1e-10)
                    ).sum().item()

                    # Encoder output statistics (phi_A on eval batch)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        phi_a = model.enc_a(Av)
                        phi_b = model.enc_b(Bv)
                    metrics["encoder/phi_a_mean"] = phi_a.mean().item()
                    metrics["encoder/phi_a_std"] = phi_a.std().item()
                    metrics["encoder/phi_b_std"] = phi_b.std().item()
                    metrics["encoder/phi_a_dead_frac"] = (phi_a.abs() < 1e-4).float().mean().item()

                    # Decoder weight spread
                    metrics["decoder/W_std"] = model.W.weight.std().item()
                    metrics["decoder/W_max"] = model.W.weight.abs().max().item()

                wb_run.log(metrics, step=step)

            if best_rel_err < SUCCESS_THRESHOLD:
                print(f"  {tag}  ✓ SUCCESS at step {step}: best={best_rel_err:.3e}")
                break

            # Plateau guard: stop wasting time if stuck
            if step - last_improvement > plateau_patience and step > warmup_steps * 2:
                print(f"  {tag}  ✗ PLATEAU at step {step}: no improvement for {plateau_patience} steps")
                break

    if wb_run is not None:
        wb_run.summary["best_rel_err"] = best_rel_err
        wb_run.summary["success"] = best_rel_err < SUCCESS_THRESHOLD
        wb_run.finish()

    success = best_rel_err < SUCCESS_THRESHOLD
    return {
        "N": N,
        "seed": seed,
        "d": d,
        "encoder_depth": encoder_depth,
        "encoder_width": encoder_width,
        "device": device_str,
        "best_rel_err": best_rel_err,
        "success": success,
        "steps_run": step,
        "total_params": total_params,
        "log": log,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Keth-Varai Machine")
    parser.add_argument("--N", type=int, default=27)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--encoder_width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="keth-varai")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Keth-Varai Machine  |  N={args.N}  seed={args.seed}")
    print(f"{'='*60}")

    result = train(
        N=args.N,
        seed=args.seed,
        d=args.d,
        encoder_depth=args.encoder_depth,
        encoder_width=args.encoder_width,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.steps,
        log_every=args.log_every,
        device_str=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    out_path = os.path.join(args.out_dir, f"run_N{args.N}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "SUCCESS" if result["success"] else "FAILED"
    print(f"\n  {status}  best_rel_err={result['best_rel_err']:.3e}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()

