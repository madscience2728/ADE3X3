"""
train_kolmogorov.py -- Training loop for the Bundle Machine.

Objective:
  min E[loss(C, C_hat)]
    + beta * KL(q(s|A,B) || p(s))     # IB penalty on strategy code
    + lambda_n * N                      # execution rank cost (fixed per run)

The beta parameter controls the information bottleneck tightness.
  beta=0: no bottleneck, strategy code is unconstrained
  beta->inf: strategy code collapses to prior, no input info passes through

Usage:
    python train_kolmogorov.py --N 1 --strategy_dim 9 --beta 0.001
    python train_kolmogorov.py --N 1 --strategy_dim 7 --beta 0.01
    python train_kolmogorov.py --N 11 --strategy_dim 9 --beta 0.0
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from data import sample_batch, frobenius_relative_error, elementwise_relative_error, worstcase_logsumexp
from model_kolmogorov import BundleMachine

CHECKPOINT_EVERY = 1_000


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_checkpoint(path, model, optimizer, scheduler, scaler,
                     step, best_rel_err, last_improvement, log, elapsed_s,
                     plateau_scheduler=None):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_rel_err": best_rel_err,
        "last_improvement": last_improvement,
        "log": log[-100:],
        "elapsed_s": elapsed_s,
    }
    if plateau_scheduler is not None:
        data["plateau_scheduler"] = plateau_scheduler.state_dict()
    torch.save(data, path)


def train(
    N: int = 1,
    strategy_dim: int = 9,
    encoder_width: int = 192,
    encoder_depth: int = 2,
    head_width: int = 256,
    head_depth: int = 1,
    beta: float = 0.001,           # IB penalty weight
    beta_warmup: int = 10_000,     # steps to linearly warm up beta from 0
    stochastic: bool = True,
    seed: int = 42,
    steps: int = 200_000,
    batch_size: int = 16384,
    lr: float = 1e-3,
    device: str = _default_device(),
    worst_case: bool = True,
    worst_temp_start: float = 1.0,
    worst_temp_end: float = 0.01,
):
    torch.manual_seed(seed)
    dev = torch.device(device)
    use_amp = (device == "cuda")

    tag = f"K_N{N}_s{strategy_dim}_b{beta}_seed{seed}"

    model = BundleMachine(
        N=N, strategy_dim=strategy_dim,
        encoder_width=encoder_width, encoder_depth=encoder_depth,
        head_width=head_width, head_depth=head_depth,
        stochastic=stochastic,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print(f"  Bundle Machine (Kolmogorov)  |  N={N}  K_s={strategy_dim}  beta={beta}  seed={seed}")
    print("=" * 60)
    print(f"  params={n_params}  device={device}  stochastic={stochastic}")

    # Optimizer: single group
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_steps = 5_000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10_000, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")

    best_rel_err = float("inf")
    last_improvement = 0
    log = []
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        A, B, C = sample_batch(batch_size, dev)

        with torch.cuda.amp.autocast(enabled=use_amp):
            C_hat, kl_loss = model(A, B)

            # Reconstruction loss
            if worst_case:
                frac = step / steps
                temp = worst_temp_start * (worst_temp_end / worst_temp_start) ** frac
                l_elem = elementwise_relative_error(C_hat, C)
                l_worst = worstcase_logsumexp(C_hat, C, temp)
                recon_loss = 0.5 * l_elem + 0.5 * l_worst
            else:
                recon_loss = elementwise_relative_error(C_hat, C)

            # Beta warmup
            effective_beta = beta * min(1.0, step / beta_warmup) if beta_warmup > 0 else beta

            # Total loss
            loss = recon_loss + effective_beta * kl_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Validation
        if step % 200 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                A_v, B_v, C_v = sample_batch(batch_size, dev)
                C_hat_v, kl_v = model(A_v, B_v)
                rel = frobenius_relative_error(C_hat_v, C_v).item()
                bits = -math.log2(rel) if rel > 0 else 99

                # Per-element worst
                diff = (C_hat_v - C_v).abs()
                scale = C_v.view(-1, 9).norm(dim=1, keepdim=True).clamp(min=1e-12)
                per_elem = (diff / scale).mean(dim=0)
                worst_idx = per_elem.argmax().item()
                worst_val = per_elem[worst_idx].item()

                max_rel = (diff / scale).max().item()

            if rel < best_rel_err:
                best_rel_err = rel
                last_improvement = step

            plateau_scheduler.step(rel)

            elapsed = time.time() - t0

            # Strategy code stats
            with torch.no_grad():
                s, _ = model.encode(A_v, B_v)
                s_std = s.std(dim=0).mean().item()
                s_norm = s.norm(dim=1).mean().item()

            current_lr = optimizer.param_groups[0]['lr']

            print(f"  [{tag}]  step {step:>6d}/{steps}  "
                  f"rel={rel:.3e}  bits={bits:.1f}  best={best_rel_err:.3e}  "
                  f"lr={current_lr:.1e}  t={elapsed:.0f}s  "
                  f"KL={kl_v.item():.2f}  s_std={s_std:.3f}  s_norm={s_norm:.3f}  "
                  f"eb={effective_beta:.4f}  "
                  f"worst=C[{worst_idx}]:{worst_val:.3f}  maxrel={max_rel:.3e}")

            log.append({
                "step": step, "rel": rel, "bits": bits,
                "best": best_rel_err, "kl": kl_v.item(),
                "s_std": s_std, "s_norm": s_norm, "beta_eff": effective_beta,
            })

            # Early plateau detection
            if step - last_improvement > 100_000:
                print(f"  [{tag}]  PLATEAU — no improvement for 100k steps. Stopping.")
                break

        if step % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                             step, best_rel_err, last_improvement, log,
                             time.time() - t0, plateau_scheduler)

    # Final save
    elapsed = time.time() - t0
    _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                     step, best_rel_err, last_improvement, log, elapsed,
                     plateau_scheduler)

    print(f"\n  [{tag}]  DONE  best_rel_err={best_rel_err:.6e}  "
          f"steps={step}  time={elapsed:.0f}s")

    # Save results JSON
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"kolmogorov_{tag}.json")
    with open(results_path, "w") as f:
        json.dump({
            "N": N, "strategy_dim": strategy_dim, "beta": beta,
            "stochastic": stochastic, "seed": seed,
            "best_rel_err": best_rel_err, "steps": step,
            "params": n_params, "elapsed_s": elapsed,
            "log": log,
        }, f, indent=2)

    return best_rel_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=27)
    p.add_argument("--strategy_dim", type=int, default=9)
    p.add_argument("--encoder_width", type=int, default=192)
    p.add_argument("--encoder_depth", type=int, default=2)
    p.add_argument("--head_width", type=int, default=256)
    p.add_argument("--head_depth", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--beta_warmup", type=int, default=10_000)
    p.add_argument("--no_stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1_200_000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=_default_device())
    p.add_argument("--no_worst_case", action="store_true")
    p.add_argument("--worst_temp_start", type=float, default=1.0)
    p.add_argument("--worst_temp_end", type=float, default=0.01)
    args = p.parse_args()

    train(
        N=args.N, strategy_dim=args.strategy_dim,
        encoder_width=args.encoder_width, encoder_depth=args.encoder_depth,
        head_width=args.head_width, head_depth=args.head_depth,
        beta=args.beta, beta_warmup=args.beta_warmup,
        stochastic=not args.no_stochastic,
        seed=args.seed, steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, device=args.device,
        worst_case=not args.no_worst_case,
        worst_temp_start=args.worst_temp_start,
        worst_temp_end=args.worst_temp_end,
    )


if __name__ == "__main__":
    main()
