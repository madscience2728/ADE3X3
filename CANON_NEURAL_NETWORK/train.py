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

from data import sample_batch, frobenius_relative_error, elementwise_relative_error, worstcase_logsumexp
from model import KethVaraiMachine

SUCCESS_THRESHOLD = 1e-8  # relative Frobenius error target
CHECKPOINT_EVERY = 1_000  # save checkpoint every N steps


def _default_device() -> str:
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
        "log": log,
        "elapsed_s": elapsed_s,
    }
    if plateau_scheduler is not None:
        data["plateau_scheduler"] = plateau_scheduler.state_dict()
    torch.save(data, path)


def train(
    N: int,
    seed: int,
    d: int = 256,
    encoder_depth: int = 8,
    encoder_width: int = 972,
    n_heads: int = 4,
    attn_every: int = 4,
    expanded_products: bool = True,
    lr: float = 3e-4,
    batch_size: int = 4096,
    max_steps: int = 200_000,
    log_every: int = 500,
    device_str: str = "auto",
    worker_id: int = 0,        # for parallel runs: prefix log lines
    use_wandb: bool = False,
    wandb_project: str = "keth-varai",
    lambda_entropy: float = 0.0,   # channel entropy reg (disabled: fights true decomposition)
    input_noise_std: float = 0.0,   # input noise (disabled: conflicts with constant-output goal)
    l1_weight: float = 0.0,    # weight for per-element relative L1 loss (0 = Frobenius only)
    checkpoint_dir: str = "checkpoints",  # directory for checkpoint files
    no_restarts: bool = True,   # single decay (default); False = warm restarts
    use_plateau: bool = True,   # ReduceLROnPlateau (adaptive); False = cosine schedule
    no_amp: bool = False,       # disable AMP, force fp32
    linear_heads: bool = False,  # pure linear heads (depth=0)
    head_depth: int = 1,         # hyper-head hidden layers
    head_width: int = 256,       # hyper-head hidden width
    worst_case: bool = False,    # use logsumexp worst-case loss instead of Frobenius mean
    worst_temp_start: float = 1.0,  # logsumexp temperature at step 0
    worst_temp_end: float = 0.01,   # logsumexp temperature at final step
) -> dict:
    if device_str == "auto":
        device_str = _default_device()
    device = torch.device(device_str)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = KethVaraiMachine(
        N=N, latent_dim=d, encoder_depth=encoder_depth, encoder_width=encoder_width,
        n_heads=n_heads, attn_every=attn_every, expanded_products=expanded_products,
        linear_heads=linear_heads, head_depth=head_depth, head_width=head_width,
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
                            encoder_width=encoder_width, n_heads=n_heads, attn_every=attn_every,
                            lr=lr, batch_size=batch_size,
                            max_steps=max_steps, device=device_str, params=total_params),
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"  {tag}  [wandb] init failed ({e.__class__.__name__}): {e} — continuing without wandb")

    # Optimizer: AdamW with weight decay on encoder, bare Adam on hyper-heads
    # Heads get 0.5× encoder LR — they're a meta-learning problem and need
    # more stable convergence than the encoder.
    enc_params = (list(model.encoder.parameters())
                  + list(model.encoder_blocks.parameters())
                  + list(model.encoder_head.parameters()))
    head_params = list(model.head_U.parameters()) + list(model.head_V.parameters()) + list(model.head_W.parameters())
    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr * 0.5, "weight_decay": 0.0},
    ])

    # Schedule
    warmup_steps = 5_000
    eta_min_ratio = 1e-3

    if use_plateau:
        # Warmup via LambdaLR, then ReduceLROnPlateau takes over
        def warmup_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10_000,
            min_lr=lr * eta_min_ratio, threshold=1e-3, threshold_mode='rel',
        )
        scheduler = warmup_scheduler  # start with warmup
    else:
        restart_period = 50_000
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = step - warmup_steps
            if no_restarts:
                total_decay = max_steps - warmup_steps
                progress = min(t / total_decay, 1.0)
            else:
                cycle_pos = t % restart_period
                progress = cycle_pos / restart_period
            return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        plateau_scheduler = None

    # AMP scaler — only active on CUDA
    use_amp = device.type == "cuda" and not no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if no_amp:
        print(f"  {tag}  AMP DISABLED — pure fp32")

    best_rel_err = math.inf
    log: list[dict] = []
    t0 = time.time()
    plateau_patience = 100_000   # steps without improvement before we give up
    last_improvement = 0
    start_step = 1

    # Checkpoint paths
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_N{N}_s{seed}.pt")

    # Resume from checkpoint if available
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_plateau and "plateau_scheduler" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
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
            # Optional input noise (annealed). Off by default — hurts convergence
            # when the ideal output is input-independent.
            if input_noise_std > 0:
                noise_scale = input_noise_std * max(0.0, 1.0 - step / max_steps)
                A_in = A + noise_scale * torch.randn_like(A)
                B_in = B + noise_scale * torch.randn_like(B)
            else:
                A_in, B_in = A, B

            C_hat = model(A_in, B_in)
            frob_loss = frobenius_relative_error(C_hat.float(), C.float())
            if worst_case:
                # Log-linear temperature anneal
                progress = min(step / max_steps, 1.0)
                wc_temp = worst_temp_start * (worst_temp_end / worst_temp_start) ** progress
                l1_loss = elementwise_relative_error(C_hat.float(), C.float())
                wc_loss = worstcase_logsumexp(C_hat.float(), C.float(), wc_temp)
                rec_loss = 0.5 * l1_loss + 0.5 * wc_loss
            elif l1_weight > 0:
                l1_loss = elementwise_relative_error(C_hat.float(), C.float())
                rec_loss = (1.0 - l1_weight) * frob_loss + l1_weight * l1_loss
            else:
                rec_loss = frob_loss

            # Optional entropy reg. Off by default — penalizes non-uniform
            # channel norms, which fights the true decomposition.
            if lambda_entropy > 0:
                act = model.channel_norms(A_in, B_in).mean(dim=0).clamp(min=1e-12)
                p_act = act / act.sum()
                neg_entropy = (p_act * p_act.log()).sum()
                loss = rec_loss + lambda_entropy * neg_entropy
            else:
                loss = rec_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Per-group grad clipping: prevents encoder gradients from starving heads
        torch.nn.utils.clip_grad_norm_(enc_params, 1.0)
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if use_plateau:
            warmup_scheduler.step()
            if step >= warmup_steps:
                plateau_scheduler.step(rec_loss.item())
        else:
            scheduler.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                Av, Bv, Cv = sample_batch(16_384, device)
                C_hat_v = model(Av, Bv)
                rel_err = frobenius_relative_error(C_hat_v.float(), Cv.float()).item()
                # Per-element diagnostics: which output element has worst error?
                elem_err = (C_hat_v.float() - Cv.float()).abs().view(-1, 9)
                elem_scale = Cv.float().abs().view(-1, 9).clamp(min=1e-12)
                per_elem = (elem_err / elem_scale).mean(dim=0)  # (9,)
                worst_elem = per_elem.argmax().item()
                worst_val = per_elem[worst_elem].item()

            # Worst-case diagnostic: max relative error across all entries in val batch
            with torch.no_grad():
                wc_diff = (C_hat_v.float() - Cv.float()).abs().view(-1, 9)
                wc_scale = Cv.float().view(-1, 9).norm(dim=1, keepdim=True).clamp(min=1e-12)
                max_rel = (wc_diff / wc_scale).max().item()

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

            # Collapse diagnostic (always computed, not just wandb)
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

            wc_info = ""
            if worst_case:
                wc_temp_now = worst_temp_start * (worst_temp_end / worst_temp_start) ** min(step / max_steps, 1.0)
                wc_info = f"  maxrel={max_rel:.3e} T={wc_temp_now:.3f}"
            else:
                wc_info = f"  maxrel={max_rel:.3e}"

            print(
                f"  {tag}  step {step:7d}/{max_steps}"
                f"  rel={rel_err:.3e}  bits={bits:.1f}  best={best_rel_err:.3e}"
                f"  lr={current_lr:.1e}  t={elapsed:.0f}s"
                f"  U={u_collapse:.3f} V={v_collapse:.3f} W={w_collapse:.3f}"
                f"  worst=C[{worst_elem}]:{worst_val:.3f}"
                f"{wc_info}"
            )

            if wb_run is not None:
                metrics = {
                    "loss/rel_err": rel_err,
                    "loss/bits": bits,
                    "loss/best_rel_err": best_rel_err,
                    "loss/noise_scale": input_noise_std * max(0.0, 1.0 - step / max_steps) if input_noise_std > 0 else 0.0,
                    "train/lr": current_lr,
                }

                # Grad norms per parameter group
                enc_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in enc_params
                    if p.grad is not None
                ) ** 0.5
                bilin_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in list(model.head_U.parameters()) + list(model.head_V.parameters()) + list(model.head_W.parameters())
                    if p.grad is not None
                ) ** 0.5
                metrics["grad/encoder_norm"] = enc_grad
                metrics["grad/bottleneck_norm"] = bilin_grad

                # Channel activity: per-channel |u_k| * |v_k| (proxy for effective rank)
                with torch.no_grad():
                    channel_activity = model.channel_norms(Av, Bv).mean(dim=0).cpu()
                    sorted_act, _ = channel_activity.sort(descending=True)
                    metrics["channels/active"] = (channel_activity > channel_activity.max() * 0.01).sum().item()
                    metrics["channels/top1_frac"] = (sorted_act[0] / sorted_act.sum()).item()
                    metrics["channels/top5_frac"] = (sorted_act[:5].sum() / sorted_act.sum()).item()
                    metrics["channels/entropy"] = (
                        -(channel_activity / channel_activity.sum() + 1e-10).log()
                        * (channel_activity / channel_activity.sum() + 1e-10)
                    ).sum().item()

                    # Encoder latent statistics
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        latent = model._encode(torch.cat([Av, Bv], dim=1))
                    metrics["encoder/latent_mean"] = latent.mean().item()
                    metrics["encoder/latent_std"] = latent.std().item()
                    metrics["encoder/latent_dead_frac"] = (latent.abs() < 1e-4).float().mean().item()

                    # Collapse diagnostic: if the encoder has learned, U/V/W
                    # should be ~constant across the batch (std ≈ 0).
                    # collapse_ratio = std(U across batch) / mean(|U|)
                    # Near 0 → encoder outputs constant weights (good).
                    # Near 1 → encoder is thrashing (bad).
                    U_batch = model.head_U(latent).view(-1, model.N, 9)
                    V_batch = model.head_V(latent).view(-1, model.N, 9)
                    W_batch = model.head_W(latent).view(-1, 9, model.N)
                    metrics["collapse/U_ratio"] = (U_batch.std(dim=0).mean() / U_batch.abs().mean().clamp(min=1e-12)).item()
                    metrics["collapse/V_ratio"] = (V_batch.std(dim=0).mean() / V_batch.abs().mean().clamp(min=1e-12)).item()
                    metrics["collapse/W_ratio"] = (W_batch.std(dim=0).mean() / W_batch.abs().mean().clamp(min=1e-12)).item()

                wb_run.log(metrics, step=step)

            if best_rel_err < SUCCESS_THRESHOLD:
                print(f"  {tag}  ✓ SUCCESS at step {step}: best={best_rel_err:.3e}")
                # Save final checkpoint
                _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                                 step, best_rel_err, last_improvement, log, time.time() - t0,
                                 plateau_scheduler=plateau_scheduler if use_plateau else None)
                break

            # Plateau guard: stop wasting time if stuck
            if step - last_improvement > plateau_patience and step > warmup_steps * 2:
                print(f"  {tag}  ✗ PLATEAU at step {step}: no improvement for {plateau_patience} steps")
                break

            # Periodic checkpoint
            if step % CHECKPOINT_EVERY == 0:
                _save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                                 step, best_rel_err, last_improvement, log, time.time() - t0,
                                 plateau_scheduler=plateau_scheduler if use_plateau else None)

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
        "n_heads": n_heads,
        "attn_every": attn_every,
        "device": device_str,
        "best_rel_err": best_rel_err,
        "success": success,
        "steps_run": step,
        "total_params": total_params,
        "log": log,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Keth-Varai Machine")
    parser.add_argument("--N", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--encoder_width", type=int, default=192,
                        help="Must be divisible by 81 (expanded) or 9 (legacy) if using attention")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Attention heads per transformer block")
    parser.add_argument("--attn_every", type=int, default=0,
                        help="Insert 1 AttnBlock every N ResBlocks (0 = no attention)")
    parser.add_argument("--head_depth", type=int, default=1,
                        help="Hyper-head hidden layers (0=linear, 1=shallow GELU, 2+=deep)")
    parser.add_argument("--head_width", type=int, default=256,
                        help="Hyper-head hidden layer width")
    parser.add_argument("--no_expanded", action="store_true",
                        help="Disable 81-product expanded input (use legacy 18-float input)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--restarts", action="store_true", help="Use cosine warm restarts instead of plateau")
    parser.add_argument("--no_plateau", action="store_true", help="Disable ReduceLROnPlateau (use cosine schedule)")
    parser.add_argument("--l1_weight", type=float, default=0.5,
                        help="Weight for per-element relative L1 loss (0=Frobenius only, 0.5=equal mix)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP, force fp32 (slower but higher precision)")
    parser.add_argument("--linear_heads", action="store_true", help="Use pure linear heads (depth=0, ignores head_depth)")
    parser.add_argument("--no_worst_case", action="store_true", help="Disable logsumexp worst-case loss, use Frobenius mean")
    parser.add_argument("--worst_temp_start", type=float, default=1.0, help="Initial logsumexp temperature")
    parser.add_argument("--worst_temp_end", type=float, default=0.01, help="Final logsumexp temperature")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="keth-varai")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Keth-Varai Machine  |  N={args.N}  seed={args.seed}")
    print(f"{'='*60}")

    expanded = not args.no_expanded
    print(f"  products={'81 expanded' if expanded else '18 legacy'}")

    result = train(
        N=args.N,
        seed=args.seed,
        d=args.d,
        encoder_depth=args.encoder_depth,
        encoder_width=args.encoder_width,
        n_heads=args.n_heads,
        attn_every=args.attn_every,
        expanded_products=expanded,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.steps,
        log_every=args.log_every,
        device_str=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        no_restarts=not args.restarts,
        use_plateau=not args.no_plateau,
        l1_weight=args.l1_weight,
        no_amp=args.no_amp,
        linear_heads=args.linear_heads,
        head_depth=args.head_depth,
        head_width=args.head_width,
        worst_case=not args.no_worst_case,
        worst_temp_start=args.worst_temp_start,
        worst_temp_end=args.worst_temp_end,
    )

    out_path = os.path.join(args.out_dir, f"run_N{args.N}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "SUCCESS" if result["success"] else "FAILED"
    print(f"\n  {status}  best_rel_err={result['best_rel_err']:.3e}")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()

