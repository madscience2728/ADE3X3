"""
raw_param_test.py — Direct parameter optimization (no encoder, no heads).

Tests whether the bilinear bottleneck C_hat = W @ diag(U@A * V@B) can
represent 3x3 matmul at rank N. If this walls, the formulation is the problem.
If it succeeds, the hyper-head path is the bottleneck.

Usage:
    python raw_param_test.py --N 27
    python raw_param_test.py --N 23
    python raw_param_test.py --N 19
"""

import argparse
import math
import time

import torch
from data import sample_batch, frobenius_relative_error, elementwise_relative_error


def run(N: int, seed: int, steps: int, lr: float, batch_size: int, device_str: str, l1_weight: float, use_fp16: bool = False):
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Raw parameters — exactly what the hyper-heads would output
    U = torch.nn.Parameter(torch.randn(N, 9, device=device) * 0.1)
    V = torch.nn.Parameter(torch.randn(N, 9, device=device) * 0.1)
    W = torch.nn.Parameter(torch.randn(9, N, device=device) * 0.1)

    optimizer = torch.optim.Adam([U, V, W], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000,
        min_lr=1e-7, threshold=1e-3, threshold_mode='rel',
    )

    best = math.inf
    t0 = time.time()
    use_amp = use_fp16 and device.type == "cuda"
    if use_amp:
        print("  AMP ENABLED — fp16 forward pass")

    for step in range(1, steps + 1):
        A, B, C = sample_batch(batch_size, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Bilinear bottleneck: C_hat = W @ diag(U@A^T * V@B^T)
            UA = U @ A.T          # (N, batch)
            VB = V @ B.T          # (N, batch)
            mid = UA * VB         # (N, batch) — elementwise
            C_hat = (W @ mid).T   # (batch, 9)

            frob = frobenius_relative_error(C_hat.float(), C.float())
            if l1_weight > 0:
                l1 = elementwise_relative_error(C_hat.float(), C.float())
                loss = (1.0 - l1_weight) * frob + l1_weight * l1
            else:
                loss = frob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        rel = frob.item()
        if rel < best:
            best = rel

        if step % 1000 == 0 or step == 1:
            bits = -math.log2(best) if best > 0 else 999
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            print(f"  step {step:7d}/{steps}  rel={rel:.3e}  bits={bits:.1f}  best={best:.3e}  lr={current_lr:.1e}  t={elapsed:.0f}s")

    bits = -math.log2(best) if best > 0 else 999
    print(f"\n  FINAL  N={N}  best_rel={best:.3e}  bits={bits:.1f}")

    # Save the raw solution
    out_path = f"raw_UVW_N{N}_s{seed}.pt"
    torch.save({"U": U.detach().cpu(), "V": V.detach().cpu(), "W": W.detach().cpu(),
                "best_rel": best, "bits": bits}, out_path)
    print(f"  Saved → {out_path}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw parameter test for bilinear bottleneck")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true", help="Enable AMP fp16 (test precision hypothesis)")
    args = parser.parse_args()

    print(f"\n  Raw Parameter Test  |  N={args.N}  seed={args.seed}")
    print(f"  No encoder, no heads — just U({args.N},9) V({args.N},9) W(9,{args.N})")
    print()

    run(args.N, args.seed, args.steps, args.lr, args.batch_size, args.device, args.l1_weight, args.fp16)
