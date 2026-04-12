"""
control.py — Direct tensor decomposition baseline (no encoder).

U, V, W are plain learnable parameters, not functions of input.
This is the control experiment: can SGD on random (A,B) pairs find
the matmul decomposition directly?

    Ĉ = W @ (U@A ⊙ V@B)

Parameters: N×9 + N×9 + 9×N = 27N  (e.g. N=19 → 513 params)
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from data import frobenius_relative_error


class DirectDecomp(nn.Module):
    """Fixed U, V, W — no encoder, no input-dependence."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.U = nn.Parameter(torch.randn(N, 9) * 0.1)
        self.V = nn.Parameter(torch.randn(N, 9) * 0.1)
        self.W = nn.Parameter(torch.randn(9, N) * 0.1)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A, B: (batch, 9)
        p = A @ self.U.T  # (batch, N)
        q = B @ self.V.T  # (batch, N)
        m = p * q          # (batch, N)
        return m @ self.W.T  # (batch, 9)


def sample_batch(batch_size, device):
    A = torch.randn(batch_size, 9, device=device)
    B = torch.randn(batch_size, 9, device=device)
    A3 = A.view(-1, 3, 3).to(torch.float64)
    B3 = B.view(-1, 3, 3).to(torch.float64)
    C = (A3 @ B3).view(-1, 9).to(torch.float32)
    return A, B, C


def relative_error(C_hat, C):
    """Per-sample mean relative Frobenius error (matches full model metric)."""
    return frobenius_relative_error(C_hat, C)


def main():
    parser = argparse.ArgumentParser(description="Control: direct tensor decomposition")
    parser.add_argument("--N", type=int, default=23)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = DirectDecomp(args.N).to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  CONTROL: Direct Decomposition  |  N={args.N}  params={params}")
    print(f"  device={device}  lr={args.lr}  batch={args.batch_size}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-5
    )

    best = math.inf
    t0 = time.time()
    log = []

    for step in range(1, args.steps + 1):
        A, B, C = sample_batch(args.batch_size, device)
        C_hat = model(A, B)
        loss = relative_error(C_hat, C)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                Av, Bv, Cv = sample_batch(4096, device)
                err = relative_error(model(Av, Bv), Cv).item()

            if err < best:
                best = err
            bits = -math.log2(err) if err > 0 else float('inf')
            elapsed = time.time() - t0
            log.append({"step": step, "err": err, "best": best, "elapsed": round(elapsed, 1)})
            print(f"  step {step:>7d}/{args.steps}  rel={err:.3e}  bits={bits:.1f}  best={best:.3e}  t={elapsed:.0f}s")

            if best < 1e-5:
                print(f"\n  ★ SUCCESS at step {step}!")
                break

    elapsed = time.time() - t0
    bits = -math.log2(best) if best > 0 else float('inf')
    print(f"\n{'='*60}")
    print(f"  Done: N={args.N}  best={best:.3e}  bits={bits:.1f}  time={elapsed:.0f}s")
    print(f"{'='*60}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open(f"results/control_N{args.N}_s{args.seed}.json", "w") as f:
        json.dump({"N": args.N, "seed": args.seed, "best": best, "params": params, "log": log}, f, indent=2)


if __name__ == "__main__":
    main()
