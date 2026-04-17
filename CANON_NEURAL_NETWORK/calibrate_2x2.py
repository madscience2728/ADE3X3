"""
calibrate_2x2.py — Strassen's Calibration Harness.

2×2 matrix multiplication:
  - 8 inputs (A flattened + B flattened), 4 outputs (C flattened)
  - Known: R=7 (Strassen 1969), R=6 open, R<6 impossible
  - Probe should show: N=7 converges fast, N=6 struggles, N=5 fails
  - If this holds: instrument calibrated for 3×3

Uses the same bilinear bottleneck architecture as the 3×3 system:
  C_hat = W @ ((U @ A) * (V @ B))

Parameters per rank: N × (4 + 4 + 4) = 12N
  N=7: 84 params, N=6: 72, N=8: 96

Usage:
    python calibrate_2x2.py                       # sweep N=5..8, 3 seeds
    python calibrate_2x2.py --N 7 --seed 0        # single run
    python calibrate_2x2.py --sweep --min_N 4 --max_N 9 --seeds 5
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data: 2×2 matmul
# ---------------------------------------------------------------------------

def sample_batch_2x2(batch_size: int, device: torch.device):
    """Random 2×2 pairs. Ground truth in float64 → float32."""
    A = torch.randn(batch_size, 4, dtype=torch.float64)
    B = torch.randn(batch_size, 4, dtype=torch.float64)
    A_mat = A.view(batch_size, 2, 2)
    B_mat = B.view(batch_size, 2, 2)
    C_mat = torch.bmm(A_mat, B_mat)
    return (A.float().to(device),
            B.float().to(device),
            C_mat.view(batch_size, 4).float().to(device))


def frobenius_relative_error_2x2(C_pred, C_true):
    diff = (C_pred - C_true).view(-1, 4)
    true_norm = C_true.view(-1, 4).norm(dim=1).clamp(min=1e-12)
    return (diff.norm(dim=1) / true_norm).mean()


# ---------------------------------------------------------------------------
# Model: Direct bilinear decomposition for 2×2
# ---------------------------------------------------------------------------

class DirectDecomp2x2(nn.Module):
    """Fixed U, V, W for 2×2 matmul. No encoder."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.U = nn.Parameter(torch.randn(N, 4) * 0.1)
        self.V = nn.Parameter(torch.randn(N, 4) * 0.1)
        self.W = nn.Parameter(torch.randn(4, N) * 0.1)

    def forward(self, A, B):
        p = A @ self.U.T   # (batch, N)
        q = B @ self.V.T   # (batch, N)
        m = p * q           # (batch, N)
        return m @ self.W.T # (batch, 4)


# ---------------------------------------------------------------------------
# Logging schema (Emmy's format)
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    """Per-run structured log — every field Emmy specified."""
    arch: str                           # architecture fingerprint
    N: int
    seed: int
    param_count: int
    final_error: float = float("inf")
    best_error: float = float("inf")
    converged: bool = False
    convergence_step: int = -1          # step where first < 1e-5
    thresholds: dict = field(default_factory=dict)  # threshold → first step crossed
    curve: list = field(default_factory=list)       # (step, error) every 10 steps
    weight_stats: dict = field(default_factory=dict)
    gradient_norms: list = field(default_factory=list)
    elapsed_s: float = 0.0
    # Feynman's failure fields
    failure_mode: str = ""              # "no_convergence" | "plateau" | "wrong_basin" | ""
    per_output_error: list = field(default_factory=list)  # per C[i] final error
    vanishing_layers: list = field(default_factory=list)  # layers with grad < 1e-6


THRESHOLDS = [0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7]


# ---------------------------------------------------------------------------
# Near-integer score (Emmy's bridge to algebra)
# ---------------------------------------------------------------------------

def near_integer_score(params: dict[str, torch.Tensor]) -> dict:
    """How close are the weights to integers? Returns stats per parameter."""
    stats = {}
    for name, p in params.items():
        d = p.detach().cpu().float()
        residuals = (d - d.round()).abs()
        stats[name] = {
            "mean_residual": residuals.mean().item(),
            "max_residual": residuals.max().item(),
            "frac_within_0.1": (residuals < 0.1).float().mean().item(),
            "frac_within_0.01": (residuals < 0.01).float().mean().item(),
        }
    return stats


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(N: int, seed: int, steps: int = 5000, batch_size: int = 4096,
               lr: float = 1e-2, device: str = "auto", log_every: int = 10) -> RunLog:
    """Train DirectDecomp2x2 at rank N. Returns structured RunLog."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)

    model = DirectDecomp2x2(N).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    rlog = RunLog(
        arch=f"DirectDecomp2x2_N{N}",
        N=N, seed=seed, param_count=n_params,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5
    )

    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        A, B, C = sample_batch_2x2(batch_size, dev)
        C_hat = model(A, B)
        loss = frobenius_relative_error_2x2(C_hat, C)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient norms per layer (Feynman's request)
        if step % (log_every * 10) == 0:
            gnorms = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    gnorms[name] = p.grad.norm().item()
            rlog.gradient_norms.append({"step": step, **gnorms})

        optimizer.step()
        scheduler.step()

        # Evaluation
        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                Av, Bv, Cv = sample_batch_2x2(4096, dev)
                C_hat_v = model(Av, Bv)
                err = frobenius_relative_error_2x2(C_hat_v, Cv).item()

                # Per-output errors (Feynman's request)
                diff = (C_hat_v - Cv).abs()
                scale = Cv.view(-1, 4).norm(dim=1, keepdim=True).clamp(min=1e-12)
                per_elem = (diff / scale).mean(dim=0).tolist()

            rlog.curve.append({"step": step, "error": err})

            if err < rlog.best_error:
                rlog.best_error = err

            # Threshold crossings
            for t in THRESHOLDS:
                key = f"{t:.0e}"
                if key not in rlog.thresholds and err < t:
                    rlog.thresholds[key] = step

    rlog.elapsed_s = time.time() - t0
    rlog.final_error = err
    rlog.per_output_error = per_elem

    # Convergence assessment
    if rlog.best_error < 1e-5:
        rlog.converged = True
        rlog.convergence_step = rlog.thresholds.get("1e-05", steps)
    else:
        # Feynman's failure taxonomy
        errors = [c["error"] for c in rlog.curve]
        if len(errors) > 10:
            early = sum(errors[:5]) / 5
            late = sum(errors[-5:]) / 5
            if late > 0.5:
                rlog.failure_mode = "no_convergence"
            elif late > early * 0.9:
                rlog.failure_mode = "plateau"
            else:
                rlog.failure_mode = "slow_descent"

    # Weight stats (near-integer score)
    rlog.weight_stats = near_integer_score(dict(model.named_parameters()))

    # Vanishing gradient check
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.norm().item() < 1e-6:
            rlog.vanishing_layers.append(name)

    return rlog, model


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(min_N: int = 5, max_N: int = 8, seeds: int = 3,
          steps: int = 5000, device: str = "auto"):
    """Run calibration sweep. The core validation experiment."""

    results = []
    print(f"\n{'='*70}")
    print(f"  2×2 MATMUL CALIBRATION SWEEP  |  N={min_N}..{max_N}  seeds={seeds}  steps={steps}")
    print(f"  Known: R=7 (Strassen). Expect N≥7 converge, N<7 struggle/fail.")
    print(f"{'='*70}\n")

    for N in range(min_N, max_N + 1):
        for s in range(seeds):
            print(f"  --- N={N}  seed={s} ---")
            rlog, model = run_single(N, seed=s, steps=steps, device=device)
            bits = -math.log2(rlog.best_error) if rlog.best_error > 0 else 99
            status = "✓ CONVERGED" if rlog.converged else f"✗ {rlog.failure_mode}"
            print(f"  N={N} s={s}  best={rlog.best_error:.3e}  bits={bits:.1f}  "
                  f"{status}  time={rlog.elapsed_s:.1f}s")

            # Save successful weights for algebraic inspection
            if rlog.converged:
                wdir = os.path.join(os.path.dirname(__file__), "results", "calibration_2x2")
                os.makedirs(wdir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(wdir, f"weights_N{N}_s{s}.pt"))

            results.append(asdict(rlog))
            print()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'N':>3}  {'Conv Rate':>10}  {'Mean Best':>12}  {'Mean Bits':>10}  {'Mean Step@1e-5':>15}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*15}")

    for N in range(min_N, max_N + 1):
        runs = [r for r in results if r["N"] == N]
        conv = sum(1 for r in runs if r["converged"])
        mean_best = sum(r["best_error"] for r in runs) / len(runs)
        bits = -math.log2(mean_best) if mean_best > 0 else 99
        conv_steps = [r["convergence_step"] for r in runs if r["converged"]]
        mean_step = sum(conv_steps) / len(conv_steps) if conv_steps else -1
        print(f"  {N:>3}  {conv}/{len(runs):>8}  {mean_best:>12.3e}  {bits:>10.1f}  "
              f"{'N/A' if mean_step < 0 else f'{mean_step:>15.0f}'}")

    # Calibration verdict
    n7_conv = sum(1 for r in results if r["N"] == 7 and r["converged"])
    n6_conv = sum(1 for r in results if r["N"] == 6 and r["converged"])
    n7_total = sum(1 for r in results if r["N"] == 7)

    print(f"\n  VERDICT:", end=" ")
    if n7_conv == n7_total and n6_conv == 0:
        print("CALIBRATED ✓ — N=7 converges, N=6 does not. Instrument valid.")
    elif n7_conv == n7_total:
        print(f"PARTIAL — N=7 converges, but N=6 also converges ({n6_conv}/{seeds}). "
              "Check if N=6 weights are algebraically valid.")
    elif n7_conv > 0:
        print(f"WEAK — N=7 converges {n7_conv}/{n7_total}. May need more steps or seeds.")
    else:
        print("FAILED — N=7 does not converge. Instrument broken. Do not proceed to 3×3.")
    print(f"{'='*70}\n")

    # Save full results
    outdir = os.path.join(os.path.dirname(__file__), "results", "calibration_2x2")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "calibration_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {outpath}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2×2 matmul calibration harness")
    parser.add_argument("--N", type=int, default=None, help="Single N to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", default=True,
                        help="Run full sweep (default)")
    parser.add_argument("--min_N", type=int, default=5)
    parser.add_argument("--max_N", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.N is not None:
        # Single run mode
        rlog, model = run_single(args.N, args.seed, args.steps, device=args.device)
        bits = -math.log2(rlog.best_error) if rlog.best_error > 0 else 99
        print(f"\n  N={args.N}  best={rlog.best_error:.3e}  bits={bits:.1f}  "
              f"converged={rlog.converged}  time={rlog.elapsed_s:.1f}s")
        print(f"  Near-integer scores:")
        for name, stats in rlog.weight_stats.items():
            print(f"    {name}: mean_residual={stats['mean_residual']:.4f}  "
                  f"frac_within_0.1={stats['frac_within_0.1']:.2%}")
    else:
        sweep(args.min_N, args.max_N, args.seeds, args.steps, args.device)


if __name__ == "__main__":
    main()
