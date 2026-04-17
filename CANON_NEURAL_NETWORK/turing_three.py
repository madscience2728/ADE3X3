"""
turing_three.py — Turing's three experiments.

1. Sweep N=5,6,7,8 on 2×2 with MLP-w12-d2. Measure bits/step. Find the cliff.
2. Sweep k (selector overhead) at N=7 on 2×2. Find optimal budget.
3. Extract MLP-w12-d2 weights at step 200. Check near-integer structure.
"""

import math
import torch
import torch.nn as nn
import json
import os
from population_probe import (
    TinyBilinear, ArchSpec, sample_batch, relative_error,
    near_integer_score, PROBLEMS,
)


def make_mlp_spec(N, width=12, depth=2, use_products=True, head_depth=0,
                  activation="gelu"):
    return ArchSpec(
        name=f"mlp_w{width}_d{depth}_N{N}",
        N=N, selector_type="mlp",
        selector_width=width, selector_depth=depth,
        selector_activation=activation,
        use_products=use_products, head_depth=head_depth,
    )


def train_and_measure(spec, problem="2x2", steps=200, batch_size=2048,
                      lr=1e-2, seeds=5, device="auto"):
    """Train, return bits/step and final model for each seed."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    cfg = PROBLEMS[problem]
    d_out = cfg["output_dim"]

    results = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = TinyBilinear(spec, problem).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for step in range(1, steps + 1):
            model.train()
            A, B, C = sample_batch(problem, batch_size, dev)
            C_hat = model(A, B)
            loss = relative_error(C_hat, C, d_out)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Final eval
        model.eval()
        with torch.no_grad():
            Av, Bv, Cv = sample_batch(problem, 4096, dev)
            err = relative_error(model(Av, Bv), Cv, d_out).item()

        bits = -math.log2(err) if err > 0 else 99
        bits_per_step = bits / steps
        results.append({
            "seed": seed,
            "error": err,
            "bits": bits,
            "bits_per_step": bits_per_step,
            "params": model.param_count(),
            "model": model,
        })

    return results


# ===================================================================
# EXPERIMENT 1: N sweep — find the cliff in bits/step
# ===================================================================

def experiment_1():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: N sweep on 2×2 with MLP-w12-d2")
    print("  Looking for cliff in bits/step at N=6→7 (Strassen's theorem)")
    print("=" * 70 + "\n")

    rows = []
    for N in [5, 6, 7, 8]:
        spec = make_mlp_spec(N, width=12, depth=2)
        results = train_and_measure(spec, problem="2x2", steps=200, seeds=5)

        mean_bits = sum(r["bits"] for r in results) / len(results)
        mean_bps = sum(r["bits_per_step"] for r in results) / len(results)
        best_err = min(r["error"] for r in results)
        params = results[0]["params"]

        rows.append({
            "N": N, "params": params,
            "mean_bits": mean_bits, "mean_bits_per_step": mean_bps,
            "best_error": best_err,
        })
        print(f"  N={N}  params={params:>5d}  bits={mean_bits:.2f}  "
              f"bits/step={mean_bps:.4f}  best_err={best_err:.3e}")

    # Show cliff
    print(f"\n  CLIFF ANALYSIS (bits/step ratio between consecutive N):")
    for i in range(1, len(rows)):
        prev = rows[i - 1]["mean_bits_per_step"]
        curr = rows[i]["mean_bits_per_step"]
        ratio = curr / prev if prev > 0 else float("inf")
        print(f"    N={rows[i-1]['N']}→{rows[i]['N']}:  "
              f"{prev:.4f} → {curr:.4f}  ratio={ratio:.2f}×")

    # Also run direct decomposition as control
    print(f"\n  CONTROL (direct decomposition, same steps):")
    for N in [5, 6, 7, 8]:
        spec = ArchSpec(
            name=f"direct_N{N}", N=N, selector_type="none",
            selector_width=0, selector_depth=0,
            selector_activation="none",
            use_products=False, head_depth=0,
        )
        results = train_and_measure(spec, problem="2x2", steps=200, seeds=5)
        mean_bits = sum(r["bits"] for r in results) / len(results)
        mean_bps = sum(r["bits_per_step"] for r in results) / len(results)
        print(f"    N={N}  bits={mean_bits:.2f}  bits/step={mean_bps:.4f}")

    return rows


# ===================================================================
# EXPERIMENT 2: Budget sweep — find optimal k
# ===================================================================

def experiment_2():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Budget sweep at N=7 on 2×2")
    print("  Bilinear cost = 84. Sweeping selector overhead multiplier k.")
    print("=" * 70 + "\n")

    bilinear_cost = 7 * 12  # N * (4+4+4) for 2×2
    rows = []

    for k in [1, 2, 3, 5, 8, 12, 18]:
        budget = int(bilinear_cost * (1 + k))
        # Find largest width that fits
        best_spec = None
        for width in [4, 6, 8, 10, 12, 16, 20, 24, 32, 48]:
            spec = make_mlp_spec(7, width=width, depth=2)
            try:
                model = TinyBilinear(spec, "2x2")
                if model.param_count() <= budget:
                    best_spec = spec
                else:
                    break
            except Exception:
                continue

        if best_spec is None:
            # Budget too small for any hypernetwork, use direct
            best_spec = ArchSpec(
                name=f"direct_N7", N=7, selector_type="none",
                selector_width=0, selector_depth=0,
                selector_activation="none",
                use_products=False, head_depth=0,
            )

        results = train_and_measure(best_spec, problem="2x2", steps=200, seeds=5)
        mean_bits = sum(r["bits"] for r in results) / len(results)
        mean_bps = sum(r["bits_per_step"] for r in results) / len(results)
        params = results[0]["params"]

        rows.append({
            "k": k, "budget": budget, "actual_params": params,
            "arch": best_spec.name,
            "mean_bits": mean_bits, "mean_bits_per_step": mean_bps,
        })
        print(f"  k={k:>2d}  budget={budget:>5d}  actual={params:>5d}  "
              f"{best_spec.name:<25s}  bits={mean_bits:.2f}  bits/step={mean_bps:.4f}")

    # Find optimal k
    best_row = max(rows, key=lambda r: r["mean_bits_per_step"])
    print(f"\n  OPTIMAL: k={best_row['k']}  budget={best_row['budget']}  "
          f"bits/step={best_row['mean_bits_per_step']:.4f}")
    print(f"  For 3×3 at N=23: bilinear=345, budget={int(345 * (1 + best_row['k']))}")

    return rows


# ===================================================================
# EXPERIMENT 3: Weight extraction — near-integer check
# ===================================================================

def experiment_3():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Weight extraction from MLP-w12-d2 at N=7")
    print("  Looking for near-integer structure in U, V, W after 200 steps")
    print("=" * 70 + "\n")

    spec = make_mlp_spec(7, width=12, depth=2)

    # Train with best seed from experiment 1
    results = train_and_measure(spec, problem="2x2", steps=200, seeds=10)
    best = min(results, key=lambda r: r["error"])
    model = best["model"]

    print(f"  Best seed={best['seed']}  error={best['error']:.3e}  bits={best['bits']:.2f}\n")

    # Extract weights for a specific input and check near-integer structure
    model.eval()
    dev = next(model.parameters()).device

    # Generate a batch and extract the generated U, V, W
    torch.manual_seed(0)
    A = torch.randn(100, 4, device=dev)
    B = torch.randn(100, 4, device=dev)

    with torch.no_grad():
        # Get products
        x = (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, 16)
        latent = model.selector(x)
        U_raw = model.head_U(latent).view(100, 7, 4)
        V_raw = model.head_V(latent).view(100, 7, 4)
        W_raw = model.head_W(latent).view(100, 4, 7)

    # Check: are U, V, W input-dependent or roughly constant?
    U_std_across_inputs = U_raw.std(dim=0).mean().item()
    V_std_across_inputs = V_raw.std(dim=0).mean().item()
    W_std_across_inputs = W_raw.std(dim=0).mean().item()

    print(f"  Input-dependence (std across 100 inputs):")
    print(f"    U: {U_std_across_inputs:.4f}")
    print(f"    V: {V_std_across_inputs:.4f}")
    print(f"    W: {W_std_across_inputs:.4f}")

    if max(U_std_across_inputs, V_std_across_inputs, W_std_across_inputs) < 0.01:
        print(f"  → COLLAPSED: U, V, W are approximately constant (input-independent)")
        print(f"  → The hypernetwork found a fixed decomposition")
    else:
        print(f"  → INPUT-DEPENDENT: U, V, W vary with (A, B)")
        print(f"  → The hypernetwork is using input-adaptive decomposition")

    # Near-integer check on mean U, V, W
    U_mean = U_raw.mean(dim=0)
    V_mean = V_raw.mean(dim=0)
    W_mean = W_raw.mean(dim=0)

    print(f"\n  Near-integer analysis (mean U, V, W):")
    for name, tensor in [("U", U_mean), ("V", V_mean), ("W", W_mean)]:
        t = tensor.cpu().float()
        residuals = (t - t.round()).abs()
        print(f"    {name}: shape={list(t.shape)}  "
              f"mean_residual={residuals.mean():.4f}  "
              f"max_residual={residuals.max():.4f}  "
              f"frac<0.1: {(residuals < 0.1).float().mean():.1%}  "
              f"frac<0.3: {(residuals < 0.3).float().mean():.1%}")

    # Print the actual mean matrices (for algebraic inspection)
    print(f"\n  Mean U (7×4):")
    for i in range(7):
        row = U_mean[i].cpu().tolist()
        rounded = [round(x, 3) for x in row]
        nearest_int = [round(x) for x in row]
        print(f"    ch{i}: {rounded}  ≈ {nearest_int}")

    print(f"\n  Mean V (7×4):")
    for i in range(7):
        row = V_mean[i].cpu().tolist()
        rounded = [round(x, 3) for x in row]
        nearest_int = [round(x) for x in row]
        print(f"    ch{i}: {rounded}  ≈ {nearest_int}")

    print(f"\n  Mean W (4×7):")
    for i in range(4):
        row = W_mean[i].cpu().tolist()
        rounded = [round(x, 3) for x in row]
        nearest_int = [round(x) for x in row]
        print(f"    ch{i}: {rounded}  ≈ {nearest_int}")

    # Check: does rounding to nearest integer still give a valid decomposition?
    print(f"\n  Verification: rounded-to-integer decomposition")
    U_int = U_mean.round()
    V_int = V_mean.round()
    W_int = W_mean.round()

    torch.manual_seed(99)
    A_test = torch.randn(10000, 4, device=dev)
    B_test = torch.randn(10000, 4, device=dev)
    C_true = torch.bmm(A_test.view(-1, 2, 2), B_test.view(-1, 2, 2)).view(-1, 4)

    p = A_test @ U_int.T
    q = B_test @ V_int.T
    m = p * q
    C_hat_int = m @ W_int.T
    err_int = ((C_hat_int - C_true).view(-1, 4).norm(dim=1) /
               C_true.view(-1, 4).norm(dim=1).clamp(min=1e-12)).mean().item()

    bits_int = -math.log2(err_int) if err_int > 0 else 99
    print(f"    Integer U,V,W error: {err_int:.3e}  bits: {bits_int:.1f}")
    if err_int < 1e-6:
        print(f"    ★ EXACT DECOMPOSITION FOUND — recovered Strassen (or equivalent)")
    elif err_int < 0.01:
        print(f"    ~ Near-exact. Rounding introduced small error.")
    else:
        print(f"    ✗ Integer rounding destroys the decomposition. Weights are not near-integer.")

    return model


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
