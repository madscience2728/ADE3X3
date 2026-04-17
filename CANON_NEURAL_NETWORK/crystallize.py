"""
crystallize.py — Phase 2: algebraic extraction via crystallization.

Two-phase training:
  Phase 1 (0-80%):  find the basin (standard loss)
  Phase 2 (80-100%): crystallize (loss += lambda * distance_to_rational)

Logs full U, V, W at checkpoints for watching crystallization in progress.
Verifies final integer/rational decomposition algebraically.

Includes Strassen's known decomposition for comparison.
"""

import math
import torch
import torch.nn as nn
import json
import os
from population_probe import (
    TinyBilinear, ArchSpec, sample_batch, relative_error, PROBLEMS,
)


# ---------------------------------------------------------------------------
# Known Strassen decomposition for verification
# ---------------------------------------------------------------------------

def strassen_UVW():
    """
    Strassen's 7-multiplication algorithm for 2×2 matmul.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)      M2 = (c+d)e        M3 = a(f-h)
    M4 = d(g-e)           M5 = (a+b)h        M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4,       M1-M2+M3+M6]]
    
    A flat: [a, b, c, d] = [A00, A01, A10, A11]
    B flat: [e, f, g, h] = [B00, B01, B10, B11]
    C flat: [C00, C01, C10, C11]
    """
    # U: how each multiplication reads from A
    # M1=a+d, M2=c+d, M3=a, M4=d, M5=a+b, M6=c-a, M7=b-d
    U = torch.tensor([
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ], dtype=torch.float32)

    # V: how each multiplication reads from B
    # M1=e+h, M2=e, M3=f-h, M4=g-e, M5=h, M6=e+f, M7=g+h
    V = torch.tensor([
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ], dtype=torch.float32)

    # W: how outputs combine multiplications
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    W = torch.tensor([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=torch.float32)

    return U, V, W


def verify_decomposition(U, V, W, n=2, n_test=50000, device="cpu"):
    """
    Verify U, V, W decomposition: C_hat = W @ ((U@A) * (V@B)).
    Returns relative error on random test set.
    """
    dev = torch.device(device)
    U, V, W = U.to(dev), V.to(dev), W.to(dev)

    d = n * n
    A = torch.randn(n_test, d, device=dev)
    B = torch.randn(n_test, d, device=dev)
    C_true = torch.bmm(
        A.view(-1, n, n).to(torch.float64),
        B.view(-1, n, n).to(torch.float64)
    ).view(-1, d).float()

    p = A @ U.T
    q = B @ V.T
    m = p * q
    C_hat = m @ W.T

    err = ((C_hat - C_true).view(-1, d).norm(dim=1) /
           C_true.view(-1, d).norm(dim=1).clamp(min=1e-12)).mean().item()
    return err


# ---------------------------------------------------------------------------
# Algebraic snapping (ADE target set)
# ---------------------------------------------------------------------------
# Target forms: i, i/k, j^(1/i), (j/k)^(1/i)
# with i (root) in [1..4], j in [0..10], k in [1..10], plus negatives.

def _build_algebraic_targets(max_val: int = 10) -> torch.Tensor:
    """Build sorted tensor of algebraic target values in ADE snap set."""
    targets = set()
    targets.add(0.0)

    for j in range(0, max_val + 1):
        for k in range(1, max_val + 1):
            base = j / k
            for root in range(1, 5):  # 1st, 2nd, 3rd, 4th roots
                val = base ** (1.0 / root) if base >= 0 else 0.0
                if val <= max_val + 1:
                    targets.add(round(val, 10))
                    targets.add(round(-val, 10))
            # Also negative base for odd roots
            if j > 0:
                neg_base = -base
                for root in [1, 3]:  # odd roots of negatives
                    val = -(abs(neg_base) ** (1.0 / root))
                    if abs(val) <= max_val + 1:
                        targets.add(round(val, 10))

    return torch.tensor(sorted(targets), dtype=torch.float32)


# Module-level cache
_ALGEBRAIC_TARGETS = None


def _get_targets() -> torch.Tensor:
    global _ALGEBRAIC_TARGETS
    if _ALGEBRAIC_TARGETS is None:
        _ALGEBRAIC_TARGETS = _build_algebraic_targets()
    return _ALGEBRAIC_TARGETS


def snap_to_algebraic(x: torch.Tensor) -> torch.Tensor:
    """Snap each element to nearest value in ADE algebraic target set."""
    targets = _get_targets().to(x.device)
    x_flat = x.flatten().unsqueeze(1)  # (n, 1)
    dists = (x_flat - targets.unsqueeze(0)).abs()  # (n, T)
    idx = dists.argmin(dim=1)
    return targets[idx].view(x.shape)


def algebraic_distance(x: torch.Tensor) -> torch.Tensor:
    """Distance from each element to nearest algebraic target."""
    snapped = snap_to_algebraic(x)
    return (x - snapped).abs()


def algebraic_label(val: float, tol: float = 0.01) -> str:
    """Human-readable label for an algebraic value."""
    if abs(val) < tol:
        return "0"
    sign = "-" if val < 0 else ""
    av = abs(val)
    # Check i/k forms
    for k in range(1, 11):
        for j in range(0, 11):
            if abs(av - j / k) < tol:
                if k == 1:
                    return f"{sign}{j}"
                return f"{sign}{j}/{k}"
    # Check root forms
    for root in [2, 3, 4]:
        for j in range(1, 11):
            for k in range(1, 11):
                target = (j / k) ** (1.0 / root)
                if abs(av - target) < tol:
                    base = f"{j}" if k == 1 else f"{j}/{k}"
                    rname = {2: "√", 3: "∛", 4: "∜"}[root]
                    return f"{sign}{rname}({base})"
    return f"{val:.4f}"


# Keep backward compat aliases
def snap_to_rational(x: torch.Tensor, Q_max: int = 1) -> torch.Tensor:
    """Legacy alias — now snaps to full algebraic target set."""
    return snap_to_algebraic(x)


def rational_distance(x: torch.Tensor, Q_max: int = 1) -> torch.Tensor:
    """Legacy alias — now uses full algebraic target set."""
    return algebraic_distance(x)


# ---------------------------------------------------------------------------
# Crystallization training
# ---------------------------------------------------------------------------

def crystallize(
    N: int = 7,
    problem: str = "2x2",
    steps: int = 2000,
    crystal_start_frac: float = 0.8,
    crystal_lambda: float = 1.0,
    Q_max: int = 1,
    width: int = 12,
    depth: int = 2,
    lr: float = 1e-2,
    seed: int = 0,
    batch_size: int = 4096,
    device: str = "auto",
    log_steps: list = None,
):
    """Train with crystallization phase. Returns model, logs, snapshots."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if log_steps is None:
        log_steps = [500, 1000, 1500, 2000]

    dev = torch.device(device)
    torch.manual_seed(seed)
    cfg = PROBLEMS[problem]
    d_out = cfg["output_dim"]
    d_in = cfg["input_dim"]

    spec = ArchSpec(
        name=f"mlp_w{width}_d{depth}_N{N}",
        N=N, selector_type="mlp",
        selector_width=width, selector_depth=depth,
        selector_activation="gelu",
        use_products=True, head_depth=0,
    )
    model = TinyBilinear(spec, problem).to(dev)
    n_params = model.param_count()

    print(f"\n{'='*70}")
    print(f"  CRYSTALLIZATION RUN  |  {problem} N={N}  Q_max={Q_max}")
    print(f"  steps={steps}  crystal_start={int(crystal_start_frac*steps)}")
    print(f"  lambda={crystal_lambda}  params={n_params}  seed={seed}")
    print(f"{'='*70}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crystal_start = int(crystal_start_frac * steps)

    snapshots = {}
    log = []

    for step in range(1, steps + 1):
        model.train()
        A, B, C = sample_batch(problem, batch_size, dev)
        C_hat = model(A, B)
        recon_loss = relative_error(C_hat, C, d_out)

        # Crystallization penalty in phase 2
        crystal_loss = torch.tensor(0.0, device=dev)
        if step >= crystal_start:
            for name, p in model.named_parameters():
                if "head" in name:
                    crystal_loss = crystal_loss + rational_distance(p, Q_max).pow(2).mean()
            # Ramp up lambda linearly in phase 2
            phase2_frac = (step - crystal_start) / (steps - crystal_start)
            effective_lambda = crystal_lambda * phase2_frac
        else:
            effective_lambda = 0.0

        loss = recon_loss + effective_lambda * crystal_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0 or step == 1 or step in log_steps:
            model.eval()
            with torch.no_grad():
                Av, Bv, Cv = sample_batch(problem, 4096, dev)
                err = relative_error(model(Av, Bv), Cv, d_out).item()
            bits = -math.log2(err) if err > 0 else 99

            # Near-integer stats
            head_params = []
            for name, p in model.named_parameters():
                if "head" in name:
                    head_params.append(p.detach())
            all_head = torch.cat([p.flatten() for p in head_params])
            int_residual = (all_head - all_head.round()).abs().mean().item()
            frac_near = (rational_distance(all_head, Q_max) < 0.1).float().mean().item()

            phase = "CRYSTAL" if step >= crystal_start else "BASIN"
            print(f"  step {step:>5d}  {phase:<8s}  err={err:.3e}  bits={bits:.1f}  "
                  f"int_res={int_residual:.4f}  frac<0.1={frac_near:.1%}  "
                  f"λ_eff={effective_lambda:.3f}")

            log.append({
                "step": step, "error": err, "bits": bits,
                "int_residual": int_residual, "frac_near_01": frac_near,
                "phase": phase, "lambda_eff": effective_lambda,
            })

        # Snapshots at requested steps
        if step in log_steps:
            model.eval()
            with torch.no_grad():
                # Get mean U, V, W
                test_A = torch.randn(200, d_in, device=dev)
                test_B = torch.randn(200, d_in, device=dev)
                x = (test_A.unsqueeze(2) * test_B.unsqueeze(1)).view(-1, cfg["products_dim"])
                latent = model.selector(x)
                U_raw = model.head_U(latent).view(200, N, d_in)
                V_raw = model.head_V(latent).view(200, N, d_in)
                W_raw = model.head_W(latent).view(200, d_out, N)

                U_mean = U_raw.mean(dim=0)
                V_mean = V_raw.mean(dim=0)
                W_mean = W_raw.mean(dim=0)

            snapshots[step] = {
                "U": U_mean.cpu(),
                "V": V_mean.cpu(),
                "W": W_mean.cpu(),
            }
            # Print snapshot
            print(f"\n  --- SNAPSHOT at step {step} ---")
            for name, mat in [("U", U_mean), ("V", V_mean), ("W", W_mean)]:
                print(f"  {name}:")
                for i in range(mat.shape[0]):
                    raw = [f"{x:.3f}" for x in mat[i].cpu().tolist()]
                    snapped = snap_to_algebraic(mat[i].cpu())
                    labels = [algebraic_label(v.item()) for v in snapped]
                    print(f"    [{', '.join(raw)}]  → [{', '.join(labels)}]")
            print()

    # Final verification
    print(f"\n{'='*70}")
    print(f"  VERIFICATION")
    print(f"{'='*70}\n")

    final = snapshots[log_steps[-1]]
    U_final, V_final, W_final = final["U"], final["V"], final["W"]

    # Algebraic snap verification
    U_snap = snap_to_algebraic(U_final)
    V_snap = snap_to_algebraic(V_final)
    W_snap = snap_to_algebraic(W_final)

    err = verify_decomposition(U_snap, V_snap, W_snap, n=cfg["mat_size"])
    bits = -math.log2(err) if err > 0 else 99
    status = "★ EXACT" if err < 1e-6 else ("~ near" if err < 0.01 else "✗ fail")
    print(f"  Algebraic snap:  error={err:.3e}  bits={bits:.1f}  {status}")

    # Show what values it snapped to
    print(f"\n  Snapped U (algebraic labels):")
    for i in range(U_snap.shape[0]):
        labels = [algebraic_label(v.item()) for v in U_snap[i]]
        print(f"    ch{i}: [{', '.join(labels)}]")
    print(f"  Snapped V (algebraic labels):")
    for i in range(V_snap.shape[0]):
        labels = [algebraic_label(v.item()) for v in V_snap[i]]
        print(f"    ch{i}: [{', '.join(labels)}]")
    print(f"  Snapped W (algebraic labels):")
    for i in range(W_snap.shape[0]):
        labels = [algebraic_label(v.item()) for v in W_snap[i]]
        print(f"    ch{i}: [{', '.join(labels)}]")

    if err < 1e-6:
        print(f"\n  ★ VALID RANK-{N} DECOMPOSITION FOUND")

        # Compare to Strassen
        U_s, V_s, W_s = strassen_UVW()
        print(f"\n  Strassen U:\n{U_s}")
        print(f"  Strassen V:\n{V_s}")
        print(f"  Strassen W:\n{W_s}")

    # Also verify Strassen directly (sanity check)
    U_s, V_s, W_s = strassen_UVW()
    strassen_err = verify_decomposition(U_s, V_s, W_s, n=2)
    print(f"\n  Strassen verification (sanity): error={strassen_err:.3e}")

    return model, log, snapshots


# ---------------------------------------------------------------------------
# Linear control at matched param count
# ---------------------------------------------------------------------------

def linear_control(N=7, steps=2000, seed=0, device="auto"):
    """Linear selector (no depth) at same param count. Is depth doing anything?"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)
    cfg = PROBLEMS["2x2"]

    spec = ArchSpec(
        name=f"linear_w12_d1_N{N}",
        N=N, selector_type="linear",
        selector_width=12, selector_depth=1,
        selector_activation="gelu",
        use_products=True, head_depth=0,
    )
    model = TinyBilinear(spec, "2x2").to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print(f"\n  LINEAR CONTROL: params={model.param_count()}")
    for step in range(1, steps + 1):
        model.train()
        A, B, C = sample_batch("2x2", 4096, dev)
        C_hat = model(A, B)
        loss = relative_error(C_hat, C, 4)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                Av, Bv, Cv = sample_batch("2x2", 4096, dev)
                err = relative_error(model(Av, Bv), Cv, 4).item()
            bits = -math.log2(err) if err > 0 else 99
            print(f"    step {step}  err={err:.3e}  bits={bits:.1f}")

    return model


# ---------------------------------------------------------------------------
# Main: run everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  STRASSEN RECOVERY EXPERIMENT (ADE algebraic snap)")
    print("  Target set: i, i/k, root-i(j), root-i(j/k) for values ≤ 10")
    print("=" * 70)

    targets = _get_targets()
    print(f"\n  Algebraic target set: {len(targets)} values")
    print(f"  Range: [{targets[0].item():.4f}, {targets[-1].item():.4f}]")
    # Show some examples
    examples = [t.item() for t in targets if 0 <= t.item() <= 2.0]
    print(f"  Values in [0, 2]: {len(examples)} targets")
    some = sorted(set(round(x, 4) for x in examples))[:30]
    print(f"  First 30: {some}\n")

    # Crystallization with algebraic targets, multiple seeds
    best_model = None
    best_err = float("inf")

    for seed in range(5):
        model, log, snaps = crystallize(
            N=7, problem="2x2", steps=2000,
            crystal_start_frac=0.8, crystal_lambda=2.0,
            Q_max=1,  # ignored now — algebraic snap used internally
            width=12, depth=2, seed=seed,
            log_steps=[500, 1000, 1500, 2000],
        )
        final_err = log[-1]["error"]
        if final_err < best_err:
            best_err = final_err
            best_model = model
            best_snaps = snaps

    # Linear control
    print(f"\n{'='*70}")
    print(f"  LINEAR CONTROL (is depth necessary?)")
    print(f"{'='*70}")
    for seed in range(3):
        linear_control(N=7, steps=2000, seed=seed)
