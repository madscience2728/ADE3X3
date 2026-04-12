"""
oscillator.py — Riemannian Langevin Tensor Decomposer

The Keth-Varai's bodies ARE tensor decomposers. This simulation
honours that: dynamics on the product manifold (S^8)^{3N} × R^N.

Phase 1: Riemannian Langevin diffusion (annealed)
    dq = -∇_M V · dt + √(2T·dt) · η_tangent
    Gradient projected to tangent space. Noise IN tangent space.
    Retract to sphere via normalization. Samples from exp(-V/T).

Phase 2: Riemannian Adam polish
    Gradients projected to tangent space before Adam step.
    Retract after. No L1. No repulsion. Pure reconstruction.

No L1 penalty. No repulsion. No broken momentum statistics.
Channel death occurs naturally during polish (amplitudes → 0).
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# THE MATMUL TENSOR
# ═══════════════════════════════════════════════════════════════

def build_matmul_tensor():
    """T ∈ R^{9×9×9} for 3×3 matmul: T[3a+b, 3b+c, 3a+c] = 1."""
    T = torch.zeros(9, 9, 9)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                T[3 * a + b, 3 * b + c, 3 * a + c] = 1.0
    return T


# ═══════════════════════════════════════════════════════════════
# SPHERE GEOMETRY
# ═══════════════════════════════════════════════════════════════

def project_tangent(g, q):
    """Project vector g onto the tangent space of S^{n-1} at q."""
    return g - (g * q).sum(-1, keepdim=True) * q


def tangent_noise(q, scale):
    """Sample isotropic noise in the tangent space of S^{n-1} at q."""
    n = torch.randn_like(q) * scale
    return n - (n * q).sum(-1, keepdim=True) * q


# ═══════════════════════════════════════════════════════════════
# ANALYTICAL GRADIENTS (no autograd overhead)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_grad_and_V(u, v, w, alpha, T_target):
    """
    V = ‖T_target − Σ_k α_k u_k⊗v_k⊗w_k‖²_F

    Returns Riemannian gradients (projected to tangent space) and V.
    """
    R = T_target - torch.einsum('ki,kj,kl,k->ijl', u, v, w, alpha)
    V = R.pow(2).sum().item()

    # Ambient gradients
    gu = -2.0 * torch.einsum('ijl,kj,kl,k->ki', R, v, w, alpha)
    gv = -2.0 * torch.einsum('ijl,ki,kl,k->kj', R, u, w, alpha)
    gw = -2.0 * torch.einsum('ijl,ki,kj,k->kl', R, u, v, alpha)
    ga = -2.0 * torch.einsum('ijl,ki,kj,kl->k', R, u, v, w)

    # Project to tangent space of spheres
    gu = project_tangent(gu, u)
    gv = project_tangent(gv, v)
    gw = project_tangent(gw, w)

    return gu, gv, gw, ga, V


# ═══════════════════════════════════════════════════════════════
# STATE (positions only — no momenta needed for Langevin)
# ═══════════════════════════════════════════════════════════════

class OscillatorState:
    """N channels: u_k, v_k, w_k ∈ S^8, α_k ∈ R."""

    def __init__(self, N: int, device: torch.device):
        self.N = N
        self.device = device
        self.u = F.normalize(torch.randn(N, 9, device=device), dim=1)
        self.v = F.normalize(torch.randn(N, 9, device=device), dim=1)
        self.w = F.normalize(torch.randn(N, 9, device=device), dim=1)
        self.alpha = torch.randn(N, device=device) * 0.5

    def active_channels(self, threshold: float = 0.01) -> int:
        return (self.alpha.abs() > threshold).sum().item()


# ═══════════════════════════════════════════════════════════════
# RIEMANNIAN LANGEVIN STEP
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def langevin_step(state, T_target, dt, temperature):
    """
    One step of Riemannian Langevin dynamics on (S^8)^{3N} × R^N.

    dq = -∇_M V · dt + √(2T·dt) · η_tangent
    Then retract to sphere.
    """
    gu, gv, gw, ga, V = compute_grad_and_V(
        state.u, state.v, state.w, state.alpha, T_target
    )

    noise_scale = math.sqrt(2.0 * temperature * dt)

    # Gradient descent + tangent-space noise on spheres
    state.u -= dt * gu
    state.v -= dt * gv
    state.w -= dt * gw
    state.alpha -= dt * ga

    if temperature > 1e-12:
        state.u += tangent_noise(state.u, noise_scale)
        state.v += tangent_noise(state.v, noise_scale)
        state.w += tangent_noise(state.w, noise_scale)
        state.alpha += noise_scale * torch.randn_like(state.alpha)

    # Retract to sphere
    state.u = F.normalize(state.u, dim=-1)
    state.v = F.normalize(state.v, dim=-1)
    state.w = F.normalize(state.w, dim=-1)

    return V


# ═══════════════════════════════════════════════════════════════
# RIEMANNIAN ADAM POLISH
# ═══════════════════════════════════════════════════════════════

def riemannian_polish(state, T_target, steps=100_000, lr_init=3e-3, lr_final=1e-6):
    """
    Phase 2: Riemannian Adam on (S^8)^{3N} × R^N.
    Project gradients to tangent space BEFORE the Adam update,
    then retract to the sphere.  No L1.  No repulsion.
    """
    u = state.u.clone().requires_grad_(True)
    v = state.v.clone().requires_grad_(True)
    w = state.w.clone().requires_grad_(True)
    alpha = state.alpha.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([u, v, w, alpha], lr=lr_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_final
    )

    best_V = float('inf')
    best_params = None
    stall = 0

    for step in range(1, steps + 1):
        T_approx = torch.einsum('ki,kj,kl,k->ijl', u, v, w, alpha)
        V = (T_target - T_approx).pow(2).sum()

        optimizer.zero_grad()
        V.backward()

        # ── Riemannian correction: project grads to tangent space ──
        with torch.no_grad():
            for param, is_sphere in [(u, True), (v, True), (w, True), (alpha, False)]:
                if param.grad is not None and is_sphere:
                    param.grad -= (param.grad * param.data).sum(-1, keepdim=True) * param.data

        optimizer.step()
        scheduler.step()

        # ── Retract to sphere ──
        with torch.no_grad():
            u.data = F.normalize(u.data, dim=-1)
            v.data = F.normalize(v.data, dim=-1)
            w.data = F.normalize(w.data, dim=-1)

        err = V.item()
        if err < best_V:
            best_V = err
            best_params = (u.data.clone(), v.data.clone(), w.data.clone(), alpha.data.clone())
            stall = 0
        else:
            stall += 1

        if step % 10_000 == 0:
            T_norm = T_target.norm().item()
            rel = math.sqrt(err) / T_norm
            bits = -math.log2(rel) if rel > 0 else float('inf')
            best_rel = math.sqrt(best_V) / T_norm
            best_bits = -math.log2(best_rel) if best_rel > 0 else float('inf')
            print(f"    polish {step:>7d}/{steps}"
                  f"  V={err:.2e}  bits={bits:.1f}"
                  f"  best={best_V:.2e}  best_bits={best_bits:.1f}")

        if best_V < 1e-20:
            print(f"    polish CONVERGED at step {step}  V={best_V:.2e}")
            break

        # Warm restart if stuck for too long
        if stall > 20_000:
            with torch.no_grad():
                noise = 0.02
                u.data += noise * torch.randn_like(u)
                v.data += noise * torch.randn_like(v)
                w.data += noise * torch.randn_like(w)
                u.data = F.normalize(u.data, dim=-1)
                v.data = F.normalize(v.data, dim=-1)
                w.data = F.normalize(w.data, dim=-1)
            stall = 0

    # Write back best
    if best_params is not None:
        state.u, state.v, state.w, state.alpha = best_params
    return best_V


# ═══════════════════════════════════════════════════════════════
# ANNEALING SCHEDULE
# ═══════════════════════════════════════════════════════════════

def temperature_schedule(step, total, T_init, T_final):
    """Exponential cooling."""
    return T_init * (T_final / T_init) ** (step / total)


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate(
    N: int = 23,
    langevin_steps: int = 200_000,
    polish_steps: int = 200_000,
    dt: float = 0.005,
    T_init: float = 0.3,
    T_final: float = 1e-8,
    log_every: int = 100,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    T_target = build_matmul_tensor().to(device)
    T_norm = T_target.norm().item()

    state = OscillatorState(N, device)

    print(f"\n{'='*70}")
    print(f"  Riemannian Langevin Tensor Decomposer")
    print(f"  N={N}  device={device}  seed={seed}")
    print(f"  Phase 1: {langevin_steps} Langevin steps, dt={dt}, T={T_init}→{T_final}")
    print(f"  Phase 2: {polish_steps} Riemannian Adam steps")
    print(f"  ‖T_target‖ = {T_norm:.4f}")
    print(f"{'='*70}\n")

    best_V = float('inf')
    best_state = None
    t0 = time.time()
    log = []

    # ═══════════ PHASE 1: RIEMANNIAN LANGEVIN ═══════════
    print("  Phase 1: Riemannian Langevin annealing\n")

    for step in range(1, langevin_steps + 1):
        temp = temperature_schedule(step, langevin_steps, T_init, T_final)
        V = langevin_step(state, T_target, dt, temp)

        if V < best_V:
            best_V = V
            best_state = (state.u.clone(), state.v.clone(),
                          state.w.clone(), state.alpha.clone())

        if step % log_every == 0 or step == 1:
            recon_err = math.sqrt(V)
            rel = recon_err / T_norm
            bits = -math.log2(rel) if rel > 0 else float('inf')
            active = state.active_channels()
            elapsed = time.time() - t0

            alpha_sorted = state.alpha.abs().sort(descending=True).values[:5]
            top5 = " ".join(f"{a:.3f}" for a in alpha_sorted.tolist())

            best_rel = math.sqrt(best_V) / T_norm
            best_bits = -math.log2(best_rel) if best_rel > 0 else float('inf')

            entry = {
                "step": step, "V": round(V, 6), "rel": round(rel, 6),
                "bits": round(bits, 2), "active": active,
                "T": round(temp, 8),
            }
            log.append(entry)

            print(
                f"  {step:>7d}/{langevin_steps}"
                f"  V={V:>10.4f}  bits={bits:>5.1f}  best={best_bits:>5.1f}"
                f"  active={active:>2d}/{N}  T={temp:.1e}"
                f"  α=[{top5}]  t={elapsed:.0f}s"
            )

            if rel < 1e-6:
                print(f"\n  ★ Phase 1 CONVERGED at step {step}!")
                break

    # Restore best from Phase 1
    if best_state is not None:
        state.u, state.v, state.w, state.alpha = best_state

    elapsed_p1 = time.time() - t0
    rel_p1 = math.sqrt(best_V) / T_norm
    bits_p1 = -math.log2(rel_p1) if rel_p1 > 0 else float('inf')
    print(f"\n  Phase 1 done: best_rel={rel_p1:.6f}  bits={bits_p1:.1f}"
          f"  time={elapsed_p1:.0f}s")

    # ═══════════ PHASE 2: RIEMANNIAN POLISH ═══════════
    print(f"\n  Phase 2: Riemannian Adam polish ({polish_steps} steps)\n")

    final_V = riemannian_polish(state, T_target, steps=polish_steps)

    elapsed_total = time.time() - t0
    final_rel = math.sqrt(final_V) / T_norm
    final_bits = -math.log2(final_rel) if final_rel > 0 else float('inf')
    active = state.active_channels()

    print(f"\n{'='*70}")
    print(f"  RESULT: rel={final_rel:.2e}  bits={final_bits:.1f}  active={active}/{N}")
    print(f"  Total time: {elapsed_total:.0f}s")

    # Channel amplitudes
    with torch.no_grad():
        alphas = state.alpha.abs().sort(descending=True)
        print(f"\n  Channel amplitudes (sorted):")
        for i, (val, idx) in enumerate(zip(alphas.values, alphas.indices)):
            marker = "●" if val > 0.01 else "○"
            print(f"    {marker} ch{idx.item():>2d}: α={val.item():>8.5f}")

    print(f"{'='*70}")

    # Save
    os.makedirs("results", exist_ok=True)
    result = {
        "N": N, "seed": seed, "final_V": final_V,
        "final_rel": final_rel, "final_bits": final_bits,
        "active": active, "time": elapsed_total, "log": log,
    }
    path = f"results/oscillator_N{N}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {path}")

    # Save factors for surgery / downstream
    pt_path = f"results/oscillator_N{N}_s{seed}.pt"
    torch.save({
        "u": state.u, "v": state.v, "w": state.w, "alpha": state.alpha,
    }, pt_path)
    print(f"  Factors → {pt_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Riemannian Langevin Tensor Decomposer"
    )
    parser.add_argument("--N", type=int, default=23)
    parser.add_argument("--langevin_steps", type=int, default=200_000)
    parser.add_argument("--polish_steps", type=int, default=200_000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--T_init", type=float, default=0.3)
    parser.add_argument("--T_final", type=float, default=1e-8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi", type=int, default=1,
                        help="Best-of-N independent runs")
    args = parser.parse_args()

    if args.multi > 1:
        best_result = None
        for i in range(args.multi):
            print(f"\n{'#'*70}")
            print(f"  RUN {i+1}/{args.multi}  (seed={args.seed + i})")
            print(f"{'#'*70}")
            result = simulate(
                N=args.N, langevin_steps=args.langevin_steps,
                polish_steps=args.polish_steps, dt=args.dt,
                T_init=args.T_init, T_final=args.T_final,
                log_every=args.log_every, seed=args.seed + i,
            )
            if best_result is None or result["final_rel"] < best_result["final_rel"]:
                best_result = result
        print(f"\n{'#'*70}")
        print(f"  BEST: seed={best_result['seed']}"
              f"  rel={best_result['final_rel']:.2e}"
              f"  bits={best_result['final_bits']:.1f}")
        print(f"{'#'*70}")
    else:
        simulate(
            N=args.N, langevin_steps=args.langevin_steps,
            polish_steps=args.polish_steps, dt=args.dt,
            T_init=args.T_init, T_final=args.T_final,
            log_every=args.log_every, seed=args.seed,
        )


# ═══════════════════════════════════════════════════════════════
# CHANNEL SURGERY
# ═══════════════════════════════════════════════════════════════

def surgery(pt_path: str, polish_steps: int = 100_000):
    """
    Load a converged decomposition, drop weakest channel,
    re-polish, repeat until error explodes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_target = build_matmul_tensor().to(device)
    T_norm = T_target.norm().item()

    data = torch.load(pt_path, map_location=device, weights_only=True)
    u, v, w, alpha = data["u"], data["v"], data["w"], data["alpha"]
    N = u.shape[0]

    print(f"\n{'='*70}")
    print(f"  Channel Surgery — loaded {pt_path}")
    print(f"  Starting N={N}")
    print(f"{'='*70}\n")

    results = []

    while N > 1:
        # Sort by |alpha|, drop weakest
        order = alpha.abs().argsort(descending=True)
        u, v, w, alpha = u[order], v[order], w[order], alpha[order]

        # Current error before dropping
        R = T_target - torch.einsum('ki,kj,kl,k->ijl', u, v, w, alpha)
        V_before = R.pow(2).sum().item()
        rel_before = math.sqrt(V_before) / T_norm
        bits_before = -math.log2(rel_before) if rel_before > 0 else float('inf')

        print(f"  N={N:>2d}  V={V_before:.2e}  bits={bits_before:.1f}")
        results.append({"N": N, "V": V_before, "bits": round(bits_before, 1)})

        # Drop weakest channel
        N -= 1
        u, v, w, alpha = u[:N], v[:N], w[:N], alpha[:N]

        # Re-polish
        state = OscillatorState.__new__(OscillatorState)
        state.N = N
        state.device = device
        state.u = u
        state.v = v
        state.w = w
        state.alpha = alpha

        V_after = riemannian_polish(state, T_target, steps=polish_steps,
                                     lr_init=1e-3, lr_final=1e-6)
        u, v, w, alpha = state.u, state.v, state.w, state.alpha

        rel_after = math.sqrt(V_after) / T_norm
        bits_after = -math.log2(rel_after) if rel_after > 0 else float('inf')

        print(f"  → N={N:>2d}  V={V_after:.2e}  bits={bits_after:.1f}  (after polish)")

        # Save checkpoint
        pt_out = f"results/surgery_N{N}.pt"
        torch.save({"u": u, "v": v, "w": w, "alpha": alpha}, pt_out)

        # If we can't get past 2 bits, the rank wall is here
        if bits_after < 2.0:
            print(f"\n  ★ RANK WALL at N={N}: bits={bits_after:.1f}")
            print(f"    Minimum rank for this basin: {N+1}")
            break

    print(f"\n{'='*70}")
    print(f"  Surgery results:")
    for r in results:
        print(f"    N={r['N']:>2d}  bits={r['bits']:.1f}")
    print(f"{'='*70}")

    # Save summary
    with open("results/surgery_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "surgery":
        pt = sys.argv[2] if len(sys.argv) > 2 else "results/oscillator_N23_s0.pt"
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
        surgery(pt, steps)
    else:
        main()
