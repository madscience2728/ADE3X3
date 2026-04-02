#!/usr/bin/env python
"""Plaquette optimizer: closed-loop 4-interface joint optimization.

Key insight from triplet research: triplets are open paths that carry phase
without neutralizing it. Plaquettes are the smallest CLOSED loops — 4 interfaces
forming a rectangle on the 3×3 grid — that achieve phase closure.

Strategy:
  1. Enumerate all 9 rectangular plaquettes (C(3,2) × C(3,2) = 3×3)
  2. For each plaquette, apply conjugate phase squeeze with sign patterns
     that enforce loop closure (alternating signs around the rectangle)
  3. Run FULL L-BFGS on all 513 variables (not just triplet subset)
  4. Accept if global fitness improves
  5. Parallelize across plaquettes × ε × sign patterns

A plaquette is: C[r1,u1], C[r1,u2], C[r2,u1], C[r2,u2]
  forming a closed loop: (r1,u1)→(r1,u2)→(r2,u2)→(r2,u1)→(r1,u1)

Phase closure constraint: signs must alternate around the loop.
  Only 2 valid sign patterns: (+,-,+,-) and (-,+,-,+)

Usage:
    python scripts/plaquette_refine.py
    python scripts/plaquette_refine.py --workers 22 --rounds 20
    python scripts/plaquette_refine.py --input my_best.json --eps-max 0.1
"""

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as sp_minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR

_T = TARGET_TENSOR

# ── Orthonormal fiber basis (signal / nuisance modes) ──────────
_p0 = np.array([1, 1, 1]) / np.sqrt(3)
_p1 = np.array([1, -1, 0]) / np.sqrt(2)
_p2 = np.array([1, 1, -2]) / np.sqrt(6)
P_BASIS = np.vstack([_p0, _p1, _p2])  # (3,3)
P_INV = P_BASIS.T

# Dead-weighted penalty
_DEAD_FLAT = (_T == 0).ravel()
_W = np.ones(729)
_W[_DEAD_FLAT] = 10.0


# ── Enumerate all rectangular plaquettes ───────────────────────
def enumerate_plaquettes():
    """All 9 rectangular plaquettes on the 3×3 grid.

    Each is ((r1,u1),(r1,u2),(r2,u2),(r2,u1)) — closed loop order.
    """
    plaquettes = []
    for r1, r2 in itertools.combinations(range(3), 2):
        for u1, u2 in itertools.combinations(range(3), 2):
            # 4 corners in loop order (clockwise)
            loop = [(r1, u1), (r1, u2), (r2, u2), (r2, u1)]
            plaquettes.append(loop)
    return plaquettes


def apply_plaquette_squeeze(alpha, beta, plaquette, signs, eps):
    """Apply conjugate phase squeeze on a 4-interface plaquette.

    Phase closure: signs alternate around the loop.
    For each interface (r,u) with sign s:
      α row-r: p1 *= (1 + s·ε), p2 *= (1 - s·ε)
      β col-u: p1 *= (1 - s·ε), p2 *= (1 + s·ε)
    """
    a_p1 = np.ones(3)
    a_p2 = np.ones(3)
    b_p1 = np.ones(3)
    b_p2 = np.ones(3)

    for (r, u), s in zip(plaquette, signs):
        a_p1[r] *= (1 + s * eps)
        a_p2[r] *= (1 - s * eps)
        b_p1[u] *= (1 - s * eps)
        b_p2[u] *= (1 + s * eps)

    a = alpha.copy()
    for i in range(3):
        if a_p1[i] == 1.0 and a_p2[i] == 1.0:
            continue
        coords = a[:, 3*i:3*i+3] @ P_INV
        coords[:, 1] *= a_p1[i]
        coords[:, 2] *= a_p2[i]
        a[:, 3*i:3*i+3] = coords @ P_BASIS

    b = beta.copy()
    for j in range(3):
        if b_p1[j] == 1.0 and b_p2[j] == 1.0:
            continue
        coords = b[:, j::3] @ P_INV
        coords[:, 1] *= b_p1[j]
        coords[:, 2] *= b_p2[j]
        b[:, j::3] = coords @ P_BASIS

    return a, b


# ── L-BFGS on all 513 variables ───────────────────────────────
def _pack(a, b, g):
    return np.concatenate([a.ravel(), b.ravel(), g.ravel()])


def _unpack(x):
    n = RANK * DIM
    return x[:n].reshape(RANK, DIM), x[n:2*n].reshape(RANK, DIM), x[2*n:].reshape(RANK, DIM)


def _grad_from_R(R, a, b, g):
    ga = np.einsum('abc,rb,rc->ra', R, b, g, optimize=True)
    gb = np.einsum('abc,ra,rc->rb', R, a, g, optimize=True)
    gg = np.einsum('abc,ra,rb->rc', R, a, b, optimize=True)
    return np.concatenate([ga.ravel(), gb.ravel(), gg.ravel()])


def full_lbfgs(alpha, beta, gamma, max_iters=200,
               beta_schedule=(10, 30, 100, 300, 1000, 3000)):
    """Full L-BFGS on all 513 variables with dead-weighted + smooth-max stages."""
    x0 = _pack(alpha, beta, gamma)
    best_x = x0.copy()
    best_mx = float(np.max(np.abs(
        np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - _T
    )))

    def max_abs(x):
        a, b, g = _unpack(x)
        return float(np.max(np.abs(
            np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
        )))

    # Phase 1: Dead-weighted Frobenius
    def dead_fro(x):
        a, b, g = _unpack(x)
        R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
        R_flat = R.ravel()
        wR = (_W * R_flat).reshape(DIM, DIM, DIM)
        f = 0.5 * np.sum(_W * R_flat**2)
        return f, _grad_from_R(wR, a, b, g)

    res = sp_minimize(dead_fro, best_x, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iters // 2, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs(res.x)
    if mx < best_mx:
        best_x = res.x.copy()
        best_mx = mx

    # Phase 2: Smooth-max with escalating beta
    def smooth_max(x, bv):
        a, b, g = _unpack(x)
        R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
        R_flat = R.ravel()
        absR = np.abs(R_flat)
        mx_val = np.max(absR)
        shifted = bv * (absR - mx_val)
        exp_s = np.exp(shifted)
        Z = np.sum(exp_s)
        f = mx_val + np.log(Z) / bv
        signs = np.sign(R_flat)
        weights = (exp_s / Z * signs).reshape(DIM, DIM, DIM)
        return f, _grad_from_R(weights, a, b, g)

    for bv in beta_schedule:
        res = sp_minimize(lambda x: smooth_max(x, bv), best_x,
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        mx = max_abs(res.x)
        if mx < best_mx:
            best_x = res.x.copy()
            best_mx = mx

    a, b, g = _unpack(best_x)
    return a, b, g, best_mx


def refit_gamma(alpha, beta):
    """Optimal γ given fixed α, β."""
    M = np.einsum('ra,rb->abr', alpha, beta).reshape(-1, RANK)
    gamma = np.zeros((RANK, DIM))
    for c in range(DIM):
        gamma[:, c], _, _, _ = np.linalg.lstsq(M, _T[:, :, c].ravel(), rcond=None)
    return gamma


def fitness(a, b, g):
    return float(np.max(np.abs(
        np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
    )))


def load_factors(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["alpha"]), np.array(d["beta"]), np.array(d["gamma"])


def save_factors(a, b, g, fit, path):
    data = {
        "fitness": fit, "rank": RANK, "dim": DIM,
        "alpha": a.tolist(), "beta": b.tolist(), "gamma": g.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Worker function ────────────────────────────────────────────
def _refine_plaquette(args_tuple):
    """Worker: squeeze plaquette → add noise → γ-refit → ALS → full L-BFGS."""
    alpha, beta, gamma, plaquette, signs, eps, plaq_idx, lbfgs_iters, noise_sigma, seed = args_tuple

    rng = np.random.default_rng(seed)

    # Step 1: Apply plaquette squeeze (modifies α, β)
    a_sq, b_sq = apply_plaquette_squeeze(alpha, beta, plaquette, signs, eps)

    # Step 1b: Add random noise to ALL factors (escape basin)
    if noise_sigma > 0:
        a_sq += rng.normal(0, noise_sigma, a_sq.shape)
        b_sq += rng.normal(0, noise_sigma, b_sq.shape)

    # Step 2: γ-refit to get good starting point
    g_sq = refit_gamma(a_sq, b_sq)

    # Step 3: ALS cycles to settle all three factors
    for _ in range(10):
        M_bg = np.einsum('rb,rc->bcr', b_sq, g_sq).reshape(-1, RANK)
        for ai in range(DIM):
            a_sq[:, ai], _, _, _ = np.linalg.lstsq(M_bg, _T[ai, :, :].ravel(), rcond=None)
        M_ag = np.einsum('ra,rc->acr', a_sq, g_sq).reshape(-1, RANK)
        for bi in range(DIM):
            b_sq[:, bi], _, _, _ = np.linalg.lstsq(M_ag, _T[:, bi, :].ravel(), rcond=None)
        g_sq = refit_gamma(a_sq, b_sq)

    fit_post_als = fitness(a_sq, b_sq, g_sq)

    # Step 4: Full L-BFGS on all 513 variables (aggressive schedule)
    a_out, b_out, g_out, fit_out = full_lbfgs(
        a_sq, b_sq, g_sq, max_iters=lbfgs_iters,
        beta_schedule=(10, 30, 100, 300, 1000, 3000, 10000),
    )

    label = f"P{plaq_idx} [{','.join(f'({r},{u})' for r,u in plaquette)}] " \
            f"s={''.join('+' if s>0 else '-' for s in signs)} ε={eps:.3f} σ={noise_sigma:.3f}"

    return a_out, b_out, g_out, fit_out, fit_post_als, label


def main():
    parser = argparse.ArgumentParser(description="Plaquette closed-loop optimizer")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("plaquette_best.json"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int,
                        default=max(1, __import__('os').cpu_count() - 2))
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    parser.add_argument("--eps-max", type=float, default=0.1,
                        help="Max squeeze epsilon")
    args = parser.parse_args()

    # Load best candidate
    if args.input:
        src = args.input
    else:
        candidates = []
        for p in [Path("shotgun_best.json"), Path("plaquette_best.json"),
                  Path("chain_best.json")]:
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                candidates.append((d["fitness"], p))
        candidates.sort()
        src = candidates[0][1]

    alpha, beta, gamma = load_factors(src)
    fit = fitness(alpha, beta, gamma)

    plaquettes = enumerate_plaquettes()  # 9 rectangles

    # ε grid: wide range — small ones preserve structure, large ones escape basins
    eps_grid = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

    # Also test non-closure sign patterns (all same, one flipped, etc.)
    all_signs = [
        (1, -1, 1, -1),   # closure pattern A
        (-1, 1, -1, 1),   # closure pattern B
        (1, 1, -1, -1),   # row-aligned
        (1, -1, -1, 1),   # column-aligned
    ]

    # Random noise levels on top of squeeze
    noise_sigmas = [0.0, 0.02, 0.05]

    n_jobs = len(plaquettes) * len(all_signs) * len(eps_grid) * len(noise_sigmas)
    print(f"Loaded: {src}  fitness={fit:.10f}")
    print(f"Plaquettes: {len(plaquettes)}  signs: {len(all_signs)}  "
          f"ε: {len(eps_grid)}  noise: {len(noise_sigmas)}  "
          f"jobs/round: {n_jobs}  workers: {args.workers}")

    best_a, best_b, best_g = alpha.copy(), beta.copy(), gamma.copy()
    best_fit = fit
    total_accepts = 0
    t_start = time.time()

    for rnd in range(args.rounds):
        rnd_t0 = time.time()
        rnd_accepts = 0

        # Build all work items
        work = []
        seed_counter = rnd * 100000
        for pi, plaq in enumerate(plaquettes):
            for signs in all_signs:
                for eps in eps_grid:
                    for ns in noise_sigmas:
                        work.append((
                            best_a.copy(), best_b.copy(), best_g.copy(),
                            plaq, signs, eps, pi, args.lbfgs_iters,
                            ns, seed_counter,
                        ))
                        seed_counter += 1

        # Dispatch in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_refine_plaquette, w): w for w in work}
            for fut in as_completed(futures):
                a, b, g, fit, fit_als, label = fut.result()
                if fit < best_fit - 1e-12:
                    delta = best_fit - fit
                    best_a, best_b, best_g = a.copy(), b.copy(), g.copy()
                    best_fit = fit
                    rnd_accepts += 1
                    total_accepts += 1
                    print(f"  R{rnd:3d} {label}  "
                          f"ALS={fit_als:.8f} → LBFGS={fit:.10f}  Δ={delta:.2e}")
                    save_factors(best_a, best_b, best_g, best_fit, args.out)

        rnd_dt = time.time() - rnd_t0
        elapsed = time.time() - t_start
        print(f"Round {rnd:3d}: {rnd_accepts}/{n_jobs} accepts  "
              f"best={best_fit:.10f}  "
              f"round={rnd_dt:.1f}s  total={elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\nDone. {total_accepts} total accepts in {elapsed:.0f}s")
    print(f"Final fitness: {best_fit:.10f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
