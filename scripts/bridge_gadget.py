#!/usr/bin/env python
"""Bridge gadget optimizer: theory-driven 3-plaquette search.

Instead of brute-force perturbation, searches the algebraic family:

    G(lambda, mu, nu, rho) = P_square + lambda*B_R + mu*B_L + nu*B_T + rho*B_B

where:
  P_square = macro outer plaquette (loop correction)
  B_R, B_L, B_T, B_B = rank-1 transport bridges (anti-phased edge-sharing plaquettes)

The gadget is applied as a fiber-space squeeze on (alpha, beta), then gamma is
refit and full L-BFGS is run. Scipy.optimize searches the continuous (eps, lambda,
mu, nu, rho) space — no discrete enumeration needed.

Usage:
    python scripts/bridge_gadget.py
    python scripts/bridge_gadget.py --mode full    # search all 4 bridge weights
    python scripts/bridge_gadget.py --mode right   # search lambda only (B_R)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg as la
from scipy.optimize import minimize as sp_minimize, differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR

_T = TARGET_TENSOR
R, D = RANK, DIM  # 19, 9

# ── Masks and weights ─────────────────────────────────────────
_DEAD_FLAT = (_T == 0).ravel()
_W = np.ones(729)
_W[_DEAD_FLAT] = 10.0

# ── Fiber basis ───────────────────────────────────────────────
_p0 = np.array([1, 1, 1]) / np.sqrt(3)
_p1 = np.array([1, -1, 0]) / np.sqrt(2)
_p2 = np.array([1, 1, -2]) / np.sqrt(6)

# ── Plaquette and bridge definitions ─────────────────────────
P_UL = np.array([[ 1,-1, 0],
                 [-1, 1, 0],
                 [ 0, 0, 0]], dtype=float)

P_UR = np.array([[ 0, 1,-1],
                 [ 0,-1, 1],
                 [ 0, 0, 0]], dtype=float)

P_LL = np.array([[ 0, 0, 0],
                 [ 1,-1, 0],
                 [-1, 1, 0]], dtype=float)

P_LR = np.array([[ 0, 0, 0],
                 [ 0, 1,-1],
                 [ 0,-1, 1]], dtype=float)

P_square = np.array([[ 1, 0,-1],
                     [ 0, 0, 0],
                     [-1, 0, 1]], dtype=float)

# Bridges: differences of adjacent plaquettes
B_R = P_LR - P_UR   # right bridge (transport along col 2)
B_L = P_LL - P_UL   # left bridge (transport along col 0)
B_T = P_UR - P_UL   # top bridge (transport along row 0)
B_B = P_LR - P_LL   # bottom bridge (transport along row 2)

# Verify bridge structure
assert np.linalg.matrix_rank(B_R) == 1, "B_R should be rank-1"
assert np.linalg.matrix_rank(B_L) == 1, "B_L should be rank-1"
assert np.linalg.matrix_rank(B_T) == 1, "B_T should be rank-1"
assert np.linalg.matrix_rank(B_B) == 1, "B_B should be rank-1"
assert np.allclose(P_square, P_UL + P_UR + P_LL + P_LR), "P_square = sum of micros"


def load_factors(path):
    with open(path) as f:
        data = json.load(f)
    return (np.array(data["alpha"]), np.array(data["beta"]),
            np.array(data["gamma"]), data["fitness"])


def save_factors(alpha, beta, gamma, fit, path):
    data = {"fitness": float(fit), "rank": R, "dim": D,
            "alpha": alpha.tolist(), "beta": beta.tolist(),
            "gamma": gamma.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def fitness(alpha, beta, gamma):
    T_hat = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    return float(np.max(np.abs(T_hat - _T)))


def refit_gamma(alpha, beta):
    M = np.einsum('ra,rb->abr', alpha, beta).reshape(-1, R)
    gamma = np.zeros((R, D))
    for c in range(D):
        gamma[:, c], _, _, _ = np.linalg.lstsq(M, _T[:, :, c].ravel(), rcond=None)
    return gamma


def als_cycles(alpha, beta, gamma, n_cycles=10):
    a, b, g = alpha.copy(), beta.copy(), gamma.copy()
    for _ in range(n_cycles):
        M_bg = np.einsum('rb,rc->bcr', b, g).reshape(-1, R)
        for ai in range(D):
            a[:, ai], _, _, _ = np.linalg.lstsq(M_bg, _T[ai, :, :].ravel(), rcond=None)
        M_ag = np.einsum('ra,rc->acr', a, g).reshape(-1, R)
        for bi in range(D):
            b[:, bi], _, _, _ = np.linalg.lstsq(M_ag, _T[:, bi, :].ravel(), rcond=None)
        g = refit_gamma(a, b)
    return a, b, g


def _pack(a, b, g):
    return np.concatenate([a.ravel(), b.ravel(), g.ravel()])


def _unpack(x):
    a = x[:R*D].reshape(R, D)
    b = x[R*D:2*R*D].reshape(R, D)
    g = x[2*R*D:].reshape(R, D)
    return a, b, g


def _grad_from_R(wR, a, b, g):
    ga = np.einsum('abc,rb,rc->ra', wR, b, g, optimize=True)
    gb = np.einsum('abc,ra,rc->rb', wR, a, g, optimize=True)
    gg = np.einsum('abc,ra,rb->rc', wR, a, b, optimize=True)
    return np.concatenate([ga.ravel(), gb.ravel(), gg.ravel()])


def full_lbfgs(alpha, beta, gamma, max_iters=200,
               beta_schedule=(10, 30, 100, 300, 1000, 3000, 10000)):
    x0 = _pack(alpha, beta, gamma)
    best_x = x0.copy()
    best_mx = fitness(alpha, beta, gamma)

    def max_abs(x):
        a, b, g = _unpack(x)
        return float(np.max(np.abs(
            np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T)))

    def dead_fro(x):
        a, b, g = _unpack(x)
        R_ = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
        R_flat = R_.ravel()
        wR = (_W * R_flat).reshape(D, D, D)
        f = 0.5 * np.sum(_W * R_flat**2)
        return f, _grad_from_R(wR, a, b, g)

    res = sp_minimize(dead_fro, best_x, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iters // 2, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs(res.x)
    if mx < best_mx:
        best_x, best_mx = res.x.copy(), mx

    def smooth_max(x, bv):
        a, b, g = _unpack(x)
        R_ = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
        R_flat = R_.ravel()
        absR = np.abs(R_flat)
        mx_val = np.max(absR)
        shifted = bv * (absR - mx_val)
        exp_s = np.exp(shifted)
        Z = np.sum(exp_s)
        f = mx_val + np.log(Z) / bv
        signs = np.sign(R_flat)
        weights = (exp_s / Z * signs).reshape(D, D, D)
        return f, _grad_from_R(weights, a, b, g)

    for bv in beta_schedule:
        res = sp_minimize(lambda x: smooth_max(x, bv), best_x,
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        mx = max_abs(res.x)
        if mx < best_mx:
            best_x, best_mx = res.x.copy(), mx

    a, b, g = _unpack(best_x)
    return a, b, g, best_mx


# ══════════════════════════════════════════════════════════════
# Core: Apply bridge gadget as fiber-space squeeze
# ══════════════════════════════════════════════════════════════

def build_gadget(lam=0.0, mu=0.0, nu=0.0, rho=0.0):
    """Build G(lambda, mu, nu, rho) = P_square + lam*B_R + mu*B_L + nu*B_T + rho*B_B."""
    return P_square + lam * B_R + mu * B_L + nu * B_T + rho * B_B


def apply_gadget_squeeze(alpha, beta, gadget, eps):
    """Apply gadget as fiber-space squeeze on alpha and beta.

    For each interface (r, u) with gadget value G[r,u]:
        alpha row-r: p1 component *= (1 + eps * G[r,u])
                     p2 component *= (1 - eps * G[r,u])
        beta  col-u: p1 component *= (1 - eps * G[r,u])
                     p2 component *= (1 + eps * G[r,u])

    This creates conjugate phase shifts that modify interference patterns
    while preserving the p0 (signal) mode.
    """
    a_out = alpha.copy()
    b_out = beta.copy()

    for r in range(3):
        for u in range(3):
            g_val = gadget[r, u]
            if abs(g_val) < 1e-15:
                continue

            s1 = 1.0 + eps * g_val
            s2 = 1.0 - eps * g_val

            # Squeeze alpha row-r in fiber space (across s-index)
            for k in range(R):
                row = a_out[k, 3*r:3*r+3].copy()
                c1 = np.dot(row, _p1)
                c2 = np.dot(row, _p2)
                row += (s1 - 1.0) * c1 * _p1 + (s2 - 1.0) * c2 * _p2
                a_out[k, 3*r:3*r+3] = row

            # Squeeze beta col-u in fiber space (across t-index)
            for k in range(R):
                col = b_out[k, 0*3+u:9:3].copy()  # indices u, 3+u, 6+u
                c1 = np.dot(col, _p1)
                c2 = np.dot(col, _p2)
                col += (s2 - 1.0) * c1 * _p1 + (s1 - 1.0) * c2 * _p2
                b_out[k, u::3] = col

    return a_out, b_out


# ══════════════════════════════════════════════════════════════
# Objective: gadget params -> fitness after refit + L-BFGS
# ══════════════════════════════════════════════════════════════

def evaluate_gadget(params, alpha, beta, mode='right', do_lbfgs=True):
    """Evaluate a gadget configuration. Returns fitness after refit + optional L-BFGS.

    params layout depends on mode:
        'right':  [eps, lambda]
        'full':   [eps, lambda, mu, nu, rho]
        'scan':   [eps]  (lambda fixed externally)
    """
    if mode == 'right':
        eps, lam = params
        gadget = build_gadget(lam=lam)
    elif mode == 'full':
        eps, lam, mu, nu, rho = params
        gadget = build_gadget(lam=lam, mu=mu, nu=nu, rho=rho)
    elif mode == 'scan':
        eps = params[0]
        gadget = build_gadget(lam=1.0)  # fixed right bridge
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Apply squeeze
    a_sq, b_sq = apply_gadget_squeeze(alpha, beta, gadget, eps)

    # Refit gamma
    g_sq = refit_gamma(a_sq, b_sq)

    # ALS to settle
    a_sq, b_sq, g_sq = als_cycles(a_sq, b_sq, g_sq, n_cycles=10)

    fit_als = fitness(a_sq, b_sq, g_sq)

    if do_lbfgs:
        a_sq, b_sq, g_sq, fit_final = full_lbfgs(a_sq, b_sq, g_sq)
    else:
        fit_final = fit_als

    return a_sq, b_sq, g_sq, fit_final, fit_als


def objective_for_scipy(params, alpha, beta, mode):
    """Scalar objective for scipy.optimize (no L-BFGS, just ALS — fast)."""
    _, _, _, _, fit_als = evaluate_gadget(params, alpha, beta, mode=mode, do_lbfgs=False)
    return fit_als


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bridge gadget optimizer")
    parser.add_argument("--input", default="shotgun_best.json")
    parser.add_argument("--out", default="bridge_best.json")
    parser.add_argument("--mode", choices=["right", "full", "scan"], default="right")
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        src = Path(__file__).resolve().parent.parent / args.input
    alpha, beta, gamma, fit = load_factors(src)

    print(f"Loaded: {src}  fitness={fit:.10f}")
    print(f"Mode: {args.mode}")
    print()

    best_a, best_b, best_g = alpha.copy(), beta.copy(), gamma.copy()
    best_fit = fit

    # ── Phase 1: Coarse grid scan to find promising region ───
    print("=" * 60)
    print("PHASE 1: Coarse grid scan (ALS only, no L-BFGS)")
    print("=" * 60)

    t0 = time.time()

    if args.mode == 'right':
        # Scan (eps, lambda) grid
        eps_vals = np.concatenate([
            -np.logspace(-3, 0, 20)[::-1],
            np.array([0.0]),
            np.logspace(-3, 0, 20)
        ])
        lam_vals = np.linspace(-2.0, 2.0, 21)

        results = []
        for eps in eps_vals:
            for lam in lam_vals:
                _, _, _, _, fit_als = evaluate_gadget(
                    [eps, lam], alpha, beta, mode='right', do_lbfgs=False)
                results.append((fit_als, eps, lam))

        results.sort()
        print(f"Grid: {len(eps_vals)} x {len(lam_vals)} = {len(results)} points  ({time.time()-t0:.1f}s)")
        print(f"\nTop 10 (eps, lambda) by ALS fitness:")
        for i, (f, e, l) in enumerate(results[:10]):
            print(f"  {i+1:2d}: fit={f:.8f}  eps={e:+.6f}  lam={l:+.4f}")

        # Best grid point
        best_grid = results[0]
        x0 = [best_grid[1], best_grid[2]]
        print(f"\nBest grid point: eps={x0[0]:+.6f}, lam={x0[1]:+.4f}, fit={best_grid[0]:.8f}")

    elif args.mode == 'full':
        # Scan (eps, lam) first, then refine with (mu, nu, rho)
        eps_vals = np.concatenate([
            -np.logspace(-3, 0, 12)[::-1],
            np.array([0.0]),
            np.logspace(-3, 0, 12)
        ])
        lam_vals = np.linspace(-2.0, 2.0, 11)

        results = []
        for eps in eps_vals:
            for lam in lam_vals:
                _, _, _, _, fit_als = evaluate_gadget(
                    [eps, lam, 0, 0, 0], alpha, beta, mode='full', do_lbfgs=False)
                results.append((fit_als, eps, lam))

        results.sort()
        print(f"Grid: {len(results)} points  ({time.time()-t0:.1f}s)")
        print(f"\nTop 5 (eps, lambda, mu=nu=rho=0):")
        for i, (f, e, l) in enumerate(results[:5]):
            print(f"  {i+1}: fit={f:.8f}  eps={e:+.6f}  lam={l:+.4f}")

        best_e, best_l = results[0][1], results[0][2]
        x0 = [best_e, best_l, 0.0, 0.0, 0.0]

    elif args.mode == 'scan':
        eps_vals = np.linspace(-1.0, 1.0, 201)
        results = []
        for eps in eps_vals:
            _, _, _, _, fit_als = evaluate_gadget(
                [eps], alpha, beta, mode='scan', do_lbfgs=False)
            results.append((fit_als, eps))
        results.sort()
        print(f"Scan: {len(results)} points  ({time.time()-t0:.1f}s)")
        print(f"\nTop 10 eps values (G_R with lambda=1):")
        for i, (f, e) in enumerate(results[:10]):
            print(f"  {i+1:2d}: fit={f:.8f}  eps={e:+.6f}")
        x0 = [results[0][1]]

    # ── Phase 2: Scipy refinement of continuous params ────────
    print(f"\n{'='*60}")
    print("PHASE 2: Scipy optimization (Nelder-Mead on ALS fitness)")
    print("=" * 60)

    t1 = time.time()
    res = sp_minimize(
        objective_for_scipy, x0, args=(alpha, beta, args.mode),
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
    )
    print(f"Nelder-Mead: {res.nfev} evals, {time.time()-t1:.1f}s")
    print(f"  Optimal params: {res.x}")
    print(f"  ALS fitness: {res.fun:.10f}")

    # ── Phase 3: Differential evolution for global search ─────
    print(f"\n{'='*60}")
    print("PHASE 3: Differential evolution (global search)")
    print("=" * 60)

    t2 = time.time()
    if args.mode == 'right':
        bounds = [(-1.0, 1.0), (-3.0, 3.0)]
    elif args.mode == 'full':
        bounds = [(-1.0, 1.0), (-3.0, 3.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    else:
        bounds = [(-1.0, 1.0)]

    de_res = differential_evolution(
        objective_for_scipy, bounds, args=(alpha, beta, args.mode),
        maxiter=300, tol=1e-12, polish=True,
        popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        workers=1,
    )
    print(f"DE: {de_res.nfev} evals, {time.time()-t2:.1f}s")
    print(f"  Optimal params: {de_res.x}")
    print(f"  ALS fitness: {de_res.fun:.10f}")

    # Take the better of NM and DE
    if de_res.fun < res.fun:
        best_params = de_res.x
        print(f"\n  DE wins: {de_res.fun:.10f} < NM {res.fun:.10f}")
    else:
        best_params = res.x
        print(f"\n  NM wins: {res.fun:.10f} < DE {de_res.fun:.10f}")

    # ── Phase 4: Full L-BFGS at the optimal gadget point ─────
    print(f"\n{'='*60}")
    print("PHASE 4: Full L-BFGS at optimal gadget configuration")
    print("=" * 60)

    t3 = time.time()
    a_out, b_out, g_out, fit_final, fit_als = evaluate_gadget(
        best_params, alpha, beta, mode=args.mode, do_lbfgs=True)
    print(f"ALS fitness: {fit_als:.10f}")
    print(f"L-BFGS fitness: {fit_final:.10f}")
    print(f"Original fitness: {fit:.10f}")
    print(f"Time: {time.time()-t3:.1f}s")

    if fit_final < best_fit - 1e-12:
        delta = best_fit - fit_final
        best_a, best_b, best_g = a_out, b_out, g_out
        best_fit = fit_final
        save_factors(best_a, best_b, best_g, best_fit, args.out)
        print(f"\n  *** IMPROVEMENT: {delta:.2e} ***")
        print(f"  Saved: {args.out}")
    else:
        print(f"\n  No improvement over original ({fit:.10f})")

    # ── Phase 5: Multi-start with perturbation around best ───
    print(f"\n{'='*60}")
    print("PHASE 5: Multi-start perturbation around optimal")
    print("=" * 60)

    t4 = time.time()
    rng = np.random.default_rng(12345)
    n_starts = 50
    for trial in range(n_starts):
        # Perturb best_params
        noise = rng.normal(0, 0.1, size=len(best_params))
        trial_params = best_params + noise

        a_t, b_t, g_t, fit_t, fit_als_t = evaluate_gadget(
            trial_params, alpha, beta, mode=args.mode, do_lbfgs=True)

        if fit_t < best_fit - 1e-12:
            delta = best_fit - fit_t
            best_a, best_b, best_g = a_t, b_t, g_t
            best_fit = fit_t
            save_factors(best_a, best_b, best_g, best_fit, args.out)
            print(f"  Trial {trial:3d}: fit={fit_t:.10f}  DELTA={delta:.2e}  params={trial_params}")

    dt5 = time.time() - t4
    print(f"Multi-start: {n_starts} trials, {dt5:.1f}s")

    # ── Summary ───────────────────────────────────────────────
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Original fitness: {fit:.10f}")
    print(f"Best fitness:     {best_fit:.10f}")
    if best_fit < fit:
        print(f"Improvement:      {fit - best_fit:.2e}")
    else:
        print(f"No improvement found.")
    print(f"Total time: {total:.0f}s")


if __name__ == "__main__":
    main()
