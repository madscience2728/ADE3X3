#!/usr/bin/env python
"""Triplet-aware joint optimization.

Strategy: identify the best-cancelling triplets from the dead Gram matrix,
then jointly optimize all 81 variables of each triplet (3 terms × 3 factors × 9 dims)
while keeping the other 16 terms frozen. The key insight is that we minimize the
FULL tensor residual, but the joint 3-term move can discover interference patterns
invisible to single-term coordinate descent.

Outer loop:
  1. Rank all C(19,3) = 969 triplets by dead cross-term potential
  2. For each top-K triplet, run L-BFGS on its 81 variables
  3. Accept if global fitness improves
  4. Repeat until convergence

Usage:
    python scripts/triplet_refine.py                     # from shotgun_best.json
    python scripts/triplet_refine.py --input my_best.json
    python scripts/triplet_refine.py --top-k 30 --rounds 50
"""

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR

_T = TARGET_TENSOR
_DEAD_MASK = (_T == 0)
_LIVE_MASK = (_T != 0)
_DEAD_FLAT = _DEAD_MASK.ravel()
_LIVE_FLAT = _LIVE_MASK.ravel()

# Dead-weighted penalty: dead entries matter more
_W = np.ones(729)
_W[_DEAD_FLAT] = 10.0


def fitness(a, b, g):
    R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T
    return float(np.max(np.abs(R)))


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


def rank_triplets(alpha, beta, gamma, top_k=30):
    """Rank all C(19,3)=969 triplets by dead-entry cancellation potential.

    Returns list of (cancel_ratio, (i,j,k)) sorted best-first.
    """
    # Per-term dead projections
    terms = np.array([
        np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])
        for k in range(RANK)
    ])
    dead_flat = np.array([terms[k][_DEAD_MASK] for k in range(RANK)])
    dead_energy = np.sum(dead_flat**2, axis=1)
    G_dead = dead_flat @ dead_flat.T

    results = []
    for combo in itertools.combinations(range(RANK), 3):
        i, j, k = combo
        trip_dead = (dead_energy[i] + dead_energy[j] + dead_energy[k]
                     + 2*(G_dead[i,j] + G_dead[i,k] + G_dead[j,k]))
        trip_sum = dead_energy[i] + dead_energy[j] + dead_energy[k]
        ratio = trip_dead / trip_sum if trip_sum > 0 else 1.0
        results.append((ratio, combo))

    results.sort()
    return results[:top_k]


def triplet_lbfgs(alpha, beta, gamma, triplet, max_iters=200,
                  beta_schedule=(10, 30, 100, 300, 1000, 3000)):
    """L-BFGS optimization of a single triplet's 81 variables.

    The other 16 terms are frozen. We compute their contribution once,
    then only recompute the 3-term subtensor during optimization.
    """
    t0, t1, t2 = triplet
    tidx = [t0, t1, t2]
    frozen = [k for k in range(RANK) if k not in tidx]

    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()

    # Frozen contribution (constant throughout optimization)
    T_frozen = np.zeros((DIM, DIM, DIM))
    for k in frozen:
        T_frozen += np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])

    # Target for the triplet: what the 3 terms need to produce
    T_target = _T - T_frozen  # (9,9,9) — the triplet must approximate this

    # Pack the 3 terms' factors into a flat vector: [a0,a1,a2, b0,b1,b2, g0,g1,g2]
    # Each is (DIM,) = 9, so 3 terms × 3 factors × 9 = 81 variables
    def pack(a, b, g):
        parts = []
        for k in tidx:
            parts.append(a[k])
        for k in tidx:
            parts.append(b[k])
        for k in tidx:
            parts.append(g[k])
        return np.concatenate(parts)

    def unpack(x):
        """Unpack 81 variables into 3 terms' (alpha, beta, gamma) slices."""
        a_slices = [x[i*DIM:(i+1)*DIM] for i in range(3)]
        b_slices = [x[3*DIM + i*DIM:3*DIM + (i+1)*DIM] for i in range(3)]
        g_slices = [x[6*DIM + i*DIM:6*DIM + (i+1)*DIM] for i in range(3)]
        return a_slices, b_slices, g_slices

    def triplet_tensor(a_slices, b_slices, g_slices):
        """Compute the sum of 3 rank-1 terms."""
        T3 = np.zeros((DIM, DIM, DIM))
        for i in range(3):
            T3 += np.einsum('a,b,c->abc', a_slices[i], b_slices[i], g_slices[i])
        return T3

    def full_residual(x):
        """Full tensor residual with triplet variables."""
        a_s, b_s, g_s = unpack(x)
        T3 = triplet_tensor(a_s, b_s, g_s)
        return T_frozen + T3 - _T  # = full_approx - target

    def max_abs_fit(x):
        return float(np.max(np.abs(full_residual(x))))

    x0 = pack(alpha, beta, gamma)
    best_x = x0.copy()
    best_mx = max_abs_fit(x0)

    # ── Phase 1: Frobenius on the triplet subtensor ──
    def fro_obj_grad(x):
        a_s, b_s, g_s = unpack(x)
        R = T_frozen + triplet_tensor(a_s, b_s, g_s) - _T
        f = 0.5 * np.sum(R**2)
        # Gradient for each of the 3 terms
        grad_parts = []
        for i in range(3):
            grad_parts.append(np.einsum('abc,b,c->a', R, b_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,c->b', R, a_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,b->c', R, a_s[i], b_s[i], optimize=True))
        return f, np.concatenate(grad_parts)

    res = minimize(fro_obj_grad, best_x, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iters // 2, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs_fit(res.x)
    if mx < best_mx:
        best_x = res.x.copy()
        best_mx = mx

    # ── Phase 1b: Dead-weighted Frobenius ──
    def dead_fro_obj_grad(x):
        a_s, b_s, g_s = unpack(x)
        R = T_frozen + triplet_tensor(a_s, b_s, g_s) - _T
        R_flat = R.ravel()
        wR = (_W * R_flat).reshape(DIM, DIM, DIM)
        f = 0.5 * np.sum(_W * R_flat**2)
        grad_parts = []
        for i in range(3):
            grad_parts.append(np.einsum('abc,b,c->a', wR, b_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,c->b', wR, a_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,b->c', wR, a_s[i], b_s[i], optimize=True))
        return f, np.concatenate(grad_parts)

    res = minimize(dead_fro_obj_grad, best_x, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iters // 2, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs_fit(res.x)
    if mx < best_mx:
        best_x = res.x.copy()
        best_mx = mx

    # ── Phase 2: smooth-max with escalating beta ──
    def smooth_max_obj_grad(x, bv):
        a_s, b_s, g_s = unpack(x)
        R = T_frozen + triplet_tensor(a_s, b_s, g_s) - _T
        R_flat = R.ravel()
        absR = np.abs(R_flat)
        mx_val = np.max(absR)
        shifted = bv * (absR - mx_val)
        exp_s = np.exp(shifted)
        Z = np.sum(exp_s)
        f = mx_val + np.log(Z) / bv
        signs = np.sign(R_flat)
        weights = (exp_s / Z * signs).reshape(DIM, DIM, DIM)
        grad_parts = []
        for i in range(3):
            grad_parts.append(np.einsum('abc,b,c->a', weights, b_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,c->b', weights, a_s[i], g_s[i], optimize=True))
        for i in range(3):
            grad_parts.append(np.einsum('abc,a,b->c', weights, a_s[i], b_s[i], optimize=True))
        return f, np.concatenate(grad_parts)

    for bv in beta_schedule:
        res = minimize(lambda x: smooth_max_obj_grad(x, bv), best_x,
                       method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        mx = max_abs_fit(res.x)
        if mx < best_mx:
            best_x = res.x.copy()
            best_mx = mx

    # Write back
    a_s, b_s, g_s = unpack(best_x)
    for i, k in enumerate(tidx):
        alpha[k] = a_s[i]
        beta[k] = b_s[i]
        gamma[k] = g_s[i]

    return alpha, beta, gamma, best_mx


def triplet_cd(alpha, beta, gamma, triplet, sweeps=5, fine_range=0.002, patience=3):
    """Coordinate descent restricted to the 81 variables of a triplet.

    Faster per-iteration than L-BFGS, good for fine-tuning after L-BFGS moves.
    Tries all 81 coefficients of the 3 terms, accepts greedy improvements.
    """
    tidx = list(triplet)
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    factors = [alpha, beta, gamma]

    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - _T
    best_mx = float(np.max(np.abs(R)))

    offsets = np.linspace(-fine_range, fine_range, 11)

    stale = 0
    for _s in range(sweeps):
        accepts = 0
        # Shuffle the 81 coefficients
        coords = [(fi, k, d) for fi in range(3) for k in tidx for d in range(DIM)]
        np.random.shuffle(coords)

        for fi, k, d in coords:
            old_val = factors[fi][k, d]
            old_outer = np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])

            best_trial_mx = best_mx
            best_trial = None

            for delta in offsets:
                if delta == 0:
                    continue
                factors[fi][k, d] = old_val + delta
                new_outer = np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])
                R_new = R - old_outer + new_outer
                new_mx = float(np.max(np.abs(R_new)))
                if new_mx < best_trial_mx - 1e-12:
                    best_trial_mx = new_mx
                    best_trial = old_val + delta

            if best_trial is not None:
                factors[fi][k, d] = best_trial
                new_outer = np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])
                R = R - old_outer + new_outer
                best_mx = best_trial_mx
                accepts += 1
            else:
                factors[fi][k, d] = old_val

        if accepts == 0:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0

    return alpha, beta, gamma, best_mx


def cross_triplet_pair(alpha, beta, gamma, triplet, sweeps=3, fine_range=0.005, n_pairs=100):
    """Cross-term pair moves: perturb one coefficient in term i, another in term j,
    both within the same triplet. This is the key move that coordinates
    interference between terms.
    """
    tidx = list(triplet)
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    factors = [alpha, beta, gamma]
    rng = np.random.default_rng()

    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - _T
    best_mx = float(np.max(np.abs(R)))

    offsets = np.linspace(-fine_range, fine_range, 7)

    for _s in range(sweeps):
        accepts = 0
        for _ in range(n_pairs):
            # Pick two different terms from the triplet
            pick = rng.choice(3, size=2, replace=False)
            k1, k2 = tidx[pick[0]], tidx[pick[1]]

            fi1 = rng.integers(3)
            fi2 = rng.integers(3)
            d1 = rng.integers(DIM)
            d2 = rng.integers(DIM)

            old_v1 = factors[fi1][k1, d1]
            old_v2 = factors[fi2][k2, d2]

            # Remove both terms' contribution
            old_outer1 = np.einsum('a,b,c->abc', alpha[k1], beta[k1], gamma[k1])
            old_outer2 = np.einsum('a,b,c->abc', alpha[k2], beta[k2], gamma[k2])

            best_pair_mx = best_mx
            best_d1, best_d2 = None, None

            for dv1 in offsets:
                factors[fi1][k1, d1] = old_v1 + dv1
                new_outer1 = np.einsum('a,b,c->abc', alpha[k1], beta[k1], gamma[k1])
                # partial residual with term 1 updated, before term 2 changes
                for dv2 in offsets:
                    factors[fi2][k2, d2] = old_v2 + dv2
                    new_outer2 = np.einsum('a,b,c->abc', alpha[k2], beta[k2], gamma[k2])
                    R_new = R - old_outer1 - old_outer2 + new_outer1 + new_outer2
                    new_mx = float(np.max(np.abs(R_new)))
                    if new_mx < best_pair_mx - 1e-12:
                        best_pair_mx = new_mx
                        best_d1, best_d2 = dv1, dv2
                # Reset term 2 for next trial
                factors[fi2][k2, d2] = old_v2

            if best_d1 is not None:
                factors[fi1][k1, d1] = old_v1 + best_d1
                factors[fi2][k2, d2] = old_v2 + best_d2
                new_outer1 = np.einsum('a,b,c->abc', alpha[k1], beta[k1], gamma[k1])
                new_outer2 = np.einsum('a,b,c->abc', alpha[k2], beta[k2], gamma[k2])
                R = R - old_outer1 - old_outer2 + new_outer1 + new_outer2
                best_mx = best_pair_mx
                accepts += 1
            else:
                factors[fi1][k1, d1] = old_v1
                factors[fi2][k2, d2] = old_v2

        if accepts == 0:
            break

    return alpha, beta, gamma, best_mx


def _refine_one_triplet(args_tuple):
    """Worker function: perturb + refine a single triplet. Runs in a child process."""
    alpha, beta, gamma, triplet, cancel_ratio, trip_rank, lbfgs_iters, perturb_sigma, seed = args_tuple

    rng = np.random.default_rng(seed)
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()

    # Perturb only the triplet's factors (the other 16 terms stay frozen)
    if perturb_sigma > 0:
        for k in triplet:
            alpha[k] += rng.normal(0, perturb_sigma, DIM)
            beta[k] += rng.normal(0, perturb_sigma, DIM)
            gamma[k] += rng.normal(0, perturb_sigma, DIM)

    # Stage 1: L-BFGS joint optimization on the triplet
    a, b, g, fit = triplet_lbfgs(
        alpha, beta, gamma, triplet,
        max_iters=lbfgs_iters,
        beta_schedule=(10, 30, 100, 300, 1000, 3000),
    )

    # Stage 2: Cross-term pair moves within the triplet
    a, b, g, fit = cross_triplet_pair(a, b, g, triplet,
                                       sweeps=3, fine_range=0.003, n_pairs=80)

    # Stage 3: Fine CD on the triplet
    a, b, g, fit = triplet_cd(a, b, g, triplet,
                               sweeps=3, fine_range=0.001, patience=2)

    return a, b, g, fit, triplet, cancel_ratio, trip_rank, perturb_sigma


def main():
    parser = argparse.ArgumentParser(description="Triplet-aware joint refinement")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("triplet_best.json"))
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of top triplets to try per round")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of outer rounds")
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    parser.add_argument("--workers", type=int,
                        default=max(1, __import__('os').cpu_count() - 2),
                        help="Number of parallel workers")
    parser.add_argument("--perturbs", type=int, default=4,
                        help="Perturbation variants per triplet (0=unperturbed + N perturbed)")
    args = parser.parse_args()

    # Perturbation schedule: one clean + progressively larger perturbations
    sigmas = [0.0] + [0.005 * (2 ** i) for i in range(args.perturbs)]
    # e.g. [0.0, 0.005, 0.01, 0.02, 0.04]

    # Load best candidate
    if args.input:
        src = args.input
    else:
        candidates = []
        for p in [Path("shotgun_best.json"), Path("chain_best.json"),
                  Path("triplet_best.json")]:
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                candidates.append((d["fitness"], p))
        candidates.sort()
        src = candidates[0][1]

    alpha, beta, gamma = load_factors(src)
    fit = fitness(alpha, beta, gamma)
    n_jobs = args.top_k * len(sigmas)
    print(f"Loaded: {src}  fitness={fit:.10f}")
    print(f"Workers: {args.workers}  top-k: {args.top_k}  "
          f"perturbs: {len(sigmas)} sigmas={[f'{s:.3f}' for s in sigmas]}  "
          f"jobs/round: {n_jobs}")

    best_a, best_b, best_g = alpha.copy(), beta.copy(), gamma.copy()
    best_fit = fit

    total_accepts = 0
    t_start = time.time()
    seed_counter = 42

    for rnd in range(args.rounds):
        rnd_t0 = time.time()
        rnd_accepts = 0

        # Re-rank triplets each round
        triplets = rank_triplets(best_a, best_b, best_g, top_k=args.top_k)

        # Build work: each triplet × each perturbation sigma
        work = []
        for trip_rank, (cancel_ratio, triplet) in enumerate(triplets):
            for sigma in sigmas:
                work.append((
                    best_a.copy(), best_b.copy(), best_g.copy(),
                    triplet, cancel_ratio, trip_rank, args.lbfgs_iters,
                    sigma, seed_counter,
                ))
                seed_counter += 1

        # Dispatch all in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_refine_one_triplet, w): w for w in work}

            for fut in as_completed(futures):
                a, b, g, fit, triplet, cancel_ratio, trip_rank, sigma = fut.result()

                if fit < best_fit - 1e-12:
                    delta = best_fit - fit
                    best_a, best_b, best_g = a.copy(), b.copy(), g.copy()
                    best_fit = fit
                    rnd_accepts += 1
                    total_accepts += 1
                    print(f"  R{rnd:3d} T{trip_rank:2d} {triplet} "
                          f"σ={sigma:.3f} cancel={1-cancel_ratio:.1%}  "
                          f"{fit + delta:.10f} → {best_fit:.10f}  Δ={delta:.2e}")
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
