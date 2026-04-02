#!/usr/bin/env python
"""Sequential Linear Programming (SLP) minimax optimizer.

Directly targets the true minimax objective without smooth-max approximation.
Each iteration:
  1. Build Jacobian at current point
  2. Solve LP: minimize t s.t. |R_i + J_i @ dx| <= t, ||dx||_inf <= trust
  3. Take step, shrink/grow trust region based on actual vs predicted improvement
  4. Repeat until convergence

This is provably convergent to a first-order minimax stationary point.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg as la
from scipy.optimize import linprog

np.set_printoptions(precision=10, linewidth=140, suppress=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T

R, D = RANK, DIM
N_ENTRIES = D**3
N_PARAMS = 3 * R * D


def load_factors(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["alpha"]), np.array(data["beta"]), np.array(data["gamma"]), data["fitness"]


def save_factors(alpha, beta, gamma, fit, path):
    data = {"fitness": float(fit), "rank": R, "dim": D,
            "alpha": alpha.tolist(), "beta": beta.tolist(), "gamma": gamma.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def pack(a, b, g):
    return np.concatenate([a.ravel(), b.ravel(), g.ravel()])


def unpack(x):
    return (x[:R*D].reshape(R, D),
            x[R*D:2*R*D].reshape(R, D),
            x[2*R*D:].reshape(R, D))


def residual_flat(x):
    a, b, g = unpack(x)
    return (np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - T).ravel()


def maxabs(x):
    return float(np.max(np.abs(residual_flat(x))))


def build_jacobian(x):
    """Build full Jacobian J (729 x 513) at point x."""
    a, b, g = unpack(x)
    J = np.zeros((N_ENTRIES, N_PARAMS))
    for k in range(R):
        # d/d alpha[k, a_idx]: beta[k,b] * gamma[k,c] at entry (a_idx, b, c)
        # Vectorized: for fixed k, J[a*D*D + b*D + c, k*D + a] += beta[k,b]*gamma[k,c]
        bg = np.outer(b[k], g[k])  # (D, D)
        for ai in range(D):
            J[ai*D*D:(ai+1)*D*D, k*D + ai] += bg.ravel()

        ag = np.outer(a[k], g[k])  # (D, D)
        for bi in range(D):
            J[bi*D + np.arange(D)*D*D + np.arange(D).reshape(-1,1), R*D + k*D + bi] = 0  # clear
        # Simpler: loop
        for bi in range(D):
            for ai in range(D):
                for ci in range(D):
                    J[ai*D*D + bi*D + ci, R*D + k*D + bi] += a[k, ai] * g[k, ci]

        ab = np.outer(a[k], b[k])  # (D, D)
        for ci in range(D):
            for ai in range(D):
                for bi in range(D):
                    J[ai*D*D + bi*D + ci, 2*R*D + k*D + ci] += a[k, ai] * b[k, bi]

    return J


def build_jacobian_fast(x):
    """Build full Jacobian J (729 x 513) — vectorized."""
    a, b, g = unpack(x)
    J = np.zeros((N_ENTRIES, N_PARAMS))

    for k in range(R):
        # alpha block: d T_hat[a,b,c] / d alpha[k, a'] = delta(a,a') * beta[k,b] * gamma[k,c]
        bg_flat = np.outer(b[k], g[k]).ravel()  # (81,)
        for ai in range(D):
            J[ai*81:(ai+1)*81, k*D + ai] += bg_flat

        # beta block: d T_hat[a,b,c] / d beta[k, b'] = alpha[k,a] * delta(b,b') * gamma[k,c]
        for bi in range(D):
            for ai in range(D):
                J[ai*81 + bi*D: ai*81 + bi*D + D, R*D + k*D + bi] += a[k, ai] * g[k]

        # gamma block: d T_hat[a,b,c] / d gamma[k, c'] = alpha[k,a] * beta[k,b] * delta(c,c')
        ab_flat = np.outer(a[k], b[k]).ravel()  # (81,)
        for ci in range(D):
            J[ci::D, 2*R*D + k*D + ci] += ab_flat  # wrong stride
    # Fix gamma: T_hat[a,b,c] index is a*81 + b*9 + c
    # So entries with c=ci are at indices a*81 + b*9 + ci for all a,b
    # That's np.arange(81)*9 + ci ... no. Let me redo.

    # Actually let me just do it cleanly with einsum-style indexing
    J = np.zeros((N_ENTRIES, N_PARAMS))
    # Index mapping: flat index = a*D*D + b*D + c
    for k in range(R):
        for ai in range(D):
            # All entries (ai, b, c) for b=0..8, c=0..8
            start = ai * D * D
            bg = np.outer(b[k], g[k]).ravel()  # (81,)
            J[start:start+D*D, k*D + ai] += bg

        for bi in range(D):
            # All entries (a, bi, c) for a=0..8, c=0..8
            # indices: a*81 + bi*9 + c for a=0..8, c=0..8
            idx = np.arange(D)[:, None] * D * D + bi * D + np.arange(D)[None, :]  # (9,9)
            vals = np.outer(a[k], g[k])  # (9,9): a[k,a] * g[k,c]
            J[idx.ravel(), R*D + k*D + bi] += vals.ravel()

        for ci in range(D):
            # All entries (a, b, ci) for a=0..8, b=0..8
            idx = np.arange(D)[:, None] * D * D + np.arange(D)[None, :] * D + ci  # (9,9)
            vals = np.outer(a[k], b[k])  # (9,9): a[k,a] * b[k,b]
            J[idx.ravel(), 2*R*D + k*D + ci] += vals.ravel()

    return J


def solve_lp_step(res_flat, J, trust):
    """Solve LP: min t s.t. |R + J@dx| <= t, ||dx||_inf <= trust.
    
    Returns (dx, t_opt) or (None, None) if infeasible.
    """
    # Variables: [dx (N_PARAMS), t (1)]
    n_vars = N_PARAMS + 1
    n = N_ENTRIES

    # Constraints: J@dx - t <= -R  and  -J@dx - t <= R
    # Stack: [J, -1; -J, -1] @ [dx; t] <= [-R; R]
    ones_col = -np.ones((n, 1))
    A_upper = np.hstack([J, ones_col])
    A_lower = np.hstack([-J, ones_col])
    A_ub = np.vstack([A_upper, A_lower])
    b_ub = np.concatenate([-res_flat, res_flat])

    c_obj = np.zeros(n_vars)
    c_obj[N_PARAMS] = 1.0  # minimize t

    bounds = [(-trust, trust)] * N_PARAMS + [(0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'time_limit': 60})

    if result.success:
        return result.x[:N_PARAMS], result.x[N_PARAMS]
    return None, None


def main():
    parser = argparse.ArgumentParser(description="SLP minimax optimizer")
    parser.add_argument("--input", default="shotgun_best.json")
    parser.add_argument("--out", default="slp_best.json")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--trust-init", type=float, default=0.01)
    parser.add_argument("--trust-max", type=float, default=0.5)
    parser.add_argument("--trust-min", type=float, default=1e-8)
    parser.add_argument("--eta-accept", type=float, default=0.01,
                        help="Accept step if actual/predicted > eta")
    parser.add_argument("--save-freq", type=int, default=10)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        src = Path(__file__).resolve().parent.parent / args.input
    a, b, g, loaded_fit = load_factors(src)
    x = pack(a, b, g)

    current_maxabs = maxabs(x)
    best_x = x.copy()
    best_fit = current_maxabs

    print(f"SLP Minimax Optimizer")
    print(f"Loaded: {src}  fitness={loaded_fit:.10f}")
    print(f"Verified max-abs: {current_maxabs:.10f}")
    print(f"Trust region: {args.trust_init}")
    print(f"Max iterations: {args.max_iters}")
    print()

    trust = args.trust_init
    n_accepts = 0
    n_rejects = 0
    t_start = time.time()

    for it in range(args.max_iters):
        it_t0 = time.time()

        # Step 1: Build Jacobian
        res_flat = residual_flat(x)
        J = build_jacobian_fast(x)

        # Step 2: Solve LP
        dx, t_pred = solve_lp_step(res_flat, J, trust)

        if dx is None:
            print(f"  iter {it:4d}: LP infeasible, shrinking trust {trust:.2e} -> {trust/4:.2e}")
            trust = max(trust / 4, args.trust_min)
            continue

        pred_improvement = current_maxabs - t_pred

        if pred_improvement < 1e-14:
            print(f"  iter {it:4d}: No predicted improvement ({pred_improvement:.2e}), converged.")
            break

        # Step 3: Take step and evaluate actual improvement
        x_new = x + dx
        new_maxabs = maxabs(x_new)
        actual_improvement = current_maxabs - new_maxabs
        rho = actual_improvement / pred_improvement  # actual/predicted ratio

        dx_norm = la.norm(dx, np.inf)
        dt = time.time() - it_t0

        # Step 4: Trust region update
        if rho >= args.eta_accept and actual_improvement > 0:
            # Accept step
            x = x_new
            current_maxabs = new_maxabs
            n_accepts += 1

            if new_maxabs < best_fit:
                best_x = x.copy()
                best_fit = new_maxabs

            # Grow trust if step was good
            if rho > 0.75:
                trust = min(trust * 2, args.trust_max)
            elif rho > 0.5:
                trust = min(trust * 1.5, args.trust_max)

            print(f"  iter {it:4d}: ACCEPT  maxabs={new_maxabs:.10f}  "
                  f"pred={t_pred:.10f}  rho={rho:.4f}  "
                  f"trust={trust:.2e}  ||dx||={dx_norm:.2e}  {dt:.1f}s")
        else:
            # Reject step, shrink trust
            n_rejects += 1
            trust = max(trust / 2, args.trust_min)

            if it < 20 or it % 50 == 0:
                print(f"  iter {it:4d}: REJECT  actual={new_maxabs:.10f}  "
                      f"pred={t_pred:.10f}  rho={rho:.4f}  "
                      f"trust={trust:.2e}  {dt:.1f}s")

        # Save periodically
        if n_accepts > 0 and n_accepts % args.save_freq == 0:
            a_best, b_best, g_best = unpack(best_x)
            save_factors(a_best, b_best, g_best, best_fit, args.out)
            print(f"    -> Saved (accepts={n_accepts}, best={best_fit:.10f})")

        # Convergence check
        if trust < args.trust_min:
            print(f"  Trust region below minimum ({args.trust_min:.2e}), converged.")
            break

    elapsed = time.time() - t_start

    # Final save
    a_best, b_best, g_best = unpack(best_x)
    save_factors(a_best, b_best, g_best, best_fit, args.out)

    print(f"\n{'='*60}")
    print(f"SLP RESULT")
    print(f"{'='*60}")
    print(f"Original:  {loaded_fit:.10f}")
    print(f"Final:     {best_fit:.10f}")
    if best_fit < loaded_fit:
        print(f"Improvement: {loaded_fit - best_fit:.2e}")
    else:
        print(f"No improvement.")
    print(f"Accepts: {n_accepts},  Rejects: {n_rejects}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
