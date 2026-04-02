#!/usr/bin/env python
"""Turbo SLP minimax — active-set LP + 24-worker parallel descents.

Key speedups vs slp_minimax.py:
  1. Active-set LP: only include constraints near the max (top-K entries),
     verify full residual after. Falls back to full LP if active set misses.
  2. Warm-start: reuse previous LP basis via HiGHS.
  3. Fully vectorized Jacobian (no Python loops over entries).
  4. Parallel multi-start: N workers each descend from perturbed seeds.
     Best result wins.
"""

import argparse
import json
import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import linalg as la
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T

R, D = RANK, DIM
N_ENTRIES = D**3  # 729
N_PARAMS = 3 * R * D  # 513


# ── I/O ──────────────────────────────────────────────────────────────────

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


# ── Core math ────────────────────────────────────────────────────────────

def residual_flat(x):
    a, b, g = unpack(x)
    return (np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - T).ravel()


def maxabs(x):
    return float(np.max(np.abs(residual_flat(x))))


def build_jacobian(x):
    """Fully vectorized Jacobian (729 x 513). Zero Python loops over entries."""
    a, b, g = unpack(x)
    J = np.zeros((N_ENTRIES, N_PARAMS))

    # Precompute index arrays (could cache these but they're cheap)
    ai_range = np.arange(D)
    
    # alpha block: dT[a,b,c]/d alpha[k,a'] = delta(a,a') * beta[k,b] * gamma[k,c]
    # For each k: block of D columns at [k*D : k*D+D]
    # For a' in 0..D-1: rows a'*D*D .. (a'+1)*D*D get outer(b[k], g[k])
    bg = np.einsum('kb,kc->kbc', b, g).reshape(R, D*D)  # (R, D*D)
    for ai in range(D):
        # J[ai*D*D:(ai+1)*D*D, k*D+ai for k in 0..R-1] += bg[k]
        J[ai*D*D:(ai+1)*D*D, ai::D][:, :R] += bg.T  # (D*D, R)

    # beta block: dT[a,b,c]/d beta[k,b'] = alpha[k,a] * delta(b,b') * gamma[k,c]
    # Rows with b=b': indices a*D*D + b'*D + c for all a,c → shape (D,D) reshaped to D*D
    ag = np.einsum('ka,kc->kac', a, g).reshape(R, D*D)  # (R, D*D)
    for bi in range(D):
        idx = (ai_range[:, None] * D * D + bi * D + ai_range[None, :]).ravel()
        J[np.ix_(idx, R*D + bi + ai_range[:R] * D)] = 0  # not needed, J is zeros
        J[idx[:, None], R*D + np.arange(R) * D + bi] += ag.T  # (D*D, R)

    # gamma block: dT[a,b,c]/d gamma[k,c'] = alpha[k,a] * beta[k,b] * delta(c,c')
    ab = np.einsum('ka,kb->kab', a, b).reshape(R, D*D)  # (R, D*D)
    for ci in range(D):
        idx = (ai_range[:, None] * D * D + ai_range[None, :] * D + ci).ravel()
        J[idx[:, None], 2*R*D + np.arange(R) * D + ci] += ab.T  # (D*D, R)

    return J


def solve_lp_active(res_flat, J, trust, active_k=150, prev_basis=None):
    """Active-set LP: use only top-K constraints, verify full residual.
    
    Returns (dx, t_pred, basis_info, used_full).
    """
    abs_res = np.abs(res_flat)
    # Select active set: top-K by |residual|
    active_idx = np.argpartition(abs_res, -active_k)[-active_k:]

    # Try active-set solve first
    dx, t_pred, basis = _solve_lp(res_flat, J, trust, active_idx, prev_basis)
    if dx is None:
        return None, None, None, False

    # Verify: does the step violate any inactive constraint?
    new_res_lin = res_flat + J @ dx
    actual_max_lin = np.max(np.abs(new_res_lin))
    
    if actual_max_lin <= t_pred * 1.001:  # active set was sufficient
        return dx, t_pred, basis, False

    # Active set missed — fall back to full LP
    full_idx = np.arange(N_ENTRIES)
    dx, t_pred, basis = _solve_lp(res_flat, J, trust, full_idx, prev_basis)
    return dx, t_pred, basis, True


def _solve_lp(res_flat, J, trust, idx, prev_basis=None):
    """Core LP solve on subset of constraints."""
    n_active = len(idx)
    n_vars = N_PARAMS + 1
    
    J_sub = J[idx]
    r_sub = res_flat[idx]

    ones_col = -np.ones((n_active, 1))
    A_ub = np.vstack([
        np.hstack([J_sub, ones_col]),
        np.hstack([-J_sub, ones_col])
    ])
    b_ub = np.concatenate([-r_sub, r_sub])

    c_obj = np.zeros(n_vars)
    c_obj[N_PARAMS] = 1.0

    bounds = [(-trust, trust)] * N_PARAMS + [(0, None)]

    options = {'presolve': True, 'time_limit': 30}
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options=options)

    if result.success:
        return result.x[:N_PARAMS], result.x[N_PARAMS], None
    return None, None, None


# ── Shared progress state ────────────────────────────────────────────────

_progress_iters = None   # shared Array: iteration count per worker
_progress_fits = None    # shared Array: current best fitness per worker
_progress_accepts = None # shared Array: accept count per worker

def _init_worker(iters_arr, fits_arr, accepts_arr):
    global _progress_iters, _progress_fits, _progress_accepts
    _progress_iters = iters_arr
    _progress_fits = fits_arr
    _progress_accepts = accepts_arr


# ── Single worker descent ────────────────────────────────────────────────

def slp_descent(args_tuple):
    """Single SLP descent worker. Returns (best_x, best_fit, accepts, trajectory)."""
    worker_id, x0, max_iters, trust_init, trust_max, trust_min, eta_accept, active_k = args_tuple
    
    x = x0.copy()
    current_fit = maxabs(x)
    best_x = x.copy()
    best_fit = current_fit
    
    trust = trust_init
    n_accepts = 0
    n_rejects = 0
    prev_basis = None
    trajectory = [(0, current_fit)]
    
    # Update shared progress
    if _progress_fits is not None:
        _progress_fits[worker_id] = current_fit
    
    for it in range(max_iters):
        res = residual_flat(x)
        J = build_jacobian(x)
        
        dx, t_pred, basis, used_full = solve_lp_active(res, J, trust, active_k, prev_basis)
        
        if dx is None:
            trust = max(trust / 4, trust_min)
            if trust <= trust_min:
                break
            continue
        
        pred_improvement = current_fit - t_pred
        if pred_improvement < 1e-14:
            break
        
        x_new = x + dx
        new_fit = maxabs(x_new)
        actual_improvement = current_fit - new_fit
        rho = actual_improvement / pred_improvement
        
        if rho >= eta_accept and actual_improvement > 0:
            x = x_new
            current_fit = new_fit
            n_accepts += 1
            prev_basis = basis
            
            if new_fit < best_fit:
                best_x = x.copy()
                best_fit = new_fit
            
            if rho > 0.75:
                trust = min(trust * 2, trust_max)
            elif rho > 0.5:
                trust = min(trust * 1.5, trust_max)
        else:
            n_rejects += 1
            trust = max(trust / 2, trust_min)
            prev_basis = None  # reset basis on reject
        
        if trust < trust_min:
            break
        
        # Update shared progress every iteration
        if _progress_iters is not None:
            _progress_iters[worker_id] = it + 1
            _progress_fits[worker_id] = best_fit
            _progress_accepts[worker_id] = n_accepts
        
        # Log every 50 accepts
        if n_accepts > 0 and n_accepts % 50 == 0 and actual_improvement > 0:
            trajectory.append((it, best_fit))
    
    trajectory.append((max_iters, best_fit))
    return worker_id, best_x, best_fit, n_accepts, n_rejects, trajectory


# ── Perturbation strategies ──────────────────────────────────────────────

def perturb_gaussian(x, scale, rng):
    """Add Gaussian noise."""
    return x + rng.normal(0, scale, size=x.shape)


def perturb_targeted(x, scale, rng):
    """Perturb only the params that contribute most to the worst entries."""
    res = residual_flat(x)
    abs_res = np.abs(res)
    # Find top-20 worst entries
    worst = np.argsort(abs_res)[-20:]
    
    J = build_jacobian(x)
    # Which params have highest |J| for worst entries?
    sensitivity = np.sum(np.abs(J[worst]), axis=0)
    sensitivity /= sensitivity.max() + 1e-30
    
    # Perturb proportional to sensitivity
    noise = rng.normal(0, scale, size=x.shape) * sensitivity
    return x + noise


def make_seeds(x0, n_workers, perturbation_scale, rng):
    """Create n_workers seed points from x0."""
    seeds = [x0.copy()]  # worker 0 = unperturbed (continue descent)
    
    for i in range(1, n_workers):
        if i % 3 == 0:
            # Targeted perturbation (every 3rd worker)
            seeds.append(perturb_targeted(x0, perturbation_scale, rng))
        elif i % 3 == 1:
            # Small Gaussian
            seeds.append(perturb_gaussian(x0, perturbation_scale * 0.5, rng))
        else:
            # Larger Gaussian
            seeds.append(perturb_gaussian(x0, perturbation_scale * 2.0, rng))
    
    return seeds


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Turbo SLP: active-set + parallel")
    parser.add_argument("--input", default="slp_best.json")
    parser.add_argument("--out", default="slp_turbo_best.json")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-iters", type=int, default=500,
                        help="Iterations per worker")
    parser.add_argument("--trust-init", type=float, default=0.003)
    parser.add_argument("--trust-max", type=float, default=0.5)
    parser.add_argument("--trust-min", type=float, default=1e-8)
    parser.add_argument("--eta-accept", type=float, default=0.01)
    parser.add_argument("--active-k", type=int, default=150,
                        help="Number of active constraints in LP subset")
    parser.add_argument("--perturb-scale", type=float, default=0.001,
                        help="Perturbation scale for worker seeds")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of parallel rounds (best feeds next)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        src = Path(__file__).resolve().parent.parent / args.input
    a, b, g, loaded_fit = load_factors(src)
    x_best = pack(a, b, g)
    verified_fit = maxabs(x_best)

    print(f"Turbo SLP — {args.workers} workers × {args.max_iters} iters/worker")
    print(f"Loaded: {src}  fitness={loaded_fit:.10f}")
    print(f"Verified: {verified_fit:.10f}")
    print(f"Active-set K={args.active_k}, trust_init={args.trust_init}")
    print(f"Perturbation scale: {args.perturb_scale}")
    print(f"Rounds: {args.rounds}")
    print()

    rng = np.random.default_rng(args.seed)
    overall_best_fit = verified_fit
    overall_best_x = x_best.copy()

    for rnd in range(args.rounds):
        t_round = time.time()
        
        seeds = make_seeds(overall_best_x, args.workers, args.perturb_scale, rng)
        
        # Log seed fitnesses
        seed_fits = [maxabs(s) for s in seeds]
        print(f"Round {rnd+1}/{args.rounds}  seed fitnesses: "
              f"min={min(seed_fits):.10f}  max={max(seed_fits):.10f}")
        
        # Build work items
        work = [
            (wid, seeds[wid], args.max_iters, args.trust_init, args.trust_max,
             args.trust_min, args.eta_accept, args.active_k)
            for wid in range(args.workers)
        ]
        
        # Shared progress arrays
        sh_iters = mp.Array('i', args.workers)    # int: iteration count
        sh_fits = mp.Array('d', args.workers)     # double: best fitness
        sh_accepts = mp.Array('i', args.workers)  # int: accept count
        for wi in range(args.workers):
            sh_fits[wi] = seed_fits[wi]
        
        # Progress monitor thread
        stop_monitor = threading.Event()
        def monitor():
            while not stop_monitor.is_set():
                stop_monitor.wait(15)  # print every 15s
                if stop_monitor.is_set():
                    break
                iters = [sh_iters[i] for i in range(args.workers)]
                fits = [sh_fits[i] for i in range(args.workers)]
                accs = [sh_accepts[i] for i in range(args.workers)]
                best_w = int(np.argmin(fits))
                min_it = min(iters)
                max_it = max(iters)
                done_pct = sum(iters) / (args.workers * args.max_iters) * 100
                print(f"  [{time.time()-t_round:.0f}s] "
                      f"progress {done_pct:.0f}%  "
                      f"iters {min_it}-{max_it}/{args.max_iters}  "
                      f"best W{best_w:02d}={fits[best_w]:.10f} ({accs[best_w]}acc)  "
                      f"global_best={min(fits):.10f}",
                      flush=True)
        
        mon_thread = threading.Thread(target=monitor, daemon=True)
        mon_thread.start()
        
        # Launch parallel descents
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(sh_iters, sh_fits, sh_accepts)) as pool:
            results = pool.map(slp_descent, work)
        
        stop_monitor.set()
        mon_thread.join(timeout=2)
        
        dt_round = time.time() - t_round
        
        # Collect results
        print(f"\n  Round {rnd+1} results ({dt_round:.0f}s):")
        results_sorted = sorted(results, key=lambda r: r[2])
        
        for wid, _, fit, acc, rej, traj in results_sorted[:5]:
            print(f"    W{wid:02d}: {fit:.10f}  (accepts={acc}, rejects={rej})")
        if len(results_sorted) > 5:
            worst_fit = results_sorted[-1][2]
            print(f"    ... worst: {worst_fit:.10f}")
        
        # Update best
        best_result = results_sorted[0]
        if best_result[2] < overall_best_fit:
            overall_best_fit = best_result[2]
            overall_best_x = best_result[1].copy()
            a_b, b_b, g_b = unpack(overall_best_x)
            save_factors(a_b, b_b, g_b, overall_best_fit, args.out)
            print(f"  >> NEW BEST: {overall_best_fit:.10f} (worker {best_result[0]})")
        else:
            print(f"  No improvement this round (best remains {overall_best_fit:.10f})")
        
        # Also save top-3 for diversity
        for rank_i, (wid, x_r, fit_r, _, _, _) in enumerate(results_sorted[:3]):
            a_r, b_r, g_r = unpack(x_r)
            save_factors(a_r, b_r, g_r, fit_r,
                         args.out.replace(".json", f"_top{rank_i}.json"))
        
        print()
    
    # Final summary
    print("=" * 60)
    print("TURBO SLP RESULT")
    print("=" * 60)
    print(f"Input:     {loaded_fit:.10f}")
    print(f"Final:     {overall_best_fit:.10f}")
    imp = loaded_fit - overall_best_fit
    if imp > 0:
        print(f"Improvement: {imp:.2e}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
