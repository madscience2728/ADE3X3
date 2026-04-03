#!/usr/bin/env python3
"""Phase 2: Grassmannian rank-constrained optimizer.

This version uses exact LP subproblems on the fixed-V variety.

For a fixed 10-dimensional subspace V of R^18, the constraint

    H(alpha, beta)[k, :] in V

is linear in alpha when beta is fixed, and linear in beta when alpha is fixed.
So we can solve alternating L-infinity LPs for alpha and beta while keeping
rank(H) <= 10 by construction.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alternating_linf.solver import T, _solve_gamma_col, als_init, compute_fitness


def compute_H_fast(alpha, beta):
    """Build H = [Eta1 | Eta2], shape (R, 18)."""
    R = alpha.shape[0]
    A = alpha.reshape(R, 3, 3)
    B = beta.reshape(R, 3, 3)
    eta1 = (A[:, :, 0:1] * B[:, 0:1, :] - A[:, :, 1:2] * B[:, 1:2, :]).reshape(R, 9)
    eta2 = (A[:, :, 1:2] * B[:, 1:2, :] - A[:, :, 2:3] * B[:, 2:3, :]).reshape(R, 9)
    return np.hstack([eta1, eta2])


def flatten_by_columns(mat):
    """Flatten (R, 9) as [col0, col1, ..., col8]."""
    return np.concatenate([mat[:, j] for j in range(mat.shape[1])])


def unflatten_by_columns(vec, rows, cols=9):
    """Inverse of flatten_by_columns."""
    out = np.zeros((rows, cols), dtype=np.float64)
    for j in range(cols):
        out[:, j] = vec[j * rows:(j + 1) * rows]
    return out


def build_alpha_jacobian(beta_k):
    """Jacobian J_alpha with H_k = J_alpha @ alpha_k for fixed beta_k."""
    B = beta_k.reshape(3, 3)
    J = np.zeros((18, 9), dtype=np.float64)
    for r in range(3):
        for u in range(3):
            row1 = 3 * r + u
            row2 = 9 + 3 * r + u
            J[row1, 3 * r + 0] += B[0, u]
            J[row1, 3 * r + 1] += -B[1, u]
            J[row2, 3 * r + 1] += B[1, u]
            J[row2, 3 * r + 2] += -B[2, u]
    return J


def build_beta_jacobian(alpha_k):
    """Jacobian J_beta with H_k = J_beta @ beta_k for fixed alpha_k."""
    A = alpha_k.reshape(3, 3)
    J = np.zeros((18, 9), dtype=np.float64)
    for r in range(3):
        for u in range(3):
            row1 = 3 * r + u
            row2 = 9 + 3 * r + u
            J[row1, u] += A[r, 0]
            J[row1, 3 + u] += -A[r, 1]
            J[row2, 3 + u] += A[r, 1]
            J[row2, 6 + u] += -A[r, 2]
    return J


def update_V(H, target_rank=10, V_perp_old=None, damping=0.0):
    """Return the orthonormal complement of the top target_rank right singular space."""
    _, singular_values, vt = np.linalg.svd(H, full_matrices=True)
    d_perp = H.shape[1] - target_rank
    V_perp_new = vt[target_rank:, :].T

    if damping > 0.0 and V_perp_old is not None:
        proj_new = V_perp_new @ V_perp_new.T
        proj_old = V_perp_old @ V_perp_old.T
        proj_mix = (1.0 - damping) * proj_new + damping * proj_old
        eigvals, eigvecs = np.linalg.eigh(proj_mix)
        order = np.argsort(eigvals)
        V_perp_new = eigvecs[:, order[-d_perp:]]

    tail_energy = float(np.sum(singular_values[target_rank:] ** 2)) if len(singular_values) > target_rank else 0.0
    return V_perp_new, singular_values, tail_energy


def solve_alpha_full_constrained(beta, gamma, V_perp):
    """Solve min L_inf over full alpha subject to H(alpha,beta) rows lying in V."""
    R = beta.shape[0]
    num_vars = 9 * R

    c_obj = np.zeros(num_vars + 1, dtype=np.float64)
    c_obj[-1] = 1.0

    A_ub_parts = []
    b_ub_parts = []
    block = np.einsum('kb,kc->bck', beta, gamma).reshape(81, R)
    for a in range(9):
        M = np.zeros((81, num_vars), dtype=np.float64)
        M[:, a * R:(a + 1) * R] = block
        target = T[a].ravel()
        ones = np.ones((81, 1), dtype=np.float64)
        A_ub_parts.append(np.hstack([M, -ones]))
        b_ub_parts.append(target)
        A_ub_parts.append(np.hstack([-M, -ones]))
        b_ub_parts.append(-target)
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)

    d_perp = V_perp.shape[1]
    A_eq = np.zeros((R * d_perp, num_vars + 1), dtype=np.float64)
    b_eq = np.zeros(R * d_perp, dtype=np.float64)
    for k in range(R):
        Ck = V_perp.T @ build_alpha_jacobian(beta[k])
        for j in range(d_perp):
            row = k * d_perp + j
            for a in range(9):
                A_eq[row, a * R + k] = Ck[j, a]

    bounds = [(None, None)] * num_vars + [(0.0, None)]
    res = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={
            'presolve': True,
            'time_limit': 60.0,
            'dual_feasibility_tolerance': 1e-10,
            'primal_feasibility_tolerance': 1e-10,
        },
    )
    if not res.success:
        return None, res
    alpha = unflatten_by_columns(res.x[:num_vars], R)
    return alpha, res


def solve_beta_full_constrained(alpha, gamma, V_perp):
    """Solve min L_inf over full beta subject to H(alpha,beta) rows lying in V."""
    R = alpha.shape[0]
    num_vars = 9 * R

    c_obj = np.zeros(num_vars + 1, dtype=np.float64)
    c_obj[-1] = 1.0

    A_ub_parts = []
    b_ub_parts = []
    block = np.einsum('ka,kc->ack', alpha, gamma).reshape(81, R)
    for b in range(9):
        M = np.zeros((81, num_vars), dtype=np.float64)
        M[:, b * R:(b + 1) * R] = block
        target = T[:, b, :].ravel()
        ones = np.ones((81, 1), dtype=np.float64)
        A_ub_parts.append(np.hstack([M, -ones]))
        b_ub_parts.append(target)
        A_ub_parts.append(np.hstack([-M, -ones]))
        b_ub_parts.append(-target)
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)

    d_perp = V_perp.shape[1]
    A_eq = np.zeros((R * d_perp, num_vars + 1), dtype=np.float64)
    b_eq = np.zeros(R * d_perp, dtype=np.float64)
    for k in range(R):
        Ck = V_perp.T @ build_beta_jacobian(alpha[k])
        for j in range(d_perp):
            row = k * d_perp + j
            for b in range(9):
                A_eq[row, b * R + k] = Ck[j, b]

    bounds = [(None, None)] * num_vars + [(0.0, None)]
    res = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={
            'presolve': True,
            'time_limit': 60.0,
            'dual_feasibility_tolerance': 1e-10,
            'primal_feasibility_tolerance': 1e-10,
        },
    )
    if not res.success:
        return None, res
    beta = unflatten_by_columns(res.x[:num_vars], R)
    return beta, res


def solve_gamma_all(alpha, beta, gamma):
    """Exact LP polish for gamma with fixed alpha, beta."""
    gamma = gamma.copy()
    for c in range(9):
        x = _solve_gamma_col(c, alpha, beta)
        if x is not None:
            gamma[:, c] = x
    return gamma


def grassmannian_optimize(
    alpha,
    beta,
    gamma,
    target_rank=10,
    outer_iters=25,
    damping=0.2,
    verbose=True,
):
    """Alternating LP optimizer on the fixed-rank variety."""
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    H = compute_H_fast(alpha, beta)
    V_perp, singular_values, tail_energy = update_V(H, target_rank)

    initial_fit = compute_fitness(alpha, beta, gamma)
    best_fit = None
    best_alpha = None
    best_beta = None
    best_gamma = None
    history = []

    if verbose:
        H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))
        sigma_10 = singular_values[10] if len(singular_values) > 10 else 0.0
        sigma_11 = singular_values[11] if len(singular_values) > 11 else 0.0
        print(
            f"  Init: fitness={initial_fit:.8f}  rank(H)={H_rank}  "
            f"tail_energy={tail_energy:.6e}  sigma10={sigma_10:.4f}  sigma11={sigma_11:.4f}"
        )

    V_perp_old = V_perp.copy()
    for outer in range(outer_iters):
        started = time.time()

        alpha_new, alpha_res = solve_alpha_full_constrained(beta, gamma, V_perp)
        if alpha_new is None:
            if verbose:
                print(f"  [{outer:3d}] alpha LP failed: {alpha_res.message}")
            break
        alpha = alpha_new

        beta_new, beta_res = solve_beta_full_constrained(alpha, gamma, V_perp)
        if beta_new is None:
            if verbose:
                print(f"  [{outer:3d}] beta LP failed: {beta_res.message}")
            break
        beta = beta_new

        gamma = solve_gamma_all(alpha, beta, gamma)

        H = compute_H_fast(alpha, beta)
        H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))
        fit = compute_fitness(alpha, beta, gamma)

        V_perp, singular_values, tail_energy = update_V(
            H,
            target_rank=target_rank,
            V_perp_old=V_perp_old,
            damping=damping,
        )
        V_perp_old = V_perp.copy()

        sigma_10 = singular_values[10] if len(singular_values) > 10 else 0.0
        sigma_11 = singular_values[11] if len(singular_values) > 11 else 0.0
        entry = {
            'outer': outer,
            'fitness': fit,
            'H_rank': H_rank,
            'tail_energy': float(tail_energy),
            'sigma_10': float(sigma_10),
            'sigma_11': float(sigma_11),
            'alpha_lp_obj': float(alpha_res.fun),
            'beta_lp_obj': float(beta_res.fun),
            'elapsed': float(time.time() - started),
        }
        history.append(entry)

        if best_fit is None or fit < best_fit:
            best_fit = fit
            best_alpha = alpha.copy()
            best_beta = beta.copy()
            best_gamma = gamma.copy()

        if verbose:
            print(
                f"  [{outer:3d}] fit={fit:.8f}  rank(H)={H_rank:2d}  "
                f"tail={tail_energy:.3e}  sigma10={sigma_10:.4f}  sigma11={sigma_11:.4f}  "
                f"alpha_lp={alpha_res.fun:.6f}  beta_lp={beta_res.fun:.6f}  "
                f"{entry['elapsed']:.1f}s"
            )

        if outer >= 2:
            recent = [item['fitness'] for item in history[-3:]]
            if max(recent) - min(recent) < 1e-10 and tail_energy < 1e-20:
                if verbose:
                    print(f"  Converged at outer={outer}")
                break

    if best_alpha is None:
        return alpha, beta, gamma, initial_fit, history
    return best_alpha, best_beta, best_gamma, best_fit, history


def load_factors(path):
    with open(path) as f:
        data = json.load(f)
    if 'alpha' in data:
        return (
            np.array(data['alpha'], dtype=np.float64),
            np.array(data['beta'], dtype=np.float64),
            np.array(data['gamma'], dtype=np.float64),
        )
    terms = data['terms']
    R = len(terms)
    alpha = np.zeros((R, 9), dtype=np.float64)
    beta = np.zeros((R, 9), dtype=np.float64)
    gamma = np.zeros((R, 9), dtype=np.float64)
    for i, term in enumerate(terms):
        for idx, val in zip(term['alpha_support'], term['alpha_values']):
            alpha[i, idx] = val
        for idx, val in zip(term['beta_support'], term['beta_values']):
            beta[i, idx] = val
        for idx, val in zip(term['gamma_support'], term['gamma_values']):
            gamma[i, idx] = val
    return alpha, beta, gamma


def save_result(path, alpha, beta, gamma, fitness, H_rank, history=None):
    payload = {
        'alpha': alpha.tolist(),
        'beta': beta.tolist(),
        'gamma': gamma.tolist(),
        'fitness': float(fitness),
        'H_rank': int(H_rank),
    }
    if history is not None:
        payload['history'] = history
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def worker_run(cfg_json):
    cfg = json.loads(cfg_json)
    wid = cfg['id']
    if cfg['init'] == 'warm':
        alpha, beta, gamma = load_factors(cfg['path'])
        sigma = cfg.get('sigma', 0.0)
        if sigma > 0.0:
            rng = np.random.default_rng(3000 + wid)
            alpha += rng.normal(0.0, sigma, alpha.shape)
            beta += rng.normal(0.0, sigma, beta.shape)
    else:
        rng = np.random.default_rng(4000 + wid)
        alpha, beta, gamma = als_init(19, rng)

    alpha, beta, gamma, fit, history = grassmannian_optimize(
        alpha,
        beta,
        gamma,
        target_rank=10,
        outer_iters=cfg.get('outer_iters', 12),
        damping=cfg.get('damping', 0.2),
        verbose=False,
    )
    H = compute_H_fast(alpha, beta)
    H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))
    tail_energy = 0.0
    sigma_11 = 0.0
    if history:
        tail_energy = history[-1]['tail_energy']
        sigma_11 = history[-1]['sigma_11']
    return {
        'worker': wid,
        'init': cfg['init'],
        'fitness': float(fit),
        'H_rank': H_rank,
        'tail_energy': float(tail_energy),
        'sigma_11': float(sigma_11),
        'alpha': alpha.tolist(),
        'beta': beta.tolist(),
        'gamma': gamma.tolist(),
    }


def main():
    print('=' * 80)
    print('  Phase 2: Grassmannian Rank-Constrained Optimizer')
    print('  LP alternating on the fixed-V variety')
    print('=' * 80)

    workdir = Path(__file__).resolve().parent
    warm_path = ROOT / 'slp_turbo_best.json'
    if warm_path.exists():
        alpha, beta, gamma = load_factors(str(warm_path))
        print(f"\n  Warm start: {warm_path.name}  fitness={compute_fitness(alpha, beta, gamma):.8f}")
        alpha, beta, gamma, fit, history = grassmannian_optimize(
            alpha,
            beta,
            gamma,
            target_rank=10,
            outer_iters=15,
            damping=0.2,
            verbose=True,
        )
        H = compute_H_fast(alpha, beta)
        H_rank = int(np.linalg.matrix_rank(H, tol=1e-10))
        print(f"\n  Warm result: fitness={fit:.8f}  rank(H)={H_rank}")
        save_result(workdir / 'phase2_best.json', alpha, beta, gamma, fit, H_rank, history)

    configs = []
    for idx, (damping, sigma) in enumerate([
        (0.0, 0.0),
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.2, 0.01),
        (0.2, 0.03),
    ]):
        configs.append({
            'id': idx,
            'init': 'warm',
            'path': str(warm_path),
            'damping': damping,
            'sigma': sigma,
            'outer_iters': 10,
        })
    for idx in range(2):
        configs.append({
            'id': len(configs) + idx,
            'init': 'cold',
            'path': '',
            'damping': 0.2,
            'outer_iters': 8,
        })

    workers = min(8, os.cpu_count() or 4)
    print(f"\n  Sweep: {len(configs)} configs across {workers} workers")
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_run, json.dumps(cfg)): cfg['id'] for cfg in configs}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"    W{result['worker']:02d} ({result['init']:>4s}) "
                    f"fit={result['fitness']:.8f}  H_rank={result['H_rank']}  "
                    f"tail={result['tail_energy']:.3e}  sigma11={result['sigma_11']:.4f}"
                )
            except Exception as exc:
                print(f"    W{wid:02d} failed: {exc}")
    print(f"\n  Sweep completed in {time.time() - started:.1f}s")

    results.sort(key=lambda item: item['fitness'])
    if results:
        best = results[0]
        save_result(
            workdir / 'phase2_sweep_best.json',
            np.array(best['alpha']),
            np.array(best['beta']),
            np.array(best['gamma']),
            best['fitness'],
            best['H_rank'],
        )
        summary = [
            {
                'worker': item['worker'],
                'init': item['init'],
                'fitness': item['fitness'],
                'H_rank': item['H_rank'],
                'tail_energy': item['tail_energy'],
                'sigma_11': item['sigma_11'],
            }
            for item in results
        ]
        with open(workdir / 'phase2_results.json', 'w') as f:
            json.dump({'results': summary}, f, indent=2)


if __name__ == '__main__':
    main()
