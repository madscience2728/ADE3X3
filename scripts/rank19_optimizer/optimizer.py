"""
optimizer.py - Joint LP reproducing phase2_grassmannian exactly.

Per iteration:
  1. Compute V_perp (orthogonal complement of top-10 singular space of H).
     V_perp shape (18, 8)  rows of H must lie in the 10-dim V = span(V10),
     equivalently V_perp.T @ H_row = 0 for all rows.
  2. Joint LP over ALL alpha[k,a] simultaneously (num_vars = 9*R).
     - Column-major layout: x[a*R + k] = alpha[k, a]
     - Equality:  V_perp.T @ J_alpha_k @ alpha_k = 0  for each k  (8*R constraints)
     - Minimize:  max over all tensor entries of |T_hat - T|
     - No box bounds on factor vars (only t >= 0).
  3. Joint LP over ALL beta[k,b] similarly.
  4. LP per column of gamma (unconstrained, R variables per column).
  5. Damped update of V_perp.

NO normalization.  NO post-projection.  NO per-term LP.
This matches phase2_grassmannian.grassmannian_optimize() exactly.
"""

import numpy as np
from scipy.optimize import linprog

from tensor import build_target_tensor, build_fiber_coordinates, build_decomposition
from gates import compute_jacobians_beta, compute_jacobians_alpha


def _unflatten_by_columns(vec, R, cols=9):
    out = np.zeros((R, cols), dtype=np.float64)
    for j in range(cols):
        out[:, j] = vec[j * R:(j + 1) * R]
    return out


def update_V(H, target_rank=10, V_perp_old=None, damping=0.0):
    _, singular_values, vt = np.linalg.svd(H, full_matrices=True)
    d_perp = H.shape[1] - target_rank
    V_perp_new = vt[target_rank:, :].T
    if damping > 0.0 and V_perp_old is not None:
        blended = (1.0 - damping) * V_perp_new + damping * V_perp_old
        q, _ = np.linalg.qr(blended)
        V_perp = q[:, :d_perp]
    else:
        V_perp = V_perp_new
    tail = float(np.sum(singular_values[target_rank:] ** 2))
    return V_perp, singular_values, tail


def solve_alpha_full(alpha, beta, gamma, V_perp):
    T = build_target_tensor()
    R = beta.shape[0]
    d_perp = V_perp.shape[1]
    num_vars = 9 * R
    c_obj = np.zeros(num_vars + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    # Our convention: T_hat[i,j,m] = sum_k gamma[k,i]*alpha[k,j]*beta[k,m]
    # alpha is mode-1. block[i*9+m, k] = gamma[k,i]*beta[k,m]
    block = np.einsum('ki,km->imk', gamma.reshape(R,9), beta.reshape(R,9)).reshape(81, R)
    ones = np.ones((81, 1), dtype=np.float64)
    A_ub_parts, b_ub_parts = [], []
    for j in range(9):
        M = np.zeros((81, num_vars), dtype=np.float64)
        M[:, j * R:(j + 1) * R] = block
        target = T[:, j, :].ravel()
        A_ub_parts.append(np.hstack([M, -ones]))
        b_ub_parts.append(target)
        A_ub_parts.append(np.hstack([-M, -ones]))
        b_ub_parts.append(-target)
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)
    A_eq = np.zeros((R * d_perp, num_vars + 1), dtype=np.float64)
    b_eq = np.zeros(R * d_perp, dtype=np.float64)
    for k in range(R):
        Ck = V_perp.T @ compute_jacobians_alpha(beta.reshape(R,3,3)[k])[0]
        for j in range(d_perp):
            row = k * d_perp + j
            for a in range(9):
                A_eq[row, a * R + k] = Ck[j, a]
    bounds = [(None, None)] * num_vars + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': 120.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if not res.success:
        return None, res
    return _unflatten_by_columns(res.x[:num_vars], R), res


def solve_beta_full(alpha, beta, gamma, V_perp):
    T = build_target_tensor()
    R = alpha.shape[0]
    d_perp = V_perp.shape[1]
    num_vars = 9 * R
    c_obj = np.zeros(num_vars + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    # Our convention: T_hat[i,j,m] = sum_k gamma[k,i]*alpha[k,j]*beta[k,m]
    # beta is mode-2. block[i*9+j, k] = gamma[k,i]*alpha[k,j]
    block = np.einsum('ki,kj->ijk', gamma.reshape(R,9), alpha.reshape(R,9)).reshape(81, R)
    ones = np.ones((81, 1), dtype=np.float64)
    A_ub_parts, b_ub_parts = [], []
    for m in range(9):
        M = np.zeros((81, num_vars), dtype=np.float64)
        M[:, m * R:(m + 1) * R] = block
        target = T[:, :, m].ravel()
        A_ub_parts.append(np.hstack([M, -ones]))
        b_ub_parts.append(target)
        A_ub_parts.append(np.hstack([-M, -ones]))
        b_ub_parts.append(-target)
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)
    A_eq = np.zeros((R * d_perp, num_vars + 1), dtype=np.float64)
    b_eq = np.zeros(R * d_perp, dtype=np.float64)
    for k in range(R):
        Ck = V_perp.T @ compute_jacobians_beta(alpha.reshape(R,3,3)[k])[0]
        for j in range(d_perp):
            row = k * d_perp + j
            for b in range(9):
                A_eq[row, b * R + k] = Ck[j, b]
    bounds = [(None, None)] * num_vars + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': 120.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if not res.success:
        return None, res
    return _unflatten_by_columns(res.x[:num_vars], R), res


def _solve_gamma_col(c, alpha, beta):
    T = build_target_tensor()
    R = alpha.shape[0]
    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    # Our convention: gamma is mode-0. T_hat[c,j,m] = sum_k gamma[k,c]*alpha[k,j]*beta[k,m]
    M = np.einsum('kj,km->jmk', A, B).reshape(81, R)
    target = T[c, :, :].ravel()
    c_obj = np.zeros(R + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    ones = np.ones((81, 1), dtype=np.float64)
    A_ub = np.vstack([np.hstack([M, -ones]), np.hstack([-M, -ones])])
    b_ub = np.concatenate([target, -target])
    bounds = [(None, None)] * R + [(0.0, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 'time_limit': 10.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if res.success:
        return res.x[:R]
    return None


def solve_gamma_lp(alpha, beta, gamma):
    R = alpha.shape[0]
    G = gamma.reshape(R, 9).copy()
    for c in range(9):
        x = _solve_gamma_col(c, alpha.reshape(R,9), beta.reshape(R,9))
        if x is not None:
            G[:, c] = x
    return G


def solve_gamma_ls(alpha, beta):
    T = build_target_tensor()
    R = alpha.shape[0]
    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    ii, jj = np.mgrid[0:9, 0:9]
    ii = ii.ravel(); jj = jj.ravel()
    W = A[:, ii].T * B[:, jj].T
    T1 = T.reshape(9, 81)
    gram = W.T @ W
    Gamma = T1 @ W @ np.linalg.pinv(gram)
    return Gamma.T.reshape(R, 9)


def _diagnostics(alpha, beta, gamma, T):
    R = alpha.shape[0]
    af = alpha.reshape(R, 9)
    bf = beta.reshape(R, 9)
    gf = gamma.reshape(R, 9)
    coords = build_fiber_coordinates(af.reshape(R,3,3), bf.reshape(R,3,3))
    H = coords['H']; Delta = coords['Delta']
    Nuisance = coords['Nuisance']; Sigma = coords['Sigma']
    rank_H   = int(np.linalg.matrix_rank(H, tol=1e-10))
    rank_nuis = int(np.linalg.matrix_rank(Nuisance, tol=1e-10))
    aug_rank = int(np.linalg.matrix_rank(np.hstack([Sigma, Nuisance]), tol=1e-10))
    Mls, _, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
    delta_resid = float(np.linalg.norm(Delta - H @ Mls, 'fro'))
    gam_delta = float(np.linalg.norm(gf.T @ Delta, 'fro'))
    a3 = af.reshape(R,3,3); b3 = bf.reshape(R,3,3); g3 = gf.reshape(R,3,3)
    T_hat = build_decomposition(a3, b3, g3)
    fitness = float(np.max(np.abs(T - T_hat)))
    return {'fitness': fitness, 'rank_H': rank_H, 'rank_nuisance': rank_nuis,
            'delta_residual': delta_resid, 'gamma_delta': gam_delta,
            'augmented_rank': aug_rank, 'conservation': R + (18 - rank_H),
            'norm_beta': float(np.linalg.norm(bf))}


def _log(it, d, elapsed=None, tag=''):
    print(
        f"iter={it:04d}  fit={d['fitness']:.5f}  "
        f"rk(H)={d['rank_H']:2d}  rk(N)={d['rank_nuisance']:2d}  "
        f"delta={d['delta_residual']:.4f}  gamDelta={d['gamma_delta']:.4f}  "
        f"aug={d['augmented_rank']:2d}  C={d['conservation']}"
        f"{'ok' if d['conservation'] == 27 else '!27'}"
        f"  normB={d['norm_beta']:.3f}"
        + (f'  [{tag}]' if tag else '')
        + (f'  {elapsed:.1f}s' if elapsed is not None else '')
    )


def optimize(alpha_init, beta_init, gamma_init,
             max_iter=500, log_every=1, fitness_tol=1e-12,
             damping=0.2, verbose=True, **kwargs):
    """Joint-LP Grassmannian optimizer matching phase2_grassmannian."""
    import time
    T = build_target_tensor()
    R = alpha_init.shape[0]
    alpha = alpha_init.reshape(R, 9).copy()
    beta  = beta_init.reshape(R, 9).copy()
    gamma = gamma_init.reshape(R, 9).copy()
    best = (alpha.copy(), beta.copy(), gamma.copy(), float('inf'))
    history = []
    t0 = time.time()

    coords = build_fiber_coordinates(alpha.reshape(R,3,3), beta.reshape(R,3,3))
    H = coords['H']
    V_perp, sv, tail = update_V(H, target_rank=10)
    V_perp_old = V_perp.copy()

    for it in range(max_iter):
        alpha_new, alpha_res = solve_alpha_full(alpha, beta, gamma, V_perp)
        if alpha_new is None:
            if verbose:
                print(f"  [{it:3d}] alpha LP failed: {alpha_res.message}")
            break
        alpha = alpha_new

        beta_new, beta_res = solve_beta_full(alpha, beta, gamma, V_perp)
        if beta_new is None:
            if verbose:
                print(f"  [{it:3d}] beta LP failed: {beta_res.message}")
            break
        beta = beta_new

        gamma = solve_gamma_lp(alpha, beta, gamma)

        coords = build_fiber_coordinates(alpha.reshape(R,3,3), beta.reshape(R,3,3))
        H = coords['H']
        V_perp, sv, tail = update_V(H, target_rank=10, V_perp_old=V_perp_old, damping=damping)
        V_perp_old = V_perp.copy()

        if it % log_every == 0:
            a3 = alpha.reshape(R,3,3); b3 = beta.reshape(R,3,3); g3 = gamma.reshape(R,3,3)
            d = _diagnostics(a3, b3, g3, T)
            d['alpha_lp_obj'] = float(alpha_res.fun)
            d['beta_lp_obj']  = float(beta_res.fun)
            history.append({'iter': it, **{k: (v.tolist() if hasattr(v,'tolist') else v)
                                            for k, v in d.items()}})
            if verbose:
                _log(it, d, elapsed=time.time()-t0,
                     tag=f'aLP={alpha_res.fun:.4f} bLP={beta_res.fun:.4f}')
            if d['fitness'] < best[3]:
                best = (alpha.copy(), beta.copy(), gamma.copy(), d['fitness'])
            if d['fitness'] < fitness_tol and d['conservation'] == 27:
                if verbose:
                    print(f'\n*** SUCCESS at iter {it}, fit={d["fitness"]:.4e} ***')
                a3 = alpha.reshape(R,3,3); b3 = beta.reshape(R,3,3); g3 = gamma.reshape(R,3,3)
                return {'alpha': a3, 'beta': b3, 'gamma': g3,
                        'diagnostics': d, 'history': history,
                        'success': True, 'iterations': it}
            if it >= 2:
                recent = [h['fitness'] for h in history[-3:]]
                if max(recent) - min(recent) < 1e-10 and tail < 1e-20:
                    if verbose:
                        print(f'  Converged at iter {it}')
                    break

    ba, bb, bg, _ = best
    a3 = ba.reshape(R,3,3); b3 = bb.reshape(R,3,3); g3 = bg.reshape(R,3,3)
    d_final = _diagnostics(a3, b3, g3, T)
    if verbose:
        _log(max_iter, d_final, tag='best')
    return {'alpha': a3, 'beta': b3, 'gamma': g3,
            'diagnostics': d_final, 'history': history,
            'success': False, 'iterations': max_iter}
