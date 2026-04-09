"""
Saddle-point escape test + Hessian negative-curvature check.

Part 1: 24 parallel L-BFGS-B runs from perturbations of current best.
         σ values: 0.001, 0.01, 0.1 (8 workers each)
         → If any escapes to lower ε: saddle confirmed.

Part 2: Negative curvature check via Lanczos power iteration on the Hessian.
         Uses Hessian-vector products via double finite differences (cheap).
         → If min eigenvalue < -threshold: saddle. If all positive: local min.
"""
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import json, os

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

R, DIM = 22, 9

def obj(x):
    A = x[:R*DIM].reshape(R,DIM); B = x[R*DIM:2*R*DIM].reshape(R,DIM); C = x[2*R*DIM:].reshape(R,DIM)
    E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
    loss = 0.5*float(np.dot(E.ravel(),E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki',E,B,C); dB = np.einsum('ijl,ki,kl->kj',E,A,C); dC = np.einsum('ijl,ki,kj->kl',E,A,B)
    return loss, np.concatenate([dA.ravel(),dB.ravel(),dC.ravel()])

def frob(x):
    return np.sqrt(max(0.0, 2*obj(x)[0]))

def worker(args):
    worker_id, seed, sigma, x_base, q = args
    rng = np.random.default_rng(seed)
    best = frob(x_base)
    best_x = x_base.copy()
    for trial in range(20):
        noise = rng.standard_normal(len(x_base)) * sigma
        x0 = x_base + noise
        res = minimize(obj, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-11})
        f = np.sqrt(max(0.0, 2*res.fun))
        if f < best:
            best = f
            best_x = res.x.copy()
    q.put((worker_id, sigma, best, best_x.tolist() if best < frob(x_base) * 0.9 else None))

def hess_vec_product(x, v, h=1e-5):
    """Hessian-vector product via finite differences on gradient."""
    _, g1 = obj(x + h*v)
    _, g0 = obj(x - h*v)
    return (g1 - g0) / (2*h)

def min_eigenvalue_lanczos(x, n_iter=40):
    """Estimate minimum eigenvalue of Hessian via Lanczos iteration."""
    n = len(x)
    rng = np.random.default_rng(0)
    q = rng.standard_normal(n); q /= np.linalg.norm(q)
    Q = [q]
    alphas, betas = [], []
    for j in range(n_iter):
        z = hess_vec_product(x, Q[-1])
        alpha = float(np.dot(Q[-1], z))
        alphas.append(alpha)
        if j > 0:
            z -= betas[-1] * Q[-2]
        z -= alpha * Q[-1]
        beta = float(np.linalg.norm(z))
        if beta < 1e-12:
            break
        betas.append(beta)
        Q.append(z / beta)
        if (j+1) % 10 == 0:
            T_mat = np.diag(alphas) + np.diag(betas[:len(alphas)-1], 1) + np.diag(betas[:len(alphas)-1], -1)
            eigs = np.linalg.eigvalsh(T_mat)
            print(f'  Lanczos iter {j+1}: min_eig estimate = {eigs.min():.4e}  max_eig = {eigs.max():.4e}')
    b = betas[:len(alphas)-1]
    T_mat = np.diag(alphas) + np.diag(b, 1) + np.diag(b, -1)
    return np.linalg.eigvalsh(T_mat)


if __name__ == '__main__':
    with open('CANON_ADE_EPSILON/results/rank22_best.json') as f:
        d = json.load(f)
    x_best = np.concatenate([np.array(d['alpha']).ravel(), np.array(d['beta']).ravel(), np.array(d['gamma']).ravel()])
    eps0 = frob(x_best)
    print(f'Current best ε = {eps0:.8f}  (ε²={eps0**2:.2e})')

    # ── Part 2: Lanczos negative-curvature check ──────────────────────────────
    print('\n── Hessian min-eigenvalue via Lanczos (40 iterations) ──')
    eigs = min_eigenvalue_lanczos(x_best, n_iter=40)
    min_eig = eigs.min()
    max_eig = eigs.max()
    print(f'\nLanczos result: min_eig = {min_eig:.4e}  max_eig = {max_eig:.4e}')
    if min_eig < -1e-6:
        print('→ SADDLE POINT confirmed (negative curvature direction exists)')
    elif min_eig < 1e-6:
        print('→ NEAR-DEGENERATE (flat direction, numerical saddle)')
    else:
        print('→ LOCAL MINIMUM (all Lanczos eigenvalues positive)')

    # ── Part 1: Parallel perturbation escape ─────────────────────────────────
    print('\n── Parallel perturbation escape (24 workers, σ ∈ {0.001, 0.01, 0.1}) ──')
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    q = manager.Queue()
    rng0 = np.random.default_rng(77)
    sigmas = [0.001]*8 + [0.01]*8 + [0.1]*8
    seeds = [int(rng0.integers(0,2**31)) for _ in range(24)]
    args = [(i, seeds[i], sigmas[i], x_best, q) for i in range(24)]
    pool = mp.Pool(24)
    pool.map(worker, args)
    pool.close(); pool.join()

    results = []
    while not q.empty():
        results.append(q.get())

    results.sort(key=lambda x: x[2])
    print(f'\n{"worker":>6}  {"sigma":>8}  {"best_ε":>12}  {"vs_base":>10}  note')
    print('-'*55)
    best_overall = eps0
    best_x_overall = None
    for wid, sigma, bfrob, bx in results:
        ratio = bfrob / eps0
        note = '*** ESCAPED ***' if ratio < 0.9 else ''
        print(f'{wid:>6}  {sigma:>8.3f}  {bfrob:>12.8f}  {ratio:>10.4f}  {note}')
        if bfrob < best_overall:
            best_overall = bfrob
            if bx is not None:
                best_x_overall = np.array(bx)

    print(f'\nBest overall ε = {best_overall:.8f}  (was {eps0:.8f})')
    if best_overall < eps0 * 0.9:
        print('CONCLUSION: saddle point — perturbation escaped to lower basin')
        if best_x_overall is not None:
            A = best_x_overall[:R*DIM].reshape(R,DIM)
            B = best_x_overall[R*DIM:2*R*DIM].reshape(R,DIM)
            C = best_x_overall[2*R*DIM:].reshape(R,DIM)
            with open('CANON_ADE_EPSILON/results/rank22_best.json','w') as f:
                json.dump({'rank':R,'frobenius':best_overall,'frob_sq':best_overall**2,
                           'alpha':A.tolist(),'beta':B.tolist(),'gamma':C.tolist()},f)
            print('Saved new best.')
    else:
        print('CONCLUSION: perturbation could not escape — likely local minimum')
