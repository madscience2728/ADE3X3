"""ALS-based optimizer for ADE3x3 rank-19 CP decomposition.
Uses alternating least squares on factor matrices with L-infinity (minimax) refinement.
Seeded from current best candidate.
"""
import json, copy, time, sys, os
import numpy as np
from scipy.optimize import minimize

# ══════════════════════════════════════════════════════════════════════════════
# Target tensor
# ══════════════════════════════════════════════════════════════════════════════
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

T_flat = T.ravel()  # 729-element vector

# ══════════════════════════════════════════════════════════════════════════════
# Candidate ↔ dense factor matrices
# ══════════════════════════════════════════════════════════════════════════════
RANK = 19

def candidate_to_factors(data):
    """Convert sparse JSON candidate to dense (9, RANK) factor matrices A, B, G."""
    A = np.zeros((9, RANK))
    B = np.zeros((9, RANK))
    G = np.zeros((9, RANK))
    for k, term in enumerate(data['terms']):
        for i, v in zip(term['alpha_support'], term['alpha_values']):
            A[i, k] = v
        for i, v in zip(term['beta_support'], term['beta_values']):
            B[i, k] = v
        for i, v in zip(term['gamma_support'], term['gamma_values']):
            G[i, k] = v
    return A, B, G

def factors_to_candidate(A, B, G, template):
    """Convert dense factors back to sparse JSON candidate. 
    Preserves support structure from template but updates values."""
    data = copy.deepcopy(template)
    for k, term in enumerate(data['terms']):
        term['alpha_values'] = [float(A[i, k]) for i in term['alpha_support']]
        term['beta_values'] = [float(B[i, k]) for i in term['beta_support']]
        term['gamma_values'] = [float(G[i, k]) for i in term['gamma_support']]
    return data

def factors_to_candidate_full(A, B, G, thresh=1e-6):
    """Convert dense factors to candidate, discovering support from nonzero entries."""
    terms = []
    for k in range(A.shape[1]):
        a_nz = np.where(np.abs(A[:, k]) > thresh)[0]
        b_nz = np.where(np.abs(B[:, k]) > thresh)[0]
        g_nz = np.where(np.abs(G[:, k]) > thresh)[0]
        terms.append({
            'alpha_support': a_nz.tolist(),
            'alpha_values': A[a_nz, k].tolist(),
            'beta_support': b_nz.tolist(),
            'beta_values': B[b_nz, k].tolist(),
            'gamma_support': g_nz.tolist(),
            'gamma_values': G[g_nz, k].tolist(),
        })
    return {'terms': terms}

def reconstruct_from_factors(A, B, G):
    """Build 9x9x9 tensor: T_hat[c,a,b] = sum_k G[c,k] * A[a,k] * B[b,k]."""
    return np.einsum('ck,ak,bk->cab', G, A, B)

def evaluate_factors(A, B, G):
    R = reconstruct_from_factors(A, B, G) - T
    return np.max(np.abs(R)), np.linalg.norm(R)

# ══════════════════════════════════════════════════════════════════════════════
# ALS: one factor at a time, least-squares solve
# ══════════════════════════════════════════════════════════════════════════════
def als_step_A(A, B, G):
    """Fix B and G, solve for A in least-squares sense.
    T[c,a,b] = sum_k G[c,k] A[a,k] B[b,k]
    For each row a: T[:, a, :].ravel() = M @ A[a, :] where M = khatri_rao(G, B)
    """
    # Khatri-Rao product of G and B: (9*9, RANK)
    # M[c*9+b, k] = G[c,k] * B[b,k]
    M = np.einsum('ck,bk->cbk', G, B).reshape(-1, RANK)
    A_new = np.zeros_like(A)
    for a in range(9):
        rhs = T[:, a, :].ravel()  # shape (81,)
        # Solve M @ x = rhs
        A_new[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return A_new

def als_step_B(A, B, G):
    """Fix A and G, solve for B."""
    M = np.einsum('ck,ak->cak', G, A).reshape(-1, RANK)
    B_new = np.zeros_like(B)
    for b in range(9):
        rhs = T[:, :, b].ravel()
        B_new[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return B_new

def als_step_G(A, B, G):
    """Fix A and B, solve for G."""
    M = np.einsum('ak,bk->abk', A, B).reshape(-1, RANK)
    G_new = np.zeros_like(G)
    for c in range(9):
        rhs = T[c, :, :].ravel()
        G_new[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return G_new

def run_als(A, B, G, max_iters=100, tol=1e-10):
    """Run standard ALS until convergence."""
    best_mx, best_fro = evaluate_factors(A, B, G)
    print(f"  ALS start: max_abs={best_mx:.10f}  fro={best_fro:.8f}")
    
    for it in range(1, max_iters + 1):
        A = als_step_A(A, B, G)
        B = als_step_B(A, B, G)
        G = als_step_G(A, B, G)
        
        mx, fro = evaluate_factors(A, B, G)
        if it % 10 == 0 or mx < best_mx - 1e-6:
            print(f"  ALS iter {it:3d}: max_abs={mx:.10f}  fro={fro:.8f}")
        
        if abs(fro - best_fro) < tol:
            print(f"  ALS converged at iter {it}")
            break
        best_mx, best_fro = mx, fro
    
    return A, B, G

# ══════════════════════════════════════════════════════════════════════════════
# L-BFGS refinement: minimize max_abs directly (smooth approximation)
# ══════════════════════════════════════════════════════════════════════════════
def pack_factors(A, B, G):
    return np.concatenate([A.ravel(), B.ravel(), G.ravel()])

def unpack_factors(x):
    n = 9 * RANK
    A = x[:n].reshape(9, RANK)
    B = x[n:2*n].reshape(9, RANK)
    G = x[2*n:3*n].reshape(9, RANK)
    return A, B, G

def smooth_max_objective(x, beta=50.0):
    """Smooth approximation to max|R| using log-sum-exp."""
    A, B, G = unpack_factors(x)
    R = (reconstruct_from_factors(A, B, G) - T).ravel()
    absR = np.abs(R)
    # log-sum-exp approximation to max
    shifted = beta * (absR - np.max(absR))
    return np.max(absR) + np.log(np.sum(np.exp(shifted))) / beta

def fro_objective(x):
    """Frobenius norm squared of residual."""
    A, B, G = unpack_factors(x)
    R = reconstruct_from_factors(A, B, G) - T
    return 0.5 * np.sum(R**2)

def max_abs_objective(x):
    """True max abs for evaluation."""
    A, B, G = unpack_factors(x)
    R = reconstruct_from_factors(A, B, G) - T
    return np.max(np.abs(R))

def run_lbfgs(A, B, G, max_iters=500, beta=50.0):
    """L-BFGS optimization of smooth max objective."""
    x0 = pack_factors(A, B, G)
    mx0 = max_abs_objective(x0)
    print(f"  L-BFGS start: max_abs={mx0:.10f}")
    
    best_x = x0.copy()
    best_mx = mx0
    
    # First minimize Frobenius, then switch to smooth max
    print(f"  Phase 1: Frobenius minimization")
    res = minimize(fro_objective, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iters // 2, 'ftol': 1e-15, 'gtol': 1e-12})
    x1 = res.x
    mx1 = max_abs_objective(x1)
    print(f"  Phase 1 done: max_abs={mx1:.10f}  fro={np.sqrt(2*res.fun):.8f}")
    
    if mx1 < best_mx:
        best_x = x1.copy()
        best_mx = mx1
    
    # Phase 2: smooth max with increasing beta
    for b in [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        print(f"  Phase 2: smooth max beta={b}")
        res = minimize(lambda x: smooth_max_objective(x, beta=b), best_x,
                       method='L-BFGS-B',
                       options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        mx = max_abs_objective(res.x)
        print(f"    result: max_abs={mx:.10f}")
        if mx < best_mx:
            best_x = res.x.copy()
            best_mx = mx
    
    A, B, G = unpack_factors(best_x)
    return A, B, G

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════
def optimize_als(input_path, output_path, als_iters=200, lbfgs_iters=1000,
                 n_restarts=1):
    data = load_candidate(input_path)
    A0, B0, G0 = candidate_to_factors(data)
    mx0, fro0 = evaluate_factors(A0, B0, G0)
    print(f"\nLoaded: max_abs={mx0:.10f}  fro={fro0:.8f}")
    print(f"{'='*90}")

    best_A, best_B, best_G = A0.copy(), B0.copy(), G0.copy()
    best_mx = mx0

    for restart in range(n_restarts):
        print(f"\n--- Restart {restart+1}/{n_restarts} ---")
        
        if restart == 0:
            A, B, G = A0.copy(), B0.copy(), G0.copy()
        else:
            # Add small noise to escape basin
            noise = 0.01 * (1 + restart * 0.5)
            A = A0 + noise * np.random.randn(*A0.shape)
            B = B0 + noise * np.random.randn(*B0.shape)
            G = G0 + noise * np.random.randn(*G0.shape)

        # L-BFGS phase (no ALS — ALS minimizes Frobenius, not minimax)
        print(f"\n[L-BFGS phase]")
        A, B, G = run_lbfgs(A, B, G, max_iters=lbfgs_iters)
        mx, fro = evaluate_factors(A, B, G)
        print(f"  L-BFGS result: max_abs={mx:.10f}  fro={fro:.8f}")

        if mx < best_mx:
            best_A, best_B, best_G = A.copy(), B.copy(), G.copy()
            best_mx = mx

    # Save result
    result = factors_to_candidate_full(best_A, best_B, best_G, thresh=1e-6)
    final_mx, final_fro = evaluate_factors(best_A, best_B, best_G)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*90}")
    print(f"FINAL: max_abs={final_mx:.10f}  fro={final_fro:.8f}")
    print(f"  Δ from input: {final_mx - mx0:+.10f}")
    print(f"  saved to {output_path}")
    return final_mx

def load_candidate(path):
    with open(path) as f:
        return json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# Parallel multi-restart
# ══════════════════════════════════════════════════════════════════════════════
def _als_worker(args):
    wid, input_path, output_path, seed, als_iters, lbfgs_iters = args
    np.random.seed(seed)
    
    data = load_candidate(input_path)
    A0, B0, G0 = candidate_to_factors(data)
    
    if wid == 0:
        A, B, G = A0.copy(), B0.copy(), G0.copy()
    else:
        noise = 0.01 * (1 + wid * 0.5)
        A = A0 + noise * np.random.randn(*A0.shape)
        B = B0 + noise * np.random.randn(*B0.shape)
        G = G0 + noise * np.random.randn(*G0.shape)
    
    t0 = time.time()
    best_A, best_B, best_G = A.copy(), B.copy(), G.copy()
    best_mx = evaluate_factors(A, B, G)[0]
    
    # L-BFGS Frobenius first
    x0 = pack_factors(A, B, G)
    res = minimize(fro_objective, x0, method='L-BFGS-B',
                   options={'maxiter': lbfgs_iters, 'ftol': 1e-15, 'gtol': 1e-12})
    A, B, G = unpack_factors(res.x)
    mx = evaluate_factors(A, B, G)[0]
    if mx < best_mx:
        best_A, best_B, best_G = A.copy(), B.copy(), G.copy()
        best_mx = mx
    
    # L-BFGS smooth max with escalating beta
    for beta in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        x0 = pack_factors(A, B, G)
        res = minimize(lambda x, b=beta: smooth_max_objective(x, beta=b), x0,
                       method='L-BFGS-B',
                       options={'maxiter': lbfgs_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        A2, B2, G2 = unpack_factors(res.x)
        mx2 = evaluate_factors(A2, B2, G2)[0]
        if mx2 < best_mx:
            best_A, best_B, best_G = A2.copy(), B2.copy(), G2.copy()
            best_mx = mx2
        # Always continue from latest for next beta
        A, B, G = A2, B2, G2
    
    elapsed = time.time() - t0
    mx, fro = evaluate_factors(best_A, best_B, best_G)
    
    result = factors_to_candidate_full(best_A, best_B, best_G)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return (wid, seed, mx, fro, elapsed, output_path)

def run_parallel_als(input_path, final_output, n_workers=24,
                     als_iters=200, lbfgs_iters=500):
    import multiprocessing as mp
    os.makedirs('optim_workers', exist_ok=True)
    
    data = load_candidate(input_path)
    A, B, G = candidate_to_factors(data)
    mx0, fro0 = evaluate_factors(A, B, G)
    
    print(f"Baseline: max_abs={mx0:.10f}  fro={fro0:.8f}")
    print(f"Launching {n_workers} ALS+L-BFGS workers")
    print(f"{'='*90}")
    
    tasks = []
    for i in range(n_workers):
        seed = 5000 + i * 251
        out_path = f'optim_workers/als_worker_{i:02d}.json'
        tasks.append((i, input_path, out_path, seed, als_iters, lbfgs_iters))
    
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_als_worker, tasks)
    total = time.time() - t0
    
    results.sort(key=lambda x: x[2])
    print(f"\n{'='*90}")
    print(f"ALL WORKERS COMPLETE ({total:.1f}s)")
    print(f"{'='*90}")
    print(f"{'Worker':>7} {'Seed':>6} {'max_abs':>14} {'fro':>12} {'Time':>7}")
    print(f"{'─'*55}")
    for wid, seed, mx, fro, el, path in results:
        flag = " ★" if mx == results[0][2] else ""
        print(f"  W{wid:02d}   {seed:6d}  {mx:14.10f}  {fro:12.8f}  {el:6.1f}s{flag}")
    
    best = results[0]
    import shutil
    shutil.copy2(best[5], final_output)
    print(f"\nBEST: W{best[0]} max_abs={best[2]:.10f}  (Δ={best[2]-mx0:+.10f})")
    print(f"  saved to {final_output}")
    return best[2]

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='optimized_v2.json')
    parser.add_argument('--output', default='optimized_als.json')
    parser.add_argument('--als-iters', type=int, default=200)
    parser.add_argument('--lbfgs-iters', type=int, default=500)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=0)
    args = parser.parse_args()
    
    if args.parallel > 0:
        run_parallel_als(args.input, args.output,
                         n_workers=args.parallel,
                         als_iters=args.als_iters,
                         lbfgs_iters=args.lbfgs_iters)
    else:
        optimize_als(args.input, args.output,
                     als_iters=args.als_iters,
                     lbfgs_iters=args.lbfgs_iters,
                     n_restarts=args.restarts)
