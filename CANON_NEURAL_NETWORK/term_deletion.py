"""term_deletion.py — Delete one term from known rank-23, refine to rank-22, ..., rank-20.

Strategy:
  1. Start with the naive rank-27 decomposition of 3x3 matmul (exact, trivial).
  2. For each deletion choice, remove one rank-1 term, refine remaining 26 via L-BFGS-B.
  3. Iterate: take best rank-r solution, try all r deletions, refine to rank-(r-1).
  4. Track the residual cliff as we descend.

The naive decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
  Term (i,j,k): U = e_{3i+j}, V = e_{3j+k}, W = e_{3i+k}
  27 terms total (one per (i,j,k) triple).
"""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# ── Build T_{3,3,3} ──
def build_T333():
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[3*i+j, 3*j+k, 3*i+k] = 1.0
    return T

T333 = build_T333()
T_NORM = np.linalg.norm(T333)

# ── Naive rank-27 decomposition ──
def naive_rank27():
    """Returns (U, V, W) each shape (27, 9) for the trivial decomposition."""
    U = np.zeros((27, 9))
    V = np.zeros((27, 9))
    W = np.zeros((27, 9))
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[idx, 3*i+j] = 1.0
                V[idx, 3*j+k] = 1.0
                W[idx, 3*i+k] = 1.0
                idx += 1
    return U, V, W

def reconstruct(U, V, W):
    return np.einsum("ra,rb,rc->abc", U, V, W)

def residual(U, V, W):
    return np.linalg.norm(reconstruct(U, V, W) - T333)

# ── L-BFGS-B refinement ──
def refine(U, V, W, maxiter=2000):
    """Refine (U, V, W) via L-BFGS-B to minimize ||UVW - T||^2."""
    r = U.shape[0]
    def objective(params):
        Uf = params[:r*9].reshape(r, 9)
        Vf = params[r*9:2*r*9].reshape(r, 9)
        Wf = params[2*r*9:].reshape(r, 9)
        R = reconstruct(Uf, Vf, Wf) - T333
        loss = 0.5 * np.dot(R.ravel(), R.ravel())
        dU = np.einsum("abc,rb,rc->ra", R, Vf, Wf)
        dV = np.einsum("abc,ra,rc->rb", R, Uf, Wf)
        dW = np.einsum("abc,ra,rb->rc", R, Uf, Vf)
        grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
        return loss, grad
    
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    res = minimize(objective, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter, "ftol": 1e-30, "gtol": 1e-12})
    Uf = res.x[:r*9].reshape(r, 9)
    Vf = res.x[r*9:2*r*9].reshape(r, 9)
    Wf = res.x[2*r*9:].reshape(r, 9)
    return Uf, Vf, Wf, res.fun

# ── ALS refinement ──
def als_refine(U, V, W, iters=100):
    """ALS refinement before L-BFGS-B."""
    r = U.shape[0]
    reg = 1e-10 * np.eye(r)
    for _ in range(iters):
        # Update U
        KR = np.einsum("ra,rb->rab", V, W).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T333.reshape(9, -1) @ KR.T
        U = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
        # Update V
        KR = np.einsum("ra,rb->rab", U, W).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T333.transpose(1,0,2).reshape(9, -1) @ KR.T
        V = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
        # Update W
        KR = np.einsum("ra,rb->rab", U, V).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T333.transpose(2,0,1).reshape(9, -1) @ KR.T
        W = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
    return U, V, W

# ── Term deletion + refinement ──
def delete_and_refine(U, V, W, n_keep_best=5):
    """Try deleting each term, refine, return sorted results."""
    r = U.shape[0]
    results = []
    for i in range(r):
        mask = [j for j in range(r) if j != i]
        Ud = U[mask].copy()
        Vd = V[mask].copy()
        Wd = W[mask].copy()
        # ALS first, then L-BFGS-B
        Ud, Vd, Wd = als_refine(Ud, Vd, Wd, iters=50)
        Ud, Vd, Wd, loss = refine(Ud, Vd, Wd, maxiter=1000)
        res = residual(Ud, Vd, Wd)
        results.append((res, i, Ud, Vd, Wd))
        print(f"  Delete term {i:2d}: residual = {res:.6e}")
    results.sort(key=lambda x: x[0])
    return results

# ── Multi-restart refinement at target rank ──
def multi_restart_refine(rank, n_restarts=20, maxiter=3000):
    """Random-init ALS + L-BFGS-B at a given rank."""
    best_res = float("inf")
    best_UVW = None
    for s in range(n_restarts):
        rng = np.random.RandomState(s)
        U = rng.randn(rank, 9) * 0.5
        V = rng.randn(rank, 9) * 0.5
        W = rng.randn(rank, 9) * 0.5
        U, V, W = als_refine(U, V, W, iters=100)
        U, V, W, loss = refine(U, V, W, maxiter=maxiter)
        res = residual(U, V, W)
        if res < best_res:
            best_res = res
            best_UVW = (U.copy(), V.copy(), W.copy())
        if s % 5 == 0:
            print(f"  restart {s:2d}: res={res:.6e}  best={best_res:.6e}")
    return best_UVW, best_res

def main():
    print("=" * 60)
    print("TERM DELETION DESCENT: rank 27 -> 20")
    print("=" * 60)
    
    # Verify naive rank-27
    U, V, W = naive_rank27()
    print(f"\nRank 27 (naive): residual = {residual(U, V, W):.6e}")
    assert residual(U, V, W) < 1e-12, "Naive decomposition broken!"
    
    # Also try random restarts at key ranks for comparison
    print("\n--- Random restart baselines ---")
    for r in [23, 22, 21, 20]:
        _, best = multi_restart_refine(r, n_restarts=10, maxiter=2000)
        print(f"  Rank {r} random-restart best: {best:.6e}")
    
    # Term deletion descent
    current_U, current_V, current_W = U, V, W
    current_rank = 27
    
    results_by_rank = {27: 0.0}
    
    for target_rank in range(26, 19, -1):
        print(f"\n{'='*60}")
        print(f"DESCENDING: rank {current_rank} -> {target_rank}")
        print(f"{'='*60}")
        
        t0 = time.time()
        results = delete_and_refine(current_U, current_V, current_W, n_keep_best=3)
        dt = time.time() - t0
        
        best_res, best_idx, best_U, best_V, best_W = results[0]
        results_by_rank[target_rank] = best_res
        
        print(f"\nBest deletion: term {best_idx}, residual = {best_res:.6e}  ({dt:.1f}s)")
        print(f"Top 3: {[('term %d: %.4e' % (r[1], r[0])) for r in results[:3]]}")
        
        # Use best as starting point for next descent
        current_U, current_V, current_W = best_U, best_V, best_W
        current_rank = target_rank
        
        if best_res < 1e-6:
            print(f"  >> EXACT at rank {target_rank}!")
        
        # Save checkpoint
        np.savez(f"term_deletion_r{target_rank}.npz",
                 U=best_U, V=best_V, W=best_W,
                 residual=np.array([best_res]),
                 deleted_term=np.array([best_idx]))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Residual by rank")
    print("=" * 60)
    for r in sorted(results_by_rank.keys(), reverse=True):
        tag = "EXACT" if results_by_rank[r] < 1e-6 else ""
        print(f"  rank {r:2d}: {results_by_rank[r]:.6e}  {tag}")

if __name__ == "__main__":
    main()
