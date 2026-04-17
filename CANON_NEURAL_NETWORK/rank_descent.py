"""rank_descent.py — Find exact rank-23 via massive ALS+LBFGS, then descend.

Phase 1: Hammer rank-23 with 500+ restarts until residual < 1e-8.
Phase 2: From exact rank-23, try term deletion + full re-optimization to rank-22.
Phase 3: Continue descent toward rank-20.

Key fix from failed term_deletion.py: the naive rank-27 has orthogonal terms
that can't rearrange. We need terms with overlapping support first.
"""
import numpy as np
from scipy.optimize import minimize
import time, sys

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
print(f"T333 norm = {T_NORM:.6f}, nnz = {int(T333.sum())}")

def reconstruct(U, V, W):
    return np.einsum("ra,rb,rc->abc", U, V, W)

def residual(U, V, W):
    return np.linalg.norm(reconstruct(U, V, W) - T333)

# ── ALS step ──
def als_refine(U, V, W, T, iters=100):
    r = U.shape[0]
    d1, d2, d3 = T.shape
    reg = 1e-10 * np.eye(r)
    for it in range(iters):
        # Update U
        KR = np.einsum("ra,rb->rab", V, W).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T.reshape(d1, -1) @ KR.T
        U = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
        # Update V
        KR = np.einsum("ra,rb->rab", U, W).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T.transpose(1,0,2).reshape(d2, -1) @ KR.T
        V = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
        # Update W
        KR = np.einsum("ra,rb->rab", U, V).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T.transpose(2,0,1).reshape(d3, -1) @ KR.T
        W = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
    return U, V, W

# ── L-BFGS-B refinement ──
def lbfgsb_refine(U, V, W, T, maxiter=3000):
    r = U.shape[0]
    d1, d2, d3 = T.shape
    def objective(params):
        Uf = params[:r*d1].reshape(r, d1)
        Vf = params[r*d1:r*(d1+d2)].reshape(r, d2)
        Wf = params[r*(d1+d2):].reshape(r, d3)
        R = np.einsum("ra,rb,rc->abc", Uf, Vf, Wf) - T
        loss = 0.5 * np.dot(R.ravel(), R.ravel())
        dU = np.einsum("abc,rb,rc->ra", R, Vf, Wf)
        dV = np.einsum("abc,ra,rc->rb", R, Uf, Wf)
        dW = np.einsum("abc,ra,rb->rc", R, Uf, Vf)
        return loss, np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
    
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    res = minimize(objective, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter, "ftol": 1e-30, "gtol": 1e-14})
    Uf = res.x[:r*d1].reshape(r, d1)
    Vf = res.x[r*d1:r*(d1+d2)].reshape(r, d2)
    Wf = res.x[r*(d1+d2):].reshape(r, d3)
    return Uf, Vf, Wf

def one_restart(rank, seed, als_iters=200, lbfgs_maxiter=5000):
    """Single ALS + LBFGS restart."""
    rng = np.random.RandomState(seed)
    # Gaussian init, moderate scale
    U = rng.randn(rank, 9) * 0.3
    V = rng.randn(rank, 9) * 0.3
    W = rng.randn(rank, 9) * 0.3
    # ALS warmup
    U, V, W = als_refine(U, V, W, T333, iters=als_iters)
    # LBFGS polish
    U, V, W = lbfgsb_refine(U, V, W, T333, maxiter=lbfgs_maxiter)
    return U, V, W, residual(U, V, W)

# ── Phase 1: Find exact rank-23 ──
def phase1_find_rank23(max_restarts=500, target_res=1e-6):
    print("\n" + "="*60)
    print("PHASE 1: Finding exact rank-23 decomposition")
    print("="*60)
    
    best_res = float("inf")
    best_UVW = None
    t0 = time.time()
    
    for s in range(max_restarts):
        U, V, W, res = one_restart(23, seed=s)
        if res < best_res:
            best_res = res
            best_UVW = (U.copy(), V.copy(), W.copy())
            np.savez("rank23_best.npz", U=U, V=V, W=W, residual=np.array([res]))
        
        if s % 10 == 0 or res < 0.1:
            dt = time.time() - t0
            print(f"  restart {s:4d}: res={res:.6e}  best={best_res:.6e}  [{dt:.0f}s]")
        
        if best_res < target_res:
            print(f"\n  ** EXACT rank-23 found at restart {s}! res={best_res:.6e} **")
            break
    
    return best_UVW, best_res

# ── Phase 2: Term deletion descent ──
def phase2_descend(U, V, W, target_rank=20, n_random_extra=20):
    print("\n" + "="*60)
    print(f"PHASE 2: Descending from rank {U.shape[0]} to {target_rank}")
    print("="*60)
    
    results = {}
    current_rank = U.shape[0]
    results[current_rank] = residual(U, V, W)
    
    while current_rank > target_rank:
        next_rank = current_rank - 1
        print(f"\n--- rank {current_rank} -> {next_rank} ---")
        
        best_res = float("inf")
        best_UVW = None
        
        # Try all term deletions
        for i in range(current_rank):
            mask = [j for j in range(current_rank) if j != i]
            Ud, Vd, Wd = U[mask].copy(), V[mask].copy(), W[mask].copy()
            # Add small noise to break symmetry
            rng = np.random.RandomState(i + 1000)
            Ud += rng.randn(*Ud.shape) * 0.01
            Vd += rng.randn(*Vd.shape) * 0.01
            Wd += rng.randn(*Wd.shape) * 0.01
            # Refine
            Ud, Vd, Wd = als_refine(Ud, Vd, Wd, T333, iters=100)
            Ud, Vd, Wd = lbfgsb_refine(Ud, Vd, Wd, T333, maxiter=3000)
            res = residual(Ud, Vd, Wd)
            if res < best_res:
                best_res = res
                best_UVW = (Ud.copy(), Vd.copy(), Wd.copy())
            if i % 5 == 0:
                print(f"  del {i:2d}: res={res:.6e}  (best so far: {best_res:.6e})")
        
        # Also try random restarts at this rank
        print(f"  + {n_random_extra} random restarts...")
        for s in range(n_random_extra):
            Ur, Vr, Wr, res = one_restart(next_rank, seed=s + 5000 + next_rank * 100,
                                            als_iters=150, lbfgs_maxiter=3000)
            if res < best_res:
                best_res = res
                best_UVW = (Ur.copy(), Vr.copy(), Wr.copy())
        
        U, V, W = best_UVW
        results[next_rank] = best_res
        current_rank = next_rank
        
        tag = "EXACT!" if best_res < 1e-6 else ""
        print(f"  >> rank {next_rank}: best residual = {best_res:.6e}  {tag}")
        
        np.savez(f"rank_descent_r{next_rank}.npz", U=U, V=V, W=W,
                 residual=np.array([best_res]))
    
    return results

def main():
    print("="*60)
    print("RANK DESCENT: 3x3 matmul tensor decomposition")
    print("="*60)
    
    # Phase 1
    uvw, res23 = phase1_find_rank23(max_restarts=500, target_res=1e-6)
    U, V, W = uvw
    
    if res23 > 1e-6:
        print(f"\nWARNING: Best rank-23 residual = {res23:.6e} (not exact)")
        print("Proceeding with best available...")
    
    # Phase 2 
    results = phase2_descend(U, V, W, target_rank=20, n_random_extra=30)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in sorted(results.keys(), reverse=True):
        tag = "EXACT" if results[r] < 1e-6 else "WALL" if results[r] > 0.5 else "close"
        print(f"  rank {r:2d}: {results[r]:.6e}  [{tag}]")

if __name__ == "__main__":
    main()
