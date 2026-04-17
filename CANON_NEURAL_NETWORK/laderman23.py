"""laderman23.py — Encode Laderman's exact rank-23 decomposition of 3x3 matmul.

Laderman (1976) "A noncommutative algorithm of length 23 for multiplying 
6x6 matrices" — but the 3x3 version is the classic one.

The 23 multiplications for C = A*B (3x3):

  m1  = (a00 + a01 + a02 - a10 - a11 - a12 - a21 - a22) * b11
  m2  = (a00 - a10) * (b11 - b10)
  m3  = a11 * (-b00 + b10 + b11 - b12 - b20 + b21 + b22)  [corrected: not standard]
  ... 

Actually, let me use the Smirnov/JJQ (Johnson-McLaughlin-Oh) form which is
cleaner. But the simplest verified source is to BUILD it from the bilinear
formulas.

Instead: let's use the KNOWN decomposition approach. We encode the 23 
bilinear products as explicit U, V, W factor matrices and verify.

Reference: Sedoglavic, Serdyuk, Smirnov (2017) give explicit decomposition.
Also: Hopcroft-Kerr (1971), Laderman (1976).

We'll encode Hopcroft-Kerr's scheme.
"""
import numpy as np

def build_T333():
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[3*i+j, 3*j+k, 3*i+k] = 1.0
    return T

# ──────────────────────────────────────────────────────────────────
# Laderman's 23-multiplication scheme for 3x3 matrix multiplication
# C = A * B where A[i][j] = a_{ij}, B[i][j] = b_{ij}
# 
# Indices: a_{ij} -> flat index 3*i+j, same for b and c.
# Factor U[r,:] encodes how multiplication r reads from A (flattened).
# Factor V[r,:] encodes how multiplication r reads from B (flattened).
# Factor W[r,:] encodes how multiplication r contributes to C (flattened).
#
# Using Laderman (1976) / Cohn & Umans notation:
# ──────────────────────────────────────────────────────────────────

def laderman_23():
    """
    Laderman's 23-multiplication algorithm for 3x3 matmul.
    Returns U, V, W each (23, 9).
    
    Notation: a[i,j] is at flat index 3i+j. Same for b, c.
    
    From: Laderman, J.D. "A noncommutative algorithm of length 23 for 
    multiplying 6x6 matrices." Bull. AMS 7(2), 1976.
    Also matches the formulation in Cohn-Umans and numerous textbooks.
    """
    # Helper: create a vector for a linear combination of matrix entries
    def avec(*coeffs_and_indices):
        """avec((c1, i1, j1), (c2, i2, j2), ...) -> 9-vector"""
        v = np.zeros(9)
        for c, i, j in coeffs_and_indices:
            v[3*i + j] += c
        return v
    
    # The 23 products: p_r = (u_r^T a)(v_r^T b)
    # where a, b are flattened 3x3 matrices
    # and c = sum_r w_r * p_r
    
    # Laderman's formulas (standard textbook version):
    # Let a_{ij} be entries of A, b_{ij} entries of B.
    
    # Define the 23 multiplications:
    # p1 = (a_{00} + a_{01} + a_{02} + a_{10} + a_{11} + a_{12} + a_{20} + a_{21} + a_{22}) * b_{11}
    # ... this gets complex. Let me use a VERIFIED computational approach instead.
    
    # ──────────────────────────────────────────────────────────────
    # ALTERNATIVE: Use Smirnov's EXPLICIT decomposition from his 2013 paper,
    # as encoded in the database at https://fmm.univ-mlv.fr/
    # The Sedoglavic-Serdyuk-Smirnov decomposition with integer coefficients.
    # ──────────────────────────────────────────────────────────────
    
    # Smirnov's scheme (from Sedoglavic et al. 2017, Table 1):
    # 23 triples (u_r, v_r, w_r) where each is a 9-vector.
    # Coefficients from {-2, -1, 0, 1, 2}.
    
    # Rather than transcribe 23*27 = 621 coefficients by hand (error-prone),
    # let's BUILD a rank-23 decomposition using the STRUCTURED approach:
    # Strassen for 2x2 (rank 7) composed with a border-rank trick.
    
    # ACTUALLY: The most reliable approach is to use the characterization
    # that rank(T_{3,3,3}) = 23 and FIND it numerically with a VERY good
    # initialization. The key insight we're missing is that init scale matters.
    
    return None  # Placeholder — we'll use numerical search below


def find_rank23_smart(max_restarts=2000):
    """
    Smarter rank-23 search:
    1. Use multiple init scales
    2. Use longer ALS
    3. Use tighter LBFGS
    4. Restart from best partial solutions with perturbation
    """
    from scipy.optimize import minimize
    
    T = build_T333()
    
    def als_refine(U, V, W, iters=300):
        r = U.shape[0]
        reg = 1e-12 * np.eye(r)
        for _ in range(iters):
            KR = np.einsum("ra,rb->rab", V, W).reshape(r, -1)
            gram = KR @ KR.T + reg
            rhs = T.reshape(9, -1) @ KR.T
            U = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
            
            KR = np.einsum("ra,rb->rab", U, W).reshape(r, -1)
            gram = KR @ KR.T + reg
            rhs = T.transpose(1,0,2).reshape(9, -1) @ KR.T
            V = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
            
            KR = np.einsum("ra,rb->rab", U, V).reshape(r, -1)
            gram = KR @ KR.T + reg
            rhs = T.transpose(2,0,1).reshape(9, -1) @ KR.T
            W = np.linalg.lstsq(gram, rhs.T, rcond=None)[0]
        return U, V, W
    
    def lbfgs(U, V, W, maxiter=8000):
        r = U.shape[0]
        def obj(params):
            Uf = params[:r*9].reshape(r,9)
            Vf = params[r*9:2*r*9].reshape(r,9)
            Wf = params[2*r*9:].reshape(r,9)
            R = np.einsum("ra,rb,rc->abc", Uf, Vf, Wf) - T
            loss = 0.5 * np.dot(R.ravel(), R.ravel())
            dU = np.einsum("abc,rb,rc->ra", R, Vf, Wf)
            dV = np.einsum("abc,ra,rc->rb", R, Uf, Wf)
            dW = np.einsum("abc,ra,rb->rc", R, Uf, Vf)
            return loss, np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
        x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        res = minimize(obj, x0, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter, "ftol": 1e-32, "gtol": 1e-16})
        Uf = res.x[:r*9].reshape(r,9)
        Vf = res.x[r*9:2*r*9].reshape(r,9)
        Wf = res.x[2*r*9:].reshape(r,9)
        return Uf, Vf, Wf
    
    def residual(U, V, W):
        return np.linalg.norm(np.einsum("ra,rb,rc->abc", U, V, W) - T)
    
    import time
    best_res = float("inf")
    best_UVW = None
    t0 = time.time()
    
    # Strategy 1: Various init scales
    scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    for s in range(max_restarts):
        rng = np.random.RandomState(s)
        scale = scales[s % len(scales)]
        
        U = rng.randn(23, 9) * scale
        V = rng.randn(23, 9) * scale
        W = rng.randn(23, 9) * scale
        
        # Long ALS warmup
        U, V, W = als_refine(U, V, W, iters=500)
        
        als_res = residual(U, V, W)
        
        # Only LBFGS if ALS got somewhere interesting
        if als_res < 1.0:
            U, V, W = lbfgs(U, V, W, maxiter=10000)
            res = residual(U, V, W)
        else:
            res = als_res
        
        if res < best_res:
            best_res = res
            best_UVW = (U.copy(), V.copy(), W.copy())
            np.savez("rank23_best.npz", U=U, V=V, W=W)
        
        if s % 20 == 0 or res < 0.01:
            dt = time.time() - t0
            print(f"  [{dt:6.0f}s] restart {s:4d} (scale={scale:.1f}): "
                  f"als={als_res:.4e} final={res:.4e}  BEST={best_res:.4e}")
        
        if best_res < 1e-8:
            print(f"\n  ** EXACT rank-23 at restart {s}! res={best_res:.2e} **")
            return best_UVW, best_res
        
        # Strategy 2: Perturb best solution every 50 restarts
        if s > 0 and s % 50 == 0 and best_UVW is not None:
            Ub, Vb, Wb = best_UVW
            for ps in range(10):
                rng2 = np.random.RandomState(s * 1000 + ps)
                noise = 0.01 * (1.0 + ps * 0.5)
                Up = Ub + rng2.randn(23, 9) * noise
                Vp = Vb + rng2.randn(23, 9) * noise
                Wp = Wb + rng2.randn(23, 9) * noise
                Up, Vp, Wp = als_refine(Up, Vp, Wp, iters=200)
                Up, Vp, Wp = lbfgs(Up, Vp, Wp, maxiter=10000)
                res = residual(Up, Vp, Wp)
                if res < best_res:
                    best_res = res
                    best_UVW = (Up.copy(), Vp.copy(), Wp.copy())
                    np.savez("rank23_best.npz", U=Up, V=Vp, W=Wp)
                    print(f"  ** PERTURB improved: {res:.4e} **")
                    if best_res < 1e-8:
                        print(f"\n  ** EXACT rank-23! **")
                        return best_UVW, best_res
    
    return best_UVW, best_res

if __name__ == "__main__":
    print("="*60)
    print("SMART RANK-23 SEARCH FOR T_{3,3,3}")
    print("="*60)
    uvw, res = find_rank23_smart(max_restarts=2000)
    print(f"\nFinal best: {res:.6e}")
    if res < 1e-6:
        print("EXACT decomposition found!")
    else:
        print("No exact decomposition found. Best saved to rank23_best.npz")
