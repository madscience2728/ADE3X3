"""
Real-valued CP decomposition optimizer for 3x3 matrix multiplication tensor.

Goal: find rank-19 decomposition T_matmul = sum_{r=1}^{19} u_r ⊗ v_r ⊗ w_r
over R (real-valued) such that reconstruction error < float64 machine epsilon.

The 3x3 matmul tensor T has T[m,p,q] = 1 iff there exist i,j,k in {0,1,2}
such that m = C[i,k], p = A[i,j], q = B[j,k] indices (= row-major flattening).

CP decomposition: T[m,p,q] = sum_r U[r,m] * V[r,p] * W[r,q]
  U: gamma (output coefficients), V: alpha (A-factor), W: beta (B-factor)

We initialize from the Phase-4 best integer packet (Gates 1+2 satisfied)
and optimize U,V,W jointly using L-BFGS-B via scipy.

Strategy:
  - Fix V,W → solve U by least squares (closed form, no gradient needed)
  - Optimize V,W jointly by L-BFGS-B minimizing ||T - CP(U_lsq, V, W)||_F^2
  - Alternate ALS passes with joint L-BFGS-B refinement
  - Track best reconstruction across all restarts
"""

import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "CANON_DATABASE" / "data"
CKPT = ROOT / "CANON_DATABASE" / "swap_checkpoint.json"   # Phase-4 best integer seed

R = 19   # target rank

# ── build the 3x3 matmul tensor ───────────────────────────────────────────────

def build_matmul_tensor():
    """T[m,p,q] = 1 iff C[i,k] = sum_j A[i,j]*B[j,k] contributes entry m from p,q."""
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                m = i * 3 + k   # C index
                p = i * 3 + j   # A index
                q = j * 3 + k   # B index
                T[m, p, q] += 1.0
    return T

T_TARGET = build_matmul_tensor()   # (9, 9, 9)
T_NORM_SQ = float(np.sum(T_TARGET ** 2))
# Mode-0 unfolding: shape (9, 81)
T0 = T_TARGET.reshape(9, 81)

def khatri_rao(V, W):
    """Row-wise Khatri-Rao product: KR[r, p*Q+q] = V[r,p]*W[r,q]. Shape (R, P*Q)."""
    Rr, P = V.shape
    _,  Q = W.shape
    return (V[:, :, None] * W[:, None, :]).reshape(Rr, P * Q)

def cp_reconstruct(U, V, W):
    """Reconstruct T from factor matrices U(R,9), V(R,9), W(R,9)."""
    # T[m,p,q] = sum_r U[r,m]*V[r,p]*W[r,q]
    KR = khatri_rao(V, W)          # (R, 81)
    T0_hat = U.T @ KR              # (9, 81)
    return T0_hat.reshape(9, 9, 9)

def solve_u(V, W):
    """Solve U = argmin ||T0 - U.T @ KR(V,W)||_F by least squares."""
    KR = khatri_rao(V, W)          # (R, 81)
    # T0 = U.T @ KR  =>  T0.T = KR.T @ U  =>  shape (81,9) = (81,R) @ (R,9)
    U_T, _, _, _ = lstsq(KR.T, T0.T)   # (R,9)
    return U_T   # (R,9)

def recon_error(U, V, W):
    T_hat = cp_reconstruct(U, V, W)
    diff  = T_hat - T_TARGET
    return float(np.sum(diff ** 2))

# ── load integer seed from Phase-4 best ──────────────────────────────────────

def load_integer_seed():
    """Load the Phase-4 best 19-term integer packet and extract V, W factors."""
    tpl      = np.load(DB / "templates.npy")           # (19683, 3, 3)
    ai       = np.load(DB / "alpha_idx.npy", mmap_mode="r")
    bi       = np.load(DB / "beta_idx.npy",  mmap_mode="r")

    ckpt = json.loads(CKPT.read_text())
    idx  = np.array(ckpt["global_best_indices"], dtype=np.int64)

    alpha = tpl[ai[idx]].astype(np.float64).reshape(R, 9)   # (19, 9)
    beta  = tpl[bi[idx]].astype(np.float64).reshape(R, 9)   # (19, 9)
    return alpha, beta

# ── objective function for scipy (optimize V, W jointly; U solved analytically)

def pack(V, W):
    return np.concatenate([V.ravel(), W.ravel()])

def unpack(x):
    V = x[:R*9].reshape(R, 9)
    W = x[R*9:].reshape(R, 9)
    return V, W

def objective_and_grad(x):
    V, W = unpack(x)
    KR  = khatri_rao(V, W)             # (R, 81)
    U_T, _, _, _ = lstsq(KR.T, T0.T)  # (R, 9)  — U.T solved
    U   = U_T                          # U[r,m]

    T0_hat = U.T @ KR                  # (9, 81)
    Res    = T0_hat - T0               # (9, 81)
    err    = float(np.sum(Res ** 2))

    # Gradients dL/dV and dL/dW
    # L = ||U.T @ KR - T0||^2
    # dL/dKR = 2 * U @ Res           shape (R, 81)
    dL_dKR = 2.0 * U @ Res            # (R, 81)

    # KR[r, p*Q+q] = V[r,p]*W[r,q]
    # dL/dV[r,p] = sum_q dL/dKR[r, p*Q+q] * W[r,q]
    # dL/dW[r,q] = sum_p dL/dKR[r, p*Q+q] * V[r,p]
    dL_dKR_3d = dL_dKR.reshape(R, 9, 9)   # (R, P, Q)
    dL_dV = np.einsum('rpq,rq->rp', dL_dKR_3d, W)   # (R, 9)
    dL_dW = np.einsum('rpq,rp->rq', dL_dKR_3d, V)   # (R, 9)

    grad = np.concatenate([dL_dV.ravel(), dL_dW.ravel()])
    return err, grad

# ── ALS step (alternating least squares) ─────────────────────────────────────

def als_step(U, V, W, n_iter=10):
    """Standard ALS: cycle through U, V, W solving each by least squares."""
    for _ in range(n_iter):
        # Solve U given V, W
        KR = khatri_rao(V, W)
        U, _, _, _ = lstsq(KR.T, T0.T)   # (R,9)

        # Solve V given U, W
        # T_(1)[p, m*Q+q] = sum_r V[r,p] * (U[r,m]*W[r,q])
        T1 = T_TARGET.transpose(1,0,2).reshape(9, 81)  # mode-1 unfolding
        KR_UW = khatri_rao(U, W)
        V, _, _, _ = lstsq(KR_UW.T, T1.T)

        # Solve W given U, V
        T2 = T_TARGET.transpose(2,0,1).reshape(9, 81)  # mode-2 unfolding
        KR_UV = khatri_rao(U, V)
        W, _, _, _ = lstsq(KR_UV.T, T2.T)

    return U, V, W

# ── main optimization loop ────────────────────────────────────────────────────

def run(n_restarts=50, seed=0):
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best_UVW = None

    print(f"Target tensor ||T||^2 = {T_NORM_SQ}")
    print(f"Machine epsilon (float64): {np.finfo(np.float64).eps:.2e}")
    print(f"Goal: recon_err < 1e-25 (well outside float64 meaningful range)")
    print(f"\nRunning {n_restarts} restarts (ALS init + L-BFGS-B refinement)...\n")
    print(f"{'restart':>8}  {'after_ALS':>12}  {'after_LBFGS':>13}  {'best':>13}")
    print("-" * 55)

    # Restart 0: use integer seed
    try:
        V0_int, W0_int = load_integer_seed()
        U0_int = solve_u(V0_int, W0_int)
        restarts_queue = [(V0_int, W0_int, "integer_seed")]
    except Exception as e:
        print(f"  [integer seed load failed: {e}]")
        restarts_queue = []

    # Remaining restarts: random initializations
    for i in range(n_restarts - len(restarts_queue)):
        scale = rng.choice([0.1, 0.5, 1.0, 2.0])
        V_init = rng.standard_normal((R, 9)) * scale
        W_init = rng.standard_normal((R, 9)) * scale
        restarts_queue.append((V_init, W_init, f"random_{i}"))

    for restart_idx, (V_init, W_init, label) in enumerate(restarts_queue):
        # Phase 1: ALS warm-up
        U_als = solve_u(V_init, W_init)
        U_als, V_als, W_als = als_step(U_als, V_init, W_init, n_iter=200)
        err_als = recon_error(U_als, V_als, W_als)

        # Phase 2: L-BFGS-B on V, W (U solved analytically each evaluation)
        x0 = pack(V_als, W_als)
        result = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-20, 'maxfun': 100000},
        )
        V_opt, W_opt = unpack(result.x)
        U_opt = solve_u(V_opt, W_opt)
        err_lbfgs = recon_error(U_opt, V_opt, W_opt)

        if err_lbfgs < best_err:
            best_err = err_lbfgs
            best_UVW = (U_opt.copy(), V_opt.copy(), W_opt.copy())
            tag = " ← best"
        else:
            tag = ""

        print(f"{restart_idx:>8}  {err_als:>12.4e}  {err_lbfgs:>13.4e}  {best_err:>13.4e}{tag}")

        if best_err < 1e-25:
            print(f"\n*** MACHINE PRECISION ACHIEVED! err = {best_err:.6e} ***")
            break

    print(f"\n{'='*55}")
    print(f"Best reconstruction error: {best_err:.6e}")
    print(f"Relative error: {best_err / T_NORM_SQ:.6e}")
    print(f"sqrt(err): {np.sqrt(best_err):.6e}  (Frobenius norm of residual)")
    print()

    U, V, W = best_UVW
    T_hat = cp_reconstruct(U, V, W)
    residual = T_hat - T_TARGET
    max_abs = float(np.max(np.abs(residual)))
    print(f"Max absolute entry error: {max_abs:.6e}")
    print(f"float64 machine eps:      {np.finfo(np.float64).eps:.6e}")

    if max_abs < np.finfo(np.float64).eps * 10:
        print("\n*** RESULT: Reconstruction is at/below float64 machine precision! ***")
        print("*** Rank-19 real CP decomposition found (up to floating-point) ***")
    elif max_abs < 1e-10:
        print(f"\nClose but not quite machine precision. Max error = {max_abs:.2e}")
        print("Consider: more restarts, higher precision solver, or perturbation refinement.")
    else:
        print(f"\nNot converged to machine precision. Residual = {max_abs:.2e}")

    # Save result
    out = {
        'best_err': best_err,
        'max_abs_entry_error': max_abs,
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist(),
    }
    out_path = ROOT / "CANON_DATABASE" / "real_cp_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path.name}")
    return best_err, best_UVW

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts", type=int, default=50)
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()
    run(n_restarts=args.restarts, seed=args.seed)
