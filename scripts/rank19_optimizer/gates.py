"""
gates.py — Gate 1 (hard), Gate 2 (soft via gradient), Gate 3 (exact solve).

Gate 1: rank(H) = 10.
  Hard constraint: (I18 - V10 V10^T) @ A_k @ β = 0.
  Enforced by PROJECTING OUT the violation (minimal perturbation):
    correction = pinv(Perp_A) @ (Perp_A @ β_old)
    β_new = β_old - correction
  This preserves 8/9 dimensions of β, not 1/9.

Gate 2: Δ ⊂ span(H).
  SOFT — improves naturally as Γ·Δ decreases when fitness improves.
  No explicit enforcement needed.

Gate 3: Γ solve — exact least squares given (α, β).
"""

import numpy as np
from typing import Tuple
from tensor import DEAD_PAIRS

RANK_TOL = 1e-10


def compute_jacobians_beta(alpha_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobians of H[k,:], Sigma[k,:], Delta[k,:] w.r.t. vec(beta_k), for fixed alpha_k.

    beta layout: beta_k[t,u] → index t*3+u.

    Returns A_k (18,9), S_k (9,9), D_k (54,9).
    """
    A_k = np.zeros((18, 9))
    S_k = np.zeros((9,  9))
    D_k = np.zeros((54, 9))

    for r in range(3):
        for u in range(3):
            fiber = r * 3 + u
            for s in range(3):
                bidx = s * 3 + u   # beta_k[s,u]
                S_k[fiber, bidx] += alpha_k[r, s]
                if s == 0:
                    A_k[fiber,     bidx] += alpha_k[r, 0]
                elif s == 1:
                    A_k[fiber,     bidx] -= alpha_k[r, 1]
                    A_k[9 + fiber, bidx] += alpha_k[r, 1]
                elif s == 2:
                    A_k[9 + fiber, bidx] -= alpha_k[r, 2]

    for dp_idx, (s, t) in enumerate(DEAD_PAIRS):
        for r in range(3):
            for u in range(3):
                dcol = dp_idx * 9 + r * 3 + u
                bidx = t * 3 + u
                D_k[dcol, bidx] += alpha_k[r, s]

    return A_k, S_k, D_k


def compute_jacobians_alpha(beta_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobians of H[k,:], Sigma[k,:], Delta[k,:] w.r.t. vec(alpha_k), for fixed beta_k.

    alpha layout: alpha_k[r,s] → index r*3+s.

    Returns A_k (18,9), S_k (9,9), D_k (54,9).
    """
    A_k = np.zeros((18, 9))
    S_k = np.zeros((9,  9))
    D_k = np.zeros((54, 9))

    for r in range(3):
        for u in range(3):
            fiber = r * 3 + u
            for s in range(3):
                aidx = r * 3 + s
                S_k[fiber, aidx] += beta_k[s, u]
                if s == 0:
                    A_k[fiber,     aidx] += beta_k[0, u]
                elif s == 1:
                    A_k[fiber,     aidx] -= beta_k[1, u]
                    A_k[9 + fiber, aidx] += beta_k[1, u]
                elif s == 2:
                    A_k[9 + fiber, aidx] -= beta_k[2, u]

    for dp_idx, (s, t) in enumerate(DEAD_PAIRS):
        for r in range(3):
            for u in range(3):
                dcol = dp_idx * 9 + r * 3 + u
                aidx = r * 3 + s
                D_k[dcol, aidx] += beta_k[t, u]

    return A_k, S_k, D_k


def compute_V10(H: np.ndarray) -> np.ndarray:
    """Top-10 right singular vectors of H (R×18). Returns V10 (18, 10)."""
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:10, :].T   # (18, 10)


def project_to_gate1(x_old: np.ndarray,
                     A_k: np.ndarray,
                     V10: np.ndarray) -> np.ndarray:
    """Project OUT the Gate-1 violation from x_old (minimal perturbation).

    Finds x_new = x_old - correction where correction is the minimum-norm vector
    such that (I - V10 V10^T) @ A_k @ x_new = 0.

    This is: x_new = x_old - pinv(Perp_A) @ (Perp_A @ x_old)

    Preserves 8/9 dimensions of x; does NOT collapse x to near-zero.
    """
    P = V10 @ V10.T                         # (18, 18)
    Perp_A = (np.eye(18) - P) @ A_k         # (18, 9), rank ~8
    violation = Perp_A @ x_old              # (18,)
    correction, _, _, _ = np.linalg.lstsq(Perp_A, violation, rcond=None)   # (9,)
    return x_old - correction


def project_gradient_to_gate1(grad: np.ndarray,
                               A_k: np.ndarray,
                               V10: np.ndarray) -> np.ndarray:
    """Project a gradient/direction onto the Gate-1 tangent space.

    Removes the component of grad that would violate Gate 1.
    The feasible directions satisfy: (I - P) A_k δ = 0.

    Returns the projected (feasible) gradient direction.
    """
    P = V10 @ V10.T
    Perp_A = (np.eye(18) - P) @ A_k   # (18, 9), rank ~8
    # pinv(Perp_A) @ Perp_A is the projection onto row(Perp_A)
    # We want to remove the row(Perp_A) component from grad
    pinv_PA = np.linalg.pinv(Perp_A)   # (9, 18)
    return grad - pinv_PA @ (Perp_A @ grad)


def solve_gamma(Sigma: np.ndarray, Nuisance: np.ndarray) -> Tuple[np.ndarray, float]:
    """Gate 3: solve Γ·Σ = 3·I₉ and Γ·Nuisance = 0 (least squares per row of Γ).

    Returns (Gamma (9, R), residual_norm).
    """
    A = np.hstack([Sigma, Nuisance])   # (R, 81)
    n = A.shape[1]
    Gamma = np.zeros((9, Sigma.shape[0]))
    total_res = 0.0

    for i in range(9):
        rhs = np.zeros(n)
        rhs[i] = 3.0
        x, res, _, _ = np.linalg.lstsq(A.T, rhs, rcond=None)
        Gamma[i, :] = x
        resval = float(res[0]) if len(res) else float(np.linalg.norm(A.T @ x - rhs) ** 2)
        total_res += resval

    return Gamma, float(np.sqrt(max(total_res, 0.0)))
