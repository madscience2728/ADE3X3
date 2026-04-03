"""
gates.py — Jacobians for bilinear constraints and gamma solve.

Provides compute_jacobians_beta, compute_jacobians_alpha, solve_gamma.
"""

import numpy as np
from typing import Tuple
from tensor import DEAD_PAIRS


def compute_jacobians_beta(alpha_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobians of H[k,:], Sigma[k,:], Delta[k,:] w.r.t. vec(beta_k).

    beta layout: beta_k[t,u] -> index t*3+u.
    Returns A_k (18,9), S_k (9,9), D_k (54,9).
    """
    A_k = np.zeros((18, 9))
    S_k = np.zeros((9,  9))
    D_k = np.zeros((54, 9))

    for r in range(3):
        for u in range(3):
            fiber = r * 3 + u
            for s in range(3):
                bidx = s * 3 + u
                S_k[fiber, bidx] += alpha_k[r, s]
                if s == 0:
                    A_k[fiber, bidx] += alpha_k[r, 0]
                elif s == 1:
                    A_k[fiber, bidx] -= alpha_k[r, 1]
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
    """Jacobians of H[k,:], Sigma[k,:], Delta[k,:] w.r.t. vec(alpha_k).

    alpha layout: alpha_k[r,s] -> index r*3+s.
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
                    A_k[fiber, aidx] += beta_k[0, u]
                elif s == 1:
                    A_k[fiber, aidx] -= beta_k[1, u]
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


def solve_gamma(Sigma, Nuisance):
    """Solve Gamma @ Sigma = 3*I_9 and Gamma @ Nuisance = 0 (least squares).

    Returns (Gamma (9, R), residual_norm).
    """
    A = np.hstack([Sigma, Nuisance])  # (R, 81)
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
