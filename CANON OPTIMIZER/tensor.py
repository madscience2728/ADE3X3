"""
tensor.py - Core tensor construction and fiber-mode coordinate blocks.

Ordering convention for dead pairs (s, t) with s != t (54 columns in Delta):
  (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)  -- 6 pairs x 9 fibers (r, u) = 54
"""

from __future__ import annotations

import numpy as np


DEAD_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def build_target_tensor() -> np.ndarray:
    """Build the 3x3 matrix multiplication tensor T in R^(9x9x9)."""
    tensor = np.zeros((9, 9, 9), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            for summation in range(3):
                gamma_idx = row * 3 + col
                alpha_idx = row * 3 + summation
                beta_idx = summation * 3 + col
                tensor[gamma_idx, alpha_idx, beta_idx] = 1.0
    return tensor


def build_fiber_coordinates(alpha: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    """Build all fiber-mode coordinate blocks from alpha and beta.

    Parameters
    ----------
    alpha : (R, 3, 3)
    beta  : (R, 3, 3)
    """
    rank = alpha.shape[0]
    sigma = np.zeros((rank, 9), dtype=np.float64)
    eta1 = np.zeros((rank, 9), dtype=np.float64)
    eta2 = np.zeros((rank, 9), dtype=np.float64)
    delta = np.zeros((rank, 54), dtype=np.float64)

    for term_idx in range(rank):
        a_term = alpha[term_idx]
        b_term = beta[term_idx]

        for row in range(3):
            for col in range(3):
                fiber_idx = row * 3 + col
                s0 = a_term[row, 0] * b_term[0, col]
                s1 = a_term[row, 1] * b_term[1, col]
                s2 = a_term[row, 2] * b_term[2, col]
                sigma[term_idx, fiber_idx] = s0 + s1 + s2
                eta1[term_idx, fiber_idx] = s0 - s1
                eta2[term_idx, fiber_idx] = s1 - s2

        delta_col = 0
        for source, target in DEAD_PAIRS:
            for row in range(3):
                for col in range(3):
                    delta[term_idx, delta_col] = a_term[row, source] * b_term[target, col]
                    delta_col += 1

    h_block = np.hstack([eta1, eta2])
    nuisance = np.hstack([h_block, delta])
    return {
        "Sigma": sigma,
        "Eta1": eta1,
        "Eta2": eta2,
        "H": h_block,
        "Delta": delta,
        "Nuisance": nuisance,
    }


def build_decomposition(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Reconstruct T_hat = sum_k gamma_k \otimes alpha_k \otimes beta_k."""
    tensor_hat = np.zeros((9, 9, 9), dtype=np.float64)
    for a_term, b_term, g_term in zip(alpha, beta, gamma, strict=True):
        tensor_hat += np.einsum("i,j,k->ijk", g_term.ravel(), a_term.ravel(), b_term.ravel())
    return tensor_hat