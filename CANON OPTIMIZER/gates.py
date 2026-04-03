"""
gates.py - Diagnostic helpers for the symmetry-reduced solver.
"""

from __future__ import annotations

import numpy as np

from tensor import build_decomposition, build_fiber_coordinates, build_target_tensor


def solve_gamma_least_squares(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Least-squares fit for gamma with fixed alpha and beta."""
    target = build_target_tensor()
    rank = alpha.shape[0]
    alpha_flat = alpha.reshape(rank, 9)
    beta_flat = beta.reshape(rank, 9)
    row_ids, col_ids = np.mgrid[0:9, 0:9]
    design = alpha_flat[:, row_ids.ravel()].T * beta_flat[:, col_ids.ravel()].T
    target_mode0 = target.reshape(9, 81)
    gram = design.T @ design
    gamma = target_mode0 @ design @ np.linalg.pinv(gram)
    return gamma.T.reshape(rank, 3, 3)


def compute_diagnostics(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict[str, float | int]:
    """Compute the gate diagnostics used elsewhere in the repo."""
    coords = build_fiber_coordinates(alpha, beta)
    sigma = coords["Sigma"]
    h_block = coords["H"]
    delta = coords["Delta"]
    nuisance = coords["Nuisance"]

    target = build_target_tensor()
    tensor_hat = build_decomposition(alpha, beta, gamma)
    residual = target - tensor_hat

    delta_fit, _, _, _ = np.linalg.lstsq(h_block, delta, rcond=None)
    gamma_flat = gamma.reshape(alpha.shape[0], 9)
    augmented = np.hstack([sigma, nuisance])

    return {
        "loss_fro": float(np.sum(residual * residual)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "rank_H": int(np.linalg.matrix_rank(h_block, tol=1e-10)),
        "rank_nuisance": int(np.linalg.matrix_rank(nuisance, tol=1e-10)),
        "rank_augmented": int(np.linalg.matrix_rank(augmented, tol=1e-10)),
        "rank_Delta": int(np.linalg.matrix_rank(delta, tol=1e-10)),
        "delta_residual": float(np.linalg.norm(delta - h_block @ delta_fit, ord="fro")),
        "gamma_delta": float(np.linalg.norm(gamma_flat.T @ delta, ord="fro")),
        "conservation": int(alpha.shape[0] + (18 - np.linalg.matrix_rank(h_block, tol=1e-10))),
    }