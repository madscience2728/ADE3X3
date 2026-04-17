"""Structured initialization strategies for rank-(r-1) search from rank-r factors.

The key insight: if rank-r factors are known, the rank-(r-1) solution (if it exists)
lives near the set of rank-(r-1) tensors, which is close to the rank-r solution.
Term deletion gives structured starting points that are orders of magnitude better
than random initialization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def load_factors(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load U,V,W factor arrays from a .npz file saved by run_matmul.py."""
    data = np.load(path)
    return data["U"], data["V"], data["W"]


def score_deletions(
    T: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Score each term by its deletion residual.

    deletion_residual[t] = ||T - sum_{s != t} u_s ⊗ v_s ⊗ w_s||_F

    Lower = more redundant = better candidate for deletion.
    """
    r = U.shape[0]
    T_full = np.einsum("ra,rb,rc->abc", U, V, W)
    residuals = np.zeros(r)

    for term_index in range(r):
        term_tensor = np.einsum("a,b,c->abc", U[term_index], V[term_index], W[term_index])
        T_minus_term = T_full - term_tensor
        residuals[term_index] = float(np.linalg.norm(T - T_minus_term))

    return residuals


def deletion_inits(
    T: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    n_noise_variants: int = 3,
    noise_eps: float = 0.01,
    rng_seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str, float]]:
    """Generate structured rank-(r-1) initialization triples by term deletion."""
    r = U.shape[0]
    rng = np.random.default_rng(rng_seed)

    deletion_residuals = score_deletions(T, U, V, W)
    order = np.argsort(deletion_residuals)

    print("  [STRUCT_INIT] Deletion residuals (sorted):", flush=True)
    for term_index in order:
        print(f"    term {term_index:2d}: del_res={deletion_residuals[term_index]:.4e}", flush=True)

    for term_index in order:
        keep_indices = [s for s in range(r) if s != term_index]
        U_del = U[keep_indices].copy()
        V_del = V[keep_indices].copy()
        W_del = W[keep_indices].copy()
        deletion_residual = float(deletion_residuals[term_index])

        desc = f"delete_term{term_index}_delres{deletion_residual:.3e}"
        yield U_del.copy(), V_del.copy(), W_del.copy(), desc, deletion_residual

        factor_scale = np.mean([
            np.linalg.norm(U_del),
            np.linalg.norm(V_del),
            np.linalg.norm(W_del),
        ]) / max(U_del.shape[0], 1)
        eps = noise_eps * max(float(factor_scale), 0.1)

        for variant_index in range(n_noise_variants):
            U_noisy = U_del + rng.standard_normal(U_del.shape) * eps
            V_noisy = V_del + rng.standard_normal(V_del.shape) * eps
            W_noisy = W_del + rng.standard_normal(W_del.shape) * eps
            desc_noisy = f"delete_term{term_index}_noise{variant_index}_eps{eps:.3e}"
            yield U_noisy, V_noisy, W_noisy, desc_noisy, deletion_residual