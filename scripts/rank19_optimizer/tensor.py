"""
tensor.py — Core tensor construction and fiber-mode coordinate blocks.

Ordering convention for dead pairs (s,t) with s≠t (54 columns in Delta):
  (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)  — 6 pairs × 9 fibers (r,u) = 54 columns
"""

import numpy as np

# Fixed ordering of dead (s,t) pairs — must match everywhere.
DEAD_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def build_target_tensor() -> np.ndarray:
    """Build the 3×3 matrix multiplication tensor T ∈ ℝ^{9×9×9}.

    T[i, j, k] = 1 iff:
        i = (r', u')  — C output index
        j = (r', s)   — A input index  (same r', free s)
        k = (s,  u')  — B input index  (same s, same u')

    The 27 live entries correspond to C[r',u'] = Σ_s A[r',s]·B[s,u'].
    """
    T = np.zeros((9, 9, 9))
    for rp in range(3):
        for up in range(3):
            for s in range(3):
                i = rp * 3 + up   # C index
                j = rp * 3 + s    # A index
                k = s  * 3 + up   # B index
                T[i, j, k] = 1.0
    return T


def build_fiber_coordinates(alpha: np.ndarray, beta: np.ndarray) -> dict:
    """Build all fiber-mode coordinate blocks from (alpha, beta).

    Parameters
    ----------
    alpha : (R, 3, 3)
    beta  : (R, 3, 3)

    Returns dict with keys:
        Sigma    (R, 9)  — fiber sums
        Eta1     (R, 9)  — first anisotropy
        Eta2     (R, 9)  — second anisotropy
        H        (R, 18) — [Eta1 | Eta2]
        Delta    (R, 54) — dead-X cross-products
        Nuisance (R, 72) — [H | Delta]
    """
    R = alpha.shape[0]
    Sigma = np.zeros((R, 9))
    Eta1  = np.zeros((R, 9))
    Eta2  = np.zeros((R, 9))
    Delta = np.zeros((R, 54))

    for k in range(R):
        a = alpha[k]   # (3,3)
        b = beta[k]    # (3,3)

        for r in range(3):
            for u in range(3):
                idx = r * 3 + u
                s0 = a[r, 0] * b[0, u]
                s1 = a[r, 1] * b[1, u]
                s2 = a[r, 2] * b[2, u]
                Sigma[k, idx] = s0 + s1 + s2
                Eta1[k,  idx] = s0 - s1
                Eta2[k,  idx] = s1 - s2

        col = 0
        for (s, t) in DEAD_PAIRS:
            for r in range(3):
                for u in range(3):
                    Delta[k, col] = a[r, s] * b[t, u]
                    col += 1

    H        = np.hstack([Eta1, Eta2])       # (R, 18)
    Nuisance = np.hstack([H, Delta])          # (R, 72)

    return {
        'Sigma':    Sigma,
        'Eta1':     Eta1,
        'Eta2':     Eta2,
        'H':        H,
        'Delta':    Delta,
        'Nuisance': Nuisance,
    }


def build_decomposition(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Reconstruct T_hat = Σ_k γ_k ⊗ α_k ⊗ β_k.

    Parameters
    ----------
    alpha, beta, gamma : (R, 3, 3)

    Returns T_hat (9, 9, 9).
    """
    T_hat = np.zeros((9, 9, 9))
    R = alpha.shape[0]
    for k in range(R):
        g = gamma[k].ravel()
        a = alpha[k].ravel()
        b = beta[k].ravel()
        T_hat += np.einsum('i,j,k->ijk', g, a, b)
    return T_hat
