"""
Construct the 3x3 matrix multiplication tensor and compute all
coordinate blocks (Sigma, Eta1, Eta2, Delta, H, Nuisance) from
a set of (alpha, beta) factor pairs.

All arrays use int64 when working over integer coefficient fields.
Use exact integer arithmetic throughout -- no floats until SVD.
"""

import numpy as np
try:
    import numba
except ImportError:
    numba = None


def build_tensor(n: int = 3) -> np.ndarray:
    """Returns T as shape (n^2, n^2, n^2) int array.
    T[r*n+s, s*n+u, r*n+u] = 1 for all (r,s,u)."""
    nn = n * n
    T = np.zeros((nn, nn, nn), dtype=np.int64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                T[r * n + s, s * n + u, r * n + u] = 1
    return T


def compute_sigma(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """Sigma[k, r*n+u] = sum_s alpha_k[r,s] * beta_k[s,u].
    alpha: (R, n, n), beta: (R, n, n) -> (R, n^2)"""
    R = alpha.shape[0]
    # sigma_k[r,u] = (alpha_k @ beta_k)[r,u]
    # So Sigma[k,:] = (alpha_k @ beta_k).flatten()
    sigma = np.zeros((R, n * n), dtype=alpha.dtype)
    for k in range(R):
        sigma[k] = (alpha[k] @ beta[k]).ravel()
    return sigma


def compute_eta1(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """Eta1[k, r*n+u] = alpha_k[r,0]*beta_k[0,u] - alpha_k[r,1]*beta_k[1,u].
    Returns (R, n^2)."""
    R = alpha.shape[0]
    eta1 = np.zeros((R, n * n), dtype=alpha.dtype)
    for k in range(R):
        for r in range(n):
            for u in range(n):
                eta1[k, r * n + u] = (alpha[k, r, 0] * beta[k, 0, u]
                                      - alpha[k, r, 1] * beta[k, 1, u])
    return eta1


def compute_eta2(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """Eta2[k, r*n+u] = alpha_k[r,1]*beta_k[1,u] - alpha_k[r,2]*beta_k[2,u].
    Returns (R, n^2)."""
    R = alpha.shape[0]
    eta2 = np.zeros((R, n * n), dtype=alpha.dtype)
    for k in range(R):
        for r in range(n):
            for u in range(n):
                eta2[k, r * n + u] = (alpha[k, r, 1] * beta[k, 1, u]
                                      - alpha[k, r, 2] * beta[k, 2, u])
    return eta2


def compute_H(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """H = [Eta1 | Eta2], shape (R, 2*n^2)."""
    return np.hstack([compute_eta1(alpha, beta, n),
                      compute_eta2(alpha, beta, n)])


def compute_delta(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """Delta[k, ...] for off-diagonal s!=t products.
    For each (r,u) and each ordered pair (s,t) with s!=t:
      delta_k[r*n*(n-1)*n + u*(n*(n-1)) + pair_idx] = alpha_k[r,s]*beta_k[t,u]
    For n=3: 9 (r,u) pairs * 6 (s,t) pairs = 54 columns.
    Returns (R, n^2 * n*(n-1))."""
    R = alpha.shape[0]
    # Build pair list for (s,t) with s!=t
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    n_pairs = len(pairs)  # n*(n-1) = 6 for n=3
    ncols = n * n * n_pairs  # 54 for n=3
    delta = np.zeros((R, ncols), dtype=alpha.dtype)
    for k in range(R):
        col = 0
        for r in range(n):
            for u in range(n):
                for s, t in pairs:
                    delta[k, col] = alpha[k, r, s] * beta[k, t, u]
                    col += 1
    return delta


def compute_nuisance(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> np.ndarray:
    """Nuisance = [H | Delta], shape (R, 2*n^2 + n^2*n*(n-1))."""
    return np.hstack([compute_H(alpha, beta, n),
                      compute_delta(alpha, beta, n)])


def compute_all_blocks(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> dict:
    """Returns dict with all coordinate blocks."""
    eta1 = compute_eta1(alpha, beta, n)
    eta2 = compute_eta2(alpha, beta, n)
    H = np.hstack([eta1, eta2])
    delta = compute_delta(alpha, beta, n)
    nuisance = np.hstack([H, delta])
    sigma = compute_sigma(alpha, beta, n)
    return {
        'Sigma': sigma,
        'Eta1': eta1,
        'Eta2': eta2,
        'H': H,
        'Delta': delta,
        'Nuisance': nuisance,
    }


def reconstruct_tensor(alpha: np.ndarray, beta: np.ndarray,
                        gamma: np.ndarray, n: int = 3) -> np.ndarray:
    """Full reconstruction: T_hat[i,j,l] = sum_k alpha_k.flat[i] * beta_k.flat[j] * gamma_k.flat[l]."""
    nn = n * n
    R = alpha.shape[0]
    T_hat = np.zeros((nn, nn, nn), dtype=np.float64)
    for k in range(R):
        a = alpha[k].ravel().astype(np.float64)
        b = beta[k].ravel().astype(np.float64)
        g = gamma[k].ravel().astype(np.float64)
        T_hat += np.einsum('i,j,l->ijl', a, b, g)
    return T_hat


# ── Numba-accelerated helpers for hot inner loops ──

_jit = numba.njit(cache=True) if numba is not None else (lambda f: f)


@_jit
def fast_H_row(alpha_k: np.ndarray, beta_k: np.ndarray) -> np.ndarray:
    """Compute one row of H = [Eta1 | Eta2] for a single term.
    alpha_k, beta_k: (3,3) int arrays. Returns (18,) int array."""
    out = np.empty(18, dtype=np.int64)
    for r in range(3):
        for u in range(3):
            idx = r * 3 + u
            out[idx] = alpha_k[r, 0] * beta_k[0, u] - alpha_k[r, 1] * beta_k[1, u]
            out[9 + idx] = alpha_k[r, 1] * beta_k[1, u] - alpha_k[r, 2] * beta_k[2, u]
    return out


@_jit
def fast_sigma_row(alpha_k: np.ndarray, beta_k: np.ndarray) -> np.ndarray:
    """Compute one row of Sigma for a single term. Returns (9,) int array."""
    out = np.empty(9, dtype=np.int64)
    for r in range(3):
        for u in range(3):
            s = np.int64(0)
            for ss in range(3):
                s += alpha_k[r, ss] * beta_k[ss, u]
            out[r * 3 + u] = s
    return out


@_jit
def fast_delta_row(alpha_k: np.ndarray, beta_k: np.ndarray) -> np.ndarray:
    """Compute one row of Delta for a single term. Returns (54,) int array."""
    out = np.empty(54, dtype=np.int64)
    col = 0
    for r in range(3):
        for u in range(3):
            for s in range(3):
                for t in range(3):
                    if s != t:
                        out[col] = alpha_k[r, s] * beta_k[t, u]
                        col += 1
    return out


# ── Self-test ──

def self_test():
    """Verify using the standard R=27 algorithm."""
    n = 3
    T = build_tensor(n)

    # Standard algorithm: alpha_k = e_r e_s^T, beta_k = e_s e_u^T, gamma_k = e_r e_u^T
    R = 27
    alpha = np.zeros((R, n, n), dtype=np.int64)
    beta = np.zeros((R, n, n), dtype=np.int64)
    gamma = np.zeros((R, n, n), dtype=np.int64)
    k = 0
    for r in range(n):
        for s in range(n):
            for u in range(n):
                alpha[k, r, s] = 1
                beta[k, s, u] = 1
                gamma[k, r, u] = 1
                k += 1

    # Reconstruct
    T_hat = reconstruct_tensor(alpha, beta, gamma, n)
    err = np.max(np.abs(T.astype(np.float64) - T_hat))
    assert err < 1e-12, f"Standard R=27 reconstruction failed: max error = {err}"

    # Check coordinate blocks
    blocks = compute_all_blocks(alpha, beta, n)
    rk_H = np.linalg.matrix_rank(blocks['H'].astype(np.float64))
    rk_N = np.linalg.matrix_rank(blocks['Nuisance'].astype(np.float64))
    assert rk_H == 18, f"Standard R=27: rank(H) = {rk_H}, expected 18"
    assert rk_N == 18, f"Standard R=27: rank(Nuisance) = {rk_N}, expected 18"

    # Conservation law
    eta_null = 18 - rk_H
    assert R + eta_null == 27, f"Conservation: {R} + {eta_null} != 27"

    # Test numba helpers
    h_row = fast_H_row(alpha[0], beta[0])
    assert h_row.shape == (18,)
    s_row = fast_sigma_row(alpha[0], beta[0])
    assert s_row.shape == (9,)
    d_row = fast_delta_row(alpha[0], beta[0])
    assert d_row.shape == (54,)

    print("tensor.py: all self-tests passed")


if __name__ == "__main__":
    self_test()
