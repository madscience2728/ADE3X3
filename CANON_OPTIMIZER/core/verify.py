"""
Given a candidate (alpha, beta, gamma), verify independently that it
exactly reconstructs the matrix multiplication tensor.

ZERO dependencies on the rest of the codebase (except numpy).
"""

import numpy as np
import json
import sys


def build_tensor_standalone(n: int = 3) -> np.ndarray:
    """Build T from scratch: T[r*n+s, s*n+u, r*n+u] = 1."""
    nn = n * n
    T = np.zeros((nn, nn, nn), dtype=np.int64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                T[r * n + s, s * n + u, r * n + u] = 1
    return T


def reconstruct_standalone(alpha: np.ndarray, beta: np.ndarray,
                            gamma: np.ndarray, n: int = 3) -> np.ndarray:
    """Reconstruct using explicit triple loop (no library calls beyond basic numpy)."""
    nn = n * n
    R = alpha.shape[0]
    T_hat = np.zeros((nn, nn, nn), dtype=np.float64)
    for k in range(R):
        a = alpha[k].ravel().astype(np.float64)
        b = beta[k].ravel().astype(np.float64)
        g = gamma[k].ravel().astype(np.float64)
        for i in range(nn):
            for j in range(nn):
                for l in range(nn):
                    T_hat[i, j, l] += a[i] * b[j] * g[l]
    return T_hat


def compute_H_standalone(alpha, beta, n=3):
    """Compute H = [Eta1 | Eta2] from scratch."""
    R = alpha.shape[0]
    H = np.zeros((R, 2 * n * n), dtype=alpha.dtype)
    for k in range(R):
        for r in range(n):
            for u in range(n):
                idx = r * n + u
                H[k, idx] = (alpha[k, r, 0] * beta[k, 0, u]
                              - alpha[k, r, 1] * beta[k, 1, u])
                H[k, n * n + idx] = (alpha[k, r, 1] * beta[k, 1, u]
                                      - alpha[k, r, 2] * beta[k, 2, u])
    return H


def verify_solution(alpha: np.ndarray, beta: np.ndarray,
                     gamma: np.ndarray, n: int = 3) -> bool:
    """Full independent verification. Returns True if valid."""
    print(f"Verifying R={alpha.shape[0]} decomposition...")
    R = alpha.shape[0]
    nn = n * n

    # 1. Build T
    T = build_tensor_standalone(n)

    # 2. Reconstruct
    T_hat = reconstruct_standalone(alpha, beta, gamma, n)

    # 3. Entry-by-entry comparison
    T_f = T.astype(np.float64)
    max_err = 0.0
    mismatch = None
    for i in range(nn):
        for j in range(nn):
            for l in range(nn):
                err = abs(T_f[i, j, l] - T_hat[i, j, l])
                if err > max_err:
                    max_err = err
                    if err > 1e-10:
                        mismatch = (i, j, l)

    if max_err < 1e-10:
        print(f"VERIFIED: Exact R={R} decomposition confirmed.")
        print(f"  Max reconstruction error: {max_err:.2e}")
    else:
        print(f"FAILED: Max error = {max_err:.2e}")
        if mismatch:
            i, j, l = mismatch
            print(f"  First mismatch at ({i},{j},{l}): T={T_f[i,j,l]}, T_hat={T_hat[i,j,l]:.6f}")
        return False

    # 4. Gate conditions
    H = compute_H_standalone(alpha, beta, n)
    rank_H = np.linalg.matrix_rank(H.astype(np.float64))
    target_rank_H = R - nn
    eta_null = 2 * nn - rank_H  # 18 - rank_H
    conservation = R + eta_null

    print(f"  rank(H) = {rank_H} (target: {target_rank_H})")
    print(f"  η_nullity = {eta_null}")
    print(f"  Conservation: R + η_null = {conservation} {'✓' if conservation == n**3 else '✗'}")

    # 5. Print terms
    print(f"\n  {R} terms:")
    for k in range(R):
        a_str = str(alpha[k].tolist())
        b_str = str(beta[k].tolist())
        g_str = str(gamma[k].tolist())
        print(f"    Term {k}: α={a_str}")
        print(f"             β={b_str}")
        print(f"             γ={g_str}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify.py SOLUTION.json")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    alpha = np.array(data['alpha'], dtype=np.float64)
    beta = np.array(data['beta'], dtype=np.float64)
    gamma = np.array(data['gamma'], dtype=np.float64)

    # Reshape if needed
    n = 3
    R = alpha.shape[0]
    if alpha.ndim == 2:
        alpha = alpha.reshape(R, n, n)
    if beta.ndim == 2:
        beta = beta.reshape(R, n, n)
    if gamma.ndim == 2:
        gamma = gamma.reshape(R, n, n)

    ok = verify_solution(alpha, beta, gamma, n)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
