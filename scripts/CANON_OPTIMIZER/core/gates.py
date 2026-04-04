"""
Algebraic gate checks for the R=19 (and other R) constraint hierarchy.
All functions return (passed: bool, diagnostics: dict).
Use integer/rational arithmetic where possible; float SVD for rank.
"""

import numpy as np
from .tensor import (compute_all_blocks, compute_H, compute_sigma,
                     compute_delta, compute_nuisance, reconstruct_tensor,
                     build_tensor, fast_H_row, fast_sigma_row, fast_delta_row)


def _numerical_rank(M: np.ndarray, tol: float = 1e-10) -> tuple:
    """Returns (rank, singular_values)."""
    if M.size == 0:
        return 0, np.array([])
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    if sv.size == 0:
        return 0, sv
    threshold = tol * sv[0] if sv[0] > 0 else tol
    rank = int(np.sum(sv > threshold))
    return rank, sv


def check_gate1(H: np.ndarray, target_rank: int = 10,
                tol: float = 1e-10) -> tuple:
    """Gate 1: rank(H) == target_rank."""
    rank_H, sv = _numerical_rank(H, tol)
    eta_nullity = 18 - rank_H
    R = H.shape[0]
    return (rank_H == target_rank,
            {'rank_H': rank_H, 'singular_values': sv,
             'eta_nullity': eta_nullity,
             'conservation': R + eta_nullity})


def check_gate2(H: np.ndarray, Delta: np.ndarray,
                tol: float = 1e-10) -> tuple:
    """Gate 2: rank([H|Delta]) == rank(H), i.e. Delta in span(H)."""
    rank_H, _ = _numerical_rank(H, tol)
    combined = np.hstack([H, Delta])
    rank_N, _ = _numerical_rank(combined, tol)
    passed = (rank_N == rank_H)
    return (passed,
            {'rank_H': rank_H, 'rank_nuisance': rank_N,
             'delta_leak': rank_N - rank_H})


def check_gate3(Sigma: np.ndarray, Nuisance: np.ndarray,
                R: int = 19, n: int = 3,
                tol: float = 1e-10,
                residual_tol: float = 1e-12) -> tuple:
    """Gate 3: rank([Sigma|Nuisance]) == rank(Nuisance) + n^2."""
    nn = n * n
    aug = np.hstack([Sigma, Nuisance])
    aug_rk, _ = _numerical_rank(aug, tol)
    nuis_rk, _ = _numerical_rank(Nuisance, tol)
    passed = (aug_rk == nuis_rk + nn)

    gamma = None
    residual = None
    if passed:
        # Solve: Gamma @ [Sigma | Nuisance]^T = [3*I_9 | 0]^T
        # i.e. [Sigma | Nuisance]^T @ Gamma^T = [3*I_9 | 0]^T
        A = aug.astype(np.float64).T  # (R+cols, nn) after transposing... 
        # Actually: Gamma is (nn, R). We need Gamma @ Sigma = 3*I_9 and Gamma @ Nuisance = 0
        # So Gamma @ M = rhs where M = [Sigma^T; Nuisance^T]... let me be careful.
        # M = np.vstack([Sigma, Nuisance]) would be wrong - Sigma is (R, 9), Nuisance is (R, 72)
        # Gamma is (9, R). Gamma @ Sigma = 3 I_9 means (9,R) @ (R,9) = (9,9)
        # Gamma @ Nuisance = 0 means (9,R) @ (R,72) = (9,72)
        # Combined: Gamma @ [Sigma | Nuisance] = [3 I_9 | 0_{9x72}]
        # So: solve Gamma @ full_matrix = rhs
        # full_matrix: (R, 9+72) = (R, 81)
        # Gamma: (9, R)
        # rhs: (9, 81)
        full = np.hstack([Sigma, Nuisance]).astype(np.float64)  # (R, 81)
        rhs = np.zeros((nn, nn + Nuisance.shape[1]), dtype=np.float64)
        rhs[:nn, :nn] = 3.0 * np.eye(nn)
        # Gamma @ full = rhs => full^T @ Gamma^T = rhs^T
        result = np.linalg.lstsq(full.T, rhs.T, rcond=None)
        gamma_T = result[0]  # (R, 9)
        gamma = gamma_T.T     # (9, R)
        # Check residual
        actual = gamma @ full
        residual = np.max(np.abs(actual - rhs))
        if residual > residual_tol:
            passed = False

    return (passed,
            {'augmented_rank': aug_rk, 'nuisance_rank': nuis_rk,
             'gamma': gamma, 'residual': residual})


def check_all_gates(alpha: np.ndarray, beta: np.ndarray,
                    R: int = None, n: int = 3,
                    tol: float = 1e-10) -> tuple:
    """Run all gates sequentially. Return (solution_found, diagnostics)."""
    if R is None:
        R = alpha.shape[0]
    target_rank_H = R - n * n

    blocks = compute_all_blocks(alpha, beta, n)

    # Gate 1
    g1_pass, g1_diag = check_gate1(blocks['H'], target_rank_H, tol)
    if not g1_pass:
        return False, {'gate1': g1_diag, 'failed_at': 'gate1'}

    # Gate 2
    g2_pass, g2_diag = check_gate2(blocks['H'], blocks['Delta'], tol)
    if not g2_pass:
        return False, {'gate1': g1_diag, 'gate2': g2_diag, 'failed_at': 'gate2'}

    # Gate 3
    g3_pass, g3_diag = check_gate3(blocks['Sigma'], blocks['Nuisance'], R, n, tol)
    if not g3_pass:
        return False, {'gate1': g1_diag, 'gate2': g2_diag, 'gate3': g3_diag,
                       'failed_at': 'gate3'}

    # Verify reconstruction
    gamma_matrix = g3_diag['gamma']  # (9, R)
    # gamma_matrix is Gamma (9, R). Need to extract per-term gamma as (R, 3, 3)
    gamma_terms = np.zeros((R, n, n), dtype=np.float64)
    for k in range(R):
        gamma_terms[k] = gamma_matrix[:, k].reshape(n, n)

    T = build_tensor(n)
    T_hat = reconstruct_tensor(alpha, beta, gamma_terms, n)
    max_err = np.max(np.abs(T.astype(np.float64) - T_hat))

    return (max_err < 1e-10,
            {'gate1': g1_diag, 'gate2': g2_diag, 'gate3': g3_diag,
             'reconstruction_error': max_err})


def incremental_rank_check(H_partial: np.ndarray, max_rank: int,
                           tol: float = 1e-10) -> bool:
    """Returns True if we should PRUNE (rank exceeds max_rank)."""
    if H_partial.shape[0] == 0:
        return False
    rank, _ = _numerical_rank(H_partial, tol)
    return rank > max_rank


def incremental_rank_lower_bound_check(H_partial: np.ndarray,
                                        terms_placed: int,
                                        R: int, n: int = 3,
                                        tol: float = 1e-10) -> bool:
    """Returns True if we should PRUNE because rank is too LOW.
    After placing k terms, we need at least R-n^2 independent rows total.
    If current rank + remaining terms < target, prune."""
    if H_partial.shape[0] == 0:
        return False
    target = R - n * n
    rank, _ = _numerical_rank(H_partial, tol)
    remaining = R - terms_placed
    # If even adding `remaining` fully independent rows can't reach target
    if rank + remaining < target:
        return True
    return False


# ── Self-test ──

def self_test():
    """Test gate functions."""
    n = 3
    R27 = 27
    alpha27 = np.zeros((R27, n, n), dtype=np.int64)
    beta27 = np.zeros((R27, n, n), dtype=np.int64)
    k = 0
    for r in range(n):
        for s in range(n):
            for u in range(n):
                alpha27[k, r, s] = 1
                beta27[k, s, u] = 1
                k += 1

    blocks = compute_all_blocks(alpha27, beta27, n)

    # Gate 1: R=27 => target_rank_H = 18
    g1_pass, g1_diag = check_gate1(blocks['H'], 18)
    assert g1_pass, f"R=27 Gate1 failed: rank(H)={g1_diag['rank_H']}"

    # Gate 2
    g2_pass, g2_diag = check_gate2(blocks['H'], blocks['Delta'])
    assert g2_pass, f"R=27 Gate2 failed: delta_leak={g2_diag['delta_leak']}"

    # Random R=19: should fail Gate 1 generically
    rng = np.random.default_rng(42)
    alpha19 = rng.choice([-1, 0, 1], size=(19, 3, 3)).astype(np.int64)
    beta19 = rng.choice([-1, 0, 1], size=(19, 3, 3)).astype(np.int64)
    H19 = compute_H(alpha19, beta19, n)
    g1_pass19, g1_diag19 = check_gate1(H19, 10)
    # Generically rank(H) = min(19, 18) = 18, not 10
    assert not g1_pass19, f"Random R=19 should fail Gate1, got rank(H)={g1_diag19['rank_H']}"

    # Incremental rank
    assert not incremental_rank_check(H19[:5], 10)  # 5 rows can't exceed 10
    assert incremental_rank_check(H19[:15], 10)      # 15 rows likely rank > 10

    print("gates.py: all self-tests passed")


if __name__ == "__main__":
    self_test()
