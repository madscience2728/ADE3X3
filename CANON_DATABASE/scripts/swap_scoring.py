"""
swap_scoring.py — Scoring for R=19 packet configurations.

Score tuple (lower = better):
  (gate1_gap, combined_weighted)

Where combined_weighted trades Gate 2 slack for Gate 3 improvement:
  combined = 10 * delta_leak + 1 * delta_resid + 10 * augmented_gap + 0.1 * recon_err

Gate 1 remains a hard constraint (gate1_gap is the first element, lexicographic).
Gate 2 and Gate 3 are weighted equally (both coefficient 10) so the optimizer
can cross delta_leak=1 ridges if augmented_gap improves by 1 or more.

Raw diagnostics are available via score_packet_full() for logging.
"""
from __future__ import annotations

import numpy as np

_RANK_TOL = 1e-9


def _rank(M: np.ndarray, tol: float = _RANK_TOL) -> int:
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    if sv.size == 0:
        return 0
    thresh = tol * (max(M.shape) * sv[0]) if sv[0] > 0 else tol
    return int(np.sum(sv > thresh))


def score_packet(
    H: np.ndarray,       # (R, 18) float64 or int8
    sigma: np.ndarray,   # (R, 9)
    delta: np.ndarray,   # (R, 54)
    target_rank_H: int,
    rank_tol: float = _RANK_TOL,
) -> tuple[int, int, float, int, float]:
    """
    Returns (gate1_gap, delta_leak, delta_resid, augmented_gap, recon_err).
    Lower is better at every position.
    """
    H_f = H.astype(np.float64)
    R = H_f.shape[0]

    N  = np.hstack([H_f, delta.astype(np.float64)])   # (R, 72)
    SN = np.hstack([sigma.astype(np.float64), N])      # (R, 81)

    rank_H  = _rank(H_f,  rank_tol)
    rank_N  = _rank(N,    rank_tol)
    rank_SN = _rank(SN,   rank_tol)

    gate1_gap     = abs(rank_H - target_rank_H)
    delta_leak    = rank_N - rank_H
    augmented_gap = R - rank_SN

    # Continuous delta residual: Frobenius norm of Delta outside col(H)
    delta_resid = 0.0
    if rank_H > 0:
        U_H, _, _ = np.linalg.svd(H_f, full_matrices=True)
        P = U_H[:, :rank_H]
        delta_proj  = P @ (P.T @ delta.astype(np.float64))
        delta_resid = float(np.linalg.norm(delta.astype(np.float64) - delta_proj, 'fro'))

    # Reconstruction error
    target_mat = np.zeros((9, 81), dtype=np.float64)
    target_mat[:, :9] = 3.0 * np.eye(9)
    Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target_mat.T, rcond=None)
    recon_err = float(np.max(np.abs(target_mat.T - SN.T @ Gamma_T)))

    return (gate1_gap, delta_leak, delta_resid, augmented_gap, recon_err)


def score_is_better(a: tuple, b: tuple) -> bool:
    """Return True if score a is strictly lexicographically better than b."""
    return a < b


def score_is_solution(score: tuple) -> bool:
    """Return True if score represents a perfect solution."""
    g1, leak, resid, aug, err = score
    return g1 == 0 and leak == 0 and aug == 0 and err < 1e-6
