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
) -> tuple[int, float]:
    """
    Returns (gate1_gap, combined_weighted).
    Gate 1 is a hard lexicographic first element.
    combined_weighted = 10*delta_leak + delta_resid + 10*augmented_gap + 0.1*recon_err
    """
    diag = _compute_diagnostics(H, sigma, delta, target_rank_H, rank_tol)
    combined = (
        10.0 * diag['delta_leak']
        + diag['delta_resid']
        + 10.0 * diag['augmented_gap']
        + 0.1  * diag['recon_err']
    )
    return (diag['gate1_gap'], combined)


def score_packet_full(
    H: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    target_rank_H: int,
    rank_tol: float = _RANK_TOL,
) -> dict:
    """Return full diagnostic dict for logging/saving."""
    return _compute_diagnostics(H, sigma, delta, target_rank_H, rank_tol)


def _compute_diagnostics(
    H: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    target_rank_H: int,
    rank_tol: float = _RANK_TOL,
) -> dict:
    H_f = H.astype(np.float64)
    R = H_f.shape[0]

    N  = np.hstack([H_f, delta.astype(np.float64)])
    SN = np.hstack([sigma.astype(np.float64), N])

    rank_H  = _rank(H_f,  rank_tol)
    rank_N  = _rank(N,    rank_tol)
    rank_SN = _rank(SN,   rank_tol)

    gate1_gap     = abs(rank_H - target_rank_H)
    delta_leak    = rank_N - rank_H
    augmented_gap = R - rank_SN
    sigma_innovation = rank_SN - rank_N

    delta_resid = 0.0
    if rank_H > 0:
        U_H, _, _ = np.linalg.svd(H_f, full_matrices=True)
        P = U_H[:, :rank_H]
        delta_proj  = P @ (P.T @ delta.astype(np.float64))
        delta_resid = float(np.linalg.norm(delta.astype(np.float64) - delta_proj, 'fro'))

    target_mat = np.zeros((9, 81), dtype=np.float64)
    target_mat[:, :9] = 3.0 * np.eye(9)
    Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target_mat.T, rcond=None)
    recon_err = float(np.max(np.abs(target_mat.T - SN.T @ Gamma_T)))

    return {
        'gate1_gap': gate1_gap,
        'delta_leak': delta_leak,
        'delta_resid': delta_resid,
        'augmented_gap': augmented_gap,
        'sigma_innovation': sigma_innovation,
        'recon_err': recon_err,
        'rank_H': rank_H,
        'rank_N': rank_N,
        'rank_SN': rank_SN,
    }


def score_is_better(a: tuple, b: tuple) -> bool:
    """Return True if score a is strictly better than b (lexicographic on 2-tuple)."""
    return a < b


def score_is_solution(score: tuple) -> bool:
    """Return True if score represents a perfect solution."""
    g1, combined = score
    return g1 == 0 and combined < 0.1 + 1e-6   # gate1 clean, combined~0 (recon_err<1e-6)
