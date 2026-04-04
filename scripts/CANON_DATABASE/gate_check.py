from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _numerical_rank(M: np.ndarray, tol: float = 1e-10) -> tuple[int, np.ndarray]:
    if M.size == 0:
        return 0, np.array([])
    sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    if sv.size == 0:
        return 0, sv
    threshold = tol * sv[0] if sv[0] > 0 else tol
    return int(np.sum(sv > threshold)), sv


def check_gate1(H: np.ndarray, target_rank: int = 10, tol: float = 1e-10) -> tuple[bool, dict]:
    rank_H, sv = _numerical_rank(H, tol)
    eta_nullity = 18 - rank_H
    R = H.shape[0]
    return rank_H == target_rank, {
        "rank_H": rank_H,
        "singular_values": sv,
        "eta_nullity": eta_nullity,
        "conservation": R + eta_nullity,
    }


def check_gate2(H: np.ndarray, Delta: np.ndarray, tol: float = 1e-10) -> tuple[bool, dict]:
    rank_H, _ = _numerical_rank(H, tol)
    rank_N, _ = _numerical_rank(np.hstack([H, Delta]), tol)
    return rank_N == rank_H, {
        "rank_H": rank_H,
        "rank_nuisance": rank_N,
        "delta_leak": rank_N - rank_H,
    }


def check_gate3(
    Sigma: np.ndarray,
    Nuisance: np.ndarray,
    R: int = 19,
    n: int = 3,
    tol: float = 1e-10,
    residual_tol: float = 1e-12,
) -> tuple[bool, dict]:
    nn = n * n
    full = np.hstack([Sigma, Nuisance]).astype(np.float64)
    aug_rk, _ = _numerical_rank(full, tol)
    nuis_rk, _ = _numerical_rank(Nuisance, tol)
    passed = aug_rk == nuis_rk + nn

    gamma = None
    residual = None
    if passed:
        rhs = np.zeros((nn, nn + Nuisance.shape[1]), dtype=np.float64)
        rhs[:, :nn] = 3.0 * np.eye(nn)
        gamma_T, _, _, _ = np.linalg.lstsq(full.T, rhs.T, rcond=None)
        gamma = gamma_T.T
        residual = float(np.max(np.abs(gamma @ full - rhs)))
        if residual > residual_tol:
            passed = False

    return passed, {
        "augmented_rank": aug_rk,
        "nuisance_rank": nuis_rk,
        "gamma": gamma,
        "residual": residual,
    }


def build_multiplication_tensor() -> np.ndarray:
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[r * 3 + s, s * 3 + u, r * 3 + u] = 1
    return T


def reconstruct(alphas: np.ndarray, betas: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    T_hat = np.zeros((9, 9, 9), dtype=np.float64)
    for k in range(alphas.shape[0]):
        T_hat += np.einsum("i,j,l->ijl", alphas[k].ravel(), betas[k].ravel(), Gamma[:, k])
    return T_hat


def verify_solution(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, n: int = 3) -> bool:
    nn = n * n
    T = build_multiplication_tensor().astype(np.float64)
    T_hat = np.zeros((nn, nn, nn), dtype=np.float64)
    for k in range(alpha.shape[0]):
        T_hat += np.einsum("i,j,l->ijl", alpha[k].ravel(), beta[k].ravel(), gamma[k].ravel())
    return float(np.max(np.abs(T - T_hat))) < 1e-10


def full_gate_check(db, indices_19: list[int], config) -> tuple[bool, dict]:
    alphas, betas = db.get_factors_batch(np.asarray(indices_19, dtype=np.int64))
    H = np.asarray(db.H[indices_19], dtype=np.int64)
    sigma = np.asarray(db.sigma[indices_19], dtype=np.int64)
    delta = np.asarray(db.compute_delta_batch(indices_19), dtype=np.int64)
    nuisance = np.hstack([H, delta])

    g1_pass, g1_diag = check_gate1(H.astype(np.float64), target_rank=len(indices_19) - 9, tol=config.rank_tol)
    if not g1_pass:
        return False, {"gate": 1, **g1_diag}

    g2_pass, g2_diag = check_gate2(H.astype(np.float64), delta.astype(np.float64), tol=config.rank_tol)
    if not g2_pass:
        return False, {"gate": 2, **g2_diag}

    g3_pass, g3_diag = check_gate3(
        sigma.astype(np.float64),
        nuisance.astype(np.float64),
        R=len(indices_19),
        n=3,
        tol=config.rank_tol,
        residual_tol=config.gate3_residual_tol,
    )
    if not g3_pass:
        return False, {"gate": 3, **g3_diag}

    Gamma = np.asarray(g3_diag["gamma"], dtype=np.float64)
    residual = float(np.max(np.abs(build_multiplication_tensor().astype(np.float64) - reconstruct(alphas, betas, Gamma))))
    if residual >= config.gate3_residual_tol:
        return False, {"gate": 3, "solve_residual": residual, **g3_diag}

    gamma_terms = np.zeros((len(indices_19), 3, 3), dtype=np.float64)
    for k in range(len(indices_19)):
        gamma_terms[k] = Gamma[:, k].reshape(3, 3)

    return True, {
        "gate": "ALL PASSED",
        "indices": list(indices_19),
        "alpha": alphas,
        "beta": betas,
        "gamma": gamma_terms,
        "gamma_matrix": Gamma,
        "residual": residual,
        "rank_H": int(g1_diag["rank_H"]),
        "rank_N": int(g2_diag["rank_nuisance"]),
        "augmented_rank": int(g3_diag["augmented_rank"]),
    }


def save_solution(solution: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indices": solution["indices"],
        "alpha": np.asarray(solution["alpha"]).tolist(),
        "beta": np.asarray(solution["beta"]).tolist(),
        "gamma": np.asarray(solution["gamma"]).tolist(),
        "residual": float(solution["residual"]),
        "rank_H": int(solution["rank_H"]),
        "rank_N": int(solution["rank_N"]),
        "augmented_rank": int(solution["augmented_rank"]),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def verify_phase2_solution(solution: dict) -> bool:
    return verify_solution(
        np.asarray(solution["alpha"], dtype=np.float64),
        np.asarray(solution["beta"], dtype=np.float64),
        np.asarray(solution["gamma"], dtype=np.float64),
        n=3,
    )