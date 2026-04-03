"""Alternating L-infinity solver for tensor decomposition.

For fixed (β, γ), the residual T[a,b,c] - Σ_k α[k,a]·β[k,b]·γ[k,c] is
LINEAR in α. So minimizing max|residual| over α is a linear program.
Same for β with (α, γ) fixed, and γ with (α, β) fixed.

Cycling through all three factors, each step is exact-optimal and
monotonically non-increasing in L∞. No gradients, no surrogates.
27 tiny LPs per cycle (9 per factor, ~20 vars × 162 constraints each).
"""

import numpy as np
from scipy.optimize import linprog


def build_target_tensor() -> np.ndarray:
    n = 3
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                T[n * r + s, n * s + u, n * r + u] = 1.0
    return T


T = build_target_tensor()


def compute_fitness(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> float:
    R_tensor = np.einsum('ka,kb,kc->abc', alpha, beta, gamma) - T
    return float(np.max(np.abs(R_tensor)))


# ── Per-column LP solvers ────────────────────────────────────────────

def _solve_alpha_col(a: int, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray | None:
    """Solve for α[:,a] given fixed β, γ via LP. Returns R-vector or None."""
    R = beta.shape[0]
    # M[bc, k] = β[k,b] · γ[k,c]  where bc = 9b + c
    M = np.einsum('kb,kc->bck', beta, gamma).reshape(-1, R)  # (81, R)
    target = T[a].ravel()  # (81,)

    # Variables: [α[0,a], ..., α[R-1,a], s]
    # Minimize s subject to:
    #   M @ x - s ≤ target       (residual ≤ s)
    #  -M @ x - s ≤ -target      (-residual ≤ s)
    c_obj = np.zeros(R + 1)
    c_obj[-1] = 1.0

    ones = np.ones((81, 1))
    A_ub = np.vstack([
        np.hstack([M, -ones]),
        np.hstack([-M, -ones]),
    ])
    b_ub = np.concatenate([target, -target])
    bounds = [(None, None)] * R + [(0, None)]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method='highs',
                  options={'presolve': True, 'time_limit': 10.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if res.success:
        return res.x[:R]
    return None


def _solve_beta_col(b: int, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray | None:
    """Solve for β[:,b] given fixed α, γ via LP."""
    R = alpha.shape[0]
    M = np.einsum('ka,kc->ack', alpha, gamma).reshape(-1, R)  # (81, R)
    target = T[:, b, :].ravel()

    c_obj = np.zeros(R + 1)
    c_obj[-1] = 1.0

    ones = np.ones((81, 1))
    A_ub = np.vstack([
        np.hstack([M, -ones]),
        np.hstack([-M, -ones]),
    ])
    b_ub = np.concatenate([target, -target])
    bounds = [(None, None)] * R + [(0, None)]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method='highs',
                  options={'presolve': True, 'time_limit': 10.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if res.success:
        return res.x[:R]
    return None


def _solve_gamma_col(c: int, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray | None:
    """Solve for γ[:,c] given fixed α, β via LP."""
    R = alpha.shape[0]
    M = np.einsum('ka,kb->abk', alpha, beta).reshape(-1, R)  # (81, R)
    target = T[:, :, c].ravel()

    c_obj = np.zeros(R + 1)
    c_obj[-1] = 1.0

    ones = np.ones((81, 1))
    A_ub = np.vstack([
        np.hstack([M, -ones]),
        np.hstack([-M, -ones]),
    ])
    b_ub = np.concatenate([target, -target])
    bounds = [(None, None)] * R + [(0, None)]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method='highs',
                  options={'presolve': True, 'time_limit': 10.0,
                           'dual_feasibility_tolerance': 1e-10,
                           'primal_feasibility_tolerance': 1e-10})
    if res.success:
        return res.x[:R]
    return None


# ── Main alternating solver ──────────────────────────────────────────

def alternating_linf(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    max_cycles: int = 500,
    tol: float = 1e-14,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float]]:
    """Alternating L∞: cycle exact LP solves through α, β, γ.

    Each cycle solves 27 tiny LPs (9 per factor).
    Fitness is monotonically non-increasing.

    Returns (alpha, beta, gamma, final_fitness, history).
    """
    alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    R = alpha.shape[0]

    fit = compute_fitness(alpha, beta, gamma)
    history = [fit]
    best_fit = fit
    best_alpha, best_beta, best_gamma = alpha.copy(), beta.copy(), gamma.copy()

    for cycle in range(max_cycles):
        # ── Solve for α (9 independent LPs) ──
        for a in range(9):
            x = _solve_alpha_col(a, beta, gamma)
            if x is not None:
                alpha[:, a] = x

        # ── Solve for β (9 independent LPs) ──
        for b in range(9):
            x = _solve_beta_col(b, alpha, gamma)
            if x is not None:
                beta[:, b] = x

        # ── Solve for γ (9 independent LPs) ──
        for c in range(9):
            x = _solve_gamma_col(c, alpha, beta)
            if x is not None:
                gamma[:, c] = x

        fit = compute_fitness(alpha, beta, gamma)
        history.append(fit)

        if fit < best_fit:
            best_fit = fit
            best_alpha, best_beta, best_gamma = alpha.copy(), beta.copy(), gamma.copy()

        if verbose and (cycle % 10 == 0 or cycle < 5):
            print(f"  cycle {cycle:4d}: fitness = {fit:.14f} (best = {best_fit:.14f})")

        # Convergence check
        if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
            if verbose:
                print(f"  converged at cycle {cycle}: Δ = {abs(history[-1] - history[-2]):.2e}")
            break

    return best_alpha, best_beta, best_gamma, best_fit, history


def als_init(
    R: int,
    rng: np.random.Generator,
    max_iters: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ALS initialization: random → Frobenius-converged starting point."""
    alpha = rng.standard_normal((R, 9))
    beta = rng.standard_normal((R, 9))
    gamma = rng.standard_normal((R, 9))

    for _ in range(max_iters):
        # Solve for α via least-squares
        M = np.einsum('kb,kc->bck', beta, gamma).reshape(-1, R)
        for a in range(9):
            rhs = T[a].ravel()
            alpha[:, a], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Solve for β
        M = np.einsum('ka,kc->ack', alpha, gamma).reshape(-1, R)
        for b in range(9):
            rhs = T[:, b, :].ravel()
            beta[:, b], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Solve for γ
        M = np.einsum('ka,kb->abk', alpha, beta).reshape(-1, R)
        for c in range(9):
            rhs = T[:, :, c].ravel()
            gamma[:, c], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

    return alpha, beta, gamma
