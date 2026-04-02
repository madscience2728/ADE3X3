"""Tier 3: CPU FP64 minimax refinement (V2-style coordinate descent)."""

from typing import Callable

import numpy as np

from .config import RANK, DIM, TARGET_TENSOR, ALGEBRAIC_LOOKUP, CPU_MIN_SWEEPS

# Precompute dead/live masks for weighted L-BFGS objective
_LIVE_MASK_FLAT = (TARGET_TENSOR != 0).ravel()  # (729,) bool
_DEAD_MASK_FLAT = (TARGET_TENSOR == 0).ravel()   # (729,) bool
# Weight dead entries higher: they dominate the bottleneck at the 0.5 basin
_DEAD_WEIGHT = np.ones(729)
_DEAD_WEIGHT[_DEAD_MASK_FLAT] = 10.0  # 10× penalty on dead-entry leakage

# Target tensor, transposed for ALS factor layout
# Our factors are (R, 9), the tensor is T[a,b,c] = T[9*r+s, 9*s'+u, 9*r'+u']
# For ALS: T[a,b,c] = sum_k alpha[k,a] * beta[k,b] * gamma[k,c]
_T = TARGET_TENSOR  # (9, 9, 9)


# ── ALS: Alternating Least Squares initialization ────────────────────

def _als_step_alpha(alpha, beta, gamma):
    """Fix beta, gamma → solve for alpha via least-squares.
    T[a,b,c] = sum_k alpha[k,a] * beta[k,b] * gamma[k,c]
    For each a: T[a,:,:].ravel() = M @ alpha[:,a] where M[b*9+c, k] = beta[k,b]*gamma[k,c]
    """
    R = alpha.shape[0]
    M = np.einsum('kb,kc->bck', beta, gamma).reshape(-1, R)  # (81, R)
    alpha_new = np.zeros_like(alpha)
    for a in range(DIM):
        rhs = _T[a, :, :].ravel()  # (81,)
        alpha_new[:, a], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return alpha_new


def _als_step_beta(alpha, beta, gamma):
    """Fix alpha, gamma → solve for beta."""
    R = alpha.shape[0]
    M = np.einsum('ka,kc->ack', alpha, gamma).reshape(-1, R)  # (81, R)
    beta_new = np.zeros_like(beta)
    for b in range(DIM):
        rhs = _T[:, b, :].ravel()
        beta_new[:, b], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return beta_new


def _als_step_gamma(alpha, beta, gamma):
    """Fix alpha, beta → solve for gamma."""
    R = alpha.shape[0]
    M = np.einsum('ka,kb->abk', alpha, beta).reshape(-1, R)  # (81, R)
    gamma_new = np.zeros_like(gamma)
    for c in range(DIM):
        rhs = _T[:, :, c].ravel()
        gamma_new[:, c], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    return gamma_new


def cpu_als_init(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    max_iters: int = 30,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ALS initialization: converge from random to a Frobenius basin.

    This is the critical missing piece — ALS solves exact least-squares for
    each entire factor matrix per iteration, converging to a basin (typically
    0.5 or 1.0 max-abs) in ~5-10 iterations from random. Without this,
    coordinate descent from random takes exponentially many generations.

    Returns (alpha, beta, gamma, max_abs_fitness).
    """
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()

    R_res = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - _T
    prev_fro = float(np.linalg.norm(R_res))

    for _it in range(max_iters):
        alpha = _als_step_alpha(alpha, beta, gamma)
        beta = _als_step_beta(alpha, beta, gamma)
        gamma = _als_step_gamma(alpha, beta, gamma)

        R_res = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - _T
        fro = float(np.linalg.norm(R_res))

        if abs(fro - prev_fro) < tol:
            break
        prev_fro = fro

    fit = float(np.max(np.abs(R_res)))
    return alpha, beta, gamma, fit


def _nearest_algebraic(val: float, k: int = 8) -> np.ndarray:
    absv = abs(val)
    dists = np.abs(ALGEBRAIC_LOOKUP - absv)
    idx = np.argpartition(dists, min(k, len(dists) - 1))[:k]
    return ALGEBRAIC_LOOKUP[idx[np.argsort(dists[idx])]]


def _gen_trials(val: float, fine_range: float = 0.003, n_nearby: int = 8, fine_steps: int = 5) -> list[float]:
    """Generate trial values for a single coefficient."""
    trials = set()
    sign = 1.0 if val >= 0 else -1.0
    neighbors = _nearest_algebraic(val, k=n_nearby)
    for aval in neighbors:
        trials.add(sign * aval)
        if abs(val) < 0.3:
            trials.add(-sign * aval)
    # Wider continuous grid: ±fine_range with dense sampling
    for delta in np.linspace(-fine_range, fine_range, fine_steps * 2 + 1):
        if delta != 0:
            trials.add(val + delta)
    # Also sample wider: ±3× fine_range for basin escape
    for delta in np.linspace(-fine_range * 3, fine_range * 3, fine_steps):
        if delta != 0:
            trials.add(val + delta)
    for aval in neighbors[:3]:
        for delta in np.linspace(-fine_range / 2, fine_range / 2, fine_steps):
            trials.add(sign * aval + delta)
    trials.discard(val)
    return sorted(trials)


def cpu_minimax_refine(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sweeps: int = 3,
    fine_range: float = 0.003,
    patience: int = 2,
    check_preempt: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """V2-style minimax coordinate descent on CPU in FP64.

    Parameters
    ----------
    check_preempt : optional callable returning True when the worker should
        stop early (e.g. GPU has new work ready).  Only checked after
        ``CPU_MIN_SWEEPS`` sweeps have completed, so we always do at least
        that much work.

    Returns (alpha, beta, gamma, final_fitness).
    """
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    factors = [alpha, beta, gamma]

    # Current residual
    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    best_mx = float(np.max(np.abs(R)))

    stale = 0
    for sweep in range(sweeps):
        # Preemption: after min sweeps, check if caller wants us to stop early
        if check_preempt is not None and sweep >= CPU_MIN_SWEEPS:
            if check_preempt():
                break

        accepts = 0

        # Build shuffled coefficient index
        coeff_idx = []
        for fi in range(3):
            for ri in range(RANK):
                for di in range(DIM):
                    coeff_idx.append((fi, ri, di))

        order = np.random.permutation(len(coeff_idx))

        for ci in order:
            fi, ri, di = coeff_idx[ci]
            old_val = factors[fi][ri, di]

            # Build rank-1 outer product for this term (before change)
            old_outer = np.einsum('a,b,c->abc',
                                  alpha[ri], beta[ri], gamma[ri])

            trials = _gen_trials(old_val, fine_range=fine_range)
            best_trial = None
            best_trial_mx = best_mx

            for tv in trials:
                factors[fi][ri, di] = tv
                new_outer = np.einsum('a,b,c->abc',
                                      alpha[ri], beta[ri], gamma[ri])
                R_new = R - old_outer + new_outer
                new_mx = float(np.max(np.abs(R_new)))
                if new_mx < best_trial_mx - 1e-12:
                    best_trial = tv
                    best_trial_mx = new_mx

            if best_trial is not None:
                factors[fi][ri, di] = best_trial
                new_outer = np.einsum('a,b,c->abc',
                                      alpha[ri], beta[ri], gamma[ri])
                R = R - old_outer + new_outer
                best_mx = best_trial_mx
                accepts += 1
            else:
                factors[fi][ri, di] = old_val

        if accepts == 0:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0

    return alpha, beta, gamma, best_mx


def cpu_pair_refine(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sweeps: int = 2,
    fine_range: float = 0.01,
    n_pairs: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Multi-coordinate pair moves: perturb 2 coupled coefficients simultaneously.

    This breaks the single-coordinate stagnation that coordinate descent suffers from.
    Tries same-term pairs (within-axis and cross-axis) to find improvements
    invisible to single-coordinate search.
    """
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    factors = [alpha, beta, gamma]
    rng = np.random.default_rng()

    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    best_mx = float(np.max(np.abs(R)))

    offsets = np.linspace(-fine_range, fine_range, 7)

    for _sweep in range(sweeps):
        accepts = 0
        for _ in range(n_pairs):
            # Pick a random rank-1 term
            ri = rng.integers(RANK)

            # Cross-axis pair: one from factor fi1, one from fi2, same term
            fi1, fi2 = rng.choice(3, size=2, replace=False)
            di1 = rng.integers(DIM)
            di2 = rng.integers(DIM)

            old_v1 = factors[fi1][ri, di1]
            old_v2 = factors[fi2][ri, di2]
            old_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])

            best_pair_mx = best_mx
            best_d1, best_d2 = None, None

            for d1 in offsets:
                factors[fi1][ri, di1] = old_v1 + d1
                for d2 in offsets:
                    factors[fi2][ri, di2] = old_v2 + d2
                    new_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])
                    R_new = R - old_outer + new_outer
                    new_mx = float(np.max(np.abs(R_new)))
                    if new_mx < best_pair_mx - 1e-12:
                        best_pair_mx = new_mx
                        best_d1, best_d2 = d1, d2

            if best_d1 is not None:
                factors[fi1][ri, di1] = old_v1 + best_d1
                factors[fi2][ri, di2] = old_v2 + best_d2
                new_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])
                R = R - old_outer + new_outer
                best_mx = best_pair_mx
                accepts += 1
            else:
                factors[fi1][ri, di1] = old_v1
                factors[fi2][ri, di2] = old_v2

        if accepts == 0:
            break

    return alpha, beta, gamma, best_mx


def _pack_factors(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([alpha.ravel(), beta.ravel(), gamma.ravel()])


def _unpack_factors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = RANK * DIM
    return x[:n].reshape(RANK, DIM), x[n:2*n].reshape(RANK, DIM), x[2*n:].reshape(RANK, DIM)


def cpu_lbfgs_refine(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    max_iters: int = 300,
    beta_schedule: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0, 1000.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """L-BFGS smooth-max refinement on CPU with analytical gradients.

    Uses log-sum-exp approximation to max|R| with increasing sharpness (beta).
    All objectives provide analytical Jacobians — no numerical finite differences.
    """
    from scipy.optimize import minimize

    n = RANK * DIM  # 171

    x0 = _pack_factors(alpha, beta, gamma)
    best_x = x0.copy()
    best_mx = float(np.max(np.abs(
        np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    )))

    # ── Gradient helpers ─────────────────────────────────────────
    # dR/d(alpha[r,i]) = beta[r,:] ⊗ gamma[r,:] broadcast into (9,9,9)
    # Analytically: dR[a,b,c]/d(alpha[r,i]) = delta(a,i) * beta[r,b] * gamma[r,c]
    # So grad_alpha[r,i] = sum_{b,c} R[i,b,c] * beta[r,b] * gamma[r,c]
    #                     = (R[i,:,:] * (beta[r,:,None] * gamma[r,None,:])).sum()
    # Vectorised: grad_alpha = einsum('abc,rb,rc->ra', R, beta, gamma)

    def _grad_from_R(R, a, b, g):
        """Compute gradient of 0.5*||R||^2 w.r.t. packed [alpha, beta, gamma]."""
        ga = np.einsum('abc,rb,rc->ra', R, b, g, optimize=True)
        gb = np.einsum('abc,ra,rc->rb', R, a, g, optimize=True)
        gg = np.einsum('abc,ra,rb->rc', R, a, b, optimize=True)
        return np.concatenate([ga.ravel(), gb.ravel(), gg.ravel()])

    def max_abs(x):
        a, b, g = _unpack_factors(x)
        return float(np.max(np.abs(
            np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
        )))

    # Phase 1: Frobenius warm-up with analytical gradient
    def fro_obj_and_grad(x):
        a, b, g = _unpack_factors(x)
        R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
        f = 0.5 * np.sum(R ** 2)
        grad = _grad_from_R(R, a, b, g)
        return f, grad

    res = minimize(fro_obj_and_grad, best_x, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iters // 3, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs(res.x)
    if mx < best_mx:
        best_x = res.x.copy()
        best_mx = mx

    # Phase 1b: Dead-weighted Frobenius with analytical gradient
    def dead_weighted_fro_obj_and_grad(x):
        a, b, g = _unpack_factors(x)
        R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
        R_flat = R.ravel()
        wR2 = _DEAD_WEIGHT * R_flat ** 2
        f = 0.5 * np.sum(wR2)
        # weighted residual back into (9,9,9)
        wR = (_DEAD_WEIGHT * R_flat).reshape(DIM, DIM, DIM)
        grad = _grad_from_R(wR, a, b, g)
        return f, grad

    res = minimize(dead_weighted_fro_obj_and_grad, best_x, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iters // 3, 'ftol': 1e-15, 'gtol': 1e-12})
    mx = max_abs(res.x)
    if mx < best_mx:
        best_x = res.x.copy()
        best_mx = mx

    # Phase 2: smooth-max with increasing beta + analytical gradient
    def smooth_max_obj_and_grad(x, beta_val):
        a, b, g = _unpack_factors(x)
        R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
        R_flat = R.ravel()
        absR = np.abs(R_flat)
        mx_val = np.max(absR)
        shifted = beta_val * (absR - mx_val)
        exp_shifted = np.exp(shifted)
        Z = np.sum(exp_shifted)
        f = mx_val + np.log(Z) / beta_val
        # gradient: weights[i] = exp_shifted[i] / Z * sign(R_flat[i])
        signs = np.sign(R_flat)
        weights = (exp_shifted / Z * signs).reshape(DIM, DIM, DIM)
        grad = _grad_from_R(weights, a, b, g)
        return f, grad

    for bv in beta_schedule:
        res = minimize(lambda x: smooth_max_obj_and_grad(x, bv), best_x,
                       method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12})
        mx = max_abs(res.x)
        if mx < best_mx:
            best_x = res.x.copy()
            best_mx = mx

    a_out, b_out, g_out = _unpack_factors(best_x)
    return a_out, b_out, g_out, best_mx


def cpu_algebraic_snap(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Post-processing: try snapping each coefficient to its nearest algebraic
    value one at a time, accepting only improvements.

    Ternary values {-1, 0, 1} are tried FIRST (the only known exact 3×3
    decomposition — AlphaTensor rank-23 — uses exclusively these values).
    Then falls back to the algebraic lookup table.
    """
    alpha = alpha.copy()
    beta = beta.copy()
    gamma = gamma.copy()
    factors = [alpha, beta, gamma]

    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    best_mx = float(np.max(np.abs(R)))

    # Ternary targets — the AlphaTensor rank-23 alphabet
    TERNARY = np.array([-1.0, 0.0, 1.0])

    # Sort coefficients by magnitude (try biggest first — most impactful)
    coeff_list = []
    for fi in range(3):
        for ri in range(RANK):
            for di in range(DIM):
                coeff_list.append((abs(factors[fi][ri, di]), fi, ri, di))
    coeff_list.sort(reverse=True)

    snapped = 0
    for _, fi, ri, di in coeff_list:
        old_val = factors[fi][ri, di]
        sign = 1.0 if old_val >= 0 else -1.0

        old_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])

        best_trial_mx = best_mx
        best_snap = None

        # Phase 1: try ternary {-1, 0, 1} first
        for tv in TERNARY:
            if abs(tv - old_val) < 1e-12:
                continue
            factors[fi][ri, di] = tv
            new_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])
            R_new = R - old_outer + new_outer
            new_mx = float(np.max(np.abs(R_new)))
            if new_mx < best_trial_mx - 1e-12:
                best_trial_mx = new_mx
                best_snap = tv

        # Phase 2: try algebraic lookup (broader table)
        neighbors = _nearest_algebraic(old_val, k=k)
        for aval in neighbors:
            factors[fi][ri, di] = sign * aval
            new_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])
            R_new = R - old_outer + new_outer
            new_mx = float(np.max(np.abs(R_new)))
            if new_mx < best_trial_mx - 1e-12:
                best_trial_mx = new_mx
                best_snap = sign * aval

        if best_snap is not None:
            factors[fi][ri, di] = best_snap
            new_outer = np.einsum('a,b,c->abc', alpha[ri], beta[ri], gamma[ri])
            R = R - old_outer + new_outer
            best_mx = best_trial_mx
            snapped += 1
        else:
            factors[fi][ri, di] = old_val

    return alpha, beta, gamma, best_mx


# ── Tier 3.5: LBFGS threshold ────────────────────────────────────
# With analytical gradients L-BFGS runs in ~1s, so gate generously.
# ALS lands at ~0.5; L-BFGS pushes to ~0.10-0.12. Let it run on anything in the basin.
LBFGS_FITNESS_THRESHOLD = 1.0

# ── Tier 4: SLP minimax threshold ────────────────────────────────
# SLP is expensive (~1-2s per LP iteration) but breaks the L-BFGS basin floor.
# Only run on candidates that have already converged through L-BFGS.
SLP_FITNESS_THRESHOLD = 0.12
SLP_DEFAULT_ITERS = 100
SLP_ACTIVE_K = 500  # number of active constraints in LP subset


from scipy.optimize import linprog


# ── Tier 4: SLP minimax refinement ───────────────────────────────

def _slp_build_jacobian(alpha, beta, gamma):
    """Build full Jacobian (729 x 513) — vectorized over rank."""
    R, D = alpha.shape
    N = D ** 3
    P = 3 * R * D
    J = np.zeros((N, P))
    ai_range = np.arange(D)

    # alpha block: dT[a,b,c]/d alpha[k,a'] = delta(a,a') * beta[k,b] * gamma[k,c]
    bg = np.einsum('kb,kc->kbc', beta, gamma).reshape(R, D * D)
    for ai in range(D):
        J[ai * D * D:(ai + 1) * D * D, ai::D][:, :R] += bg.T

    # beta block: dT[a,b,c]/d beta[k,b'] = alpha[k,a] * delta(b,b') * gamma[k,c]
    ag = np.einsum('ka,kc->kac', alpha, gamma).reshape(R, D * D)
    for bi in range(D):
        idx = (ai_range[:, None] * D * D + bi * D + ai_range[None, :]).ravel()
        J[idx[:, None], R * D + np.arange(R) * D + bi] += ag.T

    # gamma block: dT[a,b,c]/d gamma[k,c'] = alpha[k,a] * beta[k,b] * delta(c,c')
    ab = np.einsum('ka,kb->kab', alpha, beta).reshape(R, D * D)
    for ci in range(D):
        idx = (ai_range[:, None] * D * D + ai_range[None, :] * D + ci).ravel()
        J[idx[:, None], 2 * R * D + np.arange(R) * D + ci] += ab.T

    return J


def _slp_solve_lp(res_flat, J, trust, active_k):
    """Solve LP with active-set: min t s.t. |R + J@dx| <= t, ||dx||_inf <= trust."""
    N, P = J.shape
    abs_res = np.abs(res_flat)

    # Active set: top-K constraints
    k = min(active_k, N)
    active_idx = np.argpartition(abs_res, -k)[-k:]

    J_sub = J[active_idx]
    r_sub = res_flat[active_idx]
    n_act = len(active_idx)

    ones_col = -np.ones((n_act, 1))
    A_ub = np.vstack([
        np.hstack([J_sub, ones_col]),
        np.hstack([-J_sub, ones_col]),
    ])
    b_ub = np.concatenate([-r_sub, r_sub])
    c_obj = np.zeros(P + 1)
    c_obj[P] = 1.0
    bounds = [(-trust, trust)] * P + [(0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'time_limit': 30})
    if not result.success:
        return None, None

    dx = result.x[:P]
    t_pred = result.x[P]

    # Verify active set was sufficient
    new_res_lin = res_flat + J @ dx
    if np.max(np.abs(new_res_lin)) > t_pred * 1.01:
        # Fallback to full LP
        ones_full = -np.ones((N, 1))
        A_ub = np.vstack([np.hstack([J, ones_full]), np.hstack([-J, ones_full])])
        b_ub = np.concatenate([-res_flat, res_flat])
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs', options={'presolve': True, 'time_limit': 60})
        if not result.success:
            return None, None
        dx = result.x[:P]
        t_pred = result.x[P]

    return dx, t_pred


def cpu_slp_refine(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    max_iters: int = SLP_DEFAULT_ITERS,
    trust_init: float = 0.003,
    trust_max: float = 0.5,
    trust_min: float = 1e-8,
    eta_accept: float = 0.01,
    active_k: int = SLP_ACTIVE_K,
    check_preempt: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SLP minimax refinement: iterative LP targeting true minimax objective.

    Breaks through the L-BFGS basin floor by directly optimizing max|residual|
    without smooth-max approximation. Each iteration builds a Jacobian, solves
    a linear program, and takes a trust-region-managed step.
    """
    R, D = alpha.shape
    P = 3 * R * D

    def _pack(a, b, g):
        return np.concatenate([a.ravel(), b.ravel(), g.ravel()])

    def _unpack(x):
        return (x[:R*D].reshape(R, D),
                x[R*D:2*R*D].reshape(R, D),
                x[2*R*D:].reshape(R, D))

    def _residual(x):
        a, b, g = _unpack(x)
        return (np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - _T).ravel()

    x = _pack(alpha, beta, gamma)
    current_fit = float(np.max(np.abs(_residual(x))))
    best_x = x.copy()
    best_fit = current_fit

    trust = trust_init

    for it in range(max_iters):
        if check_preempt and check_preempt():
            break

        res = _residual(x)
        a, b, g = _unpack(x)
        J = _slp_build_jacobian(a, b, g)

        dx, t_pred = _slp_solve_lp(res, J, trust, active_k)
        if dx is None:
            trust = max(trust / 4, trust_min)
            if trust <= trust_min:
                break
            continue

        pred_improvement = current_fit - t_pred
        if pred_improvement < 1e-14:
            break

        x_new = x + dx
        new_fit = float(np.max(np.abs(_residual(x_new))))
        actual_improvement = current_fit - new_fit
        rho = actual_improvement / pred_improvement

        if rho >= eta_accept and actual_improvement > 0:
            x = x_new
            current_fit = new_fit
            if new_fit < best_fit:
                best_x = x.copy()
                best_fit = new_fit
            if rho > 0.75:
                trust = min(trust * 2, trust_max)
            elif rho > 0.5:
                trust = min(trust * 1.5, trust_max)
        else:
            trust = max(trust / 2, trust_min)

        if trust < trust_min:
            break

    a_out, b_out, g_out = _unpack(best_x)
    return a_out, b_out, g_out, best_fit


def cpu_full_refine(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sweeps: int = 3,
    fine_range: float = 0.003,
    patience: int = 5,
    check_preempt: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Full CPU refinement pipeline: ALS → minimax → pair moves → L-BFGS → algebraic snap.

    Chains all available local search stages for maximum improvement.
    ALS runs first to snap random candidates to a Frobenius basin (~0.5 max-abs).
    L-BFGS and algebraic snap only run on sufficiently good candidates.
    """
    # Stage 0: ALS initialization (snap to basin — the critical cold-start step)
    # Only run on candidates that are far from converged (saves time on good ones)
    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    init_fit = float(np.max(np.abs(R)))
    if init_fit > 1.0:
        a, b, g, fit = cpu_als_init(alpha, beta, gamma, max_iters=30)
    else:
        a, b, g, fit = alpha.copy(), beta.copy(), gamma.copy(), init_fit

    # Stage 1: coordinate descent (always)
    a, b, g, fit = cpu_minimax_refine(
        a, b, g, sweeps=sweeps, fine_range=fine_range,
        patience=patience, check_preempt=check_preempt,
    )

    # Stage 2: pair moves (always — fast, catches 2D improvements)
    a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=2, fine_range=fine_range * 3, n_pairs=40)

    # Stage 3: L-BFGS (only for promising candidates)
    if fit < LBFGS_FITNESS_THRESHOLD:
        a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=200)
        # Stage 3b: another round of coord descent to polish L-BFGS output
        a, b, g, fit = cpu_minimax_refine(a, b, g, sweeps=2, fine_range=fine_range)

    # Stage 3c: pair moves after L-BFGS (escape basin) + L-BFGS polish
    if fit < LBFGS_FITNESS_THRESHOLD:
        pre_pair = fit
        a, b, g, fit = cpu_pair_refine(a, b, g, sweeps=2, fine_range=fine_range * 3, n_pairs=40)
        if fit < pre_pair - 1e-8:
            # Pair moves escaped basin — re-run L-BFGS to polish
            a, b, g, fit = cpu_lbfgs_refine(a, b, g, max_iters=100)

    # Stage 4+5: Fork — snap and SLP are competing strategies.
    # Snap discretizes coefficients (good for exact solutions, destroys SLP gradients).
    # SLP needs smooth continuous values. Run both from pre-snap state, keep best.
    if fit < LBFGS_FITNESS_THRESHOLD:
        # Save pre-snap state for SLP branch
        a_pre, b_pre, g_pre, fit_pre = a.copy(), b.copy(), g.copy(), fit

        # Branch A: algebraic snap
        a_snap, b_snap, g_snap, fit_snap = cpu_algebraic_snap(a, b, g, k=4)

        # Branch B: SLP minimax (from pre-snap continuous values)
        if fit_pre < SLP_FITNESS_THRESHOLD:
            a_slp, b_slp, g_slp, fit_slp = cpu_slp_refine(
                a_pre, b_pre, g_pre, max_iters=SLP_DEFAULT_ITERS,
                trust_init=0.003, active_k=SLP_ACTIVE_K,
                check_preempt=check_preempt,
            )
        else:
            a_slp, b_slp, g_slp, fit_slp = a_pre, b_pre, g_pre, fit_pre

        # Pick the winner
        if fit_slp < fit_snap:
            a, b, g, fit = a_slp, b_slp, g_slp, fit_slp
        else:
            a, b, g, fit = a_snap, b_snap, g_snap, fit_snap

    return a, b, g, fit
