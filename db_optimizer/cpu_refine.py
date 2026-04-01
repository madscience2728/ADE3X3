"""Tier 3: CPU FP64 minimax refinement (V2-style coordinate descent)."""

from typing import Callable

import numpy as np

from .config import RANK, DIM, TARGET_TENSOR, ALGEBRAIC_LOOKUP, CPU_MIN_SWEEPS


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
    for delta in np.linspace(-fine_range, fine_range, fine_steps * 2 + 1):
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
