"""CPU SLP engine — continuous LP-based exact minimax descent.

Runs in a daemon thread. Each cycle:
  1. Snapshot alive candidates from pool
  2. Compute Jacobians (vectorized numpy)
  3. Solve LPs in parallel (ProcessPoolExecutor, 22 processes)
     — with wall-breaker strategy assignment per worker
  4. Apply accepted steps back to pool
"""

import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linprog

from .config import R, D, N_ENTRIES, N_PARAMS, T_NP, TIER_WEIGHTS
from .wall_breaker.strategy import Strategy, assign_strategies
from .wall_breaker.binding import binding_entries, compute_residuals
from .wall_breaker.masked_lp import solve_masked_lp
from .wall_breaker.coord_rotate import random_rotation, rotate_factors, unrotate_factors


# ── Build tier weights once (will be overridden by args in CPUEngine) ──
_DEFAULT_WEIGHTS = TIER_WEIGHTS.copy()


# ── Top-level LP solver (must be picklable for multiprocessing) ──────

def _solve_lp_worker(args_tuple):
    """Solve one active-set LP. Called in child process.

    args_tuple: (res_np, J_np, trust, active_k, strategy, extra)
    strategy: int (Strategy enum value)
    extra: dict with strategy-specific data (e.g. mask_idx, Q matrix)
    """
    res_np, J_np, trust, active_k, strategy, extra = args_tuple

    if strategy == Strategy.MASK_WORST:
        # Mask the single worst binding entry
        mask_idx = extra.get('mask_idx', None)
        if mask_idx is not None:
            dx, t_pred = solve_masked_lp(res_np, J_np, trust, mask_idx, active_k)
            return dx, t_pred
        # Fallback to normal
    elif strategy == Strategy.MASK_RANDOM:
        mask_idx = extra.get('mask_idx', None)
        if mask_idx is not None:
            dx, t_pred = solve_masked_lp(res_np, J_np, trust, mask_idx, active_k)
            return dx, t_pred
    elif strategy == Strategy.ROTATED:
        # Residuals and Jacobian were already computed in rotated frame
        # Solve normal LP, caller will unrotate the dx
        pass

    # Default: normal LP
    return _solve_normal_lp(res_np, J_np, trust, active_k)


def _solve_normal_lp(res_np, J_np, trust, active_k):
    """Standard active-set LP solve."""
    abs_res = np.abs(res_np)
    active_idx = np.argpartition(abs_res, -active_k)[-active_k:]

    dx, t_pred = _solve_lp_core(res_np, J_np, trust, active_idx)
    if dx is None:
        return None, None

    # Verify active set captured the true max
    new_res_lin = res_np + J_np @ dx
    if np.max(np.abs(new_res_lin)) <= t_pred * 1.001:
        return dx, t_pred

    # Fallback to full LP
    return _solve_lp_core(res_np, J_np, trust, np.arange(len(res_np)))


def _solve_lp_core(res_np, J_np, trust, idx):
    n_active = len(idx)
    J_sub = J_np[idx]
    r_sub = res_np[idx]

    ones_col = -np.ones((n_active, 1))
    A_ub = np.vstack([
        np.hstack([J_sub, ones_col]),
        np.hstack([-J_sub, ones_col])
    ])
    b_ub = np.concatenate([-r_sub, r_sub])

    c_obj = np.zeros(N_PARAMS + 1)
    c_obj[N_PARAMS] = 1.0
    bounds = [(-trust, trust)] * N_PARAMS + [(0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'time_limit': 30})
    if result.success:
        return result.x[:N_PARAMS], result.x[N_PARAMS]
    return None, None


# ── Jacobian (vectorized numpy, ~3ms) ────────────────────────────────

def _build_jacobian(x):
    """Fully vectorized Jacobian (729 × 513)."""
    a = x[:R*D].reshape(R, D)
    b = x[R*D:2*R*D].reshape(R, D)
    g = x[2*R*D:].reshape(R, D)

    J = np.zeros((N_ENTRIES, N_PARAMS))
    ai_range = np.arange(D)

    bg = np.einsum('kb,kc->kbc', b, g).reshape(R, D*D)
    for ai in range(D):
        J[ai*D*D:(ai+1)*D*D, ai::D][:, :R] += bg.T

    ag = np.einsum('ka,kc->kac', a, g).reshape(R, D*D)
    for bi in range(D):
        idx = (ai_range[:, None] * D * D + bi * D + ai_range[None, :]).ravel()
        J[idx[:, None], R*D + np.arange(R) * D + bi] += ag.T

    ab = np.einsum('ka,kb->kab', a, b).reshape(R, D*D)
    for ci in range(D):
        idx = (ai_range[:, None] * D * D + ai_range[None, :] * D + ci).ravel()
        J[idx[:, None], 2*R*D + np.arange(R) * D + ci] += ab.T

    return J


def _residual(x):
    a = x[:R*D].reshape(R, D)
    b = x[R*D:2*R*D].reshape(R, D)
    g = x[2*R*D:].reshape(R, D)
    return (np.einsum('ra,rb,rc->abc', a, b, g) - T_NP).ravel()


def _maxabs(x):
    return float(np.max(np.abs(_residual(x))))


class CPUEngine:
    """Continuous CPU SLP descent running in a daemon thread."""

    def __init__(self, pool, args):
        self.pool = pool
        self.args = args

        self._stop = threading.Event()
        self._thread = None
        self._lp_pool = None

        # Build tier weights from args
        from .config import LIVE_IDX, DEAD_IDX
        self._weights = np.ones(N_ENTRIES, dtype=np.float64)
        self._weights[LIVE_IDX] = args.live_weight
        self._weights[DEAD_IDX] = args.dead_weight

        # Stats (read by display)
        self.iterations = 0
        self.last_cycle_time = 0.0
        self.last_lp_time = 0.0
        self.state = "idle"

    def start(self):
        n_procs = min(self.pool.n, 22)
        self._lp_pool = ProcessPoolExecutor(max_workers=n_procs)
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="cpu-engine")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)
        if self._lp_pool:
            self._lp_pool.shutdown(wait=False)

    def _run_loop(self):
        """Continuous loop: snapshot -> jacobian -> LP -> apply -> repeat."""
        while not self._stop.is_set():
            t0 = time.time()

            # 1. Snapshot alive candidates
            self.state = "snapshot"
            alive_idx, xs_snap, fits_snap, trusts_snap = self.pool.get_alive_snapshot()
            M = len(alive_idx)
            if M == 0:
                self.state = "no alive workers"
                self._stop.wait(1.0)
                continue

            # 2. Assign wall-breaker strategies
            strategies = assign_strategies(M, self.iterations)

            # Find binding entries on the current global best for masked strategies
            best_x = self.pool.global_best_x
            if best_x is not None:
                bind_idx, bind_res, _ = binding_entries(best_x, top_k=10, tol=0.95)
            else:
                bind_idx = np.array([], dtype=int)

            # Pre-generate rotation matrices for ROTATED workers
            rotations = {}  # worker_i -> Q
            for i in range(M):
                if strategies[i] == Strategy.ROTATED:
                    rotations[i] = random_rotation()

            # 3. Compute residuals + Jacobians
            #    For ROTATED workers, compute in rotated frame
            self.state = f"jacobian ({M} workers)"
            residuals = np.empty((M, N_ENTRIES))
            jacobians = np.empty((M, N_ENTRIES, N_PARAMS))

            for i in range(M):
                if strategies[i] == Strategy.ROTATED and i in rotations:
                    x_rot = rotate_factors(xs_snap[i], rotations[i])
                    residuals[i] = _residual(x_rot)
                    jacobians[i] = _build_jacobian(x_rot)
                else:
                    residuals[i] = _residual(xs_snap[i])
                    jacobians[i] = _build_jacobian(xs_snap[i])

            # Apply tier weights
            residuals *= self._weights[np.newaxis, :]
            jacobians *= self._weights[np.newaxis, :, np.newaxis]

            # 4. Parallel LP solves with strategy-specific args
            self.state = f"LP solve ({M} workers)"
            lp_start = time.time()
            lp_args = []
            for i in range(M):
                strat = strategies[i]
                extra = {}
                if strat == Strategy.MASK_WORST and len(bind_idx) > 0:
                    extra['mask_idx'] = int(bind_idx[0])  # worst binding entry
                elif strat == Strategy.MASK_RANDOM and len(bind_idx) > 1:
                    extra['mask_idx'] = int(np.random.choice(bind_idx))
                lp_args.append((residuals[i], jacobians[i], trusts_snap[i],
                                self.args.active_k, int(strat), extra))
            lp_results = list(self._lp_pool.map(_solve_lp_worker, lp_args))
            self.last_lp_time = time.time() - lp_start

            # 5. Evaluate steps and build result arrays
            self.state = "applying"
            new_xs = np.zeros_like(xs_snap)
            new_fits = np.zeros(M)
            accepted = np.zeros(M, dtype=bool)
            trust_updates = trusts_snap.copy()

            for i in range(M):
                wi = alive_idx[i]
                dx, t_pred = lp_results[i]

                if dx is None:
                    trust_updates[i] = max(trusts_snap[i] / 4, self.args.trust_min)
                    continue

                pred_imp = fits_snap[i] - t_pred
                if pred_imp < 1e-14:
                    trust_updates[i] = self.args.trust_min
                    continue

                # For ROTATED workers, unrotate the dx back to original frame
                if strategies[i] == Strategy.ROTATED and i in rotations:
                    x_rot = rotate_factors(xs_snap[i], rotations[i])
                    x_new_rot = x_rot + dx
                    x_new = unrotate_factors(x_new_rot, rotations[i])
                else:
                    x_new = xs_snap[i] + dx

                new_fit = _maxabs(x_new)
                actual_imp = fits_snap[i] - new_fit
                rho = actual_imp / pred_imp if pred_imp > 1e-30 else 0

                if rho >= self.args.eta_accept and actual_imp > 0:
                    new_xs[i] = x_new
                    new_fits[i] = new_fit
                    accepted[i] = True

                    if rho > 0.75:
                        trust_updates[i] = min(trusts_snap[i] * 2, self.args.trust_max)
                    elif rho > 0.5:
                        trust_updates[i] = min(trusts_snap[i] * 1.5, self.args.trust_max)
                else:
                    trust_updates[i] = max(trusts_snap[i] / 2, self.args.trust_min)

            # 5. Apply back to pool
            self.pool.apply_lp_results(alive_idx, new_xs, new_fits, accepted,
                                       trust_updates, self.args)

            self.iterations += 1
            self.last_cycle_time = time.time() - t0
            self.state = "idle"
