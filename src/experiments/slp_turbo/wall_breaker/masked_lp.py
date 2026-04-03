"""Masked LP: drop one binding constraint to find tunnels through walls.

Instead of minimizing max|r_i + J_i dx| over ALL entries, we mask out
one binding entry at a time and solve the relaxed LP. If the relaxed
optimum is significantly lower, there's a tunnel: improving 728 entries
at the cost of temporarily worsening 1.
"""

import numpy as np
from scipy.optimize import linprog

from ..config import N_PARAMS, N_ENTRIES


def solve_masked_lp(res_np, J_np, trust, mask_idx, active_k=500):
    """LP minimax with entry `mask_idx` excluded from the objective.

    Returns (dx, t_relaxed) or (None, None).
    """
    # Zero out the masked entry so it doesn't participate
    keep = np.ones(len(res_np), dtype=bool)
    keep[mask_idx] = False
    res_k = res_np[keep]
    J_k = J_np[keep]

    # Active-set selection on the kept entries
    abs_res = np.abs(res_k)
    k = min(active_k, len(res_k))
    active = np.argpartition(abs_res, -k)[-k:]

    J_sub = J_k[active]
    r_sub = res_k[active]
    n_active = len(active)

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


def probe_tunnels(res_np, J_np, trust, binding_idx, active_k=500):
    """Try masking each binding entry and return relaxed optima.

    Parameters
    ----------
    binding_idx : array of entry indices to try masking

    Returns
    -------
    list of (mask_idx, dx, t_relaxed) sorted by t_relaxed ascending.
    Only includes successful solves.
    """
    results = []
    for mi in binding_idx:
        dx, t_rel = solve_masked_lp(res_np, J_np, trust, mi, active_k)
        if dx is not None:
            results.append((int(mi), dx, float(t_rel)))
    results.sort(key=lambda r: r[2])
    return results
