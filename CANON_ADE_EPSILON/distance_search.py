import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

from tensor_core import build_T_matmul, build_G_invariant_E, perturbed_target
from budget import budget_E_norm as _budget_E_norm, budget_frob as _budget_frob, is_loop_closed


def flatten_modes(T):
    """Return 3 mode flattenings of T, each shape (9, 81)."""
    return [
        T.reshape(9, 81),
        T.transpose(1, 0, 2).reshape(9, 81),
        T.transpose(2, 0, 1).reshape(9, 81),
    ]


def rank_proxy_score(T, R):
    """
    Proxy for distance to rank-R variety: min over 3 mode flattenings
    of sigma_{R+1} (0-indexed: sigma[R]).
    """
    best = math.inf
    for mat in flatten_modes(T):
        sv = np.linalg.svd(mat, compute_uv=False)
        idx = min(R, len(sv) - 1)
        best = min(best, float(sv[idx]))
    return best


def E_norm_from_w(w0, w1, w2, w3):
    """
    ||E||_F for a G-invariant E with orbit weights (w0,w1,w2,w3).
    Orbit sizes: O0=1, O1=6, O2=12, O3=8.
    ||E||_F^2 = 1*w0^2 + 6*w1^2 + 12*w2^2 + 8*w3^2
    """
    return math.sqrt(1*w0**2 + 6*w1**2 + 12*w2**2 + 8*w3**2)


# ---------------------------------------------------------------------------
# Top-level worker functions (must be picklable)
# ---------------------------------------------------------------------------

def worker_grid_batch(args):
    """
    args = (batch, R, eps, budget_en)
    batch: list of (w0, w1, w2, w3)
    Prunes points outside budget, returns list of (score, E_norm, w0,w1,w2,w3)
    sorted by score ascending.
    """
    batch, R, eps, budget_en = args
    T_base = build_T_matmul()
    results = []
    for (w0, w1, w2, w3) in batch:
        en = E_norm_from_w(w0, w1, w2, w3)
        if en > budget_en:
            continue
        T_p = T_base - eps * build_G_invariant_E(w0, w1, w2, w3)
        score = rank_proxy_score(T_p, R)
        results.append((score, en, w0, w1, w2, w3))
    results.sort(key=lambda x: x[0])
    return results


def worker_refine(args):
    """
    args = (initial_w, R, eps, budget_en)
    Constrained SLSQP refinement: minimize rank_proxy_score s.t. ||E||_F <= budget_en.
    Returns (score, E_norm, w0, w1, w2, w3).
    """
    initial_w, R, eps, budget_en = args
    T_base = build_T_matmul()

    def objective(w):
        T_p = T_base - eps * build_G_invariant_E(*w)
        return rank_proxy_score(T_p, R)

    constraints = [{'type': 'ineq',
                    'fun': lambda w: budget_en - E_norm_from_w(*w)}]
    bounds = [(-budget_en, budget_en)] * 4

    res = minimize(objective, initial_w, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-10, 'maxiter': 1000})
    w = res.x
    en = E_norm_from_w(*w)
    T_p = T_base - eps * build_G_invariant_E(*w)
    score = rank_proxy_score(T_p, R)
    return (score, en, float(w[0]), float(w[1]), float(w[2]), float(w[3]))


# ---------------------------------------------------------------------------
# Main search routines
# ---------------------------------------------------------------------------

def search_rank(R, b=23, M=1.0):
    eps = 2.0 ** (-b)
    budget_en = _budget_E_norm(R, b, M)
    budget_fr = _budget_frob(R, b, M)

    # Phase 1: grid search, w in [-8,8], 17 points per axis
    vals = np.linspace(-8.0, 8.0, 17)
    grid = [(w0, w1, w2, w3)
            for w0 in vals for w1 in vals
            for w2 in vals for w3 in vals]

    max_workers = 24
    batch_size = math.ceil(len(grid) / max_workers)
    batches = [grid[i:i + batch_size] for i in range(0, len(grid), batch_size)]
    batch_args = [(b_, R, eps, budget_en) for b_ in batches]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        raw = list(pool.map(worker_grid_batch, batch_args))

    all_grid = [item for batch_result in raw for item in batch_result]
    all_grid.sort(key=lambda x: x[0])
    top = all_grid[:100]

    if not top:
        T_base = build_T_matmul()
        score0 = rank_proxy_score(T_base, R)
        return {
            'R': R, 'best_score': score0, 'best_w': [0.0, 0.0, 0.0, 0.0],
            'E_norm': 0.0, 'budget_E_norm': budget_en, 'budget_frob': budget_fr,
            'loop_closed': True, 'absolute_perturbation': 0.0,
            'T_perturbed': T_base,
        }

    # Phase 2: refine top 100 with SLSQP in parallel
    refine_args = [((t[2], t[3], t[4], t[5]), R, eps, budget_en) for t in top]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        refined = list(pool.map(worker_refine, refine_args))

    refined.sort(key=lambda x: x[0])
    best = refined[0]
    best_score, best_en, w0, w1, w2, w3 = best
    best_w = [w0, w1, w2, w3]

    T_base = build_T_matmul()
    T_perturbed = T_base - eps * build_G_invariant_E(*best_w)
    closed = best_en <= budget_en

    return {
        'R': R,
        'best_score': best_score,
        'best_w': best_w,
        'E_norm': best_en,
        'budget_E_norm': budget_en,
        'budget_frob': budget_fr,
        'loop_closed': closed,
        'absolute_perturbation': eps * best_en,
        'T_perturbed': T_perturbed,
    }


def run_all(R_list, b=23, M=1.0):
    return [search_rank(R, b=b, M=M) for R in R_list]
