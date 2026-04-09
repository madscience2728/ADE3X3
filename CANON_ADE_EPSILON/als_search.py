import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

from tensor_core import build_T_matmul
from budget import budget_E_norm as _budget_E_norm, budget_frob as _budget_frob
from irrep_decomp import build_E_from_coords, E_norm_from_coords, build_basis_vectors


# ---------------------------------------------------------------------------
# Core ALS primitives
# ---------------------------------------------------------------------------

def khatri_rao(P, Q):
    """
    Khatri-Rao (column-wise Kronecker) product.
    P, Q: shape (9, R). Returns shape (81, R).
    Column t: np.kron(P[:,t], Q[:,t]).
    """
    R = P.shape[1]
    out = np.empty((P.shape[0] * Q.shape[0], R), dtype=np.float64)
    for t in range(R):
        out[:, t] = np.kron(P[:, t], Q[:, t])
    return out


def als_step(T_p, A, B, C, R):
    """
    One full ALS sweep: update A, then B, then C.
    Returns updated (A, B, C, residual).
    """
    T0 = T_p.reshape(9, 81)           # mode-0 unfolding
    T1 = T_p.transpose(1, 0, 2).reshape(9, 81)  # mode-1
    T2 = T_p.transpose(2, 0, 1).reshape(9, 81)  # mode-2

    # Update A: solve A @ KR(C,B)^T = T0
    kr_CB = khatri_rao(C, B)          # (81, R)
    A, _, _, _ = np.linalg.lstsq(kr_CB, T0.T, rcond=None)
    A = A.T  # (9, R)

    # Update B: solve B @ KR(C,A)^T = T1
    kr_CA = khatri_rao(C, A)
    B, _, _, _ = np.linalg.lstsq(kr_CA, T1.T, rcond=None)
    B = B.T

    # Update C: solve C @ KR(B,A)^T = T2
    kr_BA = khatri_rao(B, A)
    C, _, _, _ = np.linalg.lstsq(kr_BA, T2.T, rcond=None)
    C = C.T

    # Compute residual
    recon = np.einsum('ir,jr,kr->ijk', A, B, C)
    residual = float(np.linalg.norm(T_p - recon))
    return A, B, C, residual


def als_decompose(T_p, R, n_restarts=10, max_iter=500, tol=1e-12):
    """
    ALS CP decomposition of T_p at rank R with n_restarts.
    Returns dict: best_residual, best_A, best_B, best_C, n_iter, all_residuals.
    """
    rng = np.random.default_rng()
    best_residual = math.inf
    best_A = best_B = best_C = None
    best_n_iter = 0
    all_residuals = []

    for _ in range(n_restarts):
        A = rng.standard_normal((9, R))
        B = rng.standard_normal((9, R))
        C = rng.standard_normal((9, R))
        # Normalize columns
        for M in (A, B, C):
            norms = np.linalg.norm(M, axis=0, keepdims=True)
            norms[norms < 1e-14] = 1.0
            M /= norms

        prev_res = math.inf
        n_iter = 0
        for it in range(max_iter):
            A, B, C, res = als_step(T_p, A, B, C, R)
            n_iter = it + 1
            if abs(prev_res - res) < tol:
                break
            prev_res = res

        all_residuals.append(res)
        if res < best_residual:
            best_residual = res
            best_A, best_B, best_C = A.copy(), B.copy(), C.copy()
            best_n_iter = n_iter

    return {
        'best_residual': best_residual,
        'best_A': best_A,
        'best_B': best_B,
        'best_C': best_C,
        'n_iter': best_n_iter,
        'all_residuals': all_residuals,
    }


def anti_isometric_score(T_p):
    """
    Measures how far T_p's mode flattenings are from isometric.
    Returns max over 3 modes of std(sv)/mean(sv).
    T_matmul has score = 0.0 (all svs equal).
    """
    flattenings = [
        T_p.reshape(9, 81),
        T_p.transpose(1, 0, 2).reshape(9, 81),
        T_p.transpose(2, 0, 1).reshape(9, 81),
    ]
    best = 0.0
    for mat in flattenings:
        sv = np.linalg.svd(mat, compute_uv=False)
        mn = float(np.mean(sv))
        if mn < 1e-15:
            continue
        score = float(np.std(sv)) / mn
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Top-level worker functions (picklable)
# ---------------------------------------------------------------------------

def worker_als_batch(args):
    """
    args = (batch_of_coord_arrays, R, eps, b, M)
    Returns list of result dicts sorted by best_residual ascending.
    """
    batch, R, eps, b, M = args
    T_base = build_T_matmul()
    budget_en = _budget_E_norm(R, b, M)
    budget_fr = _budget_frob(R, b, M)
    results = []
    for coords in batch:
        coords = np.asarray(coords, dtype=np.float64)
        E_norm = E_norm_from_coords(coords)
        if E_norm > budget_en:
            continue
        E = build_E_from_coords(coords)
        T_p = T_base - eps * E
        anti_score = anti_isometric_score(T_p)
        result = als_decompose(T_p, R, n_restarts=10, max_iter=500)
        results.append({
            'coords': coords.tolist(),
            'E_norm': E_norm,
            'anti_score': anti_score,
            'best_residual': result['best_residual'],
            'budget_frob': budget_fr,
            'success': result['best_residual'] < budget_fr,
        })
    results.sort(key=lambda x: x['best_residual'])
    return results


def worker_refine_als(args):
    """
    args = (initial_coords, R, eps, b, M)
    SLSQP refinement minimizing ALS residual within budget.
    Returns result dict.
    """
    initial_coords, R, eps, b, M = args
    T_base = build_T_matmul()
    budget_en = _budget_E_norm(R, b, M)
    budget_fr = _budget_frob(R, b, M)

    def objective(coords):
        E = build_E_from_coords(coords)
        T_p = T_base - eps * E
        result = als_decompose(T_p, R, n_restarts=3, max_iter=200)
        return result['best_residual']

    constraints = [{'type': 'ineq',
                    'fun': lambda w: budget_en - E_norm_from_coords(w)}]
    bounds = [(-budget_en, budget_en)] * 15

    res = minimize(objective, initial_coords, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-8, 'maxiter': 50})

    coords = np.asarray(res.x)
    E = build_E_from_coords(coords)
    E_norm = float(np.linalg.norm(E))
    T_p = T_base - eps * E
    anti_score = anti_isometric_score(T_p)
    final = als_decompose(T_p, R, n_restarts=10, max_iter=500)
    return {
        'coords': coords.tolist(),
        'E_norm': E_norm,
        'anti_score': anti_score,
        'best_residual': final['best_residual'],
        'budget_frob': budget_fr,
        'success': final['best_residual'] < budget_fr,
    }


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def search_E_for_rank(R, b=23, M=1.0, n_coords=2000):
    eps = 2.0 ** (-b)
    budget_en = _budget_E_norm(R, b, M)
    budget_fr = _budget_frob(R, b, M)

    # Baseline: ALS on unperturbed T_matmul
    T_base = build_T_matmul()
    baseline = als_decompose(T_base, R, n_restarts=10, max_iter=500)
    baseline_residual = baseline['best_residual']

    # Phase 1: sample n_coords coord vectors on/in the budget ball
    rng = np.random.default_rng(0)
    coords_list = []
    dim = 15
    for _ in range(n_coords):
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)                         # unit vector
        # Scale: draw radius uniformly over volume (u^{1/dim} trick)
        u = rng.uniform(0.0, 1.0)
        r_scale = budget_en * (u ** (1.0 / dim))
        coords_list.append(v * r_scale)

    max_workers = 24
    batch_size = math.ceil(n_coords / max_workers)
    batches = [coords_list[i:i + batch_size]
               for i in range(0, n_coords, batch_size)]
    batch_args = [([c.tolist() for c in b_], R, eps, b, M)
                  for b_ in batches]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        raw = list(pool.map(worker_als_batch, batch_args))

    all_results = [item for batch_result in raw for item in batch_result]
    all_results.sort(key=lambda x: x['best_residual'])
    top20 = all_results[:20]

    if not top20:
        return {
            'R': R, 'best_residual': baseline_residual,
            'best_coords': [0.0] * 15, 'best_E_norm': 0.0,
            'budget_frob': budget_fr, 'budget_E_norm': budget_en,
            'success': False, 'anti_score_at_best': 0.0,
            'baseline_residual': baseline_residual,
        }

    # Phase 2: refine top 20 with SLSQP in parallel
    refine_args = [(np.array(t['coords']), R, eps, b, M) for t in top20]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        refined = list(pool.map(worker_refine_als, refine_args))

    refined.sort(key=lambda x: x['best_residual'])
    best = refined[0]

    return {
        'R': R,
        'best_residual': best['best_residual'],
        'best_coords': best['coords'],
        'best_E_norm': best['E_norm'],
        'budget_frob': budget_fr,
        'budget_E_norm': budget_en,
        'success': best['success'],
        'anti_score_at_best': best['anti_score'],
        'baseline_residual': baseline_residual,
    }


def run_all_ranks(R_list, b=23, M=1.0):
    return [search_E_for_rank(R, b=b, M=M) for R in R_list]
