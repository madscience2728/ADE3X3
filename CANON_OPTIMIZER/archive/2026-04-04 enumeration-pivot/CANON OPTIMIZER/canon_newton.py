"""
canon_newton.py — The unified Gauss-Newton solver for R=19.

Philosophy:
  No loss function. No fitness. No penalty. No alternation.

  Γ is analytically eliminated (profiled out via 9×9 linear solve).
  K is fixed by irrep structure (one of 526 discrete choices).
  The ONLY equation: F(x) = profiled_tensor_residual = 0.
  The ONLY solver: Levenberg-Marquardt (damped Gauss-Newton) on F.

  This is one unified algorithm that simultaneously:
    • Enforces rank(H) = 10 (K is 10-dim by construction)
    • Enforces Δ ⊂ span(H) (F=0 ⟹ exact reconstruction ⟹ Nuisance ⊂ K)
    • Solves for Γ (G is the profiled lstsq solution on K⊥)
    • Tests solvability (F → 0 iff the kernel supports a solution)

  When F(x) = 0: the decomposition is exact. Residual 0.0 falls out.
  When F(x) ≠ 0 for all x: this kernel is provably dead.

Usage:
  python "CANON OPTIMIZER/canon_newton.py"
  python "CANON OPTIMIZER/canon_newton.py" --kernels 0-50 --restarts 20

Env vars:
  CN_RESTARTS   restarts per kernel (default 10)
  CN_KERNELS    how many kernels to sweep (default all)
  CN_WORKERS    parallel workers (default cpu_count)
"""

from __future__ import annotations

import os
import sys
import time
import json
from itertools import permutations, product as iproduct, combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from scipy.optimize import least_squares

# ════════════════════════════════════════════════════════════════
# GROUP: Z₂ ≀ S₃  (order 48)
# ════════════════════════════════════════════════════════════════

def _swap12(x):
    return x if x == 0 else 3 - x

S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
KEPT = sorted([t for t in ALL27 if 0 in t])
assert len(KEPT) == 19
KEPT_IDX = {t: i for i, t in enumerate(KEPT)}
SUPER_SEEDS = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]


def act_on_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))


def _swap_perm(flag):
    return [0, 2, 1] if flag else [0, 1, 2]


def _act_on_factors(pi, eps, a, b, g):
    ROLE_PAIRS = {(0, 1): 'a', (1, 2): 'b', (0, 2): 'g'}
    facs = {(0, 1): a, (1, 2): b, (0, 2): g}
    role_perms = [_swap_perm(e) for e in eps]
    out = {}
    for new_pair, name in ROLE_PAIRS.items():
        old = (pi[new_pair[0]], pi[new_pair[1]])
        key = (min(old), max(old))
        mat = facs[key].T if old != key else facs[key]
        out[name] = mat[np.ix_(role_perms[new_pair[0]],
                                role_perms[new_pair[1]])].copy()
    return out['a'], out['b'], out['g']


# ════════════════════════════════════════════════════════════════
# PRECOMPUTE: stabilizers, orbits, transporters
# ════════════════════════════════════════════════════════════════

_STAB_CACHE = {}
_TRANSPORT_CACHE = {}

def _precompute():
    kept_set = set(KEPT)
    for seed in SUPER_SEEDS:
        stab = [(pi, eps) for pi, eps in GROUP
                if act_on_triple(pi, eps, seed) == seed]
        _STAB_CACHE[seed] = stab

        transporters = {}
        for pi, eps in GROUP:
            t = act_on_triple(pi, eps, seed)
            if t not in transporters and t in kept_set:
                transporters[t] = (pi, eps)
        _TRANSPORT_CACHE[seed] = transporters

_precompute()


# ════════════════════════════════════════════════════════════════
# SEED EXPANSION: 81 params → 19 terms (α, β only needed)
# ════════════════════════════════════════════════════════════════

def _expand_seeds_slow(params):
    """81 raw seed params → (alpha, beta) as (19, 3, 3) arrays.  [reference impl]"""
    alpha_out = [None] * 19
    beta_out = [None] * 19

    for si, seed in enumerate(SUPER_SEEDS):
        off = si * 27
        a = params[off:off + 9].reshape(3, 3)
        b = params[off + 9:off + 18].reshape(3, 3)
        g = params[off + 18:off + 27].reshape(3, 3)

        stab = _STAB_CACHE[seed]
        aa = np.zeros((3, 3))
        bb = np.zeros((3, 3))
        gg = np.zeros((3, 3))
        for pi, eps in stab:
            a2, b2, g2 = _act_on_factors(pi, eps, a, b, g)
            aa += a2; bb += b2; gg += g2
        n = len(stab)
        aa /= n; bb /= n; gg /= n

        for t, (pi, eps) in _TRANSPORT_CACHE[seed].items():
            at, bt, gt = _act_on_factors(pi, eps, aa, bb, gg)
            k = KEPT_IDX[t]
            alpha_out[k] = at
            beta_out[k] = bt

    return np.array(alpha_out), np.array(beta_out)


def _precompute_expansion_matrix():
    """expand_seeds is LINEAR in params → precompute as matrix multiply.

    Returns M_alpha (19*9, 81), M_beta (19*9, 81) such that:
        alpha.reshape(19,9) = (M_alpha @ params).reshape(19, 9)
        beta.reshape(19,9)  = (M_beta  @ params).reshape(19, 9)
    """
    M_alpha = np.zeros((19 * 9, 81))
    M_beta = np.zeros((19 * 9, 81))
    for j in range(81):
        e = np.zeros(81)
        e[j] = 1.0
        a, b = _expand_seeds_slow(e)
        M_alpha[:, j] = a.ravel()
        M_beta[:, j] = b.ravel()
    return M_alpha, M_beta


_M_ALPHA, _M_BETA = _precompute_expansion_matrix()


def expand_seeds(params):
    """81 raw seed params → (alpha, beta) as (19, 3, 3) arrays.  [fast matrix version]"""
    alpha = (_M_ALPHA @ params).reshape(19, 3, 3)
    beta = (_M_BETA @ params).reshape(19, 3, 3)
    return alpha, beta


# ════════════════════════════════════════════════════════════════
# TARGET TENSOR
# ════════════════════════════════════════════════════════════════

def _build_target():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for c in range(3):
            for s in range(3):
                T[r * 3 + c, r * 3 + s, s * 3 + c] = 1.0
    return T

TARGET = _build_target()
T_FLAT = TARGET.reshape(9, 81)         # (9, 81)  — rows indexed by γ, cols by (α,β)


# ════════════════════════════════════════════════════════════════
# IRREP DECOMPOSITION
# ════════════════════════════════════════════════════════════════

def _build_perm_matrices():
    mats = []
    for pi, eps in GROUP:
        P = np.zeros((19, 19))
        for i, t in enumerate(KEPT):
            j = KEPT_IDX[act_on_triple(pi, eps, t)]
            P[j, i] = 1.0
        mats.append(P)
    return mats


def decompose_irreps():
    perms = _build_perm_matrices()
    rng = np.random.default_rng(42)
    C = sum(c * P for c, P in zip(rng.standard_normal(48), perms)) / 48
    C = (C + C.T) / 2
    evals, evecs = np.linalg.eigh(C)

    tol = 1e-8
    irreps = []
    used = set()
    for i in range(19):
        if i in used:
            continue
        grp = [i]
        used.add(i)
        for j in range(i + 1, 19):
            if j not in used and abs(evals[i] - evals[j]) < tol:
                grp.append(j)
                used.add(j)
        irreps.append(evecs[:, grp])

    # Split reducible subspaces
    final = []
    rng2 = np.random.default_rng(999)
    for basis in irreps:
        if basis.shape[1] == 1:
            final.append(basis)
            continue
        restricted = [basis.T @ P @ basis for P in perms]
        C2 = sum(c * R for c, R in zip(rng2.standard_normal(48), restricted)) / 48
        C2 = (C2 + C2.T) / 2
        ev2, evec2 = np.linalg.eigh(C2)
        if np.max(ev2) - np.min(ev2) < 1e-6:
            final.append(basis)
        else:
            used2 = set()
            for j in range(basis.shape[1]):
                if j in used2:
                    continue
                g = [j]
                used2.add(j)
                for k in range(j + 1, basis.shape[1]):
                    if k not in used2 and abs(ev2[j] - ev2[k]) < 1e-8:
                        g.append(k)
                        used2.add(k)
                final.append(basis @ evec2[:, g])
    return final


def enumerate_kernels(irreps, dim=10):
    dims = [b.shape[1] for b in irreps]
    return [combo for r in range(1, len(irreps) + 1)
            for combo in combinations(range(len(irreps)), r)
            if sum(dims[i] for i in combo) == dim]


# ════════════════════════════════════════════════════════════════
# THE CORE: profiled residual F(x) and Levenberg-Marquardt solve
# ════════════════════════════════════════════════════════════════

def build_phi(alpha_flat, beta_flat, Q_perp):
    """Φ_i(a,b) = Σ_k Q⊥[k,i] · α_k[a] · β_k[b].

    alpha_flat: (19, 9),  beta_flat: (19, 9),  Q_perp: (19, 9)
    Returns Φ: (9, 81)
    """
    outer = alpha_flat[:, :, None] * beta_flat[:, None, :]   # (19, 9, 9)
    return Q_perp.T @ outer.reshape(19, 81)                   # (9, 81)


def profiled_residual(x, Q_perp):
    """F(x) → R^729.  The ONE equation.

    Γ is eliminated:  G_opt = T_flat · Φ^T · (Φ Φ^T)^{-1}
    Residual = T_flat − G_opt · Φ
    """
    alpha, beta = expand_seeds(x)
    phi = build_phi(alpha.reshape(19, 9), beta.reshape(19, 9), Q_perp)

    gram = phi @ phi.T                                         # (9, 9)
    rhs = T_FLAT @ phi.T                                       # (9, 9)

    # Regularized solve to avoid blow-up when phi is rank-deficient
    lam = 1e-14 * np.trace(gram) / 9 + 1e-30
    try:
        G = np.linalg.solve(gram + lam * np.eye(9), rhs.T).T
    except np.linalg.LinAlgError:
        G, _, _, _ = np.linalg.lstsq(gram, rhs.T, rcond=None)
        G = G.T

    return (T_FLAT - G @ phi).ravel()                          # 729-vector


def solve_kernel(combo, irreps, n_restarts=10, rng_base=0):
    """Run Levenberg-Marquardt for one kernel. Returns best result dict."""
    K_basis = np.hstack([irreps[i] for i in combo])            # (19, 10)
    U, _, _ = np.linalg.svd(K_basis, full_matrices=True)
    Q_perp = U[:, K_basis.shape[1]:]                           # (19, 9)

    best_cost = np.inf
    best_x = None
    best_result = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(rng_base + restart)
        x0 = rng.standard_normal(81) * 0.5

        try:
            res = least_squares(
                profiled_residual, x0, args=(Q_perp,),
                method='lm',            # Levenberg-Marquardt
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15,
                max_nfev=5000,
            )
        except Exception:
            continue

        if res.cost < best_cost:
            best_cost = res.cost
            best_x = res.x.copy()
            best_result = res

    if best_x is None:
        return {"combo": list(combo), "cost": np.inf, "solved": False}

    # Final diagnostics
    alpha, beta = expand_seeds(best_x)
    phi = build_phi(alpha.reshape(19, 9), beta.reshape(19, 9), Q_perp)
    gram = phi @ phi.T
    sv_gram = np.linalg.svd(gram, compute_uv=False)

    F = profiled_residual(best_x, Q_perp)
    max_abs = float(np.max(np.abs(F)))

    # If solved to near-zero, extract Gamma and verify
    gamma_info = {}
    if max_abs < 1e-6:
        rhs = T_FLAT @ phi.T
        try:
            G = np.linalg.solve(gram, rhs.T).T
        except np.linalg.LinAlgError:
            G, _, _, _ = np.linalg.lstsq(gram, rhs.T, rcond=None)
            G = G.T
        Gamma = G @ Q_perp.T                                  # (9, 19)
        gamma_3d = Gamma.T.reshape(19, 3, 3)
        from tensor import build_decomposition, build_target_tensor
        T_hat = build_decomposition(alpha, beta, gamma_3d)
        T_true = build_target_tensor()
        tensor_err = float(np.max(np.abs(T_true - T_hat)))
        gamma_info = {"tensor_max_err": tensor_err, "exact": tensor_err < 1e-8}

    dims = [irreps[i].shape[1] for i in combo]
    return {
        "combo": list(combo),
        "dims": dims,
        "dims_str": "+".join(str(d) for d in dims),
        "cost": float(best_cost),
        "max_abs": max_abs,
        "sv_gram": sv_gram.tolist(),
        "gram_rank": int(np.sum(sv_gram > 1e-10)),
        "nfev": best_result.nfev if best_result else 0,
        "lm_status": best_result.status if best_result else -1,
        "solved": max_abs < 1e-6,
        **gamma_info,
    }


# ════════════════════════════════════════════════════════════════
# SWEEP
# ════════════════════════════════════════════════════════════════

def _worker(args):
    """Pickleable worker for ProcessPoolExecutor."""
    kidx, combo, irreps_list, n_restarts = args
    return kidx, solve_kernel(combo, irreps_list, n_restarts, rng_base=kidx * 1000)


def run_sweep(n_restarts=10, max_kernels=None, n_workers=None):
    irreps = decompose_irreps()
    dims = [b.shape[1] for b in irreps]
    candidates = enumerate_kernels(irreps, 10)

    N = len(candidates) if max_kernels is None else min(max_kernels, len(candidates))
    n_workers = n_workers or min(os.cpu_count() or 1, N)

    print(f"Canon Newton: {len(irreps)} irreps, dims={dims}, {len(candidates)} kernels")
    print(f"Sweeping {N} kernels × {n_restarts} restarts, workers={n_workers}")
    print("=" * 100)

    start = time.time()
    results = []

    tasks = [(i, candidates[i], irreps, n_restarts) for i in range(N)]

    if n_workers <= 1:
        for task in tasks:
            kidx, r = _worker(task)
            results.append((kidx, r))
            _print_result(kidx, r, len(results), N)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                kidx, r = fut.result()
                results.append((kidx, r))
                _print_result(kidx, r, len(results), N)

    results.sort(key=lambda kr: kr[1]["max_abs"])
    elapsed = time.time() - start

    print(f"\n{'=' * 100}")
    print(f"Done in {elapsed:.1f}s")
    print(f"\nTop 15 by max_abs residual:")
    for kidx, r in results[:15]:
        _print_result(kidx, r, prefix="  ")
    print()

    solved = [(k, r) for k, r in results if r["solved"]]
    if solved:
        print(f"*** {len(solved)} EXACT SOLUTIONS ***")
        for kidx, r in solved:
            print(f"  K_{kidx} [{r['dims_str']}]: max_err={r.get('tensor_max_err', '?')}")
    else:
        best_kidx, best_r = results[0]
        print(f"No exact solutions. Best: K_{best_kidx} [{best_r['dims_str']}] "
              f"max_abs={best_r['max_abs']:.6e} cost={best_r['cost']:.6e}")

    return results


def _print_result(kidx, r, done=None, total=None, prefix=""):
    star = " *** SOLVED ***" if r["solved"] else ""
    progress = f"[{done}/{total}] " if done is not None else ""
    print(f"{prefix}{progress}K_{kidx:>3d} [{r.get('dims_str', '?'):>15s}] "
          f"max_abs={r['max_abs']:.4e} cost={r['cost']:.3e} "
          f"gr={r['gram_rank']} nfev={r['nfev']}{star}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canon Newton: unified R=19 solver")
    parser.add_argument("--restarts", type=int,
                        default=int(os.environ.get("CN_RESTARTS", "10")))
    parser.add_argument("--kernels", type=int,
                        default=int(os.environ.get("CN_KERNELS", "0")))
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("CN_WORKERS", "0")) or None)
    parser.add_argument("--out", type=str,
                        default="CANON OPTIMIZER/canon_newton_results.json")
    args = parser.parse_args()

    max_k = args.kernels if args.kernels > 0 else None
    results = run_sweep(
        n_restarts=args.restarts,
        max_kernels=max_k,
        n_workers=args.workers,
    )

    # Save
    out = []
    for kidx, r in results:
        entry = {k: v for k, v in r.items() if k != "sv_gram"}
        entry["kernel_idx"] = kidx
        out.append(entry)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
