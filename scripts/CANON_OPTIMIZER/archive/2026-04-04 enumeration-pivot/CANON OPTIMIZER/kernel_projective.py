"""
kernel_projective.py - For each candidate kernel K, constrain Gamma to K_perp
and minimize actual tensor residual.

This is the right formulation:
  - Gamma has rows in K_perp (9-dim), so Gamma = G @ Q_perp^T where G is 9x9
  - For fixed (alpha, beta), G is solved by least-squares (linear in G)
  - The profiled loss L(alpha, beta) = min_G ||T - reconstruct||^2
  - This prevents degeneracy: if (alpha, beta) → 0, the residual → ||T||^2 ≈ 27

The solution (if it exists) has L = 0 exactly when:
  - Nuisance lies in K (automatic when Gamma rows are in K_perp and residual = 0)
  - Sigma is independent modulo K (for G to be invertible)
"""

from __future__ import annotations

import os
import sys
import time
import json
from itertools import permutations, product as iproduct, combinations
from typing import Any

import numpy as np
from scipy.optimize import minimize

# ═══════════ Group, kept triples, expansion (same as kernel_sweep) ═══════════

def _swap12(x):
    return x if x == 0 else 3 - x

S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
KEPT = sorted([t for t in ALL27 if 0 in t])
KEPT_IDX = {t: i for i, t in enumerate(KEPT)}
SUPER_SEEDS = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]
DEAD_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


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
        old_pair = (pi[new_pair[0]], pi[new_pair[1]])
        key = (min(old_pair), max(old_pair))
        matrix = facs[key]
        if old_pair != key:
            matrix = matrix.T
        row_perm = role_perms[new_pair[0]]
        col_perm = role_perms[new_pair[1]]
        out[name] = matrix[np.ix_(row_perm, col_perm)].copy()
    return out['a'], out['b'], out['g']


def stabilizer(seed):
    return [(pi, eps) for pi, eps in GROUP if act_on_triple(pi, eps, seed) == seed]


def project_to_stabilizer(seed, a, b, g):
    stab = stabilizer(seed)
    aa = np.zeros((3, 3)); bb = np.zeros((3, 3)); gg = np.zeros((3, 3))
    for pi, eps in stab:
        a2, b2, g2 = _act_on_factors(pi, eps, a, b, g)
        aa += a2; bb += b2; gg += g2
    n = len(stab)
    return aa / n, bb / n, gg / n


def expand_seeds(params):
    """81 params → (alpha, beta, gamma) arrays of shape (19, 3, 3)."""
    result_a = [None] * 19
    result_b = [None] * 19
    result_g = [None] * 19
    kept_set = set(KEPT)

    for si, seed in enumerate(SUPER_SEEDS):
        off = si * 27
        a = params[off:off + 9].reshape(3, 3)
        b = params[off + 9:off + 18].reshape(3, 3)
        g = params[off + 18:off + 27].reshape(3, 3)
        a, b, g = project_to_stabilizer(seed, a, b, g)

        seen = set()
        for pi, eps in GROUP:
            t = act_on_triple(pi, eps, seed)
            if t not in seen and t in kept_set:
                seen.add(t)
                at, bt, gt = _act_on_factors(pi, eps, a, b, g)
                k = KEPT_IDX[t]
                result_a[k] = at
                result_b[k] = bt
                result_g[k] = gt

    return np.array(result_a), np.array(result_b), np.array(result_g)


# ═══════════ Target tensor ═══════════

def build_target():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for c in range(3):
            for s in range(3):
                T[r * 3 + c, r * 3 + s, s * 3 + c] = 1.0
    return T

TARGET = build_target()  # (9, 9, 9)
TARGET_FLAT = TARGET.reshape(9, 81)  # each row is T[g, :] flattened over (a, b)


# ═══════════ Irrep decomposition ═══════════

def build_perm_matrices():
    matrices = []
    for pi, eps in GROUP:
        P = np.zeros((19, 19))
        for i, t in enumerate(KEPT):
            t2 = act_on_triple(pi, eps, t)
            j = KEPT_IDX[t2]
            P[j, i] = 1.0
        matrices.append(P)
    return matrices


def decompose_irreps():
    perms = build_perm_matrices()
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(48)
    C = sum(c * P for c, P in zip(coeffs, perms)) / 48
    C = (C + C.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    tol = 1e-8
    irreps = []
    used = set()
    for i in range(19):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, 19):
            if j not in used and abs(eigenvalues[i] - eigenvalues[j]) < tol:
                group.append(j)
                used.add(j)
        irreps.append(eigenvectors[:, group])

    # Split reducible subspaces
    final = []
    for basis in irreps:
        dim = basis.shape[1]
        if dim == 1:
            final.append(basis)
            continue
        restricted = [basis.T @ P @ basis for P in perms]
        rng2 = np.random.default_rng(999)
        coeffs2 = rng2.standard_normal(48)
        C2 = sum(c * R for c, R in zip(coeffs2, restricted)) / 48
        C2 = (C2 + C2.T) / 2
        evals, evecs = np.linalg.eigh(C2)
        spread = np.max(evals) - np.min(evals)
        if spread < 1e-6:
            final.append(basis)
        else:
            used2 = set()
            for j in range(dim):
                if j in used2:
                    continue
                grp = [j]
                used2.add(j)
                for k in range(j + 1, dim):
                    if k not in used2 and abs(evals[j] - evals[k]) < 1e-8:
                        grp.append(k)
                        used2.add(k)
                final.append(basis @ evecs[:, grp])
    return final


def enumerate_kernel_candidates(irreps, target_dim=10):
    dims = [b.shape[1] for b in irreps]
    candidates = []
    for r in range(1, len(irreps) + 1):
        for combo in combinations(range(len(irreps)), r):
            if sum(dims[i] for i in combo) == target_dim:
                candidates.append(combo)
    return candidates


# ═══════════ Profiled loss: Gamma constrained to K_perp ═══════════

def build_phi(alpha_flat, beta_flat, Q_perp):
    """Build the 9 'effective basis' tensors phi_i.

    phi_i[a, b] = sum_k Q_perp[k, i] * alpha_k[a] * beta_k[b]

    alpha_flat: (19, 9), beta_flat: (19, 9), Q_perp: (19, 9)
    Returns phi: (9, 81) — each row is phi_i flattened over (a, b).
    """
    # phi_i = sum_k Q_perp[k,i] * outer(alpha_k, beta_k)
    # = (Q_perp.T @ (alpha ⊙ beta_kron))
    # More explicitly:
    R, d = alpha_flat.shape  # 19, 9
    phi = np.zeros((9, 81))
    for i in range(9):
        for k in range(R):
            w = Q_perp[k, i]
            if abs(w) < 1e-30:
                continue
            phi[i] += w * np.outer(alpha_flat[k], beta_flat[k]).ravel()
    return phi


def profiled_loss(params, Q_perp):
    """For fixed K (via Q_perp), compute min_G ||T - reconstruct(alpha, beta, G)||^2.

    Gamma = G @ Q_perp^T, where G is 9x9.
    T_hat[g, a, b] = sum_i G[g, i] * phi_i[a, b]
    => T_hat_flat[g, :] = G[g, :] @ phi  where phi is (9, 81)

    Optimal G: G_opt = T_flat @ phi^T @ inv(phi @ phi^T)  (row-by-row lstsq)
    Residual = ||T_flat - G_opt @ phi||^2
    """
    alpha, beta, _ = expand_seeds(params)
    alpha_flat = alpha.reshape(19, 9)
    beta_flat = beta.reshape(19, 9)

    phi = build_phi(alpha_flat, beta_flat, Q_perp)  # (9, 81)

    # Solve: for each of 9 gamma-rows, G[g,:] @ phi = T_flat[g,:]
    # Combined: G @ phi = T_flat, solve by lstsq
    # phi is (9, 81), T_flat is (9, 81)
    # G = T_flat @ pinv(phi) = T_flat @ phi^T @ inv(phi @ phi^T)

    gram = phi @ phi.T  # (9, 9)
    rhs = TARGET_FLAT @ phi.T  # (9, 9)

    try:
        G = np.linalg.solve(gram, rhs.T).T  # wait, let me be careful
        # We want G @ phi = TARGET_FLAT
        # G (9x9) @ phi (9x81) = TARGET_FLAT (9x81)
        # => G = TARGET_FLAT @ phi.T @ inv(phi @ phi.T)
        # => G @ gram = rhs where rhs = TARGET_FLAT @ phi.T
        # => G = rhs @ inv(gram)
        # => gram.T @ G.T = rhs.T => G.T = solve(gram, rhs.T)
        G_T = np.linalg.solve(gram, rhs.T)  # (9, 9)
        G = G_T.T
    except np.linalg.LinAlgError:
        # Singular gram — use lstsq
        G_T, _, _, _ = np.linalg.lstsq(gram, rhs.T, rcond=None)
        G = G_T.T

    residual = TARGET_FLAT - G @ phi  # (9, 81)
    return float(np.sum(residual ** 2))


def profiled_loss_detailed(params, Q_perp, Q_K):
    """Detailed analysis of a candidate."""
    alpha, beta, _ = expand_seeds(params)
    alpha_flat = alpha.reshape(19, 9)
    beta_flat = beta.reshape(19, 9)

    phi = build_phi(alpha_flat, beta_flat, Q_perp)
    gram = phi @ phi.T
    rhs = TARGET_FLAT @ phi.T

    try:
        G_T = np.linalg.solve(gram, rhs.T)
        G = G_T.T
    except np.linalg.LinAlgError:
        G_T, _, _, _ = np.linalg.lstsq(gram, rhs.T, rcond=None)
        G = G_T.T

    residual = TARGET_FLAT - G @ phi
    loss = float(np.sum(residual ** 2))
    max_resid = float(np.max(np.abs(residual)))

    Gamma = G @ Q_perp.T  # (9, 19)

    # Diagnostics
    # Build H, Delta
    eta1 = np.zeros((19, 9))
    eta2 = np.zeros((19, 9))
    sigma = np.zeros((19, 9))
    delta = np.zeros((19, 54))

    for k in range(19):
        a, b = alpha[k], beta[k]
        for r in range(3):
            for u in range(3):
                f = r * 3 + u
                s0 = a[r, 0] * b[0, u]
                s1 = a[r, 1] * b[1, u]
                s2 = a[r, 2] * b[2, u]
                sigma[k, f] = s0 + s1 + s2
                eta1[k, f] = s0 - s1
                eta2[k, f] = s1 - s2
        col = 0
        for s, t in DEAD_PAIRS:
            for rr in range(3):
                for uu in range(3):
                    delta[k, col] = a[rr, s] * b[t, uu]
                    col += 1

    H = np.hstack([eta1, eta2])
    nuisance = np.hstack([H, delta])

    sv_gram = np.linalg.svd(gram, compute_uv=False)
    containment = float(np.sum((Q_perp.T @ nuisance) ** 2))

    return {
        "loss_fro": loss,
        "max_abs_residual": max_resid,
        "sv_gram": sv_gram.tolist(),
        "gram_cond": float(sv_gram[0] / max(sv_gram[-1], 1e-30)),
        "containment_residual": containment,
        "rank_H": int(np.linalg.matrix_rank(H, tol=1e-8)),
        "rank_nuisance": int(np.linalg.matrix_rank(nuisance, tol=1e-8)),
        "gamma_sigma_check": float(np.max(np.abs(Gamma @ sigma - 3 * np.eye(9)))),
    }


# ═══════════ Sweep ═══════════

def sweep_one(kidx, combo, irreps, n_restarts=20, maxiter=2000):
    K_basis = np.hstack([irreps[i] for i in combo])
    U, _, _ = np.linalg.svd(K_basis, full_matrices=True)
    Q_perp = U[:, K_basis.shape[1]:]
    Q_K = U[:, :K_basis.shape[1]]

    best_loss = np.inf
    best_params = None

    for restart in range(n_restarts):
        rng = np.random.default_rng(kidx * 10000 + restart)
        x0 = rng.standard_normal(81) * 0.5

        result = minimize(
            profiled_loss, x0, args=(Q_perp,),
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()

    detail = profiled_loss_detailed(best_params, Q_perp, Q_K)
    dims = [irreps[i].shape[1] for i in combo]

    return {
        "kidx": kidx,
        "combo": list(combo),
        "dims": dims,
        "dims_str": "+".join(str(d) for d in dims),
        **detail,
    }


def main():
    irreps = decompose_irreps()
    dims = [b.shape[1] for b in irreps]
    print(f"{len(irreps)} irreps, dims={dims}")

    candidates = enumerate_kernel_candidates(irreps, 10)
    print(f"{len(candidates)} kernel candidates")

    n_restarts = int(os.environ.get("KP_RESTARTS", "10"))
    maxiter = int(os.environ.get("KP_MAXITER", "1000"))
    N = int(os.environ.get("KP_N", "50"))
    N = min(N, len(candidates))

    print(f"\nSweeping {N} kernels × {n_restarts} restarts × {maxiter} maxiter")
    print("=" * 100)

    start = time.time()
    results = []

    for i in range(N):
        r = sweep_one(i, candidates[i], irreps, n_restarts, maxiter)
        results.append(r)
        star = " ***" if r["max_abs_residual"] < 1e-6 else ""
        print(
            f"  K_{r['kidx']:>3d} [{r['dims_str']:>15s}] "
            f"maxres={r['max_abs_residual']:.4e} "
            f"fro={r['loss_fro']:.3e} "
            f"cont={r['containment_residual']:.1e} "
            f"rk(H)={r['rank_H']:>2d} "
            f"cond={r['gram_cond']:.1e}"
            f"{star}"
        )

    results.sort(key=lambda r: r["max_abs_residual"])
    elapsed = time.time() - start

    print(f"\n{'=' * 100}")
    print(f"Done in {elapsed:.0f}s")
    print(f"\nTop 10 by max_abs_residual:")
    for r in results[:10]:
        print(f"  K_{r['kidx']:>3d} [{r['dims_str']:>15s}] "
              f"maxres={r['max_abs_residual']:.6e} "
              f"cont={r['containment_residual']:.1e} "
              f"rk(H)={r['rank_H']} "
              f"svG={[f'{v:.1e}' for v in r['sv_gram']]}")

    exact = [r for r in results if r["max_abs_residual"] < 1e-6]
    if exact:
        print(f"\n*** {len(exact)} NEAR-EXACT SOLUTIONS ***")
    else:
        print(f"\nBest max residual: {results[0]['max_abs_residual']:.6e}")


if __name__ == "__main__":
    main()
