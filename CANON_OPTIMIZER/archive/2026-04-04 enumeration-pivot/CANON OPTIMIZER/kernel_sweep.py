"""
kernel_sweep.py - Sweep over all candidate irrep kernels for R=19.

For each candidate kernel K (dim 10, a direct sum of Z2≀S3 irreps):
  1) Fix K as the kernel subspace
  2) Minimize ||(I - QQ^T) @ [H | Delta]||_F^2 over the 81 seed params
     This is a degree-4 polynomial — smooth and L-BFGS friendly.
  3) If residual ≈ 0, solve for Gamma linearly (Gate 3)
  4) If Gamma solves → exact R=19 decomposition

The discrete search over irrep combos replaces the impossible continuous
rank-reduction that the gate_solver.py was attempting.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from itertools import permutations, product as iproduct, combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from scipy.optimize import minimize

# ══════════════════════════════════════════════════════════════
# GROUP AND REPRESENTATION (self-contained, no import cycles)
# ══════════════════════════════════════════════════════════════

def _swap12(x):
    return x if x == 0 else 3 - x

S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
assert len(GROUP) == 48

ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
KEPT = sorted([t for t in ALL27 if 0 in t])
assert len(KEPT) == 19
KEPT_IDX = {t: i for i, t in enumerate(KEPT)}
SUPER_SEEDS = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]

DEAD_PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def act_on_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))


def _swap_perm(flag):
    return [0, 2, 1] if flag else [0, 1, 2]


def _act_on_factors(pi, eps, a, b, g):
    """Apply (pi, eps) to factor triple (alpha, beta, gamma).
    
    Role pairs: alpha=(0,1), beta=(1,2), gamma=(0,2).
    pi permutes the 3 roles, eps applies swap12 per role.
    """
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
    all_a, all_b, all_g = [], [], []
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
                all_a.append(at)
                all_b.append(bt)
                all_g.append(gt)
    
    # Sort by KEPT order
    terms = list(zip(
        [act_on_triple(pi, eps, seed)
         for seed in SUPER_SEEDS for pi, eps in GROUP],
        range(1000),  # placeholder
    ))
    # Actually let me just collect properly
    result_a = [None] * 19
    result_b = [None] * 19
    result_g = [None] * 19
    
    idx = 0
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


def build_H_Delta(alpha, beta):
    """Build H (19×18) and Delta (19×54) from alpha, beta (19×3×3)."""
    R = 19
    eta1 = np.zeros((R, 9))
    eta2 = np.zeros((R, 9))
    delta = np.zeros((R, 54))
    sigma = np.zeros((R, 9))
    
    for k in range(R):
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
            for r in range(3):
                for u in range(3):
                    delta[k, col] = a[r, s] * b[t, u]
                    col += 1
    
    H = np.hstack([eta1, eta2])
    return sigma, H, delta


# ══════════════════════════════════════════════════════════════
# IRREP DECOMPOSITION
# ══════════════════════════════════════════════════════════════

def build_perm_matrices():
    """Build the 48 permutation matrices on R^19."""
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
    """Decompose R^19 into irreps of Z2≀S3. Returns list of (19, d_i) bases."""
    perms = build_perm_matrices()
    
    # Random commutant element
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(48)
    C = sum(c * P for c, P in zip(coeffs, perms)) / 48
    C = (C + C.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Group by eigenvalue
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
    
    # Check and split non-irreducible subspaces
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
            # Split
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
    """All subsets of irreps summing to target_dim."""
    dims = [b.shape[1] for b in irreps]
    candidates = []
    for r in range(1, len(irreps) + 1):
        for combo in combinations(range(len(irreps)), r):
            if sum(dims[i] for i in combo) == target_dim:
                candidates.append(combo)
    return candidates


# ══════════════════════════════════════════════════════════════
# TARGET TENSOR
# ══════════════════════════════════════════════════════════════

def build_target():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for c in range(3):
            for s in range(3):
                T[r * 3 + c, r * 3 + s, s * 3 + c] = 1.0
    return T

TARGET = build_target()


# ══════════════════════════════════════════════════════════════
# KERNEL-FIXED SOLVER
# ══════════════════════════════════════════════════════════════

def make_kernel_loss(Q_perp):
    """Return a loss function: ||(Q_perp^T) @ [H | Delta]||_F^2.
    
    Q_perp is (19, 9) — the orthonormal basis of K^perp.
    When [H | Delta] columns all lie in K, Q_perp^T @ [H | Delta] = 0.
    """
    def loss(params):
        alpha, beta, gamma = expand_seeds(params)
        sigma, H, delta = build_H_Delta(alpha, beta)
        nuisance = np.hstack([H, delta])  # (19, 72)
        proj = Q_perp.T @ nuisance         # (9, 72)
        return float(np.sum(proj ** 2))
    return loss


def solve_gamma_from_kernel(alpha, beta):
    """Given (alpha, beta) satisfying Gates 1+2, solve for Gamma."""
    sigma, H, delta = build_H_Delta(alpha, beta)
    nuisance = np.hstack([H, delta])
    design = np.hstack([sigma, nuisance])  # (19, 81)
    
    rank_aug = np.linalg.matrix_rank(design, tol=1e-8)
    rank_nuis = np.linalg.matrix_rank(nuisance, tol=1e-8)
    
    info = {
        "rank_augmented": rank_aug,
        "rank_nuisance": rank_nuis,
        "solvable": rank_aug == rank_nuis + 9,
    }
    
    if not info["solvable"]:
        return None, info
    
    # Gamma (9×19) @ design = [3*I_9 | 0]
    target_rhs = np.zeros((9, 9 + 72))
    target_rhs[:, :9] = 3.0 * np.eye(9)
    
    gamma_T, _, _, _ = np.linalg.lstsq(design.T, target_rhs.T, rcond=None)
    gamma_flat = gamma_T.T  # (9, 19)
    
    # Verify
    sig_err = float(np.max(np.abs(gamma_flat @ sigma - 3.0 * np.eye(9))))
    nuis_err = float(np.max(np.abs(gamma_flat @ nuisance)))
    info["sigma_error"] = sig_err
    info["nuisance_error"] = nuis_err
    info["gamma_exact"] = sig_err < 1e-6 and nuis_err < 1e-6
    
    if not info["gamma_exact"]:
        return None, info
    
    # Reconstruct gamma per-term
    gamma = gamma_flat.T.reshape(19, 3, 3)
    
    # Final tensor check
    from tensor import build_decomposition
    T_hat = build_decomposition(alpha, beta, gamma)
    resid = float(np.max(np.abs(TARGET - T_hat)))
    info["tensor_residual"] = resid
    info["exact"] = resid < 1e-6
    
    return gamma, info


def sweep_kernel(kernel_idx, combo, irreps, n_restarts=50, maxiter=2000, seed_base=0):
    """Try to find (alpha, beta) for one kernel candidate."""
    # Build K and K_perp
    K_basis = np.hstack([irreps[i] for i in combo])  # (19, 10)
    # QR to get orthonormal basis for K_perp
    U, S, Vt = np.linalg.svd(K_basis, full_matrices=True)
    Q_perp = U[:, K_basis.shape[1]:]  # (19, 9)
    
    loss_fn = make_kernel_loss(Q_perp)
    
    best_loss = np.inf
    best_params = None
    
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed_base + restart)
        x0 = rng.standard_normal(81) * 0.5
        
        result = minimize(loss_fn, x0, method="L-BFGS-B",
                          options={"maxiter": maxiter, "disp": False})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()
    
    # Check if we got close enough
    alpha, beta, gamma = expand_seeds(best_params)
    sigma, H, delta = build_H_Delta(alpha, beta)
    rank_H = int(np.linalg.matrix_rank(H, tol=1e-8))
    rank_nuis = int(np.linalg.matrix_rank(np.hstack([H, delta]), tol=1e-8))
    
    gamma_result = None
    gamma_info = {"attempted": False}
    if best_loss < 1e-10:
        gamma_result, gamma_info = solve_gamma_from_kernel(alpha, beta)
        gamma_info["attempted"] = True
    
    dims = [irreps[i].shape[1] for i in combo]
    return {
        "kernel_idx": kernel_idx,
        "combo": list(combo),
        "dims": dims,
        "dims_str": "+".join(str(d) for d in dims),
        "best_loss": best_loss,
        "rank_H": rank_H,
        "rank_nuisance": rank_nuis,
        "gamma_info": gamma_info,
        "solved": gamma_result is not None,
        "params": best_params.tolist() if gamma_result is not None else None,
    }


def main():
    print("Decomposing R^19 into irreps of Z2≀S3...")
    irreps = decompose_irreps()
    dims = [b.shape[1] for b in irreps]
    print(f"  {len(irreps)} irreps, dims = {dims}, sum = {sum(dims)}")
    
    candidates = enumerate_kernel_candidates(irreps, target_dim=10)
    print(f"  {len(candidates)} kernel candidates (dim-10 irrep subsets)")
    
    # Print first few
    for i, combo in enumerate(candidates[:5]):
        d = "+".join(f"V{j}({dims[j]})" for j in combo)
        print(f"    K_{i}: {d}")
    if len(candidates) > 5:
        print(f"    ... and {len(candidates) - 5} more")
    
    n_restarts = int(os.environ.get("KERNEL_RESTARTS", "30"))
    maxiter = int(os.environ.get("KERNEL_MAXITER", "1500"))
    n_workers = int(os.environ.get("KERNEL_WORKERS", str(os.cpu_count() or 4)))
    
    print(f"\nSweeping {len(candidates)} kernels × {n_restarts} restarts, "
          f"maxiter={maxiter}, workers={n_workers}")
    print("=" * 80)
    
    start = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(sweep_kernel, i, combo, irreps, n_restarts, maxiter, i * 1000): i
            for i, combo in enumerate(candidates)
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "*** SOLVED ***" if r["solved"] else ""
            gamma_note = ""
            if r["gamma_info"].get("attempted"):
                gamma_note = f" Γ:{r['gamma_info']}"
            print(
                f"  K_{r['kernel_idx']:>3d} [{r['dims_str']:>20s}] "
                f"loss={r['best_loss']:.4e} rk(H)={r['rank_H']:>2d} "
                f"rk(N)={r['rank_nuisance']:>2d} {status}{gamma_note}"
            )
    
    results.sort(key=lambda r: r["best_loss"])
    elapsed = time.time() - start
    
    print(f"\n{'=' * 80}")
    print(f"SWEEP COMPLETE in {elapsed:.1f}s")
    print(f"\nTop 10 by loss:")
    for r in results[:10]:
        print(f"  K_{r['kernel_idx']:>3d} [{r['dims_str']:>20s}] loss={r['best_loss']:.6e} "
              f"rk(H)={r['rank_H']} rk(N)={r['rank_nuisance']}")
    
    solved = [r for r in results if r["solved"]]
    if solved:
        print(f"\n*** {len(solved)} EXACT SOLUTIONS FOUND ***")
        for r in solved:
            print(f"  K_{r['kernel_idx']} [{r['dims_str']}]: {r['gamma_info']}")
    else:
        print(f"\nNo exact solutions found. Closest loss: {results[0]['best_loss']:.6e}")
        if results[0]["best_loss"] < 1e-6:
            print("  Very close — may need tighter optimization or this kernel is degenerate.")
    
    # Save
    out_path = os.environ.get("KERNEL_OUT", "CANON OPTIMIZER/kernel_sweep_results.json")
    with open(out_path, "w") as f:
        # Strip non-serializable
        for r in results:
            r.pop("params", None)
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
