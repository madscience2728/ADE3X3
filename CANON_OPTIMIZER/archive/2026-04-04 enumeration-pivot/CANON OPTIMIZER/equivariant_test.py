"""
EQUIVARIANT EXPANSION (no stabilizer averaging).

Transport seed factors to orbit members via group action WITHOUT averaging.
Each orbit member gets its own factors derived from seed by a single transporter.
This preserves rank-1 sharpness that averaging destroys.

Then re-run the full gate diagnostic on this parameterization.
"""

import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

# ═══ Group ═══
def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _apply_factors(pi, eps, a, b, g):
    """Apply group element (pi, eps) to factor triple (a, b, g).
    Each factor is a 3×3 matrix. The group acts by permuting coordinates
    and swapping indices 1↔2."""
    def _pm(e): return np.eye(3)[[0,2,1],:] if e else np.eye(3)
    P = [_pm(eps[i]) for i in range(3)]
    pi_inv = [0]*3
    for i in range(3): pi_inv[pi[i]] = i
    facs = {(0,1): a, (1,2): b, (0,2): g}
    res = {}
    for (x,y), nm in [((0,1),'a'), ((1,2),'b'), ((0,2),'g')]:
        oa, ob = pi_inv[x], pi_inv[y]
        key = (min(oa,ob), max(oa,ob))
        F = facs[key].T if (oa,ob) != key else facs[key].copy()
        res[nm] = P[x] @ F @ P[y].T
    return res['a'], res['b'], res['g']

def _compute_orbits():
    remaining = set((r,s,u) for r in range(3) for s in range(3) for u in range(3))
    out = {}
    while remaining:
        seed = min(remaining); orb = set()
        for pi, eps in _GROUP: orb.add(_apply_triple(pi, eps, seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_ORBITS = _compute_orbits()

ALL_CONFIGS = {
    7: [1,6], 8: [8], 9: [1,8], 12: [12], 13: [1,12],
    14: [6,8], 15: [1,6,8], 18: [6,12], 19: [1,6,12],
    20: [8,12], 21: [1,8,12], 26: [6,8,12], 27: [1,6,8,12],
}


def build_equivariant_expansion(sel):
    """Build expansion matrices WITHOUT stabilizer averaging.
    For each seed, pick ONE transporter per orbit member."""
    seed_list = [_ORBITS[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _ORBITS[s][1]
    kept_sorted = sorted(kept)
    kept_idx = {t:i for i,t in enumerate(kept_sorted)}
    R = len(kept_sorted)

    transport_cache = {}
    for seed in seed_list:
        transporters = {}
        for pi, eps in _GROUP:
            t = _apply_triple(pi, eps, seed)
            if t not in transporters and t in kept:
                transporters[t] = (pi, eps)
        transport_cache[seed] = transporters

    # Each seed has 27 raw params: 9 for α, 9 for β, 9 for γ
    # NO averaging — just transport directly
    n_params = len(seed_list) * 27  # includes γ
    n_ab_params = len(seed_list) * 18  # α,β only

    M_a = np.zeros((R * 9, n_ab_params))
    M_b = np.zeros((R * 9, n_ab_params))

    for j in range(n_ab_params):
        e = np.zeros(n_ab_params); e[j] = 1.0
        for si, seed in enumerate(seed_list):
            off = si * 18
            a_seed = e[off:off+9].reshape(3, 3)
            b_seed = e[off+9:off+18].reshape(3, 3)
            g_seed = np.zeros((3, 3))
            for t, (pi, eps) in transport_cache[seed].items():
                at, bt, _ = _apply_factors(pi, eps, a_seed, b_seed, g_seed)
                k = kept_idx[t]
                M_a[k*9:(k+1)*9, j] = at.ravel()
                M_b[k*9:(k+1)*9, j] = bt.ravel()

    return M_a, M_b, R, n_ab_params, kept_sorted


def compute_blocks(alpha, beta, R):
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    for k in range(R):
        Eta1[k] = (np.outer(alpha[k][:,0], beta[k][0,:]) -
                   np.outer(alpha[k][:,1], beta[k][1,:])).ravel()
        Eta2[k] = (np.outer(alpha[k][:,1], beta[k][1,:]) -
                   np.outer(alpha[k][:,2], beta[k][2,:])).ravel()
    H = np.hstack([Eta1, Eta2])
    Delta = np.zeros((R, 54))
    col = 0
    for s in range(3):
        for t in range(3):
            if s == t: continue
            for k in range(R):
                Delta[k, col:col+9] = np.outer(alpha[k][:,s], beta[k][t,:]).ravel()
            col += 9
    Nuisance = np.hstack([H, Delta])
    Sigma = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
    return Sigma, H, Delta, Nuisance


def expand(M_a, M_b, R, n_p, x):
    return (M_a @ x).reshape(R, 3, 3), (M_b @ x).reshape(R, 3, 3)


# ═══════════════════════════════════════════════════════════════
# TEST A: Check standard algorithm is NOW in the family
# ═══════════════════════════════════════════════════════════════
def test_standard():
    flush("="*90)
    flush("TEST A: Is standard algorithm in the EQUIVARIANT (non-averaged) family?")
    flush("="*90)

    sel = [1, 6, 8, 12]
    M_a, M_b, R, n_p, kept = build_equivariant_expansion(sel)

    alpha_std = np.zeros((27, 3, 3))
    beta_std = np.zeros((27, 3, 3))
    for k, (r, s, u) in enumerate(kept):
        alpha_std[k, r, s] = 1.0
        beta_std[k, s, u] = 1.0

    target = np.concatenate([alpha_std.reshape(R*9), beta_std.reshape(R*9)])
    M = np.vstack([M_a, M_b])

    x_sol, _, rank, _ = np.linalg.lstsq(M, target, rcond=None)
    err_a = np.linalg.norm(M_a @ x_sol - alpha_std.reshape(R*9))
    err_b = np.linalg.norm(M_b @ x_sol - beta_std.reshape(R*9))

    flush(f"  M shape: {M.shape}, rank: {rank}/{n_p}")
    flush(f"  ||err_α|| = {err_a:.2e}, ||err_β|| = {err_b:.2e}")

    if err_a < 1e-10 and err_b < 1e-10:
        flush("  ✓ Standard algorithm IS in the equivariant family!")
        alpha, beta = expand(M_a, M_b, R, n_p, x_sol)
        S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
        flush(f"  rk(Σ) at standard point: {np.linalg.matrix_rank(S, tol=1e-8)}")
    else:
        flush("  ✗ Still not in family. Checking effective rank...")
        flush(f"  Effective α,β params: {rank}")


# ═══════════════════════════════════════════════════════════════
# TEST B: Generic ranks of ALL blocks for every R
# ═══════════════════════════════════════════════════════════════
def test_generic_ranks():
    flush("\n" + "="*90)
    flush("TEST B: Generic ranks with EQUIVARIANT expansion (all R)")
    flush("="*90)

    rng = np.random.default_rng(42)
    flush(f"{'R':>3s} {'orbits':>15s} {'n_p':>4s} | {'rk_S':>4s} {'rk_H':>4s} {'rk_D':>4s} {'rk_N':>4s} | "
          f"{'tgt_H':>5s} {'G2?':>4s} {'G3?':>4s}")
    flush("-"*80)

    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_equivariant_expansion(sel)

        ranks = []
        for _ in range(5):
            x = rng.standard_normal(n_p)
            alpha, beta = expand(M_a, M_b, R, n_p, x)
            S, H, D, N = compute_blocks(alpha, beta, R)
            SN = np.hstack([S, N])
            ranks.append((
                np.linalg.matrix_rank(S, tol=1e-8),
                np.linalg.matrix_rank(H, tol=1e-8),
                np.linalg.matrix_rank(D, tol=1e-8),
                np.linalg.matrix_rank(N, tol=1e-8),
                np.linalg.matrix_rank(SN, tol=1e-8),
            ))
        r = ranks[0]
        tgt = R - 9
        g2 = "YES" if r[3] == r[1] else "no"
        g3 = "YES" if r[4] == r[3] + 9 else "no"
        flush(f"{R:>3d} {str(sel):>15s} {n_p:>4d} | {r[0]:>4d} {r[1]:>4d} {r[2]:>4d} {r[3]:>4d} | "
              f"{tgt:>5d} {g2:>4s} {g3:>4s}")


# ═══════════════════════════════════════════════════════════════
# TEST C: 9×9 minor test for Σ
# ═══════════════════════════════════════════════════════════════
def test_sigma_minors():
    flush("\n" + "="*90)
    flush("TEST C: 9×9 minors of Σ in EQUIVARIANT family")
    flush("="*90)

    from itertools import combinations
    rng = np.random.default_rng(999)

    for R, sel in sorted(ALL_CONFIGS.items()):
        if R < 9: continue
        M_a, M_b, _, n_p, _ = build_equivariant_expansion(sel)

        row_combos = list(combinations(range(R), 9))
        if len(row_combos) > 200:
            row_combos = [row_combos[i] for i in rng.choice(len(row_combos), 200, replace=False)]

        max_det = 0
        for _ in range(200):
            x = rng.standard_normal(n_p)
            alpha, beta = expand(M_a, M_b, R, n_p, x)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            for rows in row_combos[:50]:
                d = abs(np.linalg.det(S[list(rows), :]))
                if d > max_det: max_det = d

        status = "✓ rk(Σ)=9 achievable!" if max_det > 1e-10 else "✗ all minors ~0"
        flush(f"  R={R:>2d}: max |det(9×9)| = {max_det:.6e}  {status}")


# ═══════════════════════════════════════════════════════════════
# TEST D: Conservation law and effective dimensions
# ═══════════════════════════════════════════════════════════════
def test_effective_dims():
    flush("\n" + "="*90)
    flush("TEST D: Effective parameter dimensions (Jacobian rank)")
    flush("="*90)

    rng = np.random.default_rng(77)
    eps = 1e-7

    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_equivariant_expansion(sel)

        x0 = rng.standard_normal(n_p)

        def residual_map(x):
            alpha, beta = expand(M_a, M_b, R, n_p, x)
            S, H, D, N = compute_blocks(alpha, beta, R)
            return np.hstack([S, H, D]).ravel()

        f0 = residual_map(x0)
        J = np.zeros((len(f0), n_p))
        for j in range(n_p):
            xp = x0.copy(); xp[j] += eps
            J[:, j] = (residual_map(xp) - f0) / eps

        rk = np.linalg.matrix_rank(J, tol=1e-4)
        flush(f"  R={R:>2d}: {n_p} params → {len(f0)} outputs, Jac rank = {rk}  "
              f"(eff params = {rk}, kernel dim = {n_p - rk})")


# ═══════════════════════════════════════════════════════════════
# TEST E: Per-seed Σ rank contributions (equivariant)
# ═══════════════════════════════════════════════════════════════
def test_per_seed_sigma():
    flush("\n" + "="*90)
    flush("TEST E: Per-seed Σ rank (equivariant)")
    flush("="*90)

    rng = np.random.default_rng(55)
    seed_names = {1: "Corner", 6: "Edge", 8: "Interior", 12: "Face"}

    for orb_size in [1, 6, 8, 12]:
        M_a, M_b, R, n_p, _ = build_equivariant_expansion([orb_size])
        x = rng.standard_normal(n_p)
        alpha, beta = expand(M_a, M_b, R, n_p, x)
        S, H, _, _ = compute_blocks(alpha, beta, R)
        flush(f"  {seed_names[orb_size]:>10s}: n_p={n_p:>2d}  rk(Σ)={np.linalg.matrix_rank(S, tol=1e-8)}  "
              f"rk(H)={np.linalg.matrix_rank(H, tol=1e-8)}")

    flush("\n  Combined (additive check):")
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_equivariant_expansion(sel)
        x = rng.standard_normal(n_p)
        alpha, beta = expand(M_a, M_b, R, n_p, x)
        S, H, _, _ = compute_blocks(alpha, beta, R)
        flush(f"    R={R:>2d}: rk(Σ)={np.linalg.matrix_rank(S, tol=1e-8):<3d} "
              f"rk(H)={np.linalg.matrix_rank(H, tol=1e-8)}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_standard()
    test_generic_ranks()
    test_sigma_minors()
    test_effective_dims()
    test_per_seed_sigma()
    flush(f"\nTotal: {time.time()-t0:.1f}s")
