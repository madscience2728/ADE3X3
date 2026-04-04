"""
PURE ALGEBRAIC EXPERIMENTS — No optimization, no search.
Evaluate structural properties at random points only.

10 scatter-shot experiments probing the algebra of the symmetric sector.
"""

import numpy as np
from itertools import permutations, product as iproduct, combinations
import sys, time

# ═══ Group machinery ═══
def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _apply_factors(pi, eps, a, b, g):
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

_ALL27 = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]

def _compute_orbits():
    remaining = set(_ALL27); out = {}
    while remaining:
        seed = min(remaining); orb = set()
        for pi,eps in _GROUP: orb.add(_apply_triple(pi,eps,seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_ORBITS = _compute_orbits()

def build_expansion(sel):
    seed_list = [_ORBITS[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _ORBITS[s][1]
    kept_sorted = sorted(kept)
    kept_idx = {t:i for i,t in enumerate(kept_sorted)}
    R = len(kept_sorted)
    
    stab_cache = {}
    transport_cache = {}
    for seed in seed_list:
        stab = [(pi,eps) for pi,eps in _GROUP if _apply_triple(pi,eps,seed)==seed]
        stab_cache[seed] = stab
        transporters = {}
        for pi,eps in _GROUP:
            t = _apply_triple(pi,eps,seed)
            if t not in transporters and t in kept:
                transporters[t] = (pi,eps)
        transport_cache[seed] = transporters
    
    n_params = len(seed_list) * 18
    M_a = np.zeros((R*9, n_params))
    M_b = np.zeros((R*9, n_params))
    
    for j in range(n_params):
        e = np.zeros(n_params); e[j] = 1.0
        for si, seed in enumerate(seed_list):
            off = si * 18
            a = e[off:off+9].reshape(3,3)
            b = e[off+9:off+18].reshape(3,3)
            g0 = np.zeros((3,3))
            stab = stab_cache[seed]
            aa = bb = np.zeros((3,3))
            for pi,eps in stab:
                a2,b2,_ = _apply_factors(pi,eps,a,b,g0)
                aa = aa+a2; bb = bb+b2
            n = len(stab); aa /= n; bb /= n
            for t,(pi,eps) in transport_cache[seed].items():
                at,bt,_ = _apply_factors(pi,eps,aa,bb,g0)
                k = kept_idx[t]
                M_a[k*9:(k+1)*9, j] = at.ravel()
                M_b[k*9:(k+1)*9, j] = bt.ravel()
    
    return M_a, M_b, R, n_params

def compute_all_blocks(alpha, beta, R):
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
    return Sigma, H, Delta, Nuisance, Eta1, Eta2


ALL_CONFIGS = {
    7:  [1, 6],
    8:  [8],
    9:  [1, 8],
    12: [12],
    13: [1, 12],
    14: [6, 8],
    15: [1, 6, 8],
    18: [6, 12],
    19: [1, 6, 12],
    20: [8, 12],
    21: [1, 8, 12],
    26: [6, 8, 12],
    27: [1, 6, 8, 12],
}

def expand_random(M_a, M_b, R, n_params, rng):
    x = rng.standard_normal(n_params)
    alpha = (M_a @ x).reshape(R, 3, 3)
    beta  = (M_b @ x).reshape(R, 3, 3)
    return alpha, beta

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Generic ranks of ALL blocks for EVERY orbit combo
# ════════════════════════════════════════════════════════════════
def exp1():
    flush_print("\n" + "="*90)
    flush_print("EXP 1: Generic ranks of Sigma, H, Delta, Nuisance for every R")
    flush_print("="*90)
    
    rng = np.random.default_rng(42)
    flush_print(f"{'R':>3s} {'orbits':>15s} {'eff_p':>5s} | {'rk_S':>4s} {'rk_H':>4s} {'rk_D':>4s} {'rk_N':>4s} | "
                f"{'tgt_H':>5s} {'gap_H':>5s} {'gap_N':>5s} | {'rk_SN':>5s} {'G3?':>4s}")
    flush_print("-"*90)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, R2, n_p = build_expansion(sel)
        M = np.vstack([M_a, M_b])
        eff = int(np.sum(np.linalg.svd(M, compute_uv=False) > 1e-10))
        
        # Average over 5 random points
        ranks = []
        for _ in range(5):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S, H, D, N, _, _ = compute_all_blocks(alpha, beta, R)
            SN = np.hstack([S, N])
            ranks.append((
                np.linalg.matrix_rank(S, tol=1e-8),
                np.linalg.matrix_rank(H, tol=1e-8),
                np.linalg.matrix_rank(D, tol=1e-8),
                np.linalg.matrix_rank(N, tol=1e-8),
                np.linalg.matrix_rank(SN, tol=1e-8),
            ))
        # All should be same (generic)
        r = ranks[0]
        target_H = R - 9
        gap_H = r[1] - target_H
        gap_N = r[3] - target_H
        g3 = "YES" if r[4] == r[3] + 9 else "no"
        flush_print(f"{R:>3d} {str(sel):>15s} {eff:>5d} | {r[0]:>4d} {r[1]:>4d} {r[2]:>4d} {r[3]:>4d} | "
                    f"{target_H:>5d} {gap_H:>5d} {gap_N:>5d} | {r[4]:>5d} {g3:>4s}")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Rank of [H | Delta] minus rank of H — the "leak"
# Is the relation rk(N) - rk(H) constant (algebraic invariant)?
# ════════════════════════════════════════════════════════════════
def exp2():
    flush_print("\n" + "="*90)
    flush_print("EXP 2: Is rk(Nuisance) - rk(H) an algebraic constant? (Delta leak dimension)")
    flush_print("="*90)
    
    rng = np.random.default_rng(99)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        leaks = []
        for _ in range(20):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            _, H, _, N, _, _ = compute_all_blocks(alpha, beta, R)
            leaks.append(np.linalg.matrix_rank(N, tol=1e-8) - np.linalg.matrix_rank(H, tol=1e-8))
        unique = sorted(set(leaks))
        flush_print(f"  R={R:>2d} {str(sel):>15s}: leak = rk(N)-rk(H) = {unique}  (constant={len(unique)==1})")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Symbolic structure — is Delta in the span of 
# Kronecker products of H columns?  (Linear algebra of bilinear maps)
# ════════════════════════════════════════════════════════════════
def exp3():
    flush_print("\n" + "="*90)
    flush_print("EXP 3: Column space containment — project Delta onto span(H), measure residual rank")
    flush_print("="*90)
    
    rng = np.random.default_rng(77)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        _, H, D, N, _, _ = compute_all_blocks(alpha, beta, R)
        
        # Project each Delta column onto column space of H^T
        # H is R×18, D is R×54.  Work in row space: H^T is 18×R
        # Project D^T (54×R) onto row space of H (R×18)
        if np.linalg.matrix_rank(H, tol=1e-8) > 0:
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            rk = np.sum(s > 1e-8)
            P = U[:, :rk] @ U[:, :rk].T  # projector onto col(H)
            D_proj = P @ D
            D_resid = D - D_proj
            rk_resid = np.linalg.matrix_rank(D_resid, tol=1e-8)
        else:
            rk_resid = np.linalg.matrix_rank(D, tol=1e-8)
        
        rk_H = np.linalg.matrix_rank(H, tol=1e-8)
        rk_D = np.linalg.matrix_rank(D, tol=1e-8)
        flush_print(f"  R={R:>2d}: rk(H)={rk_H:>2d}  rk(D)={rk_D:>2d}  "
                    f"rk(D_residual_off_H)={rk_resid:>2d}  "
                    f"(Delta needs {rk_resid} extra dims beyond H)")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Eta structure — are Eta1 and Eta2 linearly dependent?
# Do they share a common column space?
# ════════════════════════════════════════════════════════════════
def exp4():
    flush_print("\n" + "="*90)
    flush_print("EXP 4: Eta1 vs Eta2 — rank structure and overlap")
    flush_print("="*90)
    
    rng = np.random.default_rng(55)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        _, H, _, _, E1, E2 = compute_all_blocks(alpha, beta, R)
        
        rk1 = np.linalg.matrix_rank(E1, tol=1e-8)
        rk2 = np.linalg.matrix_rank(E2, tol=1e-8)
        rk_H = np.linalg.matrix_rank(H, tol=1e-8)
        # Overlap = rk1 + rk2 - rk_H
        overlap = rk1 + rk2 - rk_H
        flush_print(f"  R={R:>2d}: rk(E1)={rk1:>2d}  rk(E2)={rk2:>2d}  rk(H)={rk_H:>2d}  "
                    f"overlap={overlap:>2d}  (E1∩E2 dim)")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 5: How many linearly independent α⊗β outer products?
# The map (α,β)→α⊗β is bilinear. What's the generic rank of 
# the 81-column matrix [vec(α_k⊗β_k)] for each R?
# ════════════════════════════════════════════════════════════════
def exp5():
    flush_print("\n" + "="*90)
    flush_print("EXP 5: Rank of outer-product matrix [α_k⊗β_k] (81 cols) for each R")
    flush_print("="*90)
    
    rng = np.random.default_rng(33)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        
        # Each α_k is 3×3, β_k is 3×3. Outer product is 9×9 = 81 entries
        OP = np.array([np.kron(alpha[k].ravel(), beta[k].ravel()) for k in range(R)])
        rk = np.linalg.matrix_rank(OP, tol=1e-8)
        flush_print(f"  R={R:>2d}: {R} terms, rk(α⊗β matrix)={rk}  (max possible={min(R,81)})")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 6: The Sigma block — does it already have rank 9?
# If rk(Sigma) < 9, decomposition is immediately impossible.
# ════════════════════════════════════════════════════════════════
def exp6():
    flush_print("\n" + "="*90)
    flush_print("EXP 6: Generic rank of Sigma (need 9 for solvability)")
    flush_print("="*90)
    
    rng = np.random.default_rng(11)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        ranks = []
        for _ in range(10):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            ranks.append(np.linalg.matrix_rank(S, tol=1e-8))
        r = ranks[0]
        flush_print(f"  R={R:>2d}: rk(Σ) = {r}  {'✓ FULL' if r == 9 else '✗ DEFICIENT — impossible'}")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 7: The "missing conservation" test.
# At a generic point, what is R + (18 - rk(H))? 
# The conservation law says this should be 27 for exact decomps.
# What does it equal generically?
# ════════════════════════════════════════════════════════════════
def exp7():
    flush_print("\n" + "="*90)
    flush_print("EXP 7: Generic value of R + (18 - rk(H))  [conservation law = 27 for exact]")
    flush_print("="*90)
    
    rng = np.random.default_rng(22)
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        _, H, _, _, _, _ = compute_all_blocks(alpha, beta, R)
        rk_H = np.linalg.matrix_rank(H, tol=1e-8)
        val = R + (18 - rk_H)
        flush_print(f"  R={R:>2d}: rk(H)={rk_H:>2d}  R+η_null = {R}+{18-rk_H} = {val}  "
                    f"(target=27, deficit={27-val})")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 8: Per-seed contribution — which seed dominates H rank?
# Expand each seed alone and check its rank contribution.
# ════════════════════════════════════════════════════════════════
def exp8():
    flush_print("\n" + "="*90)
    flush_print("EXP 8: Per-seed H rank contribution")
    flush_print("="*90)
    
    rng = np.random.default_rng(44)
    seed_names = {1: "Corner(0,0,0)", 6: "Edge(0,0,1)", 8: "Interior(1,1,1)", 12: "Face(0,1,1)"}
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        parts = []
        for s in sel:
            M_a, M_b, Rs, n_p = build_expansion([s])
            alpha, beta = expand_random(M_a, M_b, Rs, n_p, rng)
            _, H, _, _, _, _ = compute_all_blocks(alpha, beta, Rs)
            rk = np.linalg.matrix_rank(H, tol=1e-8)
            parts.append(f"{seed_names[s]}:rk={rk}")
        
        # Combined
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        _, H, _, _, _, _ = compute_all_blocks(alpha, beta, R)
        rk_comb = np.linalg.matrix_rank(H, tol=1e-8)
        
        flush_print(f"  R={R:>2d}: {' + '.join(parts)} → combined rk(H)={rk_comb}")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 9: Dimension of the "Delta leak" subspace.
# For each (s,t) pair with s≠t, what's the rank of the 
# individual Delta_{s,t} block?  Which pairs leak?
# ════════════════════════════════════════════════════════════════
def exp9():
    flush_print("\n" + "="*90)
    flush_print("EXP 9: Per-channel Delta rank and leak structure")
    flush_print("="*90)
    
    rng = np.random.default_rng(66)
    for R in [13, 19, 20]:
        sel = ALL_CONFIGS[R]
        M_a, M_b, _, n_p = build_expansion(sel)
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        _, H, D, _, _, _ = compute_all_blocks(alpha, beta, R)
        
        rk_H = np.linalg.matrix_rank(H, tol=1e-8)
        U, s, _ = np.linalg.svd(H, full_matrices=False)
        rk = np.sum(s > 1e-8)
        P = U[:, :rk] @ U[:, :rk].T
        
        flush_print(f"\n  R={R}, rk(H)={rk_H}:")
        col = 0
        for si in range(3):
            for ti in range(3):
                if si == ti: continue
                block = D[:, col:col+9]
                resid = block - P @ block
                rk_block = np.linalg.matrix_rank(block, tol=1e-8)
                rk_leak = np.linalg.matrix_rank(resid, tol=1e-8)
                flush_print(f"    Δ[{si},{ti}]: rk={rk_block}  leak={rk_leak}")
                col += 9


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 10: The effective parameter space — what's the image
# of the bilinear map?  Specifically: the joint image 
# (Sigma, H, Delta) as a function of params. What's its dimension
# in the ambient R×(9+18+54) = R×81 space?
# ════════════════════════════════════════════════════════════════
def exp10():
    flush_print("\n" + "="*90)
    flush_print("EXP 10: Dimension of the image of the bilinear map params→(Σ,H,Δ)")
    flush_print("="*90)
    flush_print("  (Jacobian rank of the map from eff params to all blocks)")
    
    rng = np.random.default_rng(88)
    for R in [7, 8, 9, 12, 13, 14, 19, 20, 21, 27]:
        if R not in ALL_CONFIGS: continue
        sel = ALL_CONFIGS[R]
        M_a, M_b, _, n_p = build_expansion(sel)
        
        x0 = rng.standard_normal(n_p)
        eps = 1e-7
        
        def full_map(x):
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S, H, D, N, _, _ = compute_all_blocks(alpha, beta, R)
            return np.hstack([S, H, D]).ravel()
        
        f0 = full_map(x0)
        J = np.zeros((len(f0), n_p))
        for j in range(n_p):
            xp = x0.copy(); xp[j] += eps
            J[:, j] = (full_map(xp) - f0) / eps
        
        rk_J = np.linalg.matrix_rank(J, tol=1e-4)
        flush_print(f"  R={R:>2d}: map R^{n_p} → R^{len(f0)}, Jacobian rank = {rk_J}  "
                    f"(codim in image = {len(f0) - rk_J})")


# ═══ RUN ALL ═══
if __name__ == "__main__":
    t0 = time.time()
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    exp6()
    exp7()
    exp8()
    exp9()
    exp10()
    flush_print(f"\nTotal time: {time.time()-t0:.1f}s")
