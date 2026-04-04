"""
GATE 3-FIRST ATTACK: Sigma rank anatomy across ALL orbit configurations.

Questions:
1. Which components of the 9-dim matrix space does each seed's Σ span?
2. What's the Σ column space in the irrep basis of 3×3 matrices?
3. Can structured (non-generic) params boost rk(Σ)?
4. Which R values can potentially reach rk(Σ)=9?
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
        for pi, eps in _GROUP: orb.add(_apply_triple(pi, eps, seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_ORBITS = _compute_orbits()

ALL_CONFIGS = {
    7: [1,6], 8: [8], 9: [1,8], 12: [12], 13: [1,12],
    14: [6,8], 15: [1,6,8], 18: [6,12], 19: [1,6,12],
    20: [8,12], 21: [1,8,12], 26: [6,8,12], 27: [1,6,8,12],
}

def build_expansion(sel):
    seed_list = [_ORBITS[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _ORBITS[s][1]
    kept_sorted = sorted(kept)
    kept_idx = {t:i for i,t in enumerate(kept_sorted)}
    R = len(kept_sorted)
    stab_cache = {}; transport_cache = {}
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
                aa += a2; bb += b2
            n = len(stab); aa /= n; bb /= n
            for t,(pi,eps) in transport_cache[seed].items():
                at,bt,_ = _apply_factors(pi,eps,aa,bb,g0)
                k = kept_idx[t]
                M_a[k*9:(k+1)*9, j] = at.ravel()
                M_b[k*9:(k+1)*9, j] = bt.ravel()
    return M_a, M_b, R, n_params, seed_list, kept_sorted, kept_idx, transport_cache, stab_cache

def expand_random(M_a, M_b, R, n_p, rng):
    x = rng.standard_normal(n_p)
    return (M_a @ x).reshape(R,3,3), (M_b @ x).reshape(R,3,3)

# ═══ PART 1: Σ irrep decomposition ═══
# The 9-dim space of 3×3 matrices under Z₂≀S₃ decomposes into invariant subspaces.
# Let's find them empirically by looking at the group action on vec(M).
def sigma_irrep_analysis():
    flush("\n" + "="*90)
    flush("PART 1: Irrep decomposition of the 9-dim Σ space under Z₂≀S₃")
    flush("="*90)
    
    # Build the 9×9 representation matrices for the group acting on vec(3×3 matrix)
    # A 3×3 matrix M with M_{ij} = (α·β)_{ij} transforms as:
    # Under (π, ε): M → P_r · M · P_u^T  where P_r, P_u are the row/col permutations
    # But the group acts on (r,s,u) and Σ = α·β sums over s, giving a matrix in (r,u).
    # The group action on the (r,u) indices: π permutes which is r and which is u,
    # ε swaps 1↔2 in each.
    
    reps = []
    for pi, eps in _GROUP:
        # Action on a 3×3 matrix M[r,u]:
        # The triple (r,s,u) maps under (π,ε). For Σ = α·β, the result is in (r,u) space.
        # π permutes the 3 coordinates. If π maps coord 0→a, coord 2→c, then
        # the new row index is the image of the old row index under coord a's ε,
        # and new col index is image under coord c's ε.
        # But we also need to handle the transposition when π swaps r↔u.
        
        # Simpler: just compute how M[i,j] maps for each (i,j).
        # A term (r,s,u) gets α[r,s]·β[s,u] → contributes to Σ[r,u].
        # Under g=(π,ε), the term maps to (r',s',u') and factors transform.
        # The result Σ'[r',u'] = Σ[r,u] after relabeling.
        # So the rep on Σ-space is: which (r',u') does (r,u) map to?
        
        mat = np.zeros((9, 9))
        for r in range(3):
            for u in range(3):
                # Apply (π,ε) to a triple (r, *, u) — we need the image of (r,u)
                # in the (row, col) space.
                # The group acts on {0,1,2}³. Coord 0 = row of A, coord 2 = col of B.
                # Under π: coord i goes to position π(i).
                # Under ε: value v in coord i maps to swap12(v) if ε[i].
                
                # Create a dummy triple and apply
                triple = [r, 0, u]  # s=0 dummy
                x = [triple[pi[i]] for i in range(3)]
                r2 = _swap12(x[0]) if eps[0] else x[0]
                u2 = _swap12(x[2]) if eps[2] else x[2]
                
                # But wait — π can swap which coordinate is "row" vs "col"
                # If π sends coord 0 to position 1 (summation), then the meaning changes.
                # This is the key subtlety: when π permutes coordinates, the roles change.
                # Coord 0 = row of A (α index), coord 1 = summation, coord 2 = col of B (β index)
                # After π, the "row" role is played by whatever coord π⁻¹ maps to 0.
                
                pi_inv = [0]*3
                for i in range(3): pi_inv[pi[i]] = i
                
                # The new Σ = α'·β' product. α' uses coords at positions 0,1 and β' uses 1,2.
                # In original coords, position 0 is pi_inv[0], position 2 is pi_inv[2].
                # So the "row" in the new frame comes from original coord pi_inv[0],
                # and the "col" comes from original coord pi_inv[2].
                
                # Original value at coord pi_inv[0] from our triple:
                orig_val_row = triple[pi_inv[0]]  # value of the coord that becomes "row"
                orig_val_col = triple[pi_inv[2]]  # value of the coord that becomes "col"
                
                # Apply ε to those coords:
                new_row = _swap12(orig_val_row) if eps[pi_inv[0]] else orig_val_row
                new_col = _swap12(orig_val_col) if eps[pi_inv[2]] else orig_val_col
                
                # Hmm, this is getting complicated. Let me just do it numerically.
                pass
        
        # Actually, let's just compute it by applying the group to actual α,β matrices.
        reps.append(None)  # placeholder
    
    # NUMERICAL APPROACH: sample random α,β for each seed, compute Σ, see which
    # components of vec(Σ) are hit.
    
    flush("\nPer-seed Σ column space analysis (which entries of 3×3 output are spanned):")
    flush("-" * 90)
    
    rng = np.random.default_rng(42)
    seed_names = {1: "Corner", 6: "Edge", 8: "Interior", 12: "Face"}
    
    # For each seed alone, sample many random params and look at span of Σ vectors
    for orb_size in [1, 6, 8, 12]:
        M_a, M_b, R, n_p = build_expansion([orb_size])[:4]
        
        # Collect many Σ samples
        sigmas = []
        for _ in range(100):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            sigmas.append(S)
        
        # Stack all and find the span
        all_S = np.vstack(sigmas)
        U, s, Vt = np.linalg.svd(all_S, full_matrices=False)
        rk = np.sum(s > 1e-8 * s[0])
        
        flush(f"\n  {seed_names[orb_size]} (|orbit|={orb_size}, {n_p} params):")
        flush(f"    Σ spans {rk} dimensions of the 9-dim output space")
        flush(f"    Singular values: {np.round(s[:min(9,len(s))], 4)}")
        flush(f"    Basis vectors (as 3×3 patterns):")
        for i in range(rk):
            M = Vt[i].reshape(3, 3)
            # Show which entries are nonzero
            pattern = np.where(np.abs(M) > 0.1 * np.max(np.abs(M)), "X", ".")
            flush(f"      v{i}: {pattern[0]}  {pattern[1]}  {pattern[2]}  "
                  f"(norm pattern: {np.round(M, 3).tolist()})")

# ═══ PART 2: For each R, what's the MAXIMUM Σ rank achievable? ═══
# Instead of generic points, try structured params (sparse, integer, etc.)
def sigma_max_rank_search():
    flush("\n" + "="*90)
    flush("PART 2: Maximum Σ rank search across ALL R (structured + random params)")
    flush("="*90)
    
    rng = np.random.default_rng(123)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)[:4]
        
        best_rk = 0
        best_method = ""
        
        # Method 1: Random Gaussian
        for _ in range(50):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > best_rk: best_rk = rk; best_method = "gaussian"
        
        # Method 2: Sparse ±1 params
        for _ in range(200):
            x = rng.choice([-1, 0, 1], size=n_p, p=[0.3, 0.4, 0.3])
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > best_rk: best_rk = rk; best_method = "sparse±1"
        
        # Method 3: One-hot params (one param = 1, rest = 0)
        for j in range(n_p):
            x = np.zeros(n_p); x[j] = 1.0
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > best_rk: best_rk = rk; best_method = f"one-hot[{j}]"
        
        # Method 4: Two-param combos
        for _ in range(500):
            x = np.zeros(n_p)
            idx = rng.choice(n_p, size=2, replace=False)
            x[idx] = rng.choice([-1, 1], size=2)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > best_rk: best_rk = rk; best_method = "2-param"
        
        target = 9
        gap = target - best_rk
        status = "✓ ACHIEVABLE" if gap <= 0 else f"gap={gap}"
        flush(f"  R={R:>2d} {str(sel):>15s}: best rk(Σ)={best_rk}  (need 9)  {status}  via {best_method}")

# ═══ PART 3: The Σ linearization — Σ_k = α_k·β_k is bilinear. ═══
# Build the linear map from params to Σ-space and analyze its structure.
def sigma_linearization():
    flush("\n" + "="*90)
    flush("PART 3: Σ as bilinear map — linearized Jacobian structure")
    flush("="*90)
    
    rng = np.random.default_rng(77)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)[:4]
        
        # At a random point, compute Jacobian of the map params → vec(Σ)
        x0 = rng.standard_normal(n_p)
        eps = 1e-7
        
        def sigma_map(x):
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            return np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)]).ravel()
        
        f0 = sigma_map(x0)
        J = np.zeros((len(f0), n_p))
        for j in range(n_p):
            xp = x0.copy(); xp[j] += eps
            J[:, j] = (sigma_map(xp) - f0) / eps
        
        rk_J = np.linalg.matrix_rank(J, tol=1e-4)
        
        # Also: what's the rank of J projected to the COLUMN space of Σ
        # i.e., J reshaped as (R, 9, n_p), then for the 9-dim output
        # J_col[i,j] = d(Σ[:,i])/d(x_j) → rank tells us how many output dims are reachable
        S0 = f0.reshape(R, 9)
        rk_S = np.linalg.matrix_rank(S0, tol=1e-8)
        
        # Column-space Jacobian: can perturbations increase rk(Σ)?
        # Project J onto the orthogonal complement of current column space
        U, s, _ = np.linalg.svd(S0, full_matrices=False)
        rk = np.sum(s > 1e-8)
        P_perp = np.eye(R) - U[:,:rk] @ U[:,:rk].T  # projects off current Σ col space
        
        # Reshape J as (R, 9, n_p), project each 9-column
        J_3d = J.reshape(R, 9, n_p)
        # For each output dim i, project the R-vector dΣ[:,i]/dx_j onto P_perp
        J_perp = np.tensordot(P_perp, J_3d, axes=([1],[0]))  # (R, 9, n_p)
        J_perp_flat = J_perp.reshape(R*9, n_p)
        rk_perp = np.linalg.matrix_rank(J_perp_flat, tol=1e-4)
        
        flush(f"  R={R:>2d}: rk(Σ)={rk_S}  Jac_rank={rk_J}  "
              f"Jac_⊥Σ_rank={rk_perp}  (perturbation dims that could grow Σ)")

# ═══ PART 4: The missing Σ components — which 3×3 matrix patterns? ═══
def sigma_missing_components():
    flush("\n" + "="*90)
    flush("PART 4: Which 3×3 output directions are MISSING from Σ span?")
    flush("="*90)
    
    rng = np.random.default_rng(55)
    labels = ["M00","M01","M02","M10","M11","M12","M20","M21","M22"]
    
    for R in [7, 8, 12, 13, 14, 19, 20, 21, 27]:
        if R not in ALL_CONFIGS: continue
        sel = ALL_CONFIGS[R]
        M_a, M_b, _, n_p = build_expansion(sel)[:4]
        
        # Collect Σ from many samples to find the full reachable subspace
        all_rows = []
        for _ in range(200):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            all_rows.append(S)
        
        big = np.vstack(all_rows)  # (200*R, 9)
        U, s, Vt = np.linalg.svd(big, full_matrices=False)
        rk = np.sum(s > 1e-6 * s[0])
        
        flush(f"\n  R={R} (orbits {sel}): Σ reachable subspace dim = {rk}/9")
        
        if rk < 9:
            # Show the missing directions
            missing = Vt[rk:]  # rows of Vt beyond rank
            for i in range(9 - rk):
                v = missing[i]
                M = v.reshape(3, 3)
                flush(f"    Missing direction {i}: {np.round(M, 3).tolist()}")
                # Characterize: is it diagonal? symmetric? antisymmetric?
                diag_norm = np.linalg.norm(np.diag(np.diag(M)))
                sym_norm = np.linalg.norm(M + M.T) / 2
                asym_norm = np.linalg.norm(M - M.T) / 2
                total = np.linalg.norm(M)
                flush(f"             diag={diag_norm/total:.2f}  sym={sym_norm/total:.2f}  "
                      f"asym={asym_norm/total:.2f}")

# ═══ PART 5: The bilinear rank — theoretical maximum ═══ 
# Σ_k = α_k · β_k. If α_k has rank r_a and β_k has rank r_b,
# then α_k·β_k has rank ≤ min(r_a, r_b).
# But Σ as a SET of R matrices — what's the max rank of the R×9 matrix?
# This depends on how many independent products α·β the symmetry allows.
def sigma_bilinear_rank_bound():
    flush("\n" + "="*90)
    flush("PART 5: Bilinear rank bounds — per-seed α,β ranks and Σ constraints")
    flush("="*90)
    
    rng = np.random.default_rng(99)
    seed_names = {1: "Corner", 6: "Edge", 8: "Interior", 12: "Face"}
    
    for orb_size in [1, 6, 8, 12]:
        M_a, M_b, R, n_p = build_expansion([orb_size])[:4]
        
        # Sample and check ranks of individual α_k and β_k
        a_ranks = []; b_ranks = []; prod_ranks = []
        for _ in range(50):
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            for k in range(R):
                a_ranks.append(np.linalg.matrix_rank(alpha[k], tol=1e-8))
                b_ranks.append(np.linalg.matrix_rank(beta[k], tol=1e-8))
                prod_ranks.append(np.linalg.matrix_rank(alpha[k] @ beta[k], tol=1e-8))
        
        flush(f"\n  {seed_names[orb_size]} (|orbit|={orb_size}):")
        flush(f"    Generic rk(α_k): {sorted(set(a_ranks))}")
        flush(f"    Generic rk(β_k): {sorted(set(b_ranks))}")
        flush(f"    Generic rk(α_k·β_k): {sorted(set(prod_ranks))}")
        
        # The individual product rank bounds the contribution
        # If each α_k·β_k has rank r, the R×9 matrix Σ has rank ≤ min(R, 9)
        # But the EFFECTIVE rank is constrained by symmetry

# ═══ PART 6: Cross-orbit Σ interaction — do combined orbits create NEW Σ directions? ═══
def sigma_cross_orbit():
    flush("\n" + "="*90)
    flush("PART 6: Cross-orbit Σ interaction — does combining orbits unlock new Σ dims?")
    flush("="*90)
    
    rng = np.random.default_rng(33)
    seed_names = {1: "C", 6: "E", 8: "I", 12: "F"}
    
    # For each pair and triple of orbits, check if combined rk(Σ) > sum of individual rk(Σ)
    orbits = [1, 6, 8, 12]
    individual_rk = {}
    
    for o in orbits:
        M_a, M_b, R, n_p = build_expansion([o])[:4]
        alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
        S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
        individual_rk[o] = np.linalg.matrix_rank(S, tol=1e-8)
    
    flush(f"\n  Individual: " + "  ".join(f"{seed_names[o]}={individual_rk[o]}" for o in orbits))
    
    from itertools import combinations
    for size in [2, 3, 4]:
        flush(f"\n  {size}-orbit combos:")
        for combo in combinations(orbits, size):
            sel = list(combo)
            R = sum(sel)
            M_a, M_b, _, n_p = build_expansion(sel)[:4]
            alpha, beta = expand_random(M_a, M_b, R, n_p, rng)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk_comb = np.linalg.matrix_rank(S, tol=1e-8)
            rk_sum = sum(individual_rk[o] for o in combo)
            synergy = rk_comb - rk_sum
            name = "+".join(seed_names[o] for o in combo)
            flush(f"    {name:>10s} (R={R:>2d}): rk(Σ)={rk_comb}  "
                  f"(sum={rk_sum}, synergy={synergy:+d})")

# ═══ PART 7: The killer question — on which subvariety does rk(Σ) increase? ═══
# Compute the SECOND-ORDER conditions for Σ rank increase.
def sigma_rank_increase_conditions():
    flush("\n" + "="*90)
    flush("PART 7: Can rk(Σ) EVER exceed generic? (tangent space to rank-increase locus)")
    flush("="*90)
    
    rng = np.random.default_rng(44)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p = build_expansion(sel)[:4]
        
        # At a generic point, rk(Σ) = r. Can we find a direction in param space
        # that increases it? This happens when dΣ/dx has a component in the 
        # null space of the current Σ column space.
        
        max_rk = 0
        # Try many random points
        for trial in range(100):
            x = rng.standard_normal(n_p)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > max_rk: max_rk = rk
        
        # Now try: can we do better with COMPLEX params? 
        # (Real Σ rank might be limited but complex Σ rank might be higher)
        max_rk_complex = 0
        for trial in range(50):
            x = rng.standard_normal(n_p) + 1j * rng.standard_normal(n_p)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            if rk > max_rk_complex: max_rk_complex = rk
        
        flush(f"  R={R:>2d}: max rk(Σ) over reals = {max_rk}  "
              f"over complex = {max_rk_complex}  "
              f"{'★ COMPLEX UNLOCKS MORE' if max_rk_complex > max_rk else ''}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    sigma_irrep_analysis()
    sigma_max_rank_search()
    sigma_linearization()
    sigma_missing_components()
    sigma_bilinear_rank_bound()
    sigma_cross_orbit()
    sigma_rank_increase_conditions()
    flush(f"\nTotal time: {time.time()-t0:.1f}s")
