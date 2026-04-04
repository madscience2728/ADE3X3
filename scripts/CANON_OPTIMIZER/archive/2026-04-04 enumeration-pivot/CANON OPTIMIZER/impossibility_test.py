"""
THE IMPOSSIBILITY TEST:

Upper semicontinuity of matrix rank says: for a polynomial matrix M(x),
the generic rank is the MAXIMUM. Rank can only drop at special points.

If generic rk(Σ) = 5 for R=19, then rk(Σ) ≤ 5 EVERYWHERE in the symmetric family.
Gate 3 needs rk(Σ) = 9. → ALGEBRAICALLY IMPOSSIBLE?

But the standard R=27 algorithm has rk(Σ) = 9. Is it in our family?
If yes, our generic rk=8 measurement for R=27 is wrong.

This script settles the question.
"""

import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

# ═══ Group machinery (same as before) ═══
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
    return M_a, M_b, R, n_params, kept_sorted


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Is the standard algorithm in our symmetric family?
# ═══════════════════════════════════════════════════════════════════
def test_standard_algorithm():
    flush("="*90)
    flush("TEST 1: Is the standard R=27 algorithm in our symmetric family?")
    flush("="*90)
    
    sel = [1, 6, 8, 12]
    M_a, M_b, R, n_p, kept = build_expansion(sel)
    
    # Standard algorithm: term (r,s,u) has α = e_r e_s^T, β = e_s e_u^T
    alpha_std = np.zeros((27, 3, 3))
    beta_std  = np.zeros((27, 3, 3))
    for k, (r, s, u) in enumerate(kept):
        alpha_std[k, r, s] = 1.0
        beta_std[k, s, u] = 1.0
    
    # Check: does there exist x such that M_a @ x = vec(alpha_std) and M_b @ x = vec(beta_std)?
    target_a = alpha_std.reshape(R*9)
    target_b = beta_std.reshape(R*9)
    
    # Stack: [M_a; M_b] x = [target_a; target_b]
    M_full = np.vstack([M_a, M_b])
    target_full = np.concatenate([target_a, target_b])
    
    # Least squares
    x_sol, residuals, rank, sv = np.linalg.lstsq(M_full, target_full, rcond=None)
    
    err_a = np.linalg.norm(M_a @ x_sol - target_a)
    err_b = np.linalg.norm(M_b @ x_sol - target_b)
    
    flush(f"  Least-squares solution: ||error_α|| = {err_a:.2e}, ||error_β|| = {err_b:.2e}")
    flush(f"  System rank: {rank}/{n_p} params, M_full shape: {M_full.shape}")
    
    if err_a < 1e-10 and err_b < 1e-10:
        flush("  ✓ Standard algorithm IS in the symmetric family!")
        
        # Compute Σ at this point
        alpha = (M_a @ x_sol).reshape(R, 3, 3)
        beta  = (M_b @ x_sol).reshape(R, 3, 3)
        S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
        rk_S = np.linalg.matrix_rank(S, tol=1e-8)
        flush(f"  rk(Σ) at standard algorithm point: {rk_S}")
        flush(f"  → If {rk_S} > 8 (our 'generic' measurement), then measurement was wrong!")
    else:
        flush("  ✗ Standard algorithm is NOT in the symmetric family")
        flush("  Checking which terms don't match...")
        alpha_rec = (M_a @ x_sol).reshape(R, 3, 3)
        beta_rec  = (M_b @ x_sol).reshape(R, 3, 3)
        for k in range(R):
            ea = np.linalg.norm(alpha_rec[k] - alpha_std[k])
            eb = np.linalg.norm(beta_rec[k] - beta_std[k])
            if ea > 1e-8 or eb > 1e-8:
                flush(f"    Term {k} {kept[k]}: err_α={ea:.4e}  err_β={eb:.4e}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Careful rank measurement with SVD + tolerance sweep
# ═══════════════════════════════════════════════════════════════════
def test_careful_rank():
    flush("\n" + "="*90)
    flush("TEST 2: SVD-based Σ rank with multiple tolerances")
    flush("="*90)
    
    rng = np.random.default_rng(42)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_expansion(sel)
        
        x = rng.standard_normal(n_p) * 10  # larger scale
        alpha = (M_a @ x).reshape(R, 3, 3)
        beta  = (M_b @ x).reshape(R, 3, 3)
        S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
        
        sv = np.linalg.svd(S, compute_uv=False)
        sv_norm = sv / (sv[0] + 1e-30)
        
        flush(f"\n  R={R:>2d}: singular values of Σ (normalized):")
        flush(f"    {np.array2string(sv_norm, precision=6, max_line_width=120)}")
        
        # Ranks at different tolerances
        for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
            rk = np.sum(sv > tol * sv[0])
            flush(f"    tol={tol:.0e}: rk={rk}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: The 9×9 minors of Σ — are they identically zero?
# Sample many points; if a minor is nonzero anywhere, generic rank ≥ 9
# ═══════════════════════════════════════════════════════════════════
def test_minors():
    flush("\n" + "="*90)
    flush("TEST 3: Are all 9×9 minors of Σ identically zero?")
    flush("="*90)
    flush("  (If any minor is nonzero at any point, rk(Σ)=9 is achievable)")
    
    from itertools import combinations
    rng = np.random.default_rng(999)
    
    for R in [12, 13, 19, 20, 21, 27]:
        sel = ALL_CONFIGS[R]
        M_a, M_b, _, n_p, _ = build_expansion(sel)
        
        if R < 9:
            flush(f"\n  R={R}: fewer than 9 rows, skip")
            continue
        
        # Pick some random 9-row subsets
        row_combos = list(combinations(range(R), 9))
        if len(row_combos) > 200:
            row_combos = [row_combos[i] for i in rng.choice(len(row_combos), 200, replace=False)]
        
        max_det = 0
        best_rows = None
        n_trials = 500
        
        for trial in range(n_trials):
            x = rng.standard_normal(n_p) * (10 if trial < 250 else 0.1)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            
            for rows in row_combos[:50]:  # check subset of row combos per trial
                sub = S[list(rows), :]
                d = abs(np.linalg.det(sub))
                if d > max_det:
                    max_det = d
                    best_rows = rows
        
        flush(f"\n  R={R:>2d}: max |det(9×9 minor)| = {max_det:.6e}  "
              f"{'→ rk(Σ)=9 IS achievable!' if max_det > 1e-10 else '→ all minors ~0, rk(Σ)<9 rigidly'}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Explicit computation — what is the Σ image variety?
# For R=19, parameterize the 24 effective α,β params.
# Build Σ symbolically (as degree-2 polynomials) and check the 
# rank of the coefficient matrix.
# ═══════════════════════════════════════════════════════════════════
def test_sigma_variety():
    flush("\n" + "="*90)
    flush("TEST 4: Σ variety dimension — how many independent degree-2 monomials appear?")
    flush("="*90)
    
    rng = np.random.default_rng(77)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_expansion(sel)
        
        # The entries of Σ are bilinear in the params: Σ[k,j] = Σ_{p,q} C[k,j,p,q] x_p x_q
        # The number of degree-2 monomials is n_p*(n_p+1)/2.
        # We can extract the coefficient tensor by evaluating at many points.
        
        n_mono = n_p * (n_p + 1) // 2
        
        # Build monomial evaluation matrix: for random x, compute x⊗x (symmetrized)
        n_samples = min(n_mono + 50, 2000)
        
        X_mono = np.zeros((n_samples, n_mono))
        Y_sigma = np.zeros((n_samples, R * 9))
        
        for i in range(n_samples):
            x = rng.standard_normal(n_p)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            Y_sigma[i] = S.ravel()
            
            # Monomials x_p * x_q for p ≤ q
            idx = 0
            for p in range(n_p):
                for q in range(p, n_p):
                    X_mono[i, idx] = x[p] * x[q]
                    idx += 1
        
        # Σ = X_mono @ C for some coefficient matrix C
        # Rank of Y_sigma = rank of the image of the bilinear map
        rk_img = np.linalg.matrix_rank(Y_sigma, tol=1e-8)
        rk_mono = np.linalg.matrix_rank(X_mono, tol=1e-8)
        
        flush(f"  R={R:>2d}: n_params={n_p}, n_monomials={n_mono}, "
              f"rk(Σ image)={rk_img}, rk(monomial matrix)={rk_mono}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: For every R ≥ 12 where reachable=9, explicitly find params
# that MAXIMIZE rk(Σ) using the nuclear norm heuristic
# ═══════════════════════════════════════════════════════════════════
def test_maximize_sigma_rank():
    flush("\n" + "="*90)
    flush("TEST 5: Direct search for HIGH rk(Σ) points (structured param grid)")
    flush("="*90)
    
    rng = np.random.default_rng(42)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        if R < 9: continue
        M_a, M_b, _, n_p, _ = build_expansion(sel)
        
        best_rk = 0
        best_sv_ratio = 0
        
        # Strategy: try params that are eigenvectors of M_a^T M_b + M_b^T M_a
        # (cross-correlation matrix — these params maximize the "mixing" of α and β)
        C = M_a.T @ M_b + M_b.T @ M_a
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Try each eigenvector
        for i in range(n_p):
            x = eigvecs[:, i]
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            sv = np.linalg.svd(S, compute_uv=False)
            rk = np.sum(sv > 1e-8 * sv[0])
            if rk > best_rk:
                best_rk = rk
                best_sv_ratio = sv[min(8,len(sv)-1)] / sv[0] if len(sv) > 8 else 0
        
        # Try sums/differences of top eigenvectors
        for _ in range(1000):
            coeffs = rng.choice([-1, 0, 1], size=min(n_p, 10), p=[0.35, 0.3, 0.35])
            idx = rng.choice(n_p, size=min(n_p, 10), replace=False)
            x = np.zeros(n_p)
            for c, i in zip(coeffs, idx):
                x += c * eigvecs[:, i]
            if np.linalg.norm(x) < 1e-10: continue
            
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            sv = np.linalg.svd(S, compute_uv=False)
            rk = np.sum(sv > 1e-8 * sv[0])
            if rk > best_rk:
                best_rk = rk
        
        # Try integer/rational params
        for _ in range(2000):
            x = rng.choice([-2,-1,0,1,2], size=n_p, p=[0.1,0.2,0.4,0.2,0.1]).astype(float)
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            sv = np.linalg.svd(S, compute_uv=False)
            rk = np.sum(sv > 1e-8 * sv[0])
            if rk > best_rk:
                best_rk = rk
        
        status = "✓ FULL" if best_rk >= 9 else f"  gap={9-best_rk}"
        flush(f"  R={R:>2d}: best rk(Σ) found = {best_rk}  {status}")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: THE DEFINITIVE TEST — compute rk(Σ) at the standard
# algorithm parameters for every sub-configuration
# ═══════════════════════════════════════════════════════════════════
def test_standard_per_orbit():
    flush("\n" + "="*90)
    flush("TEST 6: Standard algorithm restricted to each orbit subset")
    flush("="*90)
    
    # Standard algorithm: for term (r,s,u), α = e_r e_s^T, β = e_s e_u^T, γ = e_r e_u^T
    # Σ_k = α_k β_k = e_r e_u^T
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        kept = set()
        for s in sel: kept |= _ORBITS[s][1]
        kept_sorted = sorted(kept)
        
        S = np.zeros((R, 9))
        for k, (r, s, u) in enumerate(kept_sorted):
            # Σ_k = e_r e_u^T
            M = np.zeros((3, 3))
            M[r, u] = 1.0
            S[k] = M.ravel()
        
        rk = np.linalg.matrix_rank(S, tol=1e-8)
        flush(f"  R={R:>2d} {str(sel):>15s}: rk(Σ)_standard = {rk}  (rows cover {len(set(map(tuple, S.tolist())))} distinct patterns)")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: If standard algo IS in the family, what went wrong with
# our generic measurement? Test: is rk=8 stable or does it depend
# on the parameter scale?
# ═══════════════════════════════════════════════════════════════════
def test_scale_dependence():
    flush("\n" + "="*90)
    flush("TEST 7: Does Σ rank depend on parameter scale/structure?")
    flush("="*90)
    
    rng = np.random.default_rng(42)
    sel = [1, 6, 8, 12]
    M_a, M_b, R, n_p, _ = build_expansion(sel)
    
    flush(f"  R=27, {n_p} params")
    
    # Try different distributions
    distributions = {
        "N(0,1)":      lambda: rng.standard_normal(n_p),
        "N(0,100)":    lambda: rng.standard_normal(n_p) * 100,
        "N(0,0.01)":   lambda: rng.standard_normal(n_p) * 0.01,
        "Uniform[-1,1]": lambda: rng.uniform(-1, 1, n_p),
        "Rademacher":  lambda: rng.choice([-1.0, 1.0], n_p),
        "Sparse(0.1)": lambda: rng.standard_normal(n_p) * (rng.random(n_p) < 0.1),
        "Integer[-5,5]": lambda: rng.integers(-5, 6, n_p).astype(float),
        "One-hot":     lambda: (np.arange(n_p) == rng.integers(0, n_p)).astype(float),
    }
    
    for name, gen in distributions.items():
        ranks = []
        for _ in range(100):
            x = gen()
            if np.linalg.norm(x) < 1e-12: continue
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            rk = np.linalg.matrix_rank(S, tol=1e-8)
            ranks.append(rk)
        
        from collections import Counter
        c = Counter(ranks)
        flush(f"  {name:>20s}: rank distribution = {dict(sorted(c.items()))}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_standard_algorithm()
    test_standard_per_orbit()
    test_careful_rank()
    test_minors()
    test_sigma_variety()
    test_maximize_sigma_rank()
    test_scale_dependence()
    flush(f"\nTotal: {time.time()-t0:.1f}s")
