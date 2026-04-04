"""
FIXED TRANSPORT: Correct group action on factors (α, β, γ).

The key insight: For term (r,s,u), the three factors are:
  α ∈ Mat(n,n) with α[r,s] ← factor for (coord0, coord1)
  β ∈ Mat(n,n) with β[s,u] ← factor for (coord1, coord2)  
  γ ∈ Mat(n,n) with γ[r,u] ← factor for (coord0, coord2)

The group element g=(π,ε) acts on the INDEX TRIPLE (r,s,u) → (r',s',u').
The new triple uses the same tensor decomposition structure, so:
  α'[r',s'] = α[r,s]    (α' is the factor for new-coord0, new-coord1)
  β'[s',u'] = β[s,u]    (β' is the factor for new-coord1, new-coord2)
  γ'[r',u'] = γ[r,u]    (γ' is the factor for new-coord0, new-coord2)

But here's the critical point: the GROUP ELEMENT permutes COORDINATES,
and the three factors correspond to PAIRS of coordinates:
  α ↔ (coord0, coord1)
  β ↔ (coord1, coord2)
  γ ↔ (coord0, coord2)

When π permutes coordinates, the NEW α (for new coord0, new coord1)
comes from the OLD factor that was associated with the pair
(π⁻¹(0), π⁻¹(1)) in the original coordinates.
"""

import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))


def _apply_factors_fixed(pi, eps, a, b, g):
    """Correct group action on factor triple.
    
    The group g=(π,ε) maps triple (r,s,u) to (r',s',u') where:
      (r',s',u') = (ε₀(t_{π⁻¹(0)}), ε₁(t_{π⁻¹(1)}), ε₂(t_{π⁻¹(2)}))
    Wait no — let me re-derive.
    
    _apply_triple does: for output position i, take input position π[i], 
    apply swap if ε[i]. So new[i] = ε_i(old[π[i]]).
    
    new_coord0 = ε₀(old_{π(0)})
    new_coord1 = ε₁(old_{π(1)})
    new_coord2 = ε₂(old_{π(2)})
    
    The factor for the pair (new_coord0, new_coord1) = α'.
    α'[new_val0, new_val1] should equal (old factor for pair (π(0), π(1)))[old_val_{π(0)}, old_val_{π(1)}]
    
    The pair (π(0), π(1)) in the original: which factor was it?
    Original pairs: α↔(0,1), β↔(1,2), γ↔(0,2)
    
    So we need to map: 
      new factor for pair (0,1) ← old factor for pair (π(0), π(1))
      new factor for pair (1,2) ← old factor for pair (π(1), π(2))
      new factor for pair (0,2) ← old factor for pair (π(0), π(2))
    
    And each comes with index permutation matrices from ε.
    """
    
    def _perm_mat(e):
        """Permutation matrix for swap12: maps index i→swap12(i) if e=True."""
        return np.eye(3)[[0, 2, 1], :] if e else np.eye(3)
    
    P = [_perm_mat(eps[i]) for i in range(3)]
    
    # Map from pair to original factor
    pair_to_factor = {(0, 1): a, (1, 2): b, (0, 2): g}
    
    # For each new factor, find which old factor it comes from
    new_factors = {}
    for (i, j), name in [((0, 1), 'a'), ((1, 2), 'b'), ((0, 2), 'g')]:
        # New factor for pair (i,j) comes from old pair (π(i), π(j))
        oi, oj = pi[i], pi[j]
        key = (min(oi, oj), max(oi, oj))
        F = pair_to_factor[key]
        
        # If (oi, oj) is in canonical order, use F directly
        # If reversed, we need to transpose
        if oi <= oj:
            F_oriented = F.copy()
        else:
            F_oriented = F.T
        
        # Apply index permutations: P[i] on rows, P[j] on columns
        new_factors[name] = P[i] @ F_oriented @ P[j].T
    
    return new_factors['a'], new_factors['b'], new_factors['g']


# ═══ Verify against standard algorithm ═══
def e(i): v = np.zeros(3); v[i] = 1; return v

flush("="*90)
flush("VERIFICATION: Fixed transport vs standard algorithm")
flush("="*90)

all_pass = True

for seed_name, seed, a_s, b_s, g_s in [
    ("Corner (0,0,0)", (0,0,0), np.outer(e(0),e(0)), np.outer(e(0),e(0)), np.outer(e(0),e(0))),
    ("Edge (0,0,1)",   (0,0,1), np.outer(e(0),e(0)), np.outer(e(0),e(1)), np.outer(e(0),e(1))),
    ("Face (0,1,1)",   (0,1,1), np.outer(e(0),e(1)), np.outer(e(1),e(1)), np.outer(e(0),e(1))),
    ("Interior (1,1,1)", (1,1,1), np.outer(e(1),e(1)), np.outer(e(1),e(1)), np.outer(e(1),e(1))),
]:
    flush(f"\n{seed_name}:")
    
    transporters = {}
    for pi, eps in _GROUP:
        t = _apply_triple(pi, eps, seed)
        if t not in transporters:
            transporters[t] = (pi, eps)
    
    n_ok = 0
    n_fail = 0
    for t in sorted(transporters):
        pi, eps = transporters[t]
        r, s, u = t
        
        a_exp = np.outer(e(r), e(s))
        b_exp = np.outer(e(s), e(u))
        g_exp = np.outer(e(r), e(u))
        
        a_got, b_got, g_got = _apply_factors_fixed(pi, eps, a_s, b_s, g_s)
        
        err = max(np.linalg.norm(a_exp-a_got), np.linalg.norm(b_exp-b_got), np.linalg.norm(g_exp-g_got))
        
        if err < 1e-10:
            n_ok += 1
        else:
            n_fail += 1
            all_pass = False
            flush(f"  ✗ {t}: π={list(pi)} ε={[int(x) for x in eps]}  max_err={err:.2e}")
            flush(f"    α expected:\n{a_exp}\n    α got:\n{a_got}")
    
    flush(f"  {n_ok}/{n_ok+n_fail} correct")

flush(f"\n{'='*90}")
if all_pass:
    flush("ALL TRANSPORT CORRECT ✓")
else:
    flush("SOME TRANSPORTS FAILED ✗")

# ═══ Now rebuild expansion with fixed transport and test ranks ═══
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

def build_fixed_expansion(sel):
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
    
    n_p = len(seed_list) * 18
    M_a = np.zeros((R*9, n_p))
    M_b = np.zeros((R*9, n_p))
    
    for j in range(n_p):
        ev = np.zeros(n_p); ev[j] = 1.0
        for si, seed in enumerate(seed_list):
            off = si * 18
            a = ev[off:off+9].reshape(3,3)
            b = ev[off+9:off+18].reshape(3,3)
            g = np.zeros((3,3))
            for t, (pi, eps) in transport_cache[seed].items():
                at, bt, _ = _apply_factors_fixed(pi, eps, a, b, g)
                k = kept_idx[t]
                M_a[k*9:(k+1)*9, j] = at.ravel()
                M_b[k*9:(k+1)*9, j] = bt.ravel()
    
    return M_a, M_b, R, n_p, kept_sorted

if all_pass:
    flush("\n" + "="*90)
    flush("FULL GATE DIAGNOSTIC WITH FIXED TRANSPORT")
    flush("="*90)
    
    # Test A: standard in family?
    sel27 = [1,6,8,12]
    M_a, M_b, R, n_p, kept = build_fixed_expansion(sel27)
    
    alpha_std = np.zeros((27,3,3))
    beta_std = np.zeros((27,3,3))
    for k, (r,s,u) in enumerate(kept):
        alpha_std[k,r,s] = 1.0
        beta_std[k,s,u] = 1.0
    
    target = np.concatenate([alpha_std.reshape(R*9), beta_std.reshape(R*9)])
    M = np.vstack([M_a, M_b])
    x_sol, _, rank, _ = np.linalg.lstsq(M, target, rcond=None)
    err_a = np.linalg.norm(M_a @ x_sol - alpha_std.reshape(R*9))
    err_b = np.linalg.norm(M_b @ x_sol - beta_std.reshape(R*9))
    flush(f"\nStandard algorithm fit: ||err_α||={err_a:.2e}  ||err_β||={err_b:.2e}  rank={rank}/{n_p}")
    
    if err_a < 1e-10 and err_b < 1e-10:
        flush("✓ Standard algorithm IN the fixed family!")
    
    # Test B: generic ranks
    rng = np.random.default_rng(42)
    flush(f"\n{'R':>3s} {'orbits':>15s} {'n_p':>4s} | {'rk_S':>4s} {'rk_H':>4s} {'rk_N':>4s} | {'tgt_H':>5s} {'G2':>3s} {'G3':>3s}")
    flush("-"*70)
    
    for R, sel in sorted(ALL_CONFIGS.items()):
        M_a, M_b, _, n_p, _ = build_fixed_expansion(sel)
        x = rng.standard_normal(n_p)
        alpha = (M_a @ x).reshape(R,3,3)
        beta = (M_b @ x).reshape(R,3,3)
        
        S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
        Eta1 = np.zeros((R,9)); Eta2 = np.zeros((R,9))
        for k in range(R):
            Eta1[k] = (np.outer(alpha[k][:,0], beta[k][0,:]) - np.outer(alpha[k][:,1], beta[k][1,:])).ravel()
            Eta2[k] = (np.outer(alpha[k][:,1], beta[k][1,:]) - np.outer(alpha[k][:,2], beta[k][2,:])).ravel()
        H = np.hstack([Eta1, Eta2])
        D = np.zeros((R, 54)); col = 0
        for s in range(3):
            for t in range(3):
                if s == t: continue
                for k in range(R):
                    D[k, col:col+9] = np.outer(alpha[k][:,s], beta[k][t,:]).ravel()
                col += 9
        N = np.hstack([H, D])
        SN = np.hstack([S, N])
        
        rk_S = np.linalg.matrix_rank(S, tol=1e-8)
        rk_H = np.linalg.matrix_rank(H, tol=1e-8)
        rk_N = np.linalg.matrix_rank(N, tol=1e-8)
        rk_SN = np.linalg.matrix_rank(SN, tol=1e-8)
        tgt = R - 9
        g2 = "Y" if rk_N == rk_H else "n"
        g3 = "Y" if rk_SN == rk_N + 9 else "n"
        flush(f"{R:>3d} {str(sel):>15s} {n_p:>4d} | {rk_S:>4d} {rk_H:>4d} {rk_N:>4d} | {tgt:>5d} {g2:>3s} {g3:>3s}")
    
    # Test C: 9×9 minors
    flush(f"\n9×9 minors of Σ:")
    from itertools import combinations
    for R in [12, 13, 19, 20, 27]:
        sel = ALL_CONFIGS[R]
        M_a, M_b, _, n_p, _ = build_fixed_expansion(sel)
        rows_combos = list(combinations(range(R), 9))
        if len(rows_combos) > 200:
            rows_combos = [rows_combos[i] for i in rng.choice(len(rows_combos), 200, replace=False)]
        
        max_det = 0
        for _ in range(500):
            x = rng.standard_normal(n_p)
            alpha = (M_a @ x).reshape(R,3,3)
            beta = (M_b @ x).reshape(R,3,3)
            S = np.array([(alpha[k] @ beta[k]).ravel() for k in range(R)])
            for rows in rows_combos[:50]:
                d = abs(np.linalg.det(S[list(rows),:]))
                if d > max_det: max_det = d
        
        status = "✓ rk=9 achievable" if max_det > 1e-10 else "✗ rigid rk<9"
        flush(f"  R={R:>2d}: max|det|={max_det:.4e}  {status}")
