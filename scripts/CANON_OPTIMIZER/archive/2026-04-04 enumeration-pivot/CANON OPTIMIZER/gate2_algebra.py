"""
ALGEBRAIC GATE-2 TEST
Does Δ ⊂ span(H) hold identically on the rank(H)=target variety?

For each R in {13, 19, 20}:
  1. Optimize to reach rank(H) = R-9 (SV tail → 0)
  2. On that subvariety, check rank(Nuisance) vs rank(H)
  3. If rk(N) == rk(H) always → Gate 2 is algebraically automatic
"""

import numpy as np
from itertools import permutations, product as iproduct
from scipy.optimize import minimize
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

# ═══ Precompute expansion matrices (like canon_newton) ═══
def _build_expansion(sel):
    """Build linear expansion matrices M_a, M_b for given orbit selection."""
    seed_list = [_ORBITS[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _ORBITS[s][1]
    kept_sorted = sorted(kept)
    kept_idx = {t:i for i,t in enumerate(kept_sorted)}
    R = len(kept_sorted)
    
    # Stabilizers and transporters
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
    
    n_params = len(seed_list) * 18  # only α, β per seed
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

def _compute_blocks(alpha, beta, R):
    """Compute H, Delta, Nuisance, Sigma from expanded alpha, beta."""
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    for k in range(R):
        Eta1[k] = (np.outer(alpha[k][:,0], beta[k][0,:]) -
                   np.outer(alpha[k][:,1], beta[k][1,:])).ravel()
        Eta2[k] = (np.outer(alpha[k][:,1], beta[k][1,:]) -
                   np.outer(alpha[k][:,2], beta[k][2,:])).ravel()
    H = np.hstack([Eta1, Eta2])
    
    # Delta: outer(alpha[:,s], beta[t,:]) for s!=t → 6 blocks of 9 each = 54
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
    return H, Delta, Nuisance, Sigma

# ═══ Main test ═══
configs = {
    13: [1, 12],
    20: [8, 12],
    19: [1, 6, 12],
}

N_TRIALS = 30

print("="*90)
print("ALGEBRAIC GATE-2 TEST: Δ ⊂ span(H) on the rank(H)=target variety?")
print("="*90)
sys.stdout.flush()

for R, sel in configs.items():
    target = R - 9
    
    # Precompute expansion
    M_a, M_b, R_check, n_params = _build_expansion(sel)
    assert R_check == R
    
    # Effective α,β param rank
    M = np.vstack([M_a, M_b])
    eff_rank = int(np.sum(np.linalg.svd(M, compute_uv=False) > 1e-10))
    
    print(f"\n{'─'*90}")
    print(f"R={R}, orbits={sel}, target rank(H)={target}")
    print(f"  {n_params} raw params, {eff_rank} effective α,β params")
    print(f"  Running {N_TRIALS} optimization trials...")
    sys.stdout.flush()
    
    on_variety = []
    off_variety = []
    
    t0 = time.time()
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(trial * 137 + R * 31)
        x0 = rng.standard_normal(n_params) * 0.5
        
        # Fast objective using precomputed matrices
        def obj(x):
            alpha = (M_a @ x).reshape(R, 3, 3)
            beta  = (M_b @ x).reshape(R, 3, 3)
            Eta1 = np.zeros((R, 9))
            Eta2 = np.zeros((R, 9))
            for k in range(R):
                Eta1[k] = (np.outer(alpha[k][:,0], beta[k][0,:]) -
                           np.outer(alpha[k][:,1], beta[k][1,:])).ravel()
                Eta2[k] = (np.outer(alpha[k][:,1], beta[k][1,:]) -
                           np.outer(alpha[k][:,2], beta[k][2,:])).ravel()
            H = np.hstack([Eta1, Eta2])
            sv = np.linalg.svd(H, compute_uv=False)
            return float(np.sum(sv[target:]**2))
        
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-30, 'gtol': 1e-15})
        
        alpha = (M_a @ res.x).reshape(R, 3, 3)
        beta  = (M_b @ res.x).reshape(R, 3, 3)
        H, Delta, Nuisance, Sigma = _compute_blocks(alpha, beta, R)
        
        sv_H = np.linalg.svd(H, compute_uv=False)
        rk_H = int(np.sum(sv_H > 1e-8))
        rk_N = int(np.linalg.matrix_rank(Nuisance, tol=1e-8))
        
        SN = np.hstack([Sigma, Nuisance])
        rk_SN = int(np.linalg.matrix_rank(SN, tol=1e-8))
        gate3 = (rk_SN == rk_N + 9)
        
        tag = "✓" if rk_H == target else " "
        g2 = "G2✓" if rk_N == rk_H else f"G2✗(N={rk_N})"
        g3 = "G3✓" if gate3 else "G3✗"
        
        print(f"  [{trial+1:>2d}/{N_TRIALS}] {tag} rk(H)={rk_H:>2d} rk(N)={rk_N:>2d} "
              f"sv_tail={sv_H[target] if target<len(sv_H) else 0:.2e} "
              f"{g2} {g3} loss={res.fun:.2e}")
        sys.stdout.flush()
        
        if rk_H == target:
            on_variety.append((rk_H, rk_N, gate3))
        else:
            off_variety.append((rk_H, rk_N, gate3))
    
    elapsed = time.time() - t0
    print(f"\n  Summary for R={R} ({elapsed:.1f}s):")
    print(f"    Hit rank(H)={target}: {len(on_variety)}/{N_TRIALS}")
    
    if on_variety:
        g2_pass = sum(1 for h,n,_ in on_variety if n == h)
        g3_pass = sum(1 for _,_,g in on_variety if g)
        rk_N_vals = sorted(set(n for _,n,_ in on_variety))
        print(f"    ON VARIETY: rank(Nuisance) = {rk_N_vals}")
        print(f"    Gate 2 pass (rk_N==rk_H): {g2_pass}/{len(on_variety)}")
        print(f"    Gate 3 pass (Γ solvable):  {g3_pass}/{len(on_variety)}")
        if g2_pass == len(on_variety):
            print(f"    *** Gate 2 appears ALGEBRAICALLY AUTOMATIC for R={R} ***")
    else:
        rk_vals = sorted(set(h for h,_,_ in off_variety))
        print(f"    Could not reach target. rank(H) values: {rk_vals}")
    sys.stdout.flush()

print(f"\n{'='*90}")
print("DONE")
