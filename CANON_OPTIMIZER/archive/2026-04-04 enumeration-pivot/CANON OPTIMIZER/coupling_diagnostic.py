"""Diagnose factor coupling: does π mix α↔β↔γ roles?
If yes, the 18-param (α,β only) expansion is wrong — needs 27 params per seed."""
import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

_S3 = list(permutations(range(3)))
_GROUP = [(list(pi), list(eps)) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _apply_triple(pi, eps, t):
    def _swap12(x): return x if x == 0 else 3 - x
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _apply_factors_fixed(pi, eps, a, b, g):
    def _pm(e): return np.eye(3)[[0,2,1],:] if e else np.eye(3)
    P = [_pm(eps[i]) for i in range(3)]
    pair_to_factor = {(0,1): a, (1,2): b, (0,2): g}
    nf = {}
    for (i,j), nm in [((0,1),'a'), ((1,2),'b'), ((0,2),'g')]:
        oi, oj = pi[i], pi[j]
        key = (min(oi,oj), max(oi,oj))
        F = pair_to_factor[key]
        F_o = F.copy() if oi <= oj else F.T
        nf[nm] = P[i] @ F_o @ P[j].T
    return nf['a'], nf['b'], nf['g']


# Part 1: Which factor does each output factor come from, for each permutation?
flush("="*70)
flush("PART 1: Factor role mixing under S3 permutations")
flush("="*70)
for pi in _S3:
    mapping = {}
    for (i,j), nm in [((0,1),'α'), ((1,2),'β'), ((0,2),'γ')]:
        oi, oj = pi[i], pi[j]
        key = (min(oi,oj), max(oi,oj))
        source = {(0,1):'α', (1,2):'β', (0,2):'γ'}[key]
        mapping[nm] = source
    flush(f"  π={list(pi)}: new_α←{mapping['α']}  new_β←{mapping['β']}  new_γ←{mapping['γ']}")


# Part 2: For every transporter in E→E orbit, does new_α ever come from old_γ?
flush("\n" + "="*70)
flush("PART 2: Edge orbit transporters — do any mix γ→α or γ→β?")
flush("="*70)
seed = (0,0,1)
ALL_TERMS = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]
edge_orbit = set()
for pi, eps in _GROUP:
    t = _apply_triple(pi, eps, seed)
    edge_orbit.add(t)

for t in sorted(edge_orbit):
    for pi, eps in _GROUP:
        if _apply_triple(pi, eps, seed) == t:
            for (i,j), nm in [((0,1),'α'), ((1,2),'β')]:
                oi, oj = pi[i], pi[j]
                key = (min(oi,oj), max(oi,oj))
                source = {(0,1):'α', (1,2):'β', (0,2):'γ'}[key]
                if source == 'γ':
                    flush(f"  {seed}→{t} via π={list(pi)}: new_{nm} ← old_γ  *** COUPLING! ***")
            break  # one transporter per target


# Part 3: Build 27-param expansion and test standard algorithm fit
flush("\n" + "="*70)
flush("PART 3: 27-param expansion — does standard algorithm fit?")
flush("="*70)

kept_sorted = sorted(ALL_TERMS)
kept_idx = {t: i for i, t in enumerate(kept_sorted)}
R = 27

seed_list = [(0,0,0), (0,0,1), (0,1,1), (0,1,2)]
n_p = len(seed_list) * 27  # 9 for α + 9 for β + 9 for γ

# Build transport cache
transport_cache = {}
for seed in seed_list:
    tc = {}
    for pi, eps in _GROUP:
        t = _apply_triple(pi, eps, seed)
        if t not in tc:
            tc[t] = (pi, eps)
    transport_cache[seed] = tc

M_a = np.zeros((R*9, n_p))
M_b = np.zeros((R*9, n_p))
M_g = np.zeros((R*9, n_p))

for j in range(n_p):
    ev = np.zeros(n_p); ev[j] = 1.0
    for si, seed in enumerate(seed_list):
        off = si * 27
        a = ev[off:off+9].reshape(3,3)
        b = ev[off+9:off+18].reshape(3,3)
        g = ev[off+18:off+27].reshape(3,3)
        for t, (pi, eps) in transport_cache[seed].items():
            at, bt, gt = _apply_factors_fixed(pi, eps, a, b, g)
            k = kept_idx[t]
            M_a[k*9:(k+1)*9, j] = at.ravel()
            M_b[k*9:(k+1)*9, j] = bt.ravel()
            M_g[k*9:(k+1)*9, j] = gt.ravel()

# Standard algorithm targets
alpha_std = np.zeros((27,3,3))
beta_std = np.zeros((27,3,3))
gamma_std = np.zeros((27,3,3))
for k, (r,s,u) in enumerate(kept_sorted):
    alpha_std[k,r,s] = 1.0
    beta_std[k,s,u] = 1.0
    gamma_std[k,r,u] = 1.0

target = np.concatenate([alpha_std.reshape(R*9), beta_std.reshape(R*9), gamma_std.reshape(R*9)])
M = np.vstack([M_a, M_b, M_g])
x_sol, _, rank, _ = np.linalg.lstsq(M, target, rcond=None)

err_a = np.linalg.norm(M_a @ x_sol - alpha_std.reshape(R*9))
err_b = np.linalg.norm(M_b @ x_sol - beta_std.reshape(R*9))
err_g = np.linalg.norm(M_g @ x_sol - gamma_std.reshape(R*9))
flush(f"27-param: ||err_α||={err_a:.2e}  ||err_β||={err_b:.2e}  ||err_γ||={err_g:.2e}  rank={rank}/{n_p}")

if err_a < 1e-10 and err_b < 1e-10 and err_g < 1e-10:
    flush("✓ Standard algorithm IN the 27-param family!")
else:
    flush("✗ Standard algorithm NOT in the 27-param family")

# Also test 18-param for comparison
flush("\n--- Comparison: 18-param (γ=0) ---")
n_p18 = len(seed_list) * 18
M_a18 = np.zeros((R*9, n_p18))
M_b18 = np.zeros((R*9, n_p18))
for j in range(n_p18):
    ev = np.zeros(n_p18); ev[j] = 1.0
    for si, seed in enumerate(seed_list):
        off = si * 18
        a = ev[off:off+9].reshape(3,3)
        b = ev[off+9:off+18].reshape(3,3)
        g = np.zeros((3,3))
        for t, (pi, eps) in transport_cache[seed].items():
            at, bt, gt = _apply_factors_fixed(pi, eps, a, b, g)
            k = kept_idx[t]
            M_a18[k*9:(k+1)*9, j] = at.ravel()
            M_b18[k*9:(k+1)*9, j] = bt.ravel()

target18 = np.concatenate([alpha_std.reshape(R*9), beta_std.reshape(R*9)])
M18 = np.vstack([M_a18, M_b18])
x18, _, rank18, _ = np.linalg.lstsq(M18, target18, rcond=None)
err_a18 = np.linalg.norm(M_a18 @ x18 - alpha_std.reshape(R*9))
err_b18 = np.linalg.norm(M_b18 @ x18 - beta_std.reshape(R*9))
flush(f"18-param: ||err_α||={err_a18:.2e}  ||err_β||={err_b18:.2e}  rank={rank18}/{n_p18}")


# Part 4: If 27-param works, recheck Σ rank
flush("\n" + "="*70)
flush("PART 4: Σ rank with 27-param expansion")
flush("="*70)

ALL_CONFIGS = {
    7: [1], 8: [6], 9: [8], 12: [12],
    13: [1,12], 14: [6,8], 15: [1,6,8],
    18: [1,6,12], 19: [1,6,12], 20: [6,8,12],
    21: [1,6,8,12], 26: [6,8,12], 27: [1,6,8,12],
}
# Fix: configs for distinct R values
ALL_CONFIGS = {
    1: [1], 6: [6], 8: [8], 12: [12],
    7: [1,6], 9: [1,8], 13: [1,12],
    14: [6,8], 18: [6,12], 20: [8,12],
    15: [1,6,8], 19: [1,6,12], 21: [1,8,12],
    26: [6,8,12], 27: [1,6,8,12],
}

orbit_seeds = {1: (0,0,0), 6: (0,0,1), 8: (0,1,2), 12: (0,1,1)}
rng = np.random.default_rng(42)

flush(f"{'R':>3s} {'n_p':>4s} | {'rk_Σ':>5s} {'rk_H':>5s} | notes")
flush("-"*50)

for R, sel in sorted(ALL_CONFIGS.items()):
    seeds = [orbit_seeds[s] for s in sel]
    n_p = len(seeds) * 27
    
    tc = {}
    for seed in seeds:
        tc[seed] = {}
        for pi, eps in _GROUP:
            t = _apply_triple(pi, eps, seed)
            if t not in tc[seed]:
                tc[seed][t] = (pi, eps)

    kept = set()
    for seed in seeds:
        kept |= set(tc[seed].keys())
    kept = sorted(kept)
    assert len(kept) == R, f"Expected {R} terms, got {len(kept)}"
    ki = {t: i for i, t in enumerate(kept)}

    Ma = np.zeros((R*9, n_p))
    Mb = np.zeros((R*9, n_p))
    Mg = np.zeros((R*9, n_p))
    for j in range(n_p):
        ev = np.zeros(n_p); ev[j] = 1.0
        for si, seed in enumerate(seeds):
            off = si * 27
            a = ev[off:off+9].reshape(3,3)
            b = ev[off+9:off+18].reshape(3,3)
            g = ev[off+18:off+27].reshape(3,3)
            for t, (pi, eps) in tc[seed].items():
                at, bt, gt = _apply_factors_fixed(pi, eps, a, b, g)
                k = ki[t]
                Ma[k*9:(k+1)*9, j] = at.ravel()
                Mb[k*9:(k+1)*9, j] = bt.ravel()
                Mg[k*9:(k+1)*9, j] = gt.ravel()
    
    # Sigma = sum of gamma_k, which in 9-vec is just summing Mg columns weighted by x
    # Σ(x) = sum_k gamma_k = (sum_k Mg[k*9:(k+1)*9, :]) @ x 
    Sig = np.zeros((9, n_p))
    for k in range(R):
        Sig += Mg[k*9:(k+1)*9, :]
    rk_sig = np.linalg.matrix_rank(Sig, tol=1e-10)
    
    # H matrix (all constraints)
    H = np.vstack([Ma, Mb])
    rk_H = np.linalg.matrix_rank(H, tol=1e-10)
    
    notes = "✓ possible!" if rk_sig >= 9 else ""
    flush(f"{R:3d} {n_p:4d} | {rk_sig:5d} {rk_H:5d} | {notes}")
