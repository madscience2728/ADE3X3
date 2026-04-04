"""
POLYNOMIAL ROOTFINDING — Full 729 equations, precomputed orbits, parallel.

Tests R=13, R=19, R=20 in the Z₂≀S₃ symmetric sector.
Uses Levenberg-Marquardt (not optimization — rootfinding).

Usage:
  python poly_rootfind.py --workers 24 --trials 10000
  python poly_rootfind.py --ranks 20 --trials 50000
"""
import numpy as np
from itertools import permutations, product as iproduct
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, time, json

# ═══════════════════════════════════════════════════════════
# GROUP (precomputed once)
# ═══════════════════════════════════════════════════════════

_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _sw(x): return x if x == 0 else 3 - x

def _gt(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_sw(x[i]) if eps[i] else x[i] for i in range(3))

def _gf(pi, eps, a, b, g):
    def _pm(e): return np.eye(3)[[0, 2, 1], :] if e else np.eye(3)
    P = [_pm(eps[i]) for i in range(3)]
    pv = [0] * 3
    for i in range(3): pv[pi[i]] = i
    fc = {(0, 1): a, (1, 2): b, (0, 2): g}
    rs = {}
    for (x, y), nm in [((0, 1), 'a'), ((1, 2), 'b'), ((0, 2), 'g')]:
        oa, ob = pv[x], pv[y]
        k = (min(oa, ob), max(oa, ob))
        F = fc[k].T if (oa, ob) != k else fc[k].copy()
        rs[nm] = P[x] @ F @ P[y].T
    return rs['a'], rs['b'], rs['g']

def _sp(seed, a, b, g):
    stab = [(pi, eps) for pi, eps in _GROUP if _gt(pi, eps, seed) == seed]
    aa = bb = gg = np.zeros((3, 3))
    for pi, eps in stab:
        x, y, z = _gf(pi, eps, a, b, g)
        aa = aa + x; bb = bb + y; gg = gg + z
    return aa / len(stab), bb / len(stab), gg / len(stab)

# ═══════════════════════════════════════════════════════════
# ORBITS AND TENSOR
# ═══════════════════════════════════════════════════════════

_ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
_OB = {}
_rem = set(_ALL27)
while _rem:
    s = min(_rem); o = set()
    for pi, eps in _GROUP: o.add(_gt(pi, eps, s))
    _OB[len(o)] = (s, frozenset(o)); _rem -= o

_T = np.zeros((9, 9, 9))
for rp in range(3):
    for up in range(3):
        for s in range(3):
            _T[rp * 3 + up, rp * 3 + s, s * 3 + up] = 1.0
_Tf = _T.flatten()

# ═══════════════════════════════════════════════════════════
# EFFECTIVE BASIS (precomputed per orbit type)
# ═══════════════════════════════════════════════════════════

def _find_basis(seed, n=500):
    np.random.seed(12345)  # deterministic basis
    samp = []
    for _ in range(n):
        a, b, g = np.random.randn(3, 3), np.random.randn(3, 3), np.random.randn(3, 3)
        ap, bp, gp = _sp(seed, a, b, g)
        samp.append(np.concatenate([ap.flat, bp.flat, gp.flat]))
    M = np.array(samp)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    d = int(np.sum(s > 1e-10))
    return Vt[:d].T, d

# Precompute bases for all orbit types
_BASES = {}
_DIMS = {}
for sz in sorted(_OB.keys()):
    seed = _OB[sz][0]
    b, d = _find_basis(seed)
    _BASES[sz] = b
    _DIMS[sz] = d

# ═══════════════════════════════════════════════════════════
# PRECOMPUTE ORBIT EXPANSION STRUCTURE
# ═══════════════════════════════════════════════════════════

def _precompute_orbit_transporters(seed, kept):
    """Find one group element that maps seed to each orbit member."""
    transporters = []
    seen = set()
    for pi, eps in _GROUP:
        t = _gt(pi, eps, seed)
        if t not in seen and t in kept:
            seen.add(t)
            transporters.append((pi, eps, t))
    return transporters

# ═══════════════════════════════════════════════════════════
# RESIDUAL FUNCTION (optimized)
# ═══════════════════════════════════════════════════════════

class SymmetricSystem:
    """Precomputed system for a specific R configuration."""
    
    def __init__(self, R, sel):
        self.R = R
        self.sel = sel
        self.seed_list = [_OB[s][0] for s in sel]
        self.kept = set()
        for s in sel:
            self.kept |= _OB[s][1]
        
        # Precompute transporters for each orbit
        self.transporters = {}
        for s in sel:
            seed = _OB[s][0]
            self.transporters[s] = _precompute_orbit_transporters(seed, self.kept)
        
        # Parameter dimensions
        self.dims = [_DIMS[s] for s in sel]
        self.total_params = sum(self.dims)
        self.offsets = []
        off = 0
        for d in self.dims:
            self.offsets.append(off)
            off += d
    
    def residual(self, theta):
        """729-dimensional residual: T_hat(θ) - T."""
        T_hat = np.zeros(729)
        
        for idx, s in enumerate(self.sel):
            seed = _OB[s][0]
            d = self.dims[idx]
            off = self.offsets[idx]
            B = _BASES[s]
            
            # Map effective params to factor triple
            raw = B @ theta[off:off + d]
            a_seed = raw[:9].reshape(3, 3)
            b_seed = raw[9:18].reshape(3, 3)
            g_seed = raw[18:27].reshape(3, 3)
            
            # Stabilizer projection
            a_seed, b_seed, g_seed = _sp(seed, a_seed, b_seed, g_seed)
            
            # Expand orbit using precomputed transporters
            for pi, eps, _ in self.transporters[s]:
                at, bt, gt = _gf(pi, eps, a_seed, b_seed, g_seed)
                # Rank-1 outer product contribution
                T_hat += np.einsum('i,j,m->ijm',
                                   gt.ravel(), at.ravel(), bt.ravel()).ravel()
        
        return T_hat - _Tf

# ═══════════════════════════════════════════════════════════
# SINGLE TRIAL (for parallel execution)
# ═══════════════════════════════════════════════════════════

def _run_trial(args):
    R, sel, trial_seed, maxiter = args
    np.random.seed(trial_seed)
    
    sys = SymmetricSystem(R, sel)
    
    # Random init with varying scale
    scale = 0.01 * (10 ** (3 * np.random.rand()))
    theta0 = np.random.randn(sys.total_params) * scale
    
    try:
        res = least_squares(
            sys.residual, theta0,
            method='lm',  # Levenberg-Marquardt: fast for m >> n
            max_nfev=maxiter,
            ftol=1e-15, xtol=1e-15, gtol=1e-15
        )
        fit_inf = float(np.max(np.abs(res.fun)))
        return {
            'R': R,
            'trial': trial_seed,
            'cost': float(res.cost),
            'fit_inf': fit_inf,
            'rnorm': float(np.linalg.norm(res.fun)),
            'params': res.x.tolist() if fit_inf < 0.1 else None,
            'success': bool(res.success),
        }
    except Exception as e:
        return {
            'R': R, 'trial': trial_seed,
            'cost': 1e30, 'fit_inf': 1e30, 'rnorm': 1e30,
            'params': None, 'success': False,
        }

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--trials', type=int, default=10000)
    p.add_argument('--maxiter', type=int, default=2000)
    p.add_argument('--ranks', type=str, default='13,19,20')
    p.add_argument('--out', type=str, default='rootfind_results.json')
    args = p.parse_args()
    
    ranks = [int(r) for r in args.ranks.split(',')]
    configs = {13: [1, 12], 19: [1, 6, 12], 20: [8, 12], 21: [1, 8, 12]}
    
    print("═" * 70)
    print("  POLYNOMIAL ROOTFINDING — Z₂ ≀ S₃ SYMMETRIC SECTOR")
    print("═" * 70)
    
    # Print system info
    for R in ranks:
        sel = configs[R]
        np_ = sum(_DIMS[s] for s in sel)
        print(f"  R={R}: {np_} params, 729 equations, ratio {729/np_:.0f}:1")
    
    print(f"  Trials: {args.trials} per rank")
    print(f"  Workers: {args.workers or 'all cores'}")
    print(f"  Method: Levenberg-Marquardt (rootfinding, not optimization)")
    print("═" * 70)
    
    # Build tasks
    tasks = []
    for R in ranks:
        sel = configs[R]
        for t in range(args.trials):
            tasks.append((R, sel, t * 100 + R, args.maxiter))
    
    # Run parallel
    t0 = time.time()
    results = {R: [] for R in ranks}
    solutions = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, task): task for task in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            res = f.result()
            results[res['R']].append(res)
            
            # Report discoveries immediately
            if res['fit_inf'] < 0.01:
                print(f"  ★ R={res['R']} trial {res['trial']}: "
                      f"fit∞={res['fit_inf']:.2e} ‖f‖={res['rnorm']:.2e}")
                solutions.append(res)
            
            if done % (len(tasks) // 10 + 1) == 0:
                e = time.time() - t0
                # Current best per rank
                bests = {R: min((r['fit_inf'] for r in results[R]), default=99)
                         for R in ranks if results[R]}
                best_str = " | ".join(f"R={R}:{v:.4f}" for R, v in sorted(bests.items()))
                print(f"  [{done}/{len(tasks)}] {e:.0f}s — {best_str}")
    
    elapsed = time.time() - t0
    
    # ═══════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  RESULTS")
    print("═" * 70)
    
    all_saved = {}
    
    for R in sorted(results.keys()):
        trials = results[R]
        if not trials: continue
        
        fits = sorted([t['fit_inf'] for t in trials])
        best = min(trials, key=lambda t: t['fit_inf'])
        
        print(f"\n  R={R}:")
        print(f"    Best fit∞:  {best['fit_inf']:.6e}")
        print(f"    Best ‖f‖:   {best['rnorm']:.6e}")
        print(f"    Best trial: {best['trial']}")
        print(f"    Top 5:      {[f'{x:.4f}' for x in fits[:5]]}")
        print(f"    Median:     {fits[len(fits)//2]:.4f}")
        
        if best['fit_inf'] < 1e-6:
            print(f"    ★★★ EXACT SOLUTION FOUND ★★★")
        elif best['fit_inf'] < 0.01:
            print(f"    ★ Near-exact — likely refinable")
        elif best['fit_inf'] < 0.1:
            print(f"    Close approach")
        elif best['fit_inf'] < 0.5:
            print(f"    Moderate basin")
        else:
            print(f"    No solution in this budget")
        
        # Save top 10
        top10 = sorted(trials, key=lambda t: t['fit_inf'])[:10]
        all_saved[str(R)] = top10
    
    print(f"\n  Wall time: {elapsed:.1f}s")
    print("═" * 70)
    
    with open(args.out, 'w') as f:
        json.dump(all_saved, f, indent=2)
    print(f"  Saved to {args.out}")

if __name__ == '__main__':
    main()
