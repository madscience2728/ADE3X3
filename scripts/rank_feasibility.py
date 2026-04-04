"""
RANK FEASIBILITY TEST — Does rank(H)=R-9 exist in the Z₂≀S₃ symmetric sector?

For each viable R under the wreath product symmetry, this script ignores fitness
entirely and optimizes ONLY to minimize the tail singular values of H.

If min achievable rank(H) > R-9: the symmetric R is IMPOSSIBLE.
If min achievable rank(H) = R-9: the variety exists and we know where to look.

Usage:
  python rank_feasibility.py                  # uses all cores
  python rank_feasibility.py --workers 24     # explicit core count
  python rank_feasibility.py --trials 200     # more restarts
"""
import numpy as np
from itertools import permutations, product as iproduct
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, time, sys

# ═══════════════════════════════════════════════════════════
# GROUP MACHINERY (Z₂ ≀ S₃, order 48)
# ═══════════════════════════════════════════════════════════

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

def _stab_project(seed, a, b, g):
    stab = [(pi,eps) for pi,eps in _GROUP if _apply_triple(pi,eps,seed)==seed]
    aa,bb,gg = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    for pi,eps in stab:
        x,y,z = _apply_factors(pi,eps,a,b,g); aa+=x; bb+=y; gg+=z
    n=len(stab)
    return aa/n, bb/n, gg/n

# ═══════════════════════════════════════════════════════════
# ORBIT CATALOG
# ═══════════════════════════════════════════════════════════

_ALL27 = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]

def _compute_orbits():
    remaining = set(_ALL27); out = {}
    while remaining:
        seed = min(remaining); orb = set()
        for pi,eps in _GROUP: orb.add(_apply_triple(pi,eps,seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_ORBITS = _compute_orbits()
# Sizes: 1 (corner), 6 (edge), 8 (interior), 12 (face)

_EFF_PARAMS = {1: 3, 6: 9, 8: 6, 12: 12}

# Viable R values as orbit-size subsets
_CONFIGS = {
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

# ═══════════════════════════════════════════════════════════
# RANK OBJECTIVE (minimize tail singular values of H)
# ═══════════════════════════════════════════════════════════

def _build_H_from_params(params, seed_list, kept):
    """Build the 19×18 H matrix from seed parameters."""
    all_alpha, all_beta = [], []
    off = 0
    for st in seed_list:
        a = params[off:off+9].reshape(3,3)
        b = params[off+9:off+18].reshape(3,3)
        # gamma not needed for H
        ap, bp, _ = _stab_project(st, a, b, np.zeros((3,3)))
        seen = set()
        for pi,eps in _GROUP:
            t = _apply_triple(pi,eps,st)
            if t not in seen and t in kept:
                seen.add(t)
                at,bt,_ = _apply_factors(pi,eps,ap,bp,np.zeros((3,3)))
                all_alpha.append(at); all_beta.append(bt)
        off += 18  # only α,β — no γ needed
    
    R = len(all_alpha)
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    for k in range(R):
        for r in range(3):
            for u in range(3):
                idx = r*3+u
                s0 = all_alpha[k][r,0]*all_beta[k][0,u]
                s1 = all_alpha[k][r,1]*all_beta[k][1,u]
                s2 = all_alpha[k][r,2]*all_beta[k][2,u]
                Eta1[k,idx] = s0 - s1
                Eta2[k,idx] = s1 - s2
    return np.hstack([Eta1, Eta2])

def _rank_objective(params, seed_list, kept, target_rank):
    """Minimize sum of squared singular values beyond target_rank."""
    H = _build_H_from_params(params, seed_list, kept)
    sv = np.linalg.svd(H, compute_uv=False)
    # Penalty: sum of sv[target_rank:]^2 — want these to be zero
    tail = sv[target_rank:]
    return np.sum(tail**2)

# ═══════════════════════════════════════════════════════════
# SINGLE TRIAL
# ═══════════════════════════════════════════════════════════

def run_trial(args):
    R, sel, trial_seed, maxiter = args
    np.random.seed(trial_seed)
    
    seed_list = [_ORBITS[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _ORBITS[s][1]
    
    target_rank = R - 9  # conservation law target
    n_params = len(sel) * 18  # only α,β per seed (no γ)
    
    x0 = np.random.randn(n_params) * 0.5
    
    res = minimize(
        _rank_objective, x0,
        args=(seed_list, kept, target_rank),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-30, 'gtol': 1e-15}
    )
    
    # Get final singular values
    H = _build_H_from_params(res.x, seed_list, kept)
    sv = np.linalg.svd(H, compute_uv=False)
    actual_rank = np.sum(sv > 1e-10)
    tail_energy = np.sum(sv[target_rank:]**2)
    
    return {
        'R': R,
        'target_rank_H': target_rank,
        'actual_rank_H': int(actual_rank),
        'tail_energy': float(tail_energy),
        'singular_values': sv.tolist(),
        'trial': trial_seed,
        'converged': res.success,
    }

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Rank feasibility test in Z₂≀S₃ symmetric sector')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: all cores)')
    parser.add_argument('--trials', type=int, default=100, help='Random restarts per rank (default: 100)')
    parser.add_argument('--maxiter', type=int, default=2000, help='L-BFGS-B iterations per trial (default: 2000)')
    parser.add_argument('--ranks', type=str, default='13,18,19,20,21', help='Comma-separated R values to test')
    args = parser.parse_args()
    
    test_ranks = [int(r.strip()) for r in args.ranks.split(',')]
    
    print("═"*75)
    print("  RANK FEASIBILITY TEST — Z₂ ≀ S₃ SYMMETRIC SECTOR")
    print("═"*75)
    print(f"  Workers: {args.workers or 'all cores'}")
    print(f"  Trials per rank: {args.trials}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Testing ranks: {test_ranks}")
    print("═"*75)
    
    # Build task list
    tasks = []
    for R in test_ranks:
        if R not in _CONFIGS:
            print(f"  ⚠ R={R} not viable under Z₂≀S₃, skipping")
            continue
        sel = _CONFIGS[R]
        eff = sum(_EFF_PARAMS[s] for s in sel)
        target = R - 9
        print(f"  R={R:2d}: orbits {sel}, {eff} eff params, target rank(H)={target}")
        for trial in range(args.trials):
            tasks.append((R, sel, trial * 1000 + R, args.maxiter))
    
    print(f"\n  Total tasks: {len(tasks)}")
    print("─"*75)
    
    # Run in parallel
    t0 = time.time()
    results = {R: [] for R in test_ranks if R in _CONFIGS}
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            results[res['R']].append(res)
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {elapsed:.0f}s", flush=True)
    
    elapsed = time.time() - t0
    
    # ═══════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "═"*75)
    print("  RESULTS")
    print("═"*75)
    print(f"\n  {'R':>3} {'target':>7} {'best_rk':>8} {'worst_rk':>9} {'median_rk':>10} "
          f"{'best_tail':>10} {'feasible':>9}")
    print("  " + "─"*65)
    
    feasibility = {}
    
    for R in sorted(results.keys()):
        trials = results[R]
        if not trials: continue
        
        target = R - 9
        ranks = [t['actual_rank_H'] for t in trials]
        tails = [t['tail_energy'] for t in trials]
        
        best_rank = min(ranks)
        worst_rank = max(ranks)
        median_rank = int(np.median(ranks))
        best_tail = min(tails)
        
        hit_target = best_rank <= target
        feasibility[R] = hit_target
        
        marker = "  ✓ YES" if hit_target else "  ✗ NO"
        print(f"  {R:3d} {target:7d} {best_rank:8d} {worst_rank:9d} {median_rank:10d} "
              f"{best_tail:10.2e} {marker}")
        
        # Singular value profile of best trial
        best_trial = min(trials, key=lambda t: t['tail_energy'])
        sv = best_trial['singular_values']
        sv_str = " ".join(f"{s:.3f}" for s in sv[:min(len(sv), 18)])
        print(f"       sv profile: [{sv_str}]")
        
        # How many trials hit each rank?
        from collections import Counter
        rank_dist = Counter(ranks)
        dist_str = " ".join(f"rk{r}:{c}" for r,c in sorted(rank_dist.items()))
        print(f"       rank distribution: {dist_str}")
    
    # ═══════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "═"*75)
    print("  VERDICT")
    print("═"*75)
    
    for R in sorted(feasibility.keys()):
        target = R - 9
        if feasibility[R]:
            print(f"  R={R}: rank(H)={target} IS REACHABLE in the symmetric sector")
        else:
            best = min(t['actual_rank_H'] for t in results[R])
            gap = best - target
            print(f"  R={R}: rank(H)={target} NOT REACHED — best={best} (gap={gap})")
            if gap <= 2:
                print(f"         → Close! May be reachable with more trials or different parameterization")
            else:
                print(f"         → Large gap suggests structural obstruction in this symmetric class")
    
    print(f"\n  Total wall time: {elapsed:.1f}s")
    print("═"*75)

if __name__ == '__main__':
    main()
