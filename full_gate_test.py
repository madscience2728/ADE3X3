"""
FULL GATE PIPELINE — No prejudice, no assumptions.

Tests R=13, 19, 20 with the complete gate hierarchy:
  Gate 1: rank(H) = R-9
  Gate 2: Delta ⊂ span(H)  →  rank([H|Delta]) = rank(H)
  Gate 3: Gamma·Sigma = 3I₉  →  exact solvability

Optimizes a COMBINED objective:
  L = w_fit · ‖T - T_hat‖² + w_rank · Σ σᵢ(H)² [i>target] + w_delta · ‖Δ - H·pinv(H)·Δ‖²

If any R produces fitness < 1e-6 with all gates satisfied: that's a solution.

Usage:
  python full_gate_test.py --workers 24 --trials 500 --maxiter 5000
  python full_gate_test.py --ranks 13,19,20 --trials 200
"""
import numpy as np
from itertools import permutations, product as iproduct
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, time, json, sys

# ═══════════════════════════════════════════════════════════
# GROUP MACHINERY
# ═══════════════════════════════════════════════════════════

def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _at(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _af(pi, eps, a, b, g):
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

def _sp(seed, a, b, g):
    stab = [(pi,eps) for pi,eps in _GROUP if _at(pi,eps,seed)==seed]
    aa,bb,gg = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    for pi,eps in stab:
        x,y,z = _af(pi,eps,a,b,g); aa+=x; bb+=y; gg+=z
    n=len(stab); return aa/n, bb/n, gg/n

# ═══════════════════════════════════════════════════════════
# ORBITS AND TENSOR
# ═══════════════════════════════════════════════════════════

_ALL27 = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]

def _compute_orbits():
    remaining = set(_ALL27); out = {}
    while remaining:
        seed = min(remaining); orb = set()
        for pi,eps in _GROUP: orb.add(_at(pi,eps,seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_OB = _compute_orbits()
_T = np.zeros((9,9,9))
for rp in range(3):
    for up in range(3):
        for s in range(3):
            _T[rp*3+up, rp*3+s, s*3+up] = 1.0

_CONFIGS = {
    13: [1, 12],
    18: [6, 12],
    19: [1, 6, 12],
    20: [8, 12],
    21: [1, 8, 12],
}

# ═══════════════════════════════════════════════════════════
# BUILD DECOMPOSITION FROM SEEDS
# ═══════════════════════════════════════════════════════════

def _expand_seeds(params, seed_list, kept):
    """Build alpha, beta, gamma arrays from seed parameters."""
    all_a, all_b, all_g = [], [], []
    off = 0
    for st in seed_list:
        a = params[off:off+9].reshape(3,3)
        b = params[off+9:off+18].reshape(3,3)
        g = params[off+18:off+27].reshape(3,3)
        ap, bp, gp = _sp(st, a, b, g)
        seen = set()
        for pi,eps in _GROUP:
            t = _at(pi,eps,st)
            if t not in seen and t in kept:
                seen.add(t)
                at,bt,gt = _af(pi,eps,ap,bp,gp)
                all_a.append(at); all_b.append(bt); all_g.append(gt)
        off += 27
    return np.array(all_a), np.array(all_b), np.array(all_g)

def _compute_fiber_coords(alpha, beta):
    """Compute Sigma, H, Delta from alpha, beta."""
    R = alpha.shape[0]
    Sigma = np.zeros((R, 9))
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    
    for k in range(R):
        for r in range(3):
            for u in range(3):
                idx = r*3+u
                s0 = alpha[k,r,0]*beta[k,0,u]
                s1 = alpha[k,r,1]*beta[k,1,u]
                s2 = alpha[k,r,2]*beta[k,2,u]
                Sigma[k,idx] = s0+s1+s2
                Eta1[k,idx] = s0-s1
                Eta2[k,idx] = s1-s2
    
    H = np.hstack([Eta1, Eta2])
    
    dead_pairs = [(s,t) for s in range(3) for t in range(3) if s != t]
    Delta = np.zeros((R, 54))
    for k in range(R):
        col = 0
        for (s,t) in dead_pairs:
            for r in range(3):
                for u in range(3):
                    Delta[k, col] = alpha[k,r,s] * beta[k,t,u]
                    col += 1
    
    return Sigma, H, Delta

def _full_diagnostics(alpha, beta, gamma):
    """Compute all gate diagnostics."""
    R = alpha.shape[0]
    Sigma, H, Delta = _compute_fiber_coords(alpha, beta)
    Nuisance = np.hstack([H, Delta])
    
    # Tensor reconstruction
    T_hat = np.zeros((9,9,9))
    for k in range(R):
        T_hat += np.einsum('i,j,m->ijm', gamma[k].flat, alpha[k].flat, beta[k].flat)
    
    fitness_inf = np.max(np.abs(_T - T_hat))
    fitness_fro = np.sum((_T - T_hat)**2)
    
    # Gate 1
    rank_H = np.linalg.matrix_rank(H, tol=1e-10)
    
    # Gate 2
    rank_nuis = np.linalg.matrix_rank(Nuisance, tol=1e-10)
    M = np.linalg.lstsq(H, Delta, rcond=None)[0]
    delta_resid = np.linalg.norm(Delta - H @ M, 'fro')
    
    # Gate 3
    Gamma = gamma.reshape(R, 9).T  # (9, R)
    gamma_delta = np.linalg.norm(Gamma @ Delta, 'fro')
    gamma_sigma_err = np.linalg.norm(Gamma @ Sigma - 3*np.eye(9), 'fro')
    
    aug_rank = np.linalg.matrix_rank(np.hstack([Sigma, Nuisance]), tol=1e-10)
    conservation = R + (18 - rank_H)
    
    return {
        'fitness_inf': float(fitness_inf),
        'fitness_fro': float(fitness_fro),
        'rank_H': int(rank_H),
        'rank_nuis': int(rank_nuis),
        'delta_resid': float(delta_resid),
        'gamma_delta': float(gamma_delta),
        'gamma_sigma_err': float(gamma_sigma_err),
        'aug_rank': int(aug_rank),
        'conservation': int(conservation),
    }

# ═══════════════════════════════════════════════════════════
# COMBINED GATE OBJECTIVE
# ═══════════════════════════════════════════════════════════

def _objective(params, seed_list, kept, R, w_fit, w_rank, w_delta):
    """Combined objective: fitness + rank penalty + delta containment."""
    alpha, beta, gamma = _expand_seeds(params, seed_list, kept)
    Sigma, H, Delta = _compute_fiber_coords(alpha, beta)
    
    # Fitness (Frobenius)
    T_hat = np.zeros((9,9,9))
    for k in range(R):
        T_hat += np.einsum('i,j,m->ijm', gamma[k].flat, alpha[k].flat, beta[k].flat)
    loss_fit = np.sum((_T - T_hat)**2)
    
    # Gate 1: penalize singular values beyond target rank
    target_rank = R - 9
    sv = np.linalg.svd(H, compute_uv=False)
    loss_rank = np.sum(sv[target_rank:]**2)
    
    # Gate 2: delta containment
    if H.shape[0] > 0 and np.linalg.matrix_rank(H, tol=1e-12) > 0:
        M = np.linalg.lstsq(H, Delta, rcond=None)[0]
        loss_delta = np.sum((Delta - H @ M)**2)
    else:
        loss_delta = np.sum(Delta**2)
    
    return w_fit * loss_fit + w_rank * loss_rank + w_delta * loss_delta

# ═══════════════════════════════════════════════════════════
# STAGED OPTIMIZATION (rank first, then fitness)
# ═══════════════════════════════════════════════════════════

def _run_trial(args):
    R, sel, trial_seed, maxiter = args
    np.random.seed(trial_seed)
    
    seed_list = [_OB[s][0] for s in sel]
    kept = set()
    for s in sel: kept |= _OB[s][1]
    n_params = len(sel) * 27
    
    x0 = np.random.randn(n_params) * 0.3
    
    # Stage 1: Achieve target rank (heavy rank penalty, ignore fitness)
    res1 = minimize(
        _objective, x0,
        args=(seed_list, kept, R, 0.01, 100.0, 10.0),
        method='L-BFGS-B',
        options={'maxiter': maxiter // 2, 'ftol': 1e-30, 'gtol': 1e-14}
    )
    
    # Stage 2: Improve fitness while maintaining rank (balanced weights)
    res2 = minimize(
        _objective, res1.x,
        args=(seed_list, kept, R, 1.0, 50.0, 20.0),
        method='L-BFGS-B',
        options={'maxiter': maxiter // 2, 'ftol': 1e-30, 'gtol': 1e-14}
    )
    
    # Diagnostics
    alpha, beta, gamma = _expand_seeds(res2.x, seed_list, kept)
    diag = _full_diagnostics(alpha, beta, gamma)
    diag['trial'] = trial_seed
    diag['R'] = R
    diag['stage1_loss'] = float(res1.fun)
    diag['stage2_loss'] = float(res2.fun)
    diag['params'] = res2.x.tolist()
    
    return diag

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Full gate pipeline test')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--maxiter', type=int, default=5000)
    parser.add_argument('--ranks', type=str, default='13,19,20')
    parser.add_argument('--out', type=str, default='gate_results.json')
    args = parser.parse_args()
    
    test_ranks = [int(r.strip()) for r in args.ranks.split(',')]
    
    print("═"*75)
    print("  FULL GATE PIPELINE — NO PREJUDICE")
    print("═"*75)
    print(f"  Testing R = {test_ranks}")
    print(f"  Trials: {args.trials} per rank, MaxIter: {args.maxiter}")
    print(f"  Stage 1: rank penalty dominant (achieve Gate 1)")
    print(f"  Stage 2: balanced fitness + rank + delta (all gates)")
    print("═"*75)
    
    tasks = []
    for R in test_ranks:
        if R not in _CONFIGS:
            print(f"  ⚠ R={R} not viable, skipping"); continue
        sel = _CONFIGS[R]
        target = R - 9
        print(f"  R={R}: target rank(H)={target}, {len(sel)} seeds, {len(sel)*27} raw params")
        for trial in range(args.trials):
            tasks.append((R, sel, trial * 1000 + R, args.maxiter))
    
    print(f"\n  Total tasks: {len(tasks)}")
    print("─"*75)
    
    t0 = time.time()
    all_results = {R: [] for R in test_ranks if R in _CONFIGS}
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_trial, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            res = f.result()
            all_results[res['R']].append(res)
            if done % 50 == 0 or done == len(tasks):
                e = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {e:.0f}s", flush=True)
    
    elapsed = time.time() - t0
    
    # ═══════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "═"*75)
    print("  RESULTS BY RANK")
    print("═"*75)
    
    best_overall = None
    
    for R in sorted(all_results.keys()):
        trials = all_results[R]
        if not trials: continue
        target = R - 9
        
        # Gate 1 pass: rank(H) == target
        g1_pass = [t for t in trials if t['rank_H'] == target]
        # Gate 2 pass: delta_resid < 0.01
        g2_pass = [t for t in g1_pass if t['delta_resid'] < 0.01]
        # Gate 3 pass: gamma_sigma_err < 0.01
        g3_pass = [t for t in g2_pass if t['gamma_sigma_err'] < 0.01]
        # Full solution: fitness < 0.001
        solutions = [t for t in g3_pass if t['fitness_inf'] < 0.001]
        
        best = min(trials, key=lambda t: t['fitness_inf'])
        best_g1 = min(g1_pass, key=lambda t: t['fitness_inf']) if g1_pass else None
        
        print(f"\n  R={R} (target rank(H)={target}):")
        print(f"    Total trials:  {len(trials)}")
        print(f"    Gate 1 pass:   {len(g1_pass):>4} (rank(H)={target})")
        print(f"    Gate 2 pass:   {len(g2_pass):>4} (Δ ⊂ span(H))")
        print(f"    Gate 3 pass:   {len(g3_pass):>4} (Γ·Σ ≈ 3I)")
        print(f"    SOLUTIONS:     {len(solutions):>4} (fitness < 0.001)")
        
        print(f"\n    Best overall:  fit∞={best['fitness_inf']:.6f}  rk(H)={best['rank_H']}  "
              f"δ={best['delta_resid']:.4f}  Γ·Σ err={best['gamma_sigma_err']:.4f}  C={best['conservation']}")
        
        if best_g1:
            print(f"    Best at Gate1: fit∞={best_g1['fitness_inf']:.6f}  rk(H)={best_g1['rank_H']}  "
                  f"δ={best_g1['delta_resid']:.4f}  Γ·Σ err={best_g1['gamma_sigma_err']:.4f}  C={best_g1['conservation']}")
        
        # Rank distribution
        from collections import Counter
        rd = Counter(t['rank_H'] for t in trials)
        print(f"    Rank dist:     {dict(sorted(rd.items()))}")
        
        # Conservation distribution
        cd = Counter(t['conservation'] for t in trials)
        print(f"    Conserv dist:  {dict(sorted(cd.items()))}")
        
        if best_g1 and (best_overall is None or best_g1['fitness_inf'] < best_overall['fitness_inf']):
            best_overall = best_g1
    
    # ═══════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════
    
    print("\n" + "═"*75)
    print("  VERDICT")
    print("═"*75)
    
    for R in sorted(all_results.keys()):
        trials = all_results[R]
        target = R - 9
        g1 = [t for t in trials if t['rank_H'] == target]
        g2 = [t for t in g1 if t['delta_resid'] < 0.1]
        g3 = [t for t in g2 if t['gamma_sigma_err'] < 0.1]
        sol = [t for t in g3 if t['fitness_inf'] < 0.01]
        
        if sol:
            print(f"  R={R}: ★ SOLUTION CANDIDATE — fitness {sol[0]['fitness_inf']:.2e}")
        elif g3:
            print(f"  R={R}: Gate 3 reached — fitness {g3[0]['fitness_inf']:.4f}, needs refinement")
        elif g2:
            print(f"  R={R}: Gate 2 reached — Δ contained, but Γ·Σ ≠ 3I")
        elif g1:
            print(f"  R={R}: Gate 1 only — rank(H)={target} achieved, {len(g1)} trials")
        else:
            best_rk = min(t['rank_H'] for t in trials)
            print(f"  R={R}: Gate 1 FAILED — best rank(H)={best_rk}, target was {target}")
    
    print(f"\n  Wall time: {elapsed:.1f}s")
    print("═"*75)
    
    # Save
    save_data = {}
    for R in all_results:
        # Save top 5 per rank
        top5 = sorted(all_results[R], key=lambda t: t['fitness_inf'])[:5]
        for t in top5:
            if 'params' in t: del t['params']  # save space
        save_data[str(R)] = top5
    
    if best_overall and 'params' not in best_overall:
        # Re-find it with params
        pass
    
    with open(args.out, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {args.out}")

if __name__ == '__main__':
    main()
