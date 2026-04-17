"""
symmetric_lm.py — Symmetric-sector Levenberg-Marquardt solver for 3×3 matmul
tensor decomposition.

Parametrizes rank-R decompositions using Z₂≀S₃ orbit structure:
  - Corner (0,0,0): 1 term
  - Edge   (0,0,1): 6 terms
  - Face   (0,1,1): 12 terms
  - Interior (1,1,1): 8 terms

For each orbit, a single seed (α,β,γ) ∈ (R^{3×3})^3 is stabilizer-projected
then transported by the group to generate all orbit members.

The free parameters live in the invariant subspace of each orbit's stabilizer
action on the 27D joint (α,β,γ) space. This gives ~24 params for R=19.

Residual: 729-dim vector (T_hat - T_matmul flattened).
Solver: scipy.optimize.least_squares with method='lm'.

Usage:
    python symmetric_lm.py --R 27 --trials 100
    python symmetric_lm.py --R 19 --trials 1000
"""
import argparse
import time
import numpy as np
from itertools import permutations, product as iproduct
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    """Transport (α,β,γ) under group element (pi, eps)."""
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

# ═══════════════════════════════════════════════════════════
# ORBIT CATALOG
# ═══════════════════════════════════════════════════════════

_ALL27 = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]

def _compute_orbits():
    remaining = set(_ALL27); out = {}
    while remaining:
        seed = min(remaining); orb = set()
        for pi, eps in _GROUP: orb.add(_apply_triple(pi, eps, seed))
        out[len(orb)] = (seed, frozenset(orb)); remaining -= orb
    return out

_ORBITS = _compute_orbits()

_CONFIGS = {
    7:  [1, 6],
    9:  [1, 8],
    13: [1, 12],
    18: [6, 12],
    19: [1, 6, 12],
    20: [8, 12],
    21: [1, 8, 12],
    26: [6, 8, 12],
    27: [1, 6, 8, 12],
}

# ═══════════════════════════════════════════════════════════
# INVARIANT SUBSPACE COMPUTATION
# ═══════════════════════════════════════════════════════════

def _stab_rep_matrix(pi, eps, seed):
    """
    Compute the 27×27 representation matrix of group element (pi, eps)
    acting on the joint (α,β,γ) space (each 3×3 = 9, total 27).
    """
    M = np.zeros((27, 27))
    for i in range(27):
        # Create a basis vector: one entry = 1 in the flattened (α,β,γ) space
        a = np.zeros((3,3)); b = np.zeros((3,3)); g = np.zeros((3,3))
        if i < 9:
            a.flat[i] = 1.0
        elif i < 18:
            b.flat[i - 9] = 1.0
        else:
            g.flat[i - 18] = 1.0
        
        a2, b2, g2 = _apply_factors(pi, eps, a, b, g)
        out = np.concatenate([a2.ravel(), b2.ravel(), g2.ravel()])
        M[:, i] = out
    return M

def _compute_invariant_basis(seed):
    """
    Compute an orthonormal basis for the invariant subspace of the
    stabilizer of `seed` acting on R^27 = (α,β,γ).
    """
    stab = [(pi, eps) for pi, eps in _GROUP if _apply_triple(pi, eps, seed) == seed]
    
    # Average of representation matrices = projection onto invariant subspace
    P = np.zeros((27, 27))
    for pi, eps in stab:
        P += _stab_rep_matrix(pi, eps, seed)
    P /= len(stab)
    
    # SVD to extract basis of image
    U, s, Vt = np.linalg.svd(P)
    rank = np.sum(s > 1e-10)
    return U[:, :rank], len(stab)

# ═══════════════════════════════════════════════════════════
# MATMUL TARGET TENSOR
# ═══════════════════════════════════════════════════════════

def build_matmul_tensor():
    """T ∈ R^{9×9×9}: T[3a+b, 3b+c, 3a+c] = 1."""
    T = np.zeros((9, 9, 9))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                T[3*a+b, 3*b+c, 3*a+c] = 1.0
    return T

_T_MATMUL = build_matmul_tensor()
_T_FLAT = _T_MATMUL.ravel()  # 729-vector

# ═══════════════════════════════════════════════════════════
# CORE: params → residual
# ═══════════════════════════════════════════════════════════

class SymmetricDecomposition:
    """Encapsulates the orbit structure and invariant bases for a given R."""
    
    def __init__(self, R):
        assert R in _CONFIGS, f"R={R} not viable under Z₂≀S₃"
        self.R = R
        self.sel = _CONFIGS[R]
        self.seeds = [_ORBITS[s][0] for s in self.sel]
        self.kept = set()
        for s in self.sel:
            self.kept |= _ORBITS[s][1]
        
        # Compute invariant bases for each orbit seed
        self.bases = []
        self.dims = []
        total_free = 0
        for seed in self.seeds:
            basis, stab_order = _compute_invariant_basis(seed)
            self.bases.append(basis)
            self.dims.append(basis.shape[1])
            total_free += basis.shape[1]
        
        self.n_free = total_free
        
        # Precompute transport maps: for each orbit, list of (group_element, triple)
        self.transport_maps = []
        for seed in self.seeds:
            members = []
            seen = set()
            for pi, eps in _GROUP:
                t = _apply_triple(pi, eps, seed)
                if t not in seen and t in self.kept:
                    seen.add(t)
                    members.append((pi, eps, t))
            self.transport_maps.append(members)
        
        # Precompute the full linear map: free params → (R, 27) stacked factors
        # For each orbit member k, the 27-vector (α_k,β_k,γ_k) is a linear function
        # of the free parameters. We precompute this as a (R*27, n_free) matrix.
        self._expansion_matrix = self._build_expansion_matrix()
    
    def _build_expansion_matrix(self):
        """Build (R*27, n_free) matrix: params → stacked [α₁;β₁;γ₁;α₂;β₂;γ₂;...]."""
        rows = []
        off = 0
        for i, (basis, tmap) in enumerate(zip(self.bases, self.transport_maps)):
            d = basis.shape[1]
            for pi, eps, triple in tmap:
                # Transport matrix: 27→27 for this group element
                M = _stab_rep_matrix(pi, eps, self.seeds[i])
                # Combined: free params → seed (via basis) → transported (via M)
                # block = M @ basis, shape (27, d)
                block = M @ basis
                # Place in full matrix
                row = np.zeros((27, self.n_free))
                row[:, off:off+d] = block
                rows.append(row)
            off += d
        return np.vstack(rows)  # (R*27, n_free)
    
    def params_to_factors(self, params):
        """Map free parameters → (R,9) arrays for α, β, γ."""
        all_abg = self._expansion_matrix @ params  # (R*27,)
        all_abg = all_abg.reshape(self.R, 27)
        return all_abg[:, :9], all_abg[:, 9:18], all_abg[:, 18:]
    
    def params_to_residual(self, params):
        """Map free parameters → 729-dim residual (T_hat - T_matmul)."""
        A, B, G = self.params_to_factors(params)  # each (R, 9)
        T_hat = np.einsum('ri,rj,rk->ijk', A, B, G)
        return (T_hat - _T_MATMUL).ravel()
    
    def params_to_residual_with_info(self, params):
        """Like params_to_residual but returns structured factors."""
        A, B, G = self.params_to_factors(params)
        alpha = [A[k].reshape(3,3) for k in range(A.shape[0])]
        beta = [B[k].reshape(3,3) for k in range(B.shape[0])]
        gamma = [G[k].reshape(3,3) for k in range(G.shape[0])]
        return alpha, beta, gamma

# ═══════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

def compute_diagnostics(decomp, params):
    """Compute rank(H), conservation law, etc."""
    alpha, beta, gamma = decomp.params_to_residual_with_info(params)
    R = len(alpha)
    
    # Build H = [Eta1 | Eta2]
    Eta1 = np.zeros((R, 9))
    Eta2 = np.zeros((R, 9))
    Sigma = np.zeros((R, 9))
    for k in range(R):
        a, b = alpha[k], beta[k]
        for r in range(3):
            for u in range(3):
                idx = r*3+u
                s0 = a[r,0]*b[0,u]
                s1 = a[r,1]*b[1,u]
                s2 = a[r,2]*b[2,u]
                Sigma[k, idx] = s0 + s1 + s2
                Eta1[k, idx] = s0 - s1
                Eta2[k, idx] = s1 - s2
    
    H = np.hstack([Eta1, Eta2])
    sv_H = np.linalg.svd(H, compute_uv=False)
    rank_H = np.sum(sv_H > 1e-10)
    
    # Conservation law: R + η_nullity = 27
    eta_nullity = 18 - rank_H
    conservation = R + eta_nullity
    
    # Trace charge
    traces_a = [np.trace(a) for a in alpha]
    traces_b = [np.trace(b) for b in beta]
    traces_g = [np.trace(g) for g in gamma]
    trace_charge = sum(ta*tb*tg for ta,tb,tg in zip(traces_a, traces_b, traces_g))
    
    return {
        'rank_H': rank_H,
        'target_rank_H': R - 9,
        'sv_H': sv_H,
        'conservation': conservation,
        'trace_charge': trace_charge,
        'rank_Sigma': np.linalg.matrix_rank(Sigma, tol=1e-10),
    }

# ═══════════════════════════════════════════════════════════
# SINGLE TRIAL
# ═══════════════════════════════════════════════════════════

def run_trial(args):
    R, trial_seed = args
    np.random.seed(trial_seed)
    decomp = SymmetricDecomposition(R)
    
    x0 = np.random.randn(decomp.n_free) * 0.5
    
    result = least_squares(
        decomp.params_to_residual, x0,
        method='lm',
        ftol=1e-15, xtol=1e-15, gtol=1e-15,
        max_nfev=10000,
    )
    
    residual_norm = np.linalg.norm(result.fun)
    rel_error = residual_norm / np.linalg.norm(_T_FLAT)
    bits = -np.log2(rel_error) if rel_error > 0 else 999
    
    diag = compute_diagnostics(decomp, result.x)
    
    return {
        'R': R,
        'trial': trial_seed,
        'residual_norm': float(residual_norm),
        'rel_error': float(rel_error),
        'bits': float(bits),
        'rank_H': diag['rank_H'],
        'target_rank_H': diag['target_rank_H'],
        'conservation': diag['conservation'],
        'trace_charge': float(diag['trace_charge']),
        'rank_Sigma': diag['rank_Sigma'],
        'nfev': result.nfev,
        'success': result.success,
        'params': result.x.copy(),
    }

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Symmetric-sector LM solver for 3×3 matmul decomposition')
    parser.add_argument('--R', type=int, required=True, help='Target rank')
    parser.add_argument('--trials', type=int, default=100, help='Number of random restarts')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (default: all cores)')
    args = parser.parse_args()
    
    R = args.R
    assert R in _CONFIGS, f"R={R} not viable under Z₂≀S₃. Viable: {sorted(_CONFIGS.keys())}"
    
    # Print setup info
    decomp = SymmetricDecomposition(R)
    print("═"*75)
    print(f"  SYMMETRIC-SECTOR LM SOLVER — R={R}")
    print("═"*75)
    print(f"  Orbits: {decomp.sel} = {' + '.join(str(s) for s in decomp.sel)} = {R}")
    print(f"  Free parameters: {decomp.n_free} (invariant subspace dims: {decomp.dims})")
    print(f"  Residual dimension: 729")
    print(f"  Jacobian shape: 729 × {decomp.n_free}")
    print(f"  Target rank(H): {R - 9}")
    print(f"  Trials: {args.trials}")
    print("═"*75)
    
    tasks = [(R, seed) for seed in range(args.trials)]
    
    t0 = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            results.append(res)
            if done % max(1, args.trials // 10) == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {elapsed:.1f}s", flush=True)
    
    elapsed = time.time() - t0
    
    # Sort by residual
    results.sort(key=lambda r: r['residual_norm'])
    
    print()
    print("═"*75)
    print("  RESULTS")
    print("═"*75)
    
    # Best result
    best = results[0]
    print(f"\n  Best trial #{best['trial']}:")
    print(f"    Residual norm:  {best['residual_norm']:.6e}")
    print(f"    Relative error: {best['rel_error']:.6e}")
    print(f"    Bits:           {best['bits']:.1f}")
    print(f"    rank(H):        {best['rank_H']} (target: {best['target_rank_H']})")
    print(f"    R + η_null:     {best['conservation']} (should be 27)")
    print(f"    Trace charge:   {best['trace_charge']:.6f} (should be 3)")
    print(f"    rank(Σ):        {best['rank_Sigma']} (need ≥ 9)")
    print(f"    nfev:           {best['nfev']}")
    
    # Distribution
    rank_H_counts = {}
    for r in results:
        rk = r['rank_H']
        rank_H_counts[rk] = rank_H_counts.get(rk, 0) + 1
    
    print(f"\n  rank(H) distribution: {dict(sorted(rank_H_counts.items()))}")
    
    # Top 5
    print(f"\n  Top 5 by residual:")
    print(f"  {'trial':>6s} {'resid':>12s} {'rel_err':>12s} {'bits':>6s} {'rk(H)':>5s} {'R+η':>4s} {'tr':>8s} {'rk(Σ)':>5s}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*6} {'─'*5} {'─'*4} {'─'*8} {'─'*5}")
    for r in results[:5]:
        print(f"  {r['trial']:6d} {r['residual_norm']:12.4e} {r['rel_error']:12.4e} "
              f"{r['bits']:6.1f} {r['rank_H']:5d} {r['conservation']:4d} "
              f"{r['trace_charge']:8.4f} {r['rank_Sigma']:5d}")
    
    # Check if exact solution found
    if best['rel_error'] < 1e-10:
        print(f"\n  *** EXACT SOLUTION FOUND! *** (rel_error = {best['rel_error']:.2e})")
        np.save(f'symmetric_lm_R{R}_solution.npy', best['params'])
        print(f"  Saved → symmetric_lm_R{R}_solution.npy")
    else:
        print(f"\n  No exact solution found. Best relative error: {best['rel_error']:.4e}")
        # Save best anyway
        np.save(f'symmetric_lm_R{R}_best.npy', best['params'])
    
    print(f"\n  Wall time: {elapsed:.1f}s")
    print("═"*75)

if __name__ == '__main__':
    main()
