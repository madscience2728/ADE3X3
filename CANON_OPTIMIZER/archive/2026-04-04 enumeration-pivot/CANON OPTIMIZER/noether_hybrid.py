"""Full Σ rank table with 27-param (α,β,γ) equivariant expansion.
The Noether hybrid: algebra defines the manifold, geometry finds solutions on it."""
import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

_S3 = list(permutations(range(3)))
_GROUP = [(list(pi), list(eps)) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _swap12(x): return x if x == 0 else 3 - x
def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _apply_factors(pi, eps, a, b, g):
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

# Step 1: Compute actual orbit sizes
ALL_TRIPLES = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]
seeds = [(0,0,0), (0,0,1), (0,1,1), (0,1,2)]
seed_names = ['Corner', 'Edge', 'Face', 'Interior']

flush("="*70)
flush("ORBIT SIZES (actual)")
flush("="*70)
orbit_of = {}
for seed in seeds:
    orb = set()
    for pi, eps in _GROUP:
        orb.add(_apply_triple(pi, eps, seed))
    orbit_of[seed] = sorted(orb)
    flush(f"  {seed}: {len(orb)} terms")

# Step 2: All configs (subsets of seeds → orbit combos)
from itertools import combinations
flush("\n" + "="*70)
flush("Σ RANK TABLE — 27-param equivariant expansion")
flush("="*70)

def build_expansion(seed_subset):
    """Build (M_a, M_b, M_g) linear maps for the 27-param expansion."""
    tc = {}
    kept = set()
    for seed in seed_subset:
        tc[seed] = {}
        for pi, eps in _GROUP:
            t = _apply_triple(pi, eps, seed)
            if t not in tc[seed]:
                tc[seed][t] = (pi, eps)
        kept |= set(tc[seed].keys())
    
    kept = sorted(kept)
    R = len(kept)
    ki = {t: i for i, t in enumerate(kept)}
    n_p = len(seed_subset) * 27
    
    Ma = np.zeros((R*9, n_p))
    Mb = np.zeros((R*9, n_p))
    Mg = np.zeros((R*9, n_p))
    
    for j in range(n_p):
        ev = np.zeros(n_p); ev[j] = 1.0
        for si, seed in enumerate(seed_subset):
            off = si * 27
            a = ev[off:off+9].reshape(3,3)
            b = ev[off+9:off+18].reshape(3,3)
            g = ev[off+18:off+27].reshape(3,3)
            for t, (pi, eps) in tc[seed].items():
                at, bt, gt = _apply_factors(pi, eps, a, b, g)
                k = ki[t]
                Ma[k*9:(k+1)*9, j] += at.ravel()
                Mb[k*9:(k+1)*9, j] += bt.ravel()
                Mg[k*9:(k+1)*9, j] += gt.ravel()
    
    return Ma, Mb, Mg, R, n_p, kept

rng = np.random.default_rng(42)

flush(f"\n{'R':>3} {'seeds':>12} {'n_p':>4} | {'rk_Σ':>5} {'rk_H':>5} {'rk_N':>5} | {'G1':>3} {'G2':>3} {'G3':>3}")
flush("-"*70)

results = []
for n_seeds in range(1, 5):
    for combo in combinations(range(4), n_seeds):
        seed_subset = [seeds[i] for i in combo]
        Ma, Mb, Mg, R, n_p, kept = build_expansion(seed_subset)
        
        # Σ = sum of all γ_k  (Σ is a 3×3 matrix = 9-vector)
        Sig = np.zeros((9, n_p))
        for k in range(R):
            Sig += Mg[k*9:(k+1)*9, :]
        rk_sig = np.linalg.matrix_rank(Sig, tol=1e-10)
        
        # H = (Ma; Mb) stacked — the full constraint matrix
        H = np.vstack([Ma, Mb])
        rk_H = np.linalg.matrix_rank(H, tol=1e-10)
        
        # N = kernel of H — need rk(Σ restricted to ker H) = 9 for Gate 3  
        # More precisely: rk([H; Σ]) - rk(H) should be 9
        HΣ = np.vstack([H, Sig])
        rk_HΣ = np.linalg.matrix_rank(HΣ, tol=1e-10)
        rk_N = rk_HΣ - rk_H  # effective Σ rank in null space of H
        
        # Gate checks
        target_H = R - 9
        g1 = "✓" if rk_H >= target_H else "✗"
        g2 = "?" # need nonlinear check
        g3 = "✓" if rk_sig >= 9 else "✗"
        
        names = "+".join(seed_names[i][0] for i in combo)
        flush(f"{R:3d} {names:>12} {n_p:4d} | {rk_sig:5d} {rk_H:5d} {rk_N:5d} | {g1:>3} {g2:>3} {g3:>3}")
        results.append((R, combo, rk_sig, rk_H, rk_N))

# Step 3: Standard algorithm fit test with 27 params
flush("\n" + "="*70)
flush("STANDARD ALGORITHM FIT (R=27, all 4 orbits)")
flush("="*70)

Ma, Mb, Mg, R, n_p, kept = build_expansion(seeds)
flush(f"  R={R} (all orbits, C+E+F+I overlap → {R} terms)")
flush("  Skipping standard fit — F and I orbits overlap, R≠27")
flush("  The standard algorithm lives outside Z₂≀S₃ — that's fine.")
flush("  We're looking for NEW decompositions, not reproducing the old one.")

flush("  → Focus: Newton on the 81-param R=19 trilinear system")

# Step 4: The Noether check — does the TENSOR equation T = Σ α⊗β⊗γ 
# give us a bilinear system we can Newton-solve?
flush("\n" + "="*70)
flush("NOETHER HYBRID: Tensor equation in 27-param space")
flush("="*70)

# For R=19 (C+E+F config), the tensor equation is:
# Note: C+E+F+I also gives 19 — the orbits of (0,1,1) and (0,1,2) overlap completely!
# We use C+E+F (3 seeds × 27 = 81 params)
# T(x) = Σ_k α_k(x) ⊗ β_k(x) ⊗ γ_k(x) = T_target
# where x ∈ R^(n_p) and α,β,γ are LINEAR in x.
# This is a TRILINEAR system: each monomial in x has degree 3.
# But we can fix γ (or any one factor) and solve the remaining BILINEAR system.

# Actually: for matrix mult tensor T_{rsu,r's'u'} = δ_{ss'}, we need:
# Σ_k α_k[r,s] β_k[s',u] γ_k[r',u'] = δ_{r,r'} δ_{s,s'} δ_{u,u'}
# Wait, that's not right. The tensor is:
# T = Σ_{r,s,u} e_r⊗e_s ⊗ e_s⊗e_u ⊗ e_r⊗e_u
# Flattened: T[i,j,k] where i=(r,s), j=(s,u), k=(r,u)
# and T[(r,s),(s',u),(r',u')] = δ_{ss'} δ_{rr'} δ_{uu'}

# The decomposition T = Σ_k α_k ⊗ β_k ⊗ γ_k means:
# Σ_k α_k[r,s] β_k[s',u] γ_k[r',u'] = δ_{ss'} δ_{rr'} δ_{uu'}

# With our parameterization, α_k(x), β_k(x), γ_k(x) are all LINEAR in x.
# So the equations are TRILINEAR in x.
# The Jacobian is quadratic in x.

# For R=19:
combo_19 = [seeds[0], seeds[1], seeds[2]]  # C+E+F = 1+6+12 = 19
Ma19, Mb19, Mg19, R19, np19, kept19 = build_expansion(combo_19)
flush(f"R=19 (C+E+F): {np19} params, {R19} terms")

# Target tensor
T_target = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            i = r*3+s  # α index
            j = s*3+u  # β index  
            k = r*3+u  # γ index
            T_target[i,j,k] = 1.0

# The expansion gives us:
# α_k(x) = Ma19[k*9:(k+1)*9, :] @ x  (as 9-vector)
# β_k(x) = Mb19[k*9:(k+1)*9, :] @ x
# γ_k(x) = Mg19[k*9:(k+1)*9, :] @ x
# T(x)[i,j,l] = Σ_k α_k[i] β_k[j] γ_k[l]

# Residual: F(x) = T(x) - T_target, a 9×9×9 = 729-dim system in np19 unknowns
# Jacobian: dF/dx[i,j,l,p] = Σ_k (dα_k[i]/dx_p β_k[j] γ_k[l] + α_k[i] dβ_k[j]/dx_p γ_k[l] + α_k[i] β_k[j] dγ_k[l]/dx_p)

def tensor_residual(x, Ma, Mb, Mg, R, T_tgt):
    T = np.zeros((9,9,9))
    for k in range(R):
        ak = Ma[k*9:(k+1)*9, :] @ x
        bk = Mb[k*9:(k+1)*9, :] @ x
        gk = Mg[k*9:(k+1)*9, :] @ x
        T += np.einsum('i,j,k->ijk', ak, bk, gk)
    return T - T_tgt

def tensor_jacobian(x, Ma, Mb, Mg, R, n_p):
    """J[flat_idx, param_idx] = dT[i,j,k]/dx[p]"""
    J = np.zeros((729, n_p))
    for k in range(R):
        Ak = Ma[k*9:(k+1)*9, :]  # (9, n_p)
        Bk = Mb[k*9:(k+1)*9, :]
        Gk = Mg[k*9:(k+1)*9, :]
        ak = Ak @ x  # (9,)
        bk = Bk @ x
        gk = Gk @ x
        # dT/dx_p = Σ_k (Ak[:,p] ⊗ bk ⊗ gk + ak ⊗ Bk[:,p] ⊗ gk + ak ⊗ bk ⊗ Gk[:,p])
        for p in range(n_p):
            term1 = np.einsum('i,j,k->ijk', Ak[:,p], bk, gk)
            term2 = np.einsum('i,j,k->ijk', ak, Bk[:,p], gk)
            term3 = np.einsum('i,j,k->ijk', ak, bk, Gk[:,p])
            J[:, p] += (term1 + term2 + term3).ravel()
    return J

# Test: Gauss-Newton from random start
flush("\nGauss-Newton on R=19 trilinear system:")
best_res = np.inf
for trial in range(20):
    x = rng.standard_normal(np19) * 0.5
    for it in range(200):
        F = tensor_residual(x, Ma19, Mb19, Mg19, R19, T_target)
        res = np.linalg.norm(F)
        if res < 1e-12:
            break
        J = tensor_jacobian(x, Ma19, Mb19, Mg19, R19, np19)
        dx, _, _, _ = np.linalg.lstsq(J, -F.ravel(), rcond=None)
        # Line search
        for ls in range(10):
            x_try = x + dx * (0.5**ls)
            F_try = tensor_residual(x_try, Ma19, Mb19, Mg19, R19, T_target)
            if np.linalg.norm(F_try) < res:
                x = x_try
                break
        else:
            break
    final_res = np.linalg.norm(tensor_residual(x, Ma19, Mb19, Mg19, R19, T_target))
    if final_res < best_res:
        best_res = final_res
    if trial < 5 or final_res < 1e-6:
        flush(f"  Trial {trial:2d}: ||F||={final_res:.6e}")

flush(f"\n  Best residual across 20 trials: {best_res:.6e}")
if best_res < 1e-10:
    flush("  ★★★ EXACT DECOMPOSITION FOUND! ★★★")
elif best_res < 1e-3:
    flush("  Close! Refining...")
else:
    flush("  No convergence — checking Jacobian rank at best point...")
    F = tensor_residual(x, Ma19, Mb19, Mg19, R19, T_target)
    J = tensor_jacobian(x, Ma19, Mb19, Mg19, R19, np19)
    rk_J = np.linalg.matrix_rank(J, tol=1e-8)
    flush(f"  Jacobian rank: {rk_J}/729 (params={np19})")
    flush(f"  System: 729 equations in {np19} unknowns")
    # Effective dimension
    flush(f"  Underdetermined by {max(0,np19-rk_J)} dims (or overdetermined by {max(0,729-np19)})")
