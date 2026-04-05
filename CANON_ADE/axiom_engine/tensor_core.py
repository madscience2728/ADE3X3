"""
tensor_core.py — Fundamental tensor objects and group actions.
All axiom evaluators build on this.
"""
import numpy as np
from itertools import permutations, product as iproduct, combinations
from functools import lru_cache
import torch

# ═══════════════════════════════════════════════════════════════
# T_MATMUL: The 3×3 matrix multiplication tensor (9×9×9)
# ═══════════════════════════════════════════════════════════════

def build_T(n=3):
    T = np.zeros((n*n, n*n, n*n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                T[n*r+s, n*s+u, n*r+u] = 1.0
    return T

T_MATMUL = build_T(3)

# Support: the 27 nonzero entries
SUPPORT = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            SUPPORT.append((r, s, u))

# ═══════════════════════════════════════════════════════════════
# GROUP: Z₂ ≀ S₃ = Z₂³ ⋊ S₃, order 48
# ═══════════════════════════════════════════════════════════════

def swap12(x):
    return x if x == 0 else 3 - x

S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
assert len(GROUP) == 48

def act_on_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(swap12(x[i]) if eps[i] else x[i] for i in range(3))

# ═══════════════════════════════════════════════════════════════
# TERM SETS: kept triples for each rank
# ═══════════════════════════════════════════════════════════════

ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
KEPT19 = sorted([t for t in ALL27 if 0 in t])
DELETED8 = sorted([t for t in ALL27 if 0 not in t])

# Orbit classification
ORBIT_0 = [(0,0,0)]                                       # 1 corner
ORBIT_1 = sorted([t for t in KEPT19 if sum(x==0 for x in t)==2])  # 6 edges
ORBIT_2 = sorted([t for t in KEPT19 if sum(x==0 for x in t)==1])  # 12 faces
ORBIT_3 = DELETED8                                          # 8 interior

def kept_triples_for_rank(R):
    """Return G-stable subsets of ALL27 of size R. Returns list of lists."""
    if R == 27:
        return [ALL27]
    if R == 19:
        return [KEPT19]
    # For other ranks, enumerate G-stable subsets
    orbits = [ORBIT_0, ORBIT_1, ORBIT_2, ORBIT_3]
    orbit_sizes = [len(o) for o in orbits]
    results = []
    for mask in range(16):
        selected = []
        for i in range(4):
            if mask & (1 << i):
                selected.extend(orbits[i])
        if len(selected) == R:
            results.append(sorted(selected))
    # Also try partial orbit selections for non-exact matches
    # R=20: keep 19 + 1 from O3 (but need G-stable — O3 is transitive, so all or none)
    # R=13: O0(1) + O2(12) = 13
    # R=20: O0(1) + O1(6) + O2(12) + partial O3 — not G-stable
    # Actually G-stable subsets must be unions of entire orbits
    # Possible: {},{O0},{O1},{O2},{O3},{O0,O1},{O0,O2},{O0,O3},...all subsets
    # Sizes: 0,1,6,12,8,7,13,9,18,14,20,19,26,21,27
    return results if results else None

# ═══════════════════════════════════════════════════════════════
# PERMUTATION MATRICES on ℝ^R
# ═══════════════════════════════════════════════════════════════

def build_perm_matrices(kept):
    """Build |G|=48 permutation matrices for group action on kept triples."""
    kept_idx = {t: i for i, t in enumerate(kept)}
    R = len(kept)
    mats = []
    for pi, eps in GROUP:
        P = np.zeros((R, R), dtype=np.float64)
        for i, t in enumerate(kept):
            t2 = act_on_triple(pi, eps, t)
            if t2 in kept_idx:
                P[kept_idx[t2], i] = 1.0
        mats.append(P)
    return mats

PERM19 = build_perm_matrices(KEPT19)

# ═══════════════════════════════════════════════════════════════
# STEP-51 COORDINATES: Sigma, H (Eta1|Eta2), Delta
# ═══════════════════════════════════════════════════════════════

def compute_step51(alpha, beta):
    """
    Compute Step-51 coordinate decomposition.
    alpha: (R, 3, 3), beta: (R, 3, 3) factor matrices
    Returns: Sigma (R,9), H (R,18), Delta (R,54)
    """
    R = alpha.shape[0]
    Sigma = np.zeros((R, 9), dtype=np.float64)
    Eta1 = np.zeros((R, 9), dtype=np.float64)
    Eta2 = np.zeros((R, 9), dtype=np.float64)
    
    for k in range(R):
        for r in range(3):
            for u in range(3):
                idx = r * 3 + u
                s0 = alpha[k, r, 0] * beta[k, 0, u]
                s1 = alpha[k, r, 1] * beta[k, 1, u]
                s2 = alpha[k, r, 2] * beta[k, 2, u]
                Sigma[k, idx] = s0 + s1 + s2
                Eta1[k, idx] = s0 - s1
                Eta2[k, idx] = s1 - s2
    
    H = np.hstack([Eta1, Eta2])
    
    dead_pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    Delta = np.zeros((R, 54), dtype=np.float64)
    for k in range(R):
        col = 0
        for (s, t) in dead_pairs:
            for r in range(3):
                for u in range(3):
                    Delta[k, col] = alpha[k, r, s] * beta[k, t, u]
                    col += 1
    
    return Sigma, H, Delta

def compute_step51_torch(alpha, beta, device='cuda'):
    """GPU-accelerated Step-51 computation."""
    a = torch.as_tensor(alpha, dtype=torch.float64, device=device)
    b = torch.as_tensor(beta, dtype=torch.float64, device=device)
    R = a.shape[0]
    
    # Sigma[k, r*3+u] = sum_s a[k,r,s]*b[k,s,u]
    # = einsum('krs,ksu->kru' then reshape)
    Sigma = torch.einsum('krs,ksu->kru', a, b).reshape(R, 9)
    
    # Eta1[k, r*3+u] = a[k,r,0]*b[k,0,u] - a[k,r,1]*b[k,1,u]
    Eta1 = (a[:, :, 0:1] * b[:, 0:1, :] - a[:, :, 1:2] * b[:, 1:2, :]).reshape(R, 9)
    Eta2 = (a[:, :, 1:2] * b[:, 1:2, :] - a[:, :, 2:3] * b[:, 2:3, :]).reshape(R, 9)
    H = torch.cat([Eta1, Eta2], dim=1)
    
    # Delta: off-diagonal s≠t products
    dead_pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    Delta_parts = []
    for (s, t) in dead_pairs:
        # a[k,r,s]*b[k,t,u] for all r,u → shape (R, 9)
        Delta_parts.append((a[:, :, s:s+1] * b[:, t:t+1, :]).reshape(R, 9))
    Delta = torch.cat(Delta_parts, dim=1)
    
    return Sigma, H, Delta

# ═══════════════════════════════════════════════════════════════
# IRREP DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def decompose_irreps(perm_matrices, dim):
    """Decompose ℝ^dim into irreps of the group defined by perm_matrices."""
    rng = np.random.default_rng(42)
    n_g = len(perm_matrices)
    
    # Random commutant element
    coeffs = rng.standard_normal(n_g)
    C = sum(c * P for c, P in zip(coeffs, perm_matrices)) / n_g
    C = (C + C.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    tol = 1e-8
    irreps = []
    used = set()
    for i in range(dim):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, dim):
            if j not in used and abs(eigenvalues[i] - eigenvalues[j]) < tol:
                group.append(j)
                used.add(j)
        basis = eigenvectors[:, group]
        irreps.append(basis)
    
    # Check irreducibility and split if needed
    final_irreps = []
    for basis in irreps:
        d = basis.shape[1]
        if d == 1:
            final_irreps.append(basis)
            continue
        restricted = [basis.T @ P @ basis for P in perm_matrices]
        coeffs2 = rng.standard_normal(n_g)
        C2 = sum(c * R for c, R in zip(coeffs2, restricted)) / n_g
        C2 = (C2 + C2.T) / 2
        evals2 = np.linalg.eigvalsh(C2)
        spread = np.max(evals2) - np.min(evals2)
        if spread < 1e-6:
            final_irreps.append(basis)
        else:
            evals_full, evecs_full = np.linalg.eigh(C2)
            used2 = set()
            for j in range(d):
                if j in used2:
                    continue
                grp = [j]
                used2.add(j)
                for k in range(j+1, d):
                    if k not in used2 and abs(evals_full[j] - evals_full[k]) < 1e-8:
                        grp.append(k)
                        used2.add(k)
                sub_basis = basis @ evecs_full[:, grp]
                final_irreps.append(sub_basis)
    
    return final_irreps

# ═══════════════════════════════════════════════════════════════
# FACTOR GENERATION WITH SYMMETRY
# ═══════════════════════════════════════════════════════════════

def _act_on_factors(pi, eps, a, b, g):
    """Apply group element to factor matrices (3×3 each)."""
    def pm(e):
        return np.eye(3)[[0, 2, 1], :] if e else np.eye(3)
    P = [pm(eps[i]) for i in range(3)]
    pi_inv = [0] * 3
    for i in range(3):
        pi_inv[pi[i]] = i
    facs = {(0, 1): a, (1, 2): b, (0, 2): g}
    res = {}
    for (x, y), nm in [((0, 1), 'a'), ((1, 2), 'b'), ((0, 2), 'g')]:
        oa, ob = pi_inv[x], pi_inv[y]
        key = (min(oa, ob), max(oa, ob))
        F = facs[key].T if (oa, ob) != key else facs[key].copy()
        res[nm] = P[x] @ F @ P[y].T
    return res['a'], res['b'], res['g']

def build_symmetric_factors(params, kept=KEPT19):
    """
    Build R-term symmetric (alpha, beta, gamma) from 3 super-seed parameters.
    params: length 81 (3 seeds × 27 params each)
    Returns: alpha (R,3,3), beta (R,3,3), gamma (R,3,3)
    """
    seeds = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]
    kept_set = set(kept)
    
    all_a, all_b, all_g = [], [], []
    off = 0
    for seed in seeds:
        a = params[off:off+9].reshape(3, 3)
        b = params[off+9:off+18].reshape(3, 3)
        g = params[off+18:off+27].reshape(3, 3)
        
        stab = [(pi, eps) for pi, eps in GROUP if act_on_triple(pi, eps, seed) == seed]
        aa = np.zeros((3, 3)); bb = np.zeros((3, 3)); gg = np.zeros((3, 3))
        for pi, eps in stab:
            a2, b2, g2 = _act_on_factors(pi, eps, a, b, g)
            aa += a2; bb += b2; gg += g2
        n = len(stab)
        aa /= n; bb /= n; gg /= n
        
        seen = set()
        for pi, eps in GROUP:
            t = act_on_triple(pi, eps, seed)
            if t not in seen and t in kept_set:
                seen.add(t)
                at, bt, gt = _act_on_factors(pi, eps, aa, bb, gg)
                all_a.append(at); all_b.append(bt); all_g.append(gt)
        off += 27
    
    return np.array(all_a), np.array(all_b), np.array(all_g)

def build_random_factors(R, rng=None, symmetric=True):
    """Generate random factor matrices, optionally with G-symmetry."""
    if rng is None:
        rng = np.random.default_rng()
    if symmetric and R == 19:
        params = rng.standard_normal(81)
        return build_symmetric_factors(params)
    else:
        alpha = rng.standard_normal((R, 3, 3))
        beta = rng.standard_normal((R, 3, 3))
        gamma = rng.standard_normal((R, 3, 3))
        return alpha, beta, gamma

# ═══════════════════════════════════════════════════════════════
# FIBER-CONSTRAINED FACTOR GENERATION (Tower Connection)
# ═══════════════════════════════════════════════════════════════

# 9 fibers over (s,u) ∈ Z₃². Each fiber = {r : (r,s,u) ∈ KEPT19}.
# 4 forced fibers (s,u ∈ {0,2}²): single element r=1, γ pinned to 3.
# 5 free fibers (s or u = 1): 3 elements, γ sums to 3.

_FIBERS_19 = {}  # (s,u) -> list of (r,s,u) triples
for r, s, u in KEPT19:
    _FIBERS_19.setdefault((s, u), []).append((r, s, u))

# Precompute index map: triple -> position in KEPT19
_KEPT19_IDX = {t: i for i, t in enumerate(KEPT19)}

def build_fiber_constrained_factors(rng=None, gamma_scale=1.0):
    """
    Generate R=19 factors respecting the Tower Connection fiber constraints.

    Fiber sum constraint: Γ(s,u) = Σ_r γ(r,s,u) = 3 for all (s,u) ∈ Z₃².
    - 4 forced fibers (s,u ∈ {0,2}²): γ(1,s,u) = 3 (single term).
    - 5 free fibers: 3 terms each, parameterized by 2 d.o.f.

    Factor directions are random unit-norm; γ scales are applied.

    Returns: alpha (19,3,3), beta (19,3,3), gamma (19,3,3)
    """
    if rng is None:
        rng = np.random.default_rng()

    R = 19
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma = np.zeros((R, 3, 3))

    for (s, u), triples in _FIBERS_19.items():
        n_terms = len(triples)

        if n_terms == 1:
            # Forced fiber: γ = 3
            gammas = [3.0]
        else:
            # Free fiber: sample 2 d.o.f., third constrained to sum=3
            g1 = rng.normal(1.0, gamma_scale)
            g2 = rng.normal(1.0, gamma_scale)
            g3 = 3.0 - g1 - g2
            gammas = [g1, g2, g3]

        for k, triple in enumerate(triples):
            idx = _KEPT19_IDX[triple]
            g_val = gammas[k]
            sign = np.sign(g_val) if g_val != 0 else 1.0
            scale = np.abs(g_val) ** (1.0 / 3.0)

            # Random unit-norm direction for each factor, scaled by cbrt(|γ|)
            a = rng.standard_normal((3, 3))
            a /= np.linalg.norm(a) + 1e-12
            b = rng.standard_normal((3, 3))
            b /= np.linalg.norm(b) + 1e-12
            c = rng.standard_normal((3, 3))
            c /= np.linalg.norm(c) + 1e-12

            # Distribute sign and scale: γ = sign * scale³
            alpha[idx] = sign * scale * a
            beta[idx] = scale * b
            gamma[idx] = scale * c

    return alpha, beta, gamma


def build_fiber_constrained_symmetric_factors(rng=None, gamma_scale=1.0):
    """
    Fiber-constrained factors with Z₂≀S₃ symmetry enforced.

    Uses 3 orbit seeds (same as build_symmetric_factors) but applies
    fiber γ constraints before symmetrization.

    Returns: alpha (19,3,3), beta (19,3,3), gamma (19,3,3)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Sample the 10 γ d.o.f. (5 free fibers × 2 free params)
    gamma_vals = {}  # triple -> γ value
    for (s, u), triples in _FIBERS_19.items():
        if len(triples) == 1:
            gamma_vals[triples[0]] = 3.0
        else:
            g1 = rng.normal(1.0, gamma_scale)
            g2 = rng.normal(1.0, gamma_scale)
            gamma_vals[triples[0]] = g1
            gamma_vals[triples[1]] = g2
            gamma_vals[triples[2]] = 3.0 - g1 - g2

    # Step 2: Build symmetric factor directions from 3 orbit seeds
    params = rng.standard_normal(81)
    a_sym, b_sym, g_sym = build_symmetric_factors(params)

    # Step 3: Rescale each term's factors by cbrt(γ/γ_current)
    # Current effective γ per term = ||a_t|| * ||b_t|| * ||g_t||
    R = 19
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_out = np.zeros((R, 3, 3))

    for i, triple in enumerate(KEPT19):
        target_gamma = gamma_vals[triple]
        current_scale = (np.linalg.norm(a_sym[i]) *
                         np.linalg.norm(b_sym[i]) *
                         np.linalg.norm(g_sym[i]))
        if current_scale < 1e-12:
            # Degenerate — replace with random
            a = rng.standard_normal((3, 3))
            b = rng.standard_normal((3, 3))
            c = rng.standard_normal((3, 3))
            scale = np.abs(target_gamma) ** (1.0 / 3.0)
            sign = np.sign(target_gamma) if target_gamma != 0 else 1.0
            alpha[i] = sign * scale * a / (np.linalg.norm(a) + 1e-12)
            beta[i] = scale * b / (np.linalg.norm(b) + 1e-12)
            gamma_out[i] = scale * c / (np.linalg.norm(c) + 1e-12)
        else:
            ratio = target_gamma / current_scale
            sign = np.sign(ratio) if ratio != 0 else 1.0
            rescale = np.abs(ratio) ** (1.0 / 3.0)
            alpha[i] = sign * rescale * a_sym[i]
            beta[i] = rescale * b_sym[i]
            gamma_out[i] = rescale * g_sym[i]

    return alpha, beta, gamma_out

# ═══════════════════════════════════════════════════════════════
# AlphaTensor reference (rank 23)
# ═══════════════════════════════════════════════════════════════

def alphatensor_factorization():
    """Return (U, V, W) each 9×23 integer matrices for rank-23 decomposition."""
    u = np.array([
        [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0,-1, 0,-1,-1,-1,-1,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,-1, 1, 1, 0, 1, 0, 0,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 0, 0,-1, 0, 0],
        [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0,-1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1,-1, 0, 0, 0, 0, 0,-1, 0,-1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0,-1, 1,-1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.float64)
    v = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [-1,-1, 0, 0,-1, 0,-1,-1, 1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1,-1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0,-1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [-1,-1, 0, 0,-1, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1,-1,-1, 0, 1, 0, 1, 0,-1, 0, 0],
        [-1,-1,-1,-1,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float64)
    w = np.array([
        [0, 0, 0, 0, 0, 0,-1, 1, 1, 0, 0,-1,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1],
        [-1, 1, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0,-1, 1, 1, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
        [0, 0, 0, 1,-1, 0, 1, 0, 0, 0,-1, 0,-1, 1, 0, 0, 0,-1, 0, 0,-1, 0, 1],
        [-1, 1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 1, 0,-1, 0, 0,-1, 0, 0],
    ], dtype=np.float64)
    return u, v, w
