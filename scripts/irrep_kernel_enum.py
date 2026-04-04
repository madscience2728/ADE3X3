"""
IRREP KERNEL ENUMERATION — The cheapest possible next test.

The symmetry group Z₂ ≀ S₃ (order 48) acts on the R=19 term-coordinate space ℝ^19.
This 19-dim representation decomposes into irreducible representations (irreps).

Any valid 10-dim kernel K = ker(Γ) must be a DIRECT SUM of irrep subspaces
(because Γ commutes with the group action on valid decompositions).

This script:
  1) Builds the 48-element group Z₂ ≀ S₃ and its permutation action on the 19 kept triples
  2) Decomposes ℝ^19 into irreps via character theory / explicit projection
  3) Enumerates ALL subsets of irreps summing to dim 10
  4) For each candidate kernel, checks structural feasibility:
     - Does the complementary 9-dim space support a valid Σ (fiber-sum)?
     - Is Gate 2 (Δ containment) generically satisfiable?
  5) Reports the viable kernel shapes — these are the ONLY places to search.

If NO shapes are viable → R=19 is impossible under this symmetry assumption.
If a few shapes are viable → we've reduced continuous search to a handful of families.

Usage:
  python scripts/irrep_kernel_enum.py
"""
import numpy as np
from itertools import permutations, product as iproduct, combinations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# ══════════════════════════════════════════════
# GROUP: Z₂ ≀ S₃ = Z₂³ ⋊ S₃, order 48
# ══════════════════════════════════════════════

def swap12(x):
    return x if x == 0 else 3 - x

S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
assert len(GROUP) == 48

def act_on_triple(pi, eps, t):
    """Apply group element (pi, eps) to triple t ∈ {0,1,2}³."""
    x = [t[pi[i]] for i in range(3)]
    return tuple(swap12(x[i]) if eps[i] else x[i] for i in range(3))

# ══════════════════════════════════════════════
# THE 19 KEPT TRIPLES (orbits O₀ ∪ O₁ ∪ O₂)
# ══════════════════════════════════════════════

ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]

# Kept = triples with at least one zero coordinate
KEPT = sorted([t for t in ALL27 if 0 in t])
assert len(KEPT) == 19, f"Expected 19 kept triples, got {len(KEPT)}"

# Build index map
KEPT_IDX = {t: i for i, t in enumerate(KEPT)}

# ══════════════════════════════════════════════
# PERMUTATION REPRESENTATION ON ℝ^19
# ══════════════════════════════════════════════

def group_perm_matrix(pi, eps):
    """19×19 permutation matrix for group element (pi, eps) acting on kept triples."""
    P = np.zeros((19, 19))
    for i, t in enumerate(KEPT):
        t2 = act_on_triple(pi, eps, t)
        assert t2 in KEPT_IDX, f"Group action maps {t} to {t2} which is not in KEPT"
        j = KEPT_IDX[t2]
        P[j, i] = 1.0
    return P

print("Building 48 permutation matrices...")
PERM_MATRICES = [group_perm_matrix(pi, eps) for pi, eps in GROUP]

def main():
    global irreps

    # Verify group: closure under multiplication
    print("Verifying group closure...")
    for i, P in enumerate(PERM_MATRICES):
        for j, Q in enumerate(PERM_MATRICES):
            PQ = P @ Q
            found = False
            for k, R in enumerate(PERM_MATRICES):
                if np.allclose(PQ, R):
                    found = True
                    break
            assert found, f"Product of elements {i} and {j} not in group!"
    print("Group verified. ✓")
    
    # ══════════════════════════════════════════════
    # IRREP DECOMPOSITION VIA SIMULTANEOUS DIAGONALIZATION
    # ══════════════════════════════════════════════
    
    def decompose_into_irreps():
        """
        Decompose ℝ^19 into irreducible subspaces of Z₂ ≀ S₃.
        
        Method: Compute the group algebra projection operators using character theory.
        Since Z₂ ≀ S₃ is a finite group, we can use the averaging trick:
          P_V = (dim V / |G|) Σ_g χ_V(g)* ρ(g)
        
        But we don't know the characters a priori for this specific representation.
        Instead, we use a numerical approach:
          1) Pick a random symmetric matrix in the commutant algebra
          2) Diagonalize it — eigenspaces are irrep subspaces (generically)
        """
        # Build a random element of the commutant (group algebra center)
        # A matrix C commutes with all group elements iff C = (1/|G|) Σ_g c_g ρ(g)
        rng = np.random.default_rng(42)
        
        # Use multiple random commutant elements to resolve multiplicities
        coeffs = rng.standard_normal(48)
        C = sum(c * P for c, P in zip(coeffs, PERM_MATRICES)) / 48
        # Symmetrize to get real eigenvalues
        C = (C + C.T) / 2
        
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Group eigenvalues by near-equality
        tol = 1e-8
        irreps = []
        used = set()
        for i in range(19):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j in range(i + 1, 19):
                if j not in used and abs(eigenvalues[i] - eigenvalues[j]) < tol:
                    group.append(j)
                    used.add(j)
            basis = eigenvectors[:, group]
            irreps.append(basis)
        
        return irreps
    
    print("\nDecomposing ℝ^19 into irreps...")
    irreps = decompose_into_irreps()
    
    print(f"Found {len(irreps)} irreducible subspaces:")
    for i, basis in enumerate(irreps):
        dim = basis.shape[1]
        print(f"  V_{i}: dim = {dim}")
    
    # Verify: dimensions sum to 19
    total_dim = sum(b.shape[1] for b in irreps)
    assert total_dim == 19, f"Irrep dimensions sum to {total_dim}, expected 19"
    print(f"  Total dimension: {total_dim} ✓")
    
    # Verify each subspace is indeed invariant under the group
    print("\nVerifying invariance of each subspace...")
    for i, basis in enumerate(irreps):
        proj = basis @ basis.T  # projector onto this subspace
        for g_idx, P in enumerate(PERM_MATRICES):
            # P should map the subspace to itself
            img = P @ basis
            # Check that img lies in the subspace: proj @ img ≈ img
            resid = np.linalg.norm(img - proj @ img)
            if resid > 1e-8:
                print(f"  WARNING: V_{i} not invariant under group element {g_idx}, residual = {resid}")
                break
        else:
            print(f"  V_{i} (dim {basis.shape[1]}): invariant ✓")
    
    # Check irreducibility: for each subspace, check no proper invariant sub-subspace
    print("\nChecking irreducibility (no proper invariant subspaces)...")
    for i, basis in enumerate(irreps):
        dim = basis.shape[1]
        if dim == 1:
            print(f"  V_{i} (dim 1): trivially irreducible ✓")
            continue
        
        # Restrict group action to this subspace
        restricted = [basis.T @ P @ basis for P in PERM_MATRICES]
        
        # Random commutant element in restricted space
        rng2 = np.random.default_rng(123 + i)
        coeffs2 = rng2.standard_normal(48)
        C2 = sum(c * R for c, R in zip(coeffs2, restricted)) / 48
        C2 = (C2 + C2.T) / 2
        
        evals2 = np.linalg.eigvalsh(C2)
        # If all eigenvalues are equal (up to tol), the subspace is irreducible
        spread = np.max(evals2) - np.min(evals2)
        if spread < 1e-6:
            print(f"  V_{i} (dim {dim}): irreducible ✓ (eigenvalue spread = {spread:.2e})")
        else:
            print(f"  V_{i} (dim {dim}): REDUCIBLE — eigenvalue spread = {spread:.2e}")
            print(f"    Eigenvalues: {evals2}")
            print(f"    → Need to split this further")
            
            # Split using eigenspaces of C2
            evals_full, evecs_full = np.linalg.eigh(C2)
            sub_irreps = []
            used2 = set()
            for j in range(dim):
                if j in used2:
                    continue
                grp = [j]
                used2.add(j)
                for k in range(j+1, dim):
                    if k not in used2 and abs(evals_full[j] - evals_full[k]) < 1e-8:
                        grp.append(k)
                        used2.add(k)
                sub_basis_local = evecs_full[:, grp]
                sub_basis_global = basis @ sub_basis_local
                sub_irreps.append(sub_basis_global)
                print(f"    Sub-irrep: dim {len(grp)}")
            
            # Replace this irrep with its sub-irreps
            idx = irreps.index(basis)
            irreps[idx:idx+1] = sub_irreps
    
    # Recount after splitting
    print(f"\nFinal irrep decomposition: {len(irreps)} subspaces")
    dims = sorted([b.shape[1] for b in irreps])
    print(f"Dimensions: {dims}")
    print(f"Sum: {sum(dims)}")
    
    # ══════════════════════════════════════════════
    # ENUMERATE ALL DIM-10 KERNEL CANDIDATES
    # ══════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ENUMERATING ALL IRREP COMBINATIONS SUMMING TO DIM 10")
    print("=" * 60)
    
    n_irreps = len(irreps)
    irrep_dims = [b.shape[1] for b in irreps]
    
    viable_kernels = []
    for r in range(1, n_irreps + 1):
        for combo in combinations(range(n_irreps), r):
            if sum(irrep_dims[i] for i in combo) == 10:
                viable_kernels.append(combo)
    
    print(f"Found {len(viable_kernels)} combinations of irreps summing to dim 10:")
    for idx, combo in enumerate(viable_kernels):
        dims_str = " + ".join(f"V_{i}({irrep_dims[i]})" for i in combo)
        print(f"  K_{idx}: {dims_str} = 10")
    
    # ══════════════════════════════════════════════
    # FEASIBILITY CHECK FOR EACH KERNEL CANDIDATE
    # ══════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("FEASIBILITY ANALYSIS FOR EACH KERNEL CANDIDATE")
    print("=" * 60)
    
    # Build the matmul tensor target vector
    # For R=19, Gamma (9×19) must satisfy Gamma · Sigma = 3I₉
    # The 9-dim image of Gamma is the complement of K in ℝ^19 (in the row space sense)
    
    # For each kernel K, the complement C = K^perp has dim 9.
    # Gamma's rows span C. So Sigma must map ℝ^9 (the (r,u) fiber indices) 
    # into C in a way that Gamma · Sigma = 3I₉.
    
    # Key constraint: Sigma's 9 columns must be linearly independent in C (the complement).
    # This is rank([Sigma | K]) = 19, equivalently, Sigma's projection onto C has rank 9.
    
    # For a GENERIC choice of alpha, beta respecting symmetry:
    # - Sigma has 9 columns in ℝ^19
    # - H = [Eta1|Eta2] has 18 columns in ℝ^19, and rank(H) = 10 means H spans exactly K^perp... 
    #   wait, no: rank(H) = 10 means H spans a 10-dim space, which should be EXACTLY the complement
    #   of ker(Gamma). But ker(Gamma) = K has dim 10. So H spans the 10-dim space K^perp... 
    #   that's wrong dimensionally. Let me re-read.
    #
    # Actually: ker(Gamma) has dimension R - 9 = 10. H has 18 columns in ℝ^R=ℝ^19.
    # rank(H) = 10 means the column space of H is 10-dimensional.
    # The conservation law says rank(H) = R - 9 = 10, and span(H) = ker(Gamma).
    # So K = span(H) = ker(Gamma), dim 10.
    #
    # For Sigma: its 9 columns live in ℝ^19. The condition Gamma · Sigma = 3I₉ means
    # Sigma's columns projected onto the image space (complement of K) give 3I₉ after 
    # applying Gamma. So Sigma's columns must have nontrivial components in K^perp (dim 9),
    # and those components must span all of K^perp.
    #
    # For Delta containment: Delta's columns (54 of them) must all lie in span(H) = K.
    # This means the projection of Delta onto K^perp must be zero.
    
    # For a STRUCTURAL check: we need to know if the sigma/delta constraints are 
    # compatible with the symmetry structure of each kernel candidate.
    
    # Build fiber coordinate structure for generic symmetric (alpha, beta)
    # For each of the 3 super-seeds, we have 27 free parameters.
    # Let's do a Monte Carlo check: sample random symmetric (alpha, beta),
    # project H onto each kernel candidate, check rank and delta containment.
    
    N_WORKERS = 12
    N_TRIALS = 1000
    
    print(f"\nRunning Monte Carlo feasibility ({N_TRIALS} random symmetric seeds per kernel, {N_WORKERS} workers)...")
    
    def build_symmetric_decomposition(params, GROUP=GROUP, KEPT=KEPT):
        """Build 19-term symmetric (alpha, beta, gamma) from 3 super-seeds."""
        seeds = [(0,0,0), (0,0,1), (0,1,1)]
        kept_set = set(KEPT)
        
        all_a, all_b, all_g = [], [], []
        off = 0
        for seed in seeds:
            a = params[off:off+9].reshape(3, 3)
            b = params[off+9:off+18].reshape(3, 3)
            g = params[off+18:off+27].reshape(3, 3)
            
            # Symmetrize: average over stabilizer
            stab = [(pi, eps) for pi, eps in GROUP if act_on_triple(pi, eps, seed) == seed]
            aa = np.zeros((3,3)); bb = np.zeros((3,3)); gg = np.zeros((3,3))
            for pi, eps in stab:
                a2, b2, g2 = _act_on_factors(pi, eps, a, b, g)
                aa += a2; bb += b2; gg += g2
            n = len(stab)
            aa /= n; bb /= n; gg /= n
            
            # Orbit expansion
            seen = set()
            for pi, eps in GROUP:
                t = act_on_triple(pi, eps, seed)
                if t not in seen and t in kept_set:
                    seen.add(t)
                    at, bt, gt = _act_on_factors(pi, eps, aa, bb, gg)
                    all_a.append(at); all_b.append(bt); all_g.append(gt)
        
        return np.array(all_a), np.array(all_b), np.array(all_g)
    
    def _act_on_factors(pi, eps, a, b, g):
        """Apply group element to factor matrices."""
        def pm(e):
            return np.eye(3)[[0,2,1], :] if e else np.eye(3)
        P = [pm(eps[i]) for i in range(3)]
        pi_inv = [0]*3
        for i in range(3):
            pi_inv[pi[i]] = i
        
        facs = {(0,1): a, (1,2): b, (0,2): g}
        res = {}
        for (x, y), nm in [((0,1), 'a'), ((1,2), 'b'), ((0,2), 'g')]:
            oa, ob = pi_inv[x], pi_inv[y]
            key = (min(oa, ob), max(oa, ob))
            F = facs[key].T if (oa, ob) != key else facs[key].copy()
            res[nm] = P[x] @ F @ P[y].T
        return res['a'], res['b'], res['g']
    
    def compute_H_Delta(alpha, beta):
        """Compute H and Delta matrices."""
        R = alpha.shape[0]
        Sigma = np.zeros((R, 9))
        Eta1 = np.zeros((R, 9))
        Eta2 = np.zeros((R, 9))
        
        for k in range(R):
            for r in range(3):
                for u in range(3):
                    idx = r*3 + u
                    s0 = alpha[k, r, 0] * beta[k, 0, u]
                    s1 = alpha[k, r, 1] * beta[k, 1, u]
                    s2 = alpha[k, r, 2] * beta[k, 2, u]
                    Sigma[k, idx] = s0 + s1 + s2
                    Eta1[k, idx] = s0 - s1
                    Eta2[k, idx] = s1 - s2
        
        H = np.hstack([Eta1, Eta2])
        
        dead_pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
        Delta = np.zeros((R, 54))
        for k in range(R):
            col = 0
            for (s, t) in dead_pairs:
                for r in range(3):
                    for u in range(3):
                        Delta[k, col] = alpha[k, r, s] * beta[k, t, u]
                        col += 1
        return Sigma, H, Delta
    
    # Feasibility results
    results = []
    
    def _eval_kernel(args):
        """Worker function: test one kernel candidate with N_TRIALS random samples."""
        kidx, combo, irrep_bases, n_trials = args
        K_basis = np.hstack([irrep_bases[i] for i in combo])  # 19 x 10
        K_proj = K_basis @ K_basis.T
        C_proj = np.eye(19) - K_proj
    
        rng = np.random.default_rng(2026 * 1000 + kidx)
        gate1_pass = 0
        gate2_pass = 0
        sigma_rank_ok = 0
    
        for trial in range(n_trials):
            params = rng.standard_normal(81)
            try:
                alpha, beta, gamma = build_symmetric_decomposition(params)
            except Exception:
                continue
            
            if alpha.shape[0] != 19:
                continue
            
            Sigma, H, Delta = compute_H_Delta(alpha, beta)
    
            rank_H = np.linalg.matrix_rank(H, tol=1e-6)
            H_leak = np.linalg.norm(C_proj @ H, 'fro') / (np.linalg.norm(H, 'fro') + 1e-30)
    
            if rank_H == 10 and H_leak < 0.01:
                gate1_pass += 1
                Delta_leak = np.linalg.norm(C_proj @ Delta, 'fro') / (np.linalg.norm(Delta, 'fro') + 1e-30)
                if Delta_leak < 0.01:
                    gate2_pass += 1
    
            Sigma_in_C = C_proj @ Sigma
            sr = np.linalg.matrix_rank(Sigma_in_C, tol=1e-6)
            if sr == 9:
                sigma_rank_ok += 1
    
        return {
            'kernel_idx': kidx,
            'irrep_combo': list(combo),
            'dims': [irrep_bases[i].shape[1] for i in combo],
            'gate1_pass': gate1_pass,
            'gate2_pass': gate2_pass,
            'sigma_rank_ok': sigma_rank_ok,
            'n_trials': n_trials,
        }
    
    # Serialize irrep bases for pickling (list of arrays)
    irrep_bases_list = [b.copy() for b in irreps]
    irrep_dims_list = [b.shape[1] for b in irreps]
    
    tasks = [(kidx, combo, irrep_bases_list, N_TRIALS) for kidx, combo in enumerate(viable_kernels)]
    
    print(f"Dispatching {len(tasks)} kernel candidates across {N_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_eval_kernel, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            dims_str = "+".join(str(d) for d in r['dims'])
            g1 = r['gate1_pass']
            g2 = r['gate2_pass']
            sr = r['sigma_rank_ok']
            print(f"  K_{r['kernel_idx']} [{dims_str}=10]: G1={g1}, G2={g2}, Σ={sr}/{r['n_trials']}")
    
    results.sort(key=lambda r: r['kernel_idx'])
    
    # ══════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    any_viable = False
    for r in results:
        dims_str = "+".join(str(d) for d in r['dims'])
        viable = r['gate1_pass'] > 0 and r['sigma_rank_ok'] > 0
        status = "VIABLE" if viable else "DEAD"
        if viable:
            any_viable = True
        print(f"  K_{r['kernel_idx']} [{dims_str}]: {status} "
              f"(G1={r['gate1_pass']}, G2={r['gate2_pass']}, Σ={r['sigma_rank_ok']})")
    
    if not any_viable:
        print("\n*** NO VIABLE KERNEL SHAPES FOUND ***")
        print("If this holds, R=19 is impossible under Z₂ ≀ S₃ symmetry.")
        print("This would mean the continuous search is fundamentally doomed.")
    else:
        print(f"\nViable kernels found. These are the ONLY shapes worth searching.")
        print("Feed these into the optimizer as hard subspace constraints.")
    
    # Also check R=13 (rank(H)=4, kernel dim=4) and R=20 (rank(H)=11, kernel dim=11)
    for R_target, kernel_dim, rank_H_target in [(13, 4, 4), (20, 11, 11)]:
        print(f"\n{'='*60}")
        print(f"R={R_target}: kernel dim={kernel_dim}, rank(H)={rank_H_target}")
        print(f"{'='*60}")
        
        combos = []
        for r in range(1, n_irreps + 1):
            for combo in combinations(range(n_irreps), r):
                if sum(irrep_dims[i] for i in combo) == kernel_dim:
                    combos.append(combo)
        
        print(f"Found {len(combos)} irrep combinations summing to dim {kernel_dim}")
        for combo in combos:
            dims_str = "+".join(f"V_{i}({irrep_dims[i]})" for i in combo)
            print(f"  {dims_str}")
    
    print("\nDone.")
    
    
if __name__ == '__main__':
    main()