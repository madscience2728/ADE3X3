"""Verify the Pythagorean theorem: ||T||² = ||R||² + ||T_hat||² iff R ⊥ T_hat.
Also check whether the 19/27 energy split is a structural invariant."""
import json
import glob
import numpy as np

def matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                c = r * n + u
                a = r * n + s
                b = s * n + u
                tensor[c, a, b] = 1.0
    return tensor

T = matrix_multiplication_tensor(3)

def reconstruct(json_path):
    with open(json_path) as f:
        data = json.load(f)
    R = len(data["terms"])
    alpha = np.zeros((R, 9))
    beta = np.zeros((R, 9))
    gamma = np.zeros((R, 9))
    for i, term in enumerate(data["terms"]):
        for idx, val in zip(term["alpha_support"], term["alpha_values"]):
            alpha[i, idx] = val
        for idx, val in zip(term["beta_support"], term["beta_values"]):
            beta[i, idx] = val
        for idx, val in zip(term["gamma_support"], term["gamma_values"]):
            gamma[i, idx] = val
    candidate = np.zeros((9, 9, 9))
    for i in range(R):
        candidate += np.einsum('c,a,b->cab', gamma[i], alpha[i], beta[i])
    residual = candidate - T
    return data, residual, candidate

print("=" * 70)
print("PYTHAGOREAN THEOREM CHECK: ||T||² = ||R||² + ||T_hat||² + 2<R, T_hat>")
print("=" * 70)
print(f"||T||_F² = {np.sum(T**2):.6f}")
print()

all_json = sorted(glob.glob("outputs/exports/step84_batches/*/copy_*/step84_best_individual.json"))

print(f"{'max_abs':>10} {'||R||²':>10} {'||T̂||²':>10} {'<R,T̂>':>12} {'||R||²+||T̂||²+2<R,T̂>':>22} {'ratio 19/27':>11}")
print("-" * 80)

for jp in sorted(all_json):
    data, residual, candidate = reconstruct(jp)
    max_abs = np.max(np.abs(residual))
    
    R_fro2 = np.sum(residual**2)
    That_fro2 = np.sum(candidate**2)
    inner = np.sum(residual * candidate)
    check = R_fro2 + That_fro2 + 2 * inner
    
    print(f"{max_abs:10.6f} {R_fro2:10.6f} {That_fro2:10.6f} {inner:12.6f} {check:22.6f} {R_fro2/27:11.6f}")

print()
print("=" * 70)
print("KEY QUESTION: Is R ⊥ T̂ (i.e., <R, T̂> ≈ 0)?")
print("=" * 70)
print()
print("If yes: ||T||² = ||R||² + ||T̂||²")
print(f"  27 = 8 + 19  ←  (rank=19, coincidence?)")
print()

# Check: what does optimal ALS residual look like?
# For rank-R CP, ALS finds local min of ||T - sum_r alpha_r ⊗ beta_r ⊗ gamma_r||²
# At convergence, the gradient w.r.t. each parameter is zero
# This means: R is orthogonal to each column of each mode's Khatri-Rao product
# But is R orthogonal to T̂ itself?

# Verify: <R, alpha_i ⊗ beta_j ⊗ gamma_k> = sum_abc R[a,b,c] * alpha_i[a] * beta_j[b] * gamma_k[c]
print("=" * 70)
print("STRUCTURAL ANALYSIS: Why ||R||² = 27 - 19 = 8?")
print("=" * 70)
print()
print("The target T is the sum of 27 unit rank-1 terms (standard algorithm).")
print("Each is e_{r,u} ⊗ e_{r,s} ⊗ e_{s,u} for (r,s,u) ∈ {0,1,2}³")
print()
print("These 27 terms are mutually orthogonal (their outer products are")
print("unit vectors in different coordinate directions), so:")
print("  ||T||² = sum of 27 unit norms² = 27")
print()
print("Hypothesis: ALS at rank 19 converges to a projection of T onto")
print("a 19-dimensional subspace of the 27-dimensional span of live entries.")
print("The residual lives in the orthogonal 8-dimensional complement.")
print()
print("If true: ||R||² = 8 entries × 1² = 8 (each unmatched live entry)")
print("         ||T̂||² = 19 entries × 1² = 19 (each matched live entry)")
print("         <R, T̂> = 0 (orthogonal subspaces)")
print()

# Deeper test: decompose residual in the live-entry basis
print("=" * 70)
print("RESIDUAL IN LIVE-ENTRY BASIS (for best ~0.5 basin example)")
print("=" * 70)

# Pick the best 0.5-basin example
best_05 = None
for jp in all_json:
    data, residual, candidate = reconstruct(jp)
    max_abs = np.max(np.abs(residual))
    if 0.45 < max_abs < 0.55:
        if best_05 is None or max_abs < best_05[0]:
            best_05 = (max_abs, residual, candidate, jp)

if best_05:
    max_abs, residual, candidate, jp = best_05
    print(f"Example: {jp}")
    print(f"max_abs = {max_abs:.8f}")
    print()
    
    # The live entries form a 27-dim orthonormal basis
    # Residual projected onto each basis vector e_{c,a,b} where T[c,a,b]=1
    live_projections = []
    for r in range(3):
        for s in range(3):
            for u in range(3):
                c = r * 3 + u
                a = r * 3 + s
                b = s * 3 + u
                live_projections.append((r, s, u, residual[c, a, b]))
    
    live_proj = np.array([p[3] for p in live_projections])
    dead_energy = np.sum(residual**2) - np.sum(live_proj**2)
    
    print(f"Residual energy in live subspace: {np.sum(live_proj**2):.6f}")
    print(f"Residual energy in dead subspace: {dead_energy:.6f}")
    print(f"Total: {np.sum(residual**2):.6f}")
    print()
    
    # Check: is the dead energy zero at convergence?
    # At true ALS convergence with no structural penalty or dead leakage,
    # the optimal rank-19 approximation should live in the 27-dim live span.
    # But our optimizer is evolutionary, not pure ALS.
    
    # Check SVD of the candidate tensor (unfolded)
    # Mode-0 unfolding: 9 x 81
    cand_mat = candidate.reshape(9, 81)
    U, s, Vt = np.linalg.svd(cand_mat, full_matrices=False)
    print(f"Candidate mode-0 singular values: {s}")
    print(f"Candidate mode-0 rank (>1e-10): {np.sum(s > 1e-10)}")
    
    # Same for target
    T_mat = T.reshape(9, 81)
    U_T, s_T, Vt_T = np.linalg.svd(T_mat, full_matrices=False)
    print(f"Target mode-0 singular values: {s_T}")
    print(f"Target mode-0 rank (>1e-10): {np.sum(s_T > 1e-10)}")

print()
print("=" * 70)
print("BASIN PREDICTION FROM S₃³ GROUP THEORY")
print("=" * 70)
print()
print("S₃ × S₃ × S₃ has order 216.")
print("S₃ irreps: trivial(1), sign(1), standard(2)")
print("S₃³ irreps: all tensor products → dims 1,1,1,2,2,2,2,2,2,4,4,4,8")
print(f"Sum of dims²: {1*3 + 4*6 + 16*3 + 64}  (= {3+24+48+64})")
print()
print("The 729-dim tensor space decomposes under S₃³ into isotypic components.")
print("The 27-dim live subspace decomposes into:")
print("  27 = 1 + 2 + 2 + 2 + 4 + 4 + 4 + 8")
print("  (trivial: trace component, standard: off-diagonal patterns, etc.)")
print()
print("If the rank-19 optimizer systematically captures some irreps and misses others,")
print("the residual would live in specific irreps with ||R||² = dim of missing irreps = 8.")
print("The 8-dimensional irrep of S₃³ is the tensor product of all three standard irreps!")
print()
print("PREDICTION: The residual at ALS convergence lives in (or near) the")
print("sign⊗sign⊗sign or standard⊗standard⊗standard irrep of S₃³.")
print("This 8-dim subspace would give ||R||² = 8 exactly.")
print()

# Verify: compute the S3^3 group average to check isotypic decomposition
# The trivial isotypic projection of a function f is (1/|G|) sum_{g in G} f(g.x)
# For S3^3 acting on the 27 live entries:
# The action permutes (r,s,u) → (π(r), σ(s), τ(u)) for πσ,τ ∈ S₃

from itertools import permutations

S3 = list(permutations(range(3)))

# Project residual onto trivial isotypic component of S3^3
# acting on the 27-dim live subspace
live_res_array = np.zeros(27)
live_index_map = {}
idx = 0
for r in range(3):
    for s in range(3):
        for u in range(3):
            c_idx = r * 3 + u
            a_idx = r * 3 + s
            b_idx = s * 3 + u
            live_res_array[idx] = best_05[1][c_idx, a_idx, b_idx]  # residual at live entry
            live_index_map[(r, s, u)] = idx
            idx += 1

print("=" * 70)
print("S₃³ ISOTYPIC DECOMPOSITION OF RESIDUAL (live subspace)")
print("=" * 70)

# Build the group action matrices on the 27-dim live space
def apply_perm(perm, coords):
    """Apply (pi, sigma, tau) to (r, s, u)."""
    pi, sigma, tau = perm
    r, s, u = coords
    return (pi[r], sigma[s], tau[u])

# Project onto trivial irrep: average over all g
trivial_proj = np.zeros(27)
for pi in S3:
    for sigma in S3:
        for tau in S3:
            perm = (pi, sigma, tau)
            for r in range(3):
                for s in range(3):
                    for u in range(3):
                        src = live_index_map[(r, s, u)]
                        dst_coords = apply_perm(perm, (r, s, u))
                        dst = live_index_map[dst_coords]
                        trivial_proj[dst] += live_res_array[src]

trivial_proj /= 216  # |G| = 216
trivial_energy = np.sum(trivial_proj**2)

# Project onto sign⊗sign⊗sign irrep
def sign_char(perm):
    """Sign of permutation."""
    # Count inversions
    n = len(perm)
    inv = 0
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inv += 1
    return (-1)**inv

signnn_proj = np.zeros(27)
for pi in S3:
    for sigma in S3:
        for tau in S3:
            chi = sign_char(pi) * sign_char(sigma) * sign_char(tau)
            for r in range(3):
                for s in range(3):
                    for u in range(3):
                        src = live_index_map[(r, s, u)]
                        dst_coords = apply_perm((pi, sigma, tau), (r, s, u))
                        dst = live_index_map[dst_coords]
                        signnn_proj[dst] += chi * live_res_array[src]

signnn_proj /= 216
signnn_energy = np.sum(signnn_proj**2)

print(f"Trivial (1⊗1⊗1) projection energy:    {trivial_energy:.8f}")
print(f"Sign³ (sgn⊗sgn⊗sgn) projection energy: {signnn_energy:.8f}")
print(f"Live residual total energy:              {np.sum(live_res_array**2):.8f}")
print(f"Dead residual energy:                    {np.sum(best_05[1]**2) - np.sum(live_res_array**2):.8f}")
print()

# For each irrep of S3^3, compute projection
# S3 character table:
# trivial: all = 1
# sign: (-1)^inversions
# standard (2-dim): need the actual representation matrices

# Actually, let's just use the character-theoretic projection
# For each irrep ρ, the projection onto the ρ-isotypic component is:
# P_ρ = (dim ρ / |G|) sum_g chi_ρ(g)^* ρ(g)
#
# For 1-d irreps (trivial, sign), the projection is straightforward
# For 2-d irreps, we need the full matrix representation

# Simpler: compute the invariant subspace dimensions by brute force
# Build the 27x27 permutation matrix for each group element,
# then compute the isotypic projections

print("Building full S₃³ representation on 27-dim live space...")
group_elements = []
for pi in S3:
    for sigma in S3:
        for tau in S3:
            mat = np.zeros((27, 27))
            for r in range(3):
                for s in range(3):
                    for u in range(3):
                        src = live_index_map[(r, s, u)]
                        dst = live_index_map[apply_perm((pi, sigma, tau), (r, s, u))]
                        mat[dst, src] = 1.0
            group_elements.append((pi, sigma, tau, mat))

# Character table of S3:
# Class:     {e}   {(12),(13),(23)}   {(123),(132)}
# Size:       1          3                 2
# trivial:    1          1                 1
# sign:       1         -1                 1
# standard:   2          0                -1

def s3_class(perm):
    """Return class index: 0=identity, 1=transposition, 2=3-cycle"""
    if perm == (0, 1, 2):
        return 0
    elif perm[0] == 0 or perm[1] == 1 or perm[2] == 2:
        # has a fixed point → transposition
        return 1
    else:
        return 2

s3_chars = {
    'triv': [1, 1, 1],
    'sign': [1, -1, 1],
    'std':  [2, 0, -1],
}

# For S3^3, the character of (pi, sigma, tau) under irrep rho_a ⊗ rho_b ⊗ rho_c is:
# chi(pi, sigma, tau) = chi_a(pi) * chi_b(sigma) * chi_c(tau)

s3_irreps = ['triv', 'sign', 'std']
s3_class_sizes = [1, 3, 2]

print("\nIsotypic decomposition of residual in 27-dim live space:")
print(f"{'Irrep (a⊗b⊗c)':>25} {'dim':>5} {'proj_energy':>14} {'frac':>8}")
print("-" * 60)

total_proj_energy = 0
for a in s3_irreps:
    for b in s3_irreps:
        for c in s3_irreps:
            dim_abc = s3_chars[a][0] * s3_chars[b][0] * s3_chars[c][0]
            
            # Project: P_rho = (dim/|G|) sum_g conj(chi(g)) * rho(g)
            # For real reps, conj(chi) = chi
            proj = np.zeros(27)
            for pi_idx, pi in enumerate(S3):
                for sigma_idx, sigma in enumerate(S3):
                    for tau_idx, tau in enumerate(S3):
                        chi_val = (s3_chars[a][s3_class(pi)] * 
                                   s3_chars[b][s3_class(sigma)] * 
                                   s3_chars[c][s3_class(tau)])
                        
                        # Find the permutation matrix
                        perm_vec = np.zeros(27)
                        for r in range(3):
                            for s in range(3):
                                for u in range(3):
                                    src = live_index_map[(r, s, u)]
                                    dst = live_index_map[apply_perm((pi, sigma, tau), (r, s, u))]
                                    perm_vec[dst] += chi_val * live_res_array[src]
                        proj += perm_vec
            
            proj *= dim_abc / 216.0
            energy = np.sum(proj**2) / dim_abc  # Normalize by multiplicity
            # Actually the isotypic projection energy 
            proj_energy = np.sum(proj**2)
            total_proj_energy += proj_energy
            
            label = f"{a}⊗{b}⊗{c}"
            if proj_energy > 1e-10:
                print(f"{label:>25} {dim_abc:5d} {proj_energy:14.8f} {proj_energy/np.sum(live_res_array**2)*100:7.2f}%")

print(f"{'TOTAL':>25} {'':>5} {total_proj_energy:14.8f}")
print(f"{'Live energy':>25} {'':>5} {np.sum(live_res_array**2):14.8f}")
