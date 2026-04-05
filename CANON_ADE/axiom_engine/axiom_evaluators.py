"""
axiom_evaluators.py — All axiom evaluators: classical + novel.

Each evaluator takes a candidate (alpha, beta, gamma or params) and returns
a dict of scores/diagnostics. All are pure functions, safe for multiprocessing.
"""
import numpy as np
from itertools import combinations
from typing import Optional

# Lazy imports to avoid circular deps at module level
_CORE = None
def _core():
    global _CORE
    if _CORE is None:
        from . import tensor_core as tc
        _CORE = tc
    return _CORE

SVD_TOL = 1e-10

def matrix_rank(A, tol=SVD_TOL):
    if A.size == 0:
        return 0
    sv = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    t = tol * sv[0] if len(sv) > 0 and sv[0] > 0 else tol
    return int(np.sum(sv > t))


# ═══════════════════════════════════════════════════════════════
# A1: CONSERVATION ENVELOPE — rank(H) = R - 9
# ═══════════════════════════════════════════════════════════════

def eval_A1_conservation(Sigma, H, Delta, R):
    """Check conservation law: R + η_nullity = 27 ↔ rank(H) = R - 9."""
    rk_H = matrix_rank(H)
    target = R - 9
    eta_nullity = 18 - rk_H
    conservation = R + eta_nullity
    return {
        'A1_rank_H': rk_H,
        'A1_target': target,
        'A1_pass': int(rk_H == target),
        'A1_deficit': target - rk_H,
        'A1_conservation': conservation,
        'A1_conservation_ok': int(conservation == 27),
    }


# ═══════════════════════════════════════════════════════════════
# A2: PARITY BLOCK PARTITION — orbit structure validation
# ═══════════════════════════════════════════════════════════════

def eval_A2_parity(alpha, beta, gamma, R, kept=None):
    """Check parity/orbit structure of the decomposition."""
    tc = _core()
    if kept is None:
        kept = tc.KEPT19 if R == 19 else tc.ALL27

    # Fiber classification: forced (single-element) vs free (multi-element)
    fibers = {}
    for idx, t in enumerate(kept):
        r, s, u = t
        key = (s, u)
        fibers.setdefault(key, []).append(idx)
    
    n_forced = sum(1 for v in fibers.values() if len(v) == 1)
    n_free = sum(1 for v in fibers.values() if len(v) > 1)
    
    # Reconstruct tensor and check support match
    T_recon = np.zeros((9, 9, 9), dtype=np.float64)
    for k in range(R):
        a_flat = alpha[k].ravel()
        b_flat = beta[k].ravel()
        c_flat = gamma[k].ravel()
        T_recon += np.einsum('i,j,k', a_flat, b_flat, c_flat)
    
    fitness = np.max(np.abs(T_recon - tc.T_MATMUL))
    
    return {
        'A2_n_fibers_forced': n_forced,
        'A2_n_fibers_free': n_free,
        'A2_fitness': fitness,
        'A2_pass': int(fitness < 1e-6),
    }


# ═══════════════════════════════════════════════════════════════
# A3: FIBER MODE DECOMPOSITION — 2D lens
# ═══════════════════════════════════════════════════════════════

def eval_A3_fiber(Sigma, R, kept=None):
    """
    Fiber sum analysis: Γ(s,u) = 3 for all 9 (s,u) pairs.
    Sigma[k, r*3+u] = sum_s alpha[k,r,s]*beta[k,s,u]
    """
    tc = _core()
    
    # For each fiber (s,u), sum over contributing terms
    # Sigma already encodes fiber sums per term: Sigma[k, idx] where idx = r*3+u
    # The total fiber sum for output position (r,u) is sum_k gamma[k] * Sigma[k, r*3+u]
    # But without gamma solved, we check rank properties instead.
    
    rank_sigma = matrix_rank(Sigma)
    
    # Fiber sum achievability: can we find Gamma s.t. Gamma·Sigma = 3*I_9?
    # Requires: Sigma's 9 columns are lin. independent (rank_sigma >= 9)
    # If rank_sigma < 9, some output positions are in the null space.
    sigma_svs = np.linalg.svd(Sigma, compute_uv=False)
    
    return {
        'A3_rank_sigma': rank_sigma,
        'A3_sigma_cond': sigma_svs[0] / sigma_svs[-1] if sigma_svs[-1] > 1e-15 else np.inf,
        'A3_achievable': int(rank_sigma >= 9),
        'A3_sigma_svs': sigma_svs[:min(12, len(sigma_svs))].tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# A4: RELATION MODULE SPECTRUM — kernel intersections
# ═══════════════════════════════════════════════════════════════

def eval_A4_relation_module(alpha, beta, gamma, R):
    """
    Compute the relation module: K_A, K_B, K_C and their intersections.
    K_X = {λ ∈ ℝ^R : Σ λ_t X_t = 0} where X_t are the flattened factor vectors.
    """
    # Flatten each factor family  
    A_mat = alpha.reshape(R, 9)   # each a_t is 3×3 → 9-vec
    B_mat = beta.reshape(R, 9)
    C_mat = gamma.reshape(R, 9)
    
    def null_space(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=True)
        rank = np.sum(s > SVD_TOL * s[0]) if len(s) > 0 and s[0] > 0 else 0
        return Vt[rank:].T  # columns are null vectors
    
    K_A = null_space(A_mat.T)  # A_mat.T is 9×R, null space has dim R-rank(A)
    K_B = null_space(B_mat.T)
    K_C = null_space(C_mat.T)
    
    dim_KA = K_A.shape[1] if K_A.ndim == 2 else 0
    dim_KB = K_B.shape[1] if K_B.ndim == 2 else 0
    dim_KC = K_C.shape[1] if K_C.ndim == 2 else 0
    
    # Pairwise intersections via stacking null bases
    def intersection_dim(N1, N2):
        if N1.shape[1] == 0 or N2.shape[1] == 0:
            return 0
        # λ ∈ span(N1) ∩ span(N2) ⟺ N1·x = N2·y for some x,y
        # Equivalently: [N1 | -N2] · [x;y] = 0 has solutions
        # dim(intersection) = dim(N1) + dim(N2) - rank([N1|N2])
        combined = np.hstack([N1, N2])
        return dim_KA + dim_KB - matrix_rank(combined) if N1 is K_A and N2 is K_B else \
               N1.shape[1] + N2.shape[1] - matrix_rank(combined)

    dim_AB = K_A.shape[1] + K_B.shape[1] - matrix_rank(np.hstack([K_A, K_B])) if K_A.shape[1] > 0 and K_B.shape[1] > 0 else 0
    dim_AC = K_A.shape[1] + K_C.shape[1] - matrix_rank(np.hstack([K_A, K_C])) if K_A.shape[1] > 0 and K_C.shape[1] > 0 else 0
    dim_BC = K_B.shape[1] + K_C.shape[1] - matrix_rank(np.hstack([K_B, K_C])) if K_B.shape[1] > 0 and K_C.shape[1] > 0 else 0
    
    # Triple intersection
    if K_A.shape[1] > 0 and K_B.shape[1] > 0 and K_C.shape[1] > 0:
        combined_all = np.hstack([K_A, K_B, K_C])
        # dim(K_A ∩ K_B ∩ K_C) via inclusion-exclusion on subspace arrangement
        # More reliable: find basis of K_A ∩ K_B first, then intersect with K_C
        if dim_AB > 0:
            # Get basis of K_A ∩ K_B
            M = np.hstack([K_A, -K_B])
            U, s, Vt = np.linalg.svd(M, full_matrices=True)
            null_rank = np.sum(s > SVD_TOL * s[0])
            null_vecs = Vt[null_rank:]  # each row is [x; y] with K_A·x = K_B·y
            if null_vecs.shape[0] > 0:
                # The intersection vectors are K_A @ x for each null vector's x-part
                x_parts = null_vecs[:, :K_A.shape[1]]
                AB_basis = (K_A @ x_parts.T)  # columns are the intersection vectors
                # Now intersect with K_C
                combined_abc = np.hstack([AB_basis, K_C])
                dim_ABC = AB_basis.shape[1] + K_C.shape[1] - matrix_rank(combined_abc)
            else:
                dim_ABC = 0
        else:
            dim_ABC = 0
    else:
        dim_ABC = 0
    
    # Expected dimensions for generic position
    expected_AB = max(dim_KA + dim_KB - R, 0)
    expected_AC = max(dim_KA + dim_KC - R, 0)
    expected_BC = max(dim_KB + dim_KC - R, 0)
    
    return {
        'A4_dim_KA': dim_KA, 'A4_dim_KB': dim_KB, 'A4_dim_KC': dim_KC,
        'A4_dim_AB': dim_AB, 'A4_dim_AC': dim_AC, 'A4_dim_BC': dim_BC,
        'A4_dim_ABC': dim_ABC,
        'A4_expected_AB': expected_AB, 'A4_expected_AC': expected_AC,
        'A4_expected_BC': expected_BC,
        'A4_generic_position': int(dim_AB == expected_AB and dim_AC == expected_AC and dim_BC == expected_BC),
        'A4_minimal_rank': int(dim_ABC == 0),  # no redundant terms
    }


# ═══════════════════════════════════════════════════════════════
# A5: COORDINATE LIBERATION — Gate 2/3 compatibility in adapted basis
# ═══════════════════════════════════════════════════════════════

def eval_A5_coord_liberation(Sigma, H, Delta, R):
    """
    Check Gate 2/3 in relation-module-adapted coordinates.
    The key question: is aug_gap > 0 achievable outside Step-51?
    """
    nuisance = np.hstack([H, Delta])
    SN = np.hstack([Sigma, nuisance])
    
    rank_N = matrix_rank(nuisance)
    rank_SN = matrix_rank(SN)
    aug_gap = rank_SN - rank_N
    delta_leak = rank_N - matrix_rank(H)
    
    # Gate 2: Delta ⊂ span(H)
    gate2 = int(delta_leak == 0)
    # Gate 3: aug_gap = 9
    gate3 = int(aug_gap == 9)
    
    # D0 containment check (Phase-9 theorem)
    # D0[k,u] = alpha[k,r,0]*beta[k,0,u] — we can check if Sigma ⊂ span(H)
    # by projecting Sigma onto H's null space
    if matrix_rank(H) > 0:
        U_H, s_H, Vt_H = np.linalg.svd(H, full_matrices=True)
        rk_H = np.sum(s_H > SVD_TOL * s_H[0])
        H_null = U_H[:, rk_H:]  # null space projector basis
        sigma_on_null = H_null.T @ Sigma
        sigma_null_norm = np.linalg.norm(sigma_on_null, 'fro')
    else:
        sigma_null_norm = np.linalg.norm(Sigma, 'fro')
    
    return {
        'A5_rank_nuisance': rank_N,
        'A5_rank_SN': rank_SN,
        'A5_aug_gap': aug_gap,
        'A5_delta_leak': delta_leak,
        'A5_gate2': gate2,
        'A5_gate3': gate3,
        'A5_sigma_null_norm': sigma_null_norm,
        'A5_gates_compatible': int(gate2 and gate3),
    }


# ═══════════════════════════════════════════════════════════════
# A6: CAYLEY-DICKSON PARITY BRIDGE
# ═══════════════════════════════════════════════════════════════

def eval_A6_cayley_dickson(alpha, beta, gamma, R):
    """
    Check Cayley-Dickson parity alignment.
    Map (r,s,u) → (r%2, s%2, u%2) and check if the factor structure
    matches sedenion block-type signatures.
    """
    tc = _core()
    
    # XOR parity of each term's triple
    parities = []
    kept = tc.KEPT19 if R == 19 else tc.ALL27[:R]
    for t in kept:
        xor = (t[0] % 2) ^ (t[1] % 2) ^ (t[2] % 2)
        parities.append(xor)
    
    n_odd = sum(parities)
    n_even = R - n_odd
    
    # Block type distribution
    block_types = {}
    for t in kept:
        bt = (t[0] % 2, t[1] % 2, t[2] % 2)
        block_types[bt] = block_types.get(bt, 0) + 1
    
    # Violating types (odd XOR): 001, 010, 100, 111
    violating = sum(v for k, v in block_types.items() if (k[0]^k[1]^k[2]) == 1)
    non_violating = R - violating
    
    # Factor alignment: check if factors within same parity class are more aligned
    if R >= 2:
        A_flat = alpha.reshape(R, 9)
        gram_A = A_flat @ A_flat.T
        parity_arr = np.array(parities)
        same_parity = np.outer(parity_arr, parity_arr) + np.outer(1-parity_arr, 1-parity_arr)
        diff_parity = 1 - same_parity
        np.fill_diagonal(same_parity, 0)
        np.fill_diagonal(diff_parity, 0)
        
        gram_abs = np.abs(gram_A)
        same_coherence = gram_abs[same_parity > 0].mean() if same_parity.sum() > 0 else 0
        diff_coherence = gram_abs[diff_parity > 0].mean() if diff_parity.sum() > 0 else 0
    else:
        same_coherence = diff_coherence = 0
    
    # Convert tuple keys to strings for JSON serialization
    block_types_str = {str(k): v for k, v in block_types.items()}
    
    return {
        'A6_n_odd_parity': n_odd,
        'A6_n_even_parity': n_even,
        'A6_n_violating': violating,
        'A6_n_non_violating': non_violating,
        'A6_block_types': block_types_str,
        'A6_same_coherence': same_coherence,
        'A6_diff_coherence': diff_coherence,
        'A6_parity_separation': same_coherence - diff_coherence,
    }


# ═══════════════════════════════════════════════════════════════
# A6b: CD TOWER FIBER PARITY — Tower Connection constraints
# ═══════════════════════════════════════════════════════════════

def eval_A6b_cd_fiber_parity(alpha, beta, gamma, R):
    """
    Check Cayley-Dickson tower fiber constraints from TOWER_CONNECTION.md.
    
    The Fourier analysis shows r drops out, leaving a 2D problem in (s,u) ∈ Z₃².
    For any rank, the kept triples form fibers over (s,u). The constraint is:
      Γ(s,u) = Σ_r γ_eff(r,s,u) = 3  for all (s,u) ∈ Z₃² that have kept triples.
    
    For R=19: 4 forced fibers (size 1) + 5 free fibers (size 3).
    For R=27: all 9 fibers have 3 elements.
    For R=13: subset of fibers present.
    """
    tc = _core()
    
    # Determine kept triples for this rank
    if R == 19:
        kept = tc.KEPT19
    elif R == 27:
        kept = tc.ALL27
    elif R == 13:
        kept_lists = tc.kept_triples_for_rank(R)
        kept = kept_lists[0] if kept_lists else tc.ALL27[:R]
    else:
        kept_lists = tc.kept_triples_for_rank(R)
        kept = kept_lists[0] if kept_lists else tc.ALL27[:R]
    
    # Ensure we have the right number of terms
    if len(kept) != R:
        kept = kept[:R]
    
    # Compute effective γ per term: the actual scalar contribution to T_matmul
    # For term t at support (r,s,u): T_matmul[3r+s, 3s+u, 3r+u] gets contribution
    # = α_t[r,s] · β_t[s,u] · γ_t[r,u]... but factors are 3×3 matrices contributing
    # to full 9×9×9 tensor. The effective weight for the Fourier constraint is
    # the trace of the rank-1 outer product projected onto the support direction.
    # Simplest correct measure: sum of a_ij * b_jk * c_ik over the 9 support entries
    # that correspond to this term's (r,s,u) fiber.
    gamma_eff = np.zeros(R)
    for i, (r, s, u) in enumerate(kept):
        # The contribution of term i to T[3r+s, 3s+u, 3r+u]:
        # = (a_flat[3r+s]) * (b_flat[3s+u]) * (c_flat[3r+u])
        # But with full factor matrices, all 27 entries get contributions.
        # For fiber sum, what matters is the total weight: use Frobenius inner product
        # <a⊗b⊗c, e_{3r+s} ⊗ e_{3s+u} ⊗ e_{3r+u}> = a[3r+s]*b[3s+u]*c[3r+u]
        # across all (r',s',u') in the support. For a single term, the effective γ
        # is simply: a[r,s]*b[s,u]*c[r,u] (matching the original T_matmul entry).
        gamma_eff[i] = alpha[i].flatten()[3*r+s] * beta[i].flatten()[3*s+u] * gamma[i].flatten()[3*r+u]
    
    # Build fiber sums
    fibers = {}  # (s,u) -> list of indices
    for i, (r, s, u) in enumerate(kept):
        fibers.setdefault((s, u), []).append(i)
    
    fiber_sums = {}
    for (s, u), indices in fibers.items():
        fiber_sums[(s, u)] = sum(gamma_eff[i] for i in indices)
    
    # Evaluate constraints
    forced_keys = [(s, u) for (s, u) in fibers if s % 2 == 0 and u % 2 == 0]
    free_keys = [(s, u) for (s, u) in fibers if not (s % 2 == 0 and u % 2 == 0)]
    
    forced_errs = [abs(fiber_sums[k] - 3.0) for k in forced_keys]
    free_errs = [abs(fiber_sums[k] - 3.0) for k in free_keys]
    all_errs = [abs(v - 3.0) for v in fiber_sums.values()]
    
    mean_err = np.mean(all_errs)
    forced_err = np.mean(forced_errs) if forced_errs else 0.0
    free_err = np.mean(free_errs) if free_errs else 0.0
    max_dev = max(all_errs)
    
    # Pass if all fiber sums within tolerance of 3
    tol = 0.1
    passes = int(max_dev < tol)
    
    return {
        'A6b_fiber_sum_err': float(mean_err),
        'A6b_forced_err': float(forced_err),
        'A6b_free_err': float(free_err),
        'A6b_fiber_pass': passes,
        'A6b_max_fiber_dev': float(max_dev),
    }


# ═══════════════════════════════════════════════════════════════
# A7: GRASSMANNIAN QUANTIZATION — irrep enumeration
# ═══════════════════════════════════════════════════════════════

def eval_A7_grassmannian(H, R, irreps=None, perm_matrices=None):
    """
    Check which irrep combination the kernel of Gamma belongs to.
    H's column space should be the kernel of Gamma (dim R-9).
    """
    tc = _core()
    if irreps is None and perm_matrices is None:
        # No group action available for this rank — skip Grassmannian quantization
        return {
            'A7_rank_H': matrix_rank(H),
            'A7_target_dim': R - 9,
            'A7_n_irreps': 0,
            'A7_irrep_dims': [],
            'A7_irrep_overlaps': [],
            'A7_n_valid_shapes': 0,
            'A7_best_shape': None,
            'A7_best_score': -1,
            'A7_quantized': 0,
        }
    if irreps is None:
        irreps = tc.decompose_irreps(perm_matrices or tc.PERM19, R)
    
    target_dim = R - 9
    irrep_dims = [b.shape[1] for b in irreps]
    
    # Project H's column space onto each irrep
    U_H, s_H, _ = np.linalg.svd(H, full_matrices=True)
    rk_H = np.sum(s_H > SVD_TOL * (s_H[0] if s_H[0] > 0 else 1.0))
    H_basis = U_H[:, :rk_H]  # orthonormal basis for span(H)
    
    irrep_overlaps = []
    for basis in irreps:
        proj = basis @ basis.T
        overlap = H_basis.T @ proj @ H_basis
        sv = np.linalg.svd(overlap, compute_uv=False)
        n_in = int(np.sum(sv > 0.5))  # approximately integer
        irrep_overlaps.append(n_in)
    
    # Find best-matching irrep combination
    # The H subspace should decompose exactly into irrep blocks
    total_matched = sum(irrep_overlaps)
    residual = rk_H - total_matched
    
    # Enumerate all valid kernel shapes (subsets summing to target_dim)
    valid_shapes = []
    for r in range(1, len(irreps) + 1):
        for combo in combinations(range(len(irreps)), r):
            if sum(irrep_dims[i] for i in combo) == target_dim:
                valid_shapes.append(combo)
    
    # Score current H against each valid shape
    best_score = -1
    best_shape = None
    for combo in valid_shapes:
        K_basis = np.hstack([irreps[i] for i in combo])
        K_proj = K_basis @ K_basis.T
        overlap_mat = H_basis.T @ K_proj @ H_basis
        score = np.trace(overlap_mat) / max(rk_H, 1)
        if score > best_score:
            best_score = score
            best_shape = combo
    
    return {
        'A7_rank_H': rk_H,
        'A7_target_dim': target_dim,
        'A7_n_irreps': len(irreps),
        'A7_irrep_dims': irrep_dims,
        'A7_irrep_overlaps': irrep_overlaps,
        'A7_n_valid_shapes': len(valid_shapes),
        'A7_best_shape': best_shape,
        'A7_best_score': best_score,
        'A7_quantized': int(best_score > 0.99) if best_score >= 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# CLASSICAL AXIOMS — standard algebraic properties
# ═══════════════════════════════════════════════════════════════

def eval_classical_associativity(f):
    """Check associativity of an R×R×R structure tensor f."""
    R = f.shape[0]
    # (f*f)[i,j,k,l] via contraction: f[i,j,m]*f[m,k,l] vs f[i,m,l]*f[j,k,m]
    lhs = np.einsum('ijm,mkl->ijkl', f, f)
    rhs = np.einsum('iml,jkm->ijkl', f, f)
    err = np.max(np.abs(lhs - rhs))
    return {'CL_assoc_err': err, 'CL_assoc_pass': int(err < 1e-6)}

def eval_classical_commutativity(f):
    """Check f[i,j,k] = f[j,i,k] for all i,j,k."""
    err = np.max(np.abs(f - f.transpose(1, 0, 2)))
    return {'CL_comm_err': err, 'CL_comm_pass': int(err < 1e-6)}

def eval_classical_jacobi(f):
    """Check Jacobi identity: f[f[a,b],c] + cyclic = 0."""
    R = f.shape[0]
    # [a,[b,c]] + [b,[c,a]] + [c,[a,b]] = 0
    # With bracket [x,y]_k = f[x,y,k] - f[y,x,k]
    bracket = f - f.transpose(1, 0, 2)
    # Jacobi: bracket(a, bracket(b,c)) + cyclic
    term1 = np.einsum('ijk,klm->ijlm', bracket, bracket)
    term2 = np.einsum('jlk,kim->ijlm', bracket, bracket)
    term3 = np.einsum('lik,jkm->ijlm', bracket, bracket)
    jacobi = term1 + term2 + term3
    err = np.max(np.abs(jacobi))
    return {'CL_jacobi_err': err, 'CL_jacobi_pass': int(err < 1e-6)}

def eval_classical_nilpotency(f, max_power=5):
    """Check if algebra is nilpotent via left multiplication operators."""
    R = f.shape[0]
    results = {'CL_nilpotent': 0}
    for i in range(R):
        Li = f[i]  # R×R matrix: (Li)_{jk} = f[i,j,k]
        power = Li.copy()
        nilp = True
        for n in range(2, max_power + 1):
            power = power @ Li
            if np.max(np.abs(power)) > 1e-10:
                nilp = False
                break
        if nilp:
            results['CL_nilpotent'] = 1
            results['CL_nilpotent_idx'] = i
            break
    return results

def eval_classical_killing_form(f):
    """Compute Killing form K[i,j] = Tr(ad_i . ad_j) and its rank — vectorized."""
    R = f.shape[0]
    bracket = f - f.transpose(1, 0, 2)
    # ad_i is (R,R) matrix: bracket[i,:,:]
    # K[i,j] = Tr(bracket[i] @ bracket[j]) = sum_m,n bracket[i,m,n]*bracket[j,n,m]
    # Reshape: ad as (R, R*R) then K = ad @ ad.T with appropriate reshape
    ad_flat = bracket.reshape(R, R * R)  # ad_i flattened
    # Tr(A@B) = sum_{m,n} A[m,n]*B[n,m] = (A.ravel()) . (B.T.ravel())
    # bracket[i] is (R,R), bracket[j] is (R,R)
    # K[i,j] = sum_{m,n} bracket[i,m,n]*bracket[j,n,m]
    #        = vec(bracket[i]) . vec(bracket[j].T)
    bt = bracket.transpose(0, 2, 1).reshape(R, R * R)  # bracket[j].T flattened
    K = ad_flat @ bt.T
    
    rk = matrix_rank(K)
    return {
        'CL_killing_rank': rk,
        'CL_killing_trace': np.trace(K),
        'CL_semisimple': int(rk == R),
        'CL_killing_det': np.linalg.det(K) if R < 20 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# NOVEL AXIOMS — beyond classical
# ═══════════════════════════════════════════════════════════════

def eval_novel_spectral_signature(alpha, beta, gamma, R):
    """
    Spectral signature: eigenvalues of the Gram matrices A^T A, B^T B, C^T C.
    For a "good" decomposition, these should have specific spectral shapes.
    """
    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    C = gamma.reshape(R, 9)
    
    def spectral(M):
        G = M.T @ M
        evals = np.sort(np.linalg.eigvalsh(G))[::-1]
        return evals
    
    spec_A = spectral(A)
    spec_B = spectral(B)
    spec_C = spectral(C)
    
    # Spectral entropy (how spread out the singular values are)
    def entropy(evals):
        evals = evals[evals > 1e-15]
        p = evals / evals.sum()
        return -np.sum(p * np.log(p + 1e-30))
    
    # Spectral symmetry: how similar are the three spectra?
    spec_diff_AB = np.linalg.norm(spec_A - spec_B)
    spec_diff_AC = np.linalg.norm(spec_A - spec_C)
    spec_diff_BC = np.linalg.norm(spec_B - spec_C)
    
    return {
        'NV_spec_entropy_A': entropy(spec_A),
        'NV_spec_entropy_B': entropy(spec_B),
        'NV_spec_entropy_C': entropy(spec_C),
        'NV_spec_symmetry': spec_diff_AB + spec_diff_AC + spec_diff_BC,
        'NV_spec_A': spec_A.tolist(),
        'NV_spec_B': spec_B.tolist(),
        'NV_spec_C': spec_C.tolist(),
    }

def eval_novel_coupling_matrix(alpha, beta, gamma, R):
    """
    Coupling matrix Λ = C^T·A (R×R). Its rank is ≤ 9 always.
    The diagonal structure Λ[t1,t2]·δ(t2,t3) gives composition.
    """
    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    C = gamma.reshape(R, 9)
    
    Lambda_CA = C.T @ A  # should be ≤ 9 rank
    Lambda_CB = C.T @ B
    Lambda_AB = A.T @ B
    
    rk_CA = matrix_rank(Lambda_CA)
    rk_CB = matrix_rank(Lambda_CB)
    rk_AB = matrix_rank(Lambda_AB)
    
    # Diagonal dominance of Lambda
    diag_CA = np.abs(np.diag(Lambda_CA[:min(R,9), :min(R,9)])).sum() if min(R,9) > 0 else 0
    total_CA = np.abs(Lambda_CA).sum()
    
    return {
        'NV_coupling_rank_CA': rk_CA,
        'NV_coupling_rank_CB': rk_CB,
        'NV_coupling_rank_AB': rk_AB,
        'NV_coupling_diag_ratio': diag_CA / (total_CA + 1e-30),
        'NV_coupling_frobenius': np.linalg.norm(Lambda_CA, 'fro'),
    }

def eval_novel_frame_coherence(alpha, beta, gamma, R):
    """
    Frame coherence: max |<a_i, a_j>| / (|a_i||a_j|) for i≠j.
    Low coherence → frame vectors are well-spread → better conditioning.
    Welch bound: coherence ≥ √((R-9)/(9(R-1))) for R vectors in ℝ^9.
    """
    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    C = gamma.reshape(R, 9)
    
    def coherence(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        M_normed = M / norms
        G = M_normed @ M_normed.T
        np.fill_diagonal(G, 0)
        return np.max(np.abs(G))
    
    coh_A = coherence(A)
    coh_B = coherence(B)
    coh_C = coherence(C)
    
    # Welch bound
    if R > 9:
        welch = np.sqrt((R - 9) / (9 * (R - 1)))
    else:
        welch = 0.0
    
    return {
        'NV_coherence_A': coh_A,
        'NV_coherence_B': coh_B,
        'NV_coherence_C': coh_C,
        'NV_coherence_max': max(coh_A, coh_B, coh_C),
        'NV_welch_bound': welch,
        'NV_coherence_excess': max(coh_A, coh_B, coh_C) - welch,
    }

def eval_novel_tensor_network_rank(alpha, beta, gamma, R):
    """
    Tensor network diagnostics: flattening ranks of the reconstructed tensor.
    For T_matmul, all three mode-k flattenings have rank 9.
    """
    T_recon = np.zeros((9, 9, 9), dtype=np.float64)
    for k in range(R):
        a = alpha[k].ravel()
        b = beta[k].ravel()
        c = gamma[k].ravel()
        T_recon += np.outer(np.outer(a, b).ravel(), c).reshape(9, 9, 9)
    
    # Flattenings
    flat_0 = T_recon.reshape(9, 81)
    flat_1 = T_recon.transpose(1, 0, 2).reshape(9, 81)
    flat_2 = T_recon.transpose(2, 0, 1).reshape(9, 81)
    
    return {
        'NV_flat_rank_0': matrix_rank(flat_0),
        'NV_flat_rank_1': matrix_rank(flat_1),
        'NV_flat_rank_2': matrix_rank(flat_2),
        'NV_recon_err': np.max(np.abs(T_recon - _core().T_MATMUL)),
    }


# ═══════════════════════════════════════════════════════════════
# MASTER EVALUATOR — runs all axioms
# ═══════════════════════════════════════════════════════════════

def evaluate_all(alpha, beta, gamma, R, irreps=None, perm_matrices=None, 
                 skip_classical=False, skip_novel=False):
    """Run all axiom evaluators on a candidate decomposition."""
    results = {'R': R}
    
    # Step-51 coordinates
    Sigma, H, Delta = _core().compute_step51(alpha, beta)
    
    # Novel axioms A1-A7
    results.update(eval_A1_conservation(Sigma, H, Delta, R))
    results.update(eval_A2_parity(alpha, beta, gamma, R))
    results.update(eval_A3_fiber(Sigma, R))
    results.update(eval_A4_relation_module(alpha, beta, gamma, R))
    results.update(eval_A5_coord_liberation(Sigma, H, Delta, R))
    results.update(eval_A6_cayley_dickson(alpha, beta, gamma, R))
    results.update(eval_A6b_cd_fiber_parity(alpha, beta, gamma, R))
    results.update(eval_A7_grassmannian(H, R, irreps=irreps, perm_matrices=perm_matrices))
    
    if not skip_novel:
        results.update(eval_novel_spectral_signature(alpha, beta, gamma, R))
        results.update(eval_novel_coupling_matrix(alpha, beta, gamma, R))
        results.update(eval_novel_frame_coherence(alpha, beta, gamma, R))
        results.update(eval_novel_tensor_network_rank(alpha, beta, gamma, R))
    
    if not skip_classical:
        # Build structure tensor — vectorized over 27 sparse entries of T_matmul
        A_flat = alpha.reshape(R, 9)
        B_flat = beta.reshape(R, 9)
        C_flat = gamma.reshape(R, 9)
        # T_matmul[3r+s, 3s+u, 3r+u] = 1 for r,s,u in {0,1,2}
        # f[i,j,k] = sum_{r,s,u} A[i,3r+s]*B[j,3s+u]*C[k,3r+u]
        # Collect column indices for sparse entries
        a_cols = []; b_cols = []; c_cols = []
        for r in range(3):
            for s in range(3):
                for u in range(3):
                    a_cols.append(3*r+s)
                    b_cols.append(3*s+u)
                    c_cols.append(3*r+u)
        # Shape: (27, R) for each
        Av = A_flat[:, a_cols].T  # (27, R)
        Bv = B_flat[:, b_cols].T  # (27, R)
        Cv = C_flat[:, c_cols].T  # (27, R)
        # f[i,j,k] = sum_t Av[t,i]*Bv[t,j]*Cv[t,k]
        f = np.einsum('ti,tj,tk->ijk', Av, Bv, Cv)
        
        results.update(eval_classical_associativity(f))
        results.update(eval_classical_commutativity(f))
        results.update(eval_classical_jacobi(f))
        results.update(eval_classical_nilpotency(f))
        results.update(eval_classical_killing_form(f))
    
    return results
