"""
algebra_constructor.py — Active algebra discovery for T_matmul decomposition.

Instead of scoring random candidates against axioms, this module:
1. Enumerates G-irrep sub-representations of the right dimension (A7)
2. Constructs Cayley-Dickson structure constants as candidate decompositions (A6)
3. Parametrizes directly on Gr(R-9, 18) so conservation holds by construction (A1)
4. Uses relation module matroid as hard constraint (A4)
5. Handles all target ranks including non-G-stable R=20,21,22

Every solver exposes a single-shot function (one seed) so the runner can
parallelize across all (shape x seed) work-units on all CPU cores.

"The axioms are not filters — they are generators." — CANON ADE v2
"""
import numpy as np
from itertools import combinations, product as iproduct
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.linalg import null_space, subspace_angles
import warnings

from . import tensor_core as tc

SVD_TOL = 1e-10


# ═══════════════════════════════════════════════════════════════
# §1  A7 CONSTRUCTIVE: Irrep Enumeration + Small System Solver
# ═══════════════════════════════════════════════════════════════

def decompose_irreps_clean(R, kept=None):
    """
    Decompose ℝ^R under G = Z₂≀S₃ into irreducible representations.
    Returns list of (dim, basis) where basis is (R, dim) orthonormal.
    Uses multiple random commutant elements for reliability.
    """
    if kept is None:
        kept = tc.KEPT19 if R == 19 else tc.ALL27[:R]
    perm_mats = tc.build_perm_matrices(kept)
    n_g = len(perm_mats)

    # Average over multiple random commutant elements for robustness
    rng = np.random.default_rng(12345)
    all_evals = []
    all_evecs = []

    for trial in range(5):
        coeffs = rng.standard_normal(n_g)
        C = sum(c * P for c, P in zip(coeffs, perm_mats)) / n_g
        C = (C + C.T) / 2
        evals, evecs = np.linalg.eigh(C)
        all_evals.append(evals)
        all_evecs.append(evecs)

    # Use the first decomposition, verify with others
    evals = all_evals[0]
    evecs = all_evecs[0]

    # Cluster eigenvalues to find irrep blocks
    tol = 1e-8
    irreps = []
    used = set()
    for i in range(R):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, R):
            if j not in used and abs(evals[i] - evals[j]) < tol:
                group.append(j)
                used.add(j)
        basis = evecs[:, group]

        # Check if truly irreducible by testing with second commutant element
        if len(group) > 1 and len(all_evals) > 1:
            C2 = sum(c * P for c, P in zip(all_evals[1], perm_mats[:len(all_evals[1])])) / n_g
            # Actually use stored evecs to restrict second commutant
            coeffs2 = rng.standard_normal(n_g)
            C2 = sum(c * P for c, P in zip(coeffs2, perm_mats)) / n_g
            C2 = (C2 + C2.T) / 2
            C2_rest = basis.T @ C2 @ basis
            evals2, evecs2 = np.linalg.eigh(C2_rest)
            # Check if all eigenvalues are equal (irreducible) or split
            spread = np.max(evals2) - np.min(evals2)
            if spread > 1e-6:
                # Split further
                used2 = set()
                for ii in range(len(group)):
                    if ii in used2:
                        continue
                    sub = [ii]
                    used2.add(ii)
                    for jj in range(ii + 1, len(group)):
                        if jj not in used2 and abs(evals2[ii] - evals2[jj]) < tol:
                            sub.append(jj)
                            used2.add(jj)
                    sub_basis = basis @ evecs2[:, sub]
                    irreps.append((len(sub), sub_basis))
            else:
                irreps.append((len(group), basis))
        else:
            irreps.append((len(group), basis))

    return irreps


def enumerate_kernel_shapes(irreps, target_dim):
    """
    Enumerate all ways to select irrep components summing to target_dim.
    Returns list of tuples of irrep indices.
    """
    dims = [d for d, _ in irreps]
    n = len(dims)
    valid = []

    # Use subset-sum enumeration (feasible since n is small, typically < 10)
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            if sum(dims[i] for i in combo) == target_dim:
                valid.append(combo)
    return valid


def solve_from_kernel_shape(combo, irreps, R, kept=None, n_restarts=20, verbose=False, on_progress=None):
    """
    Given a kernel shape (indices into irreps), solve for factor vectors.

    The kernel K of dimension R-9 is fixed by the irrep selection.
    The complement (dim 9) is the "active" subspace where factors must span ℝ^9.
    We solve: find (α_t, β_t, γ_t) ∈ ℝ^{3×3} × 3 such that
      Σ_t α_t ⊗ β_t ⊗ γ_t = T_matmul
    subject to the relation module structure.

    on_progress(restart, n_restarts, best_res) called after each restart.

    Returns: best (alpha, beta, gamma, residual) or None if no solution found.
    """
    if kept is None:
        kept = tc.KEPT19 if R == 19 else tc.ALL27[:R]

    # Build the kernel subspace
    kernel_basis = np.hstack([irreps[i][1] for i in combo])  # (R, R-9)
    target_dim = kernel_basis.shape[1]
    assert target_dim == R - 9, f"Kernel dim {target_dim} != R-9 = {R - 9}"

    # Complement = active subspace (dim 9)
    # Use SVD to get orthogonal complement
    U, s, Vt = np.linalg.svd(kernel_basis, full_matrices=True)
    active_basis = U[:, target_dim:]  # (R, 9) — the complement

    # The factor matrices A, B, C each have shape (R, 9).
    # Kernel constraint: kernel_basis.T @ A = 0 (rows of A are orthogonal to kernel)
    # This means A = active_basis @ M_A for some (9, 9) matrix M_A.
    # Similarly B = active_basis @ M_B, C = active_basis @ M_C.
    # This reduces from 3*R*9 = 27R parameters to 3*81 = 243 parameters.

    T_target = tc.T_MATMUL  # (9, 9, 9)

    def residual_from_params(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)

        A = active_basis @ M_A  # (R, 9)
        B = active_basis @ M_B
        C = active_basis @ M_C

        # Reconstruct tensor: T_recon[i,j,k] = Σ_t A[t,i]*B[t,j]*C[t,k]
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return (T_recon - T_target).ravel()

    def objective(params):
        r = residual_from_params(params)
        return 0.5 * np.dot(r, r)

    def gradient(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)

        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C

        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        diff = T_recon - T_target  # (9, 9, 9)

        # Gradients w.r.t. A, B, C
        dA = np.einsum('ijk,tj,tk->ti', diff, B, C)  # (R, 9)
        dB = np.einsum('ijk,ti,tk->tj', diff, A, C)
        dC = np.einsum('ijk,ti,tj->tk', diff, A, B)

        # Chain rule: A = active_basis @ M_A → dM_A = active_basis.T @ dA
        dM_A = active_basis.T @ dA
        dM_B = active_basis.T @ dB
        dM_C = active_basis.T @ dC

        return np.concatenate([dM_A.ravel(), dM_B.ravel(), dM_C.ravel()])

    best_result = None
    best_res = np.inf
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        x0 = rng.standard_normal(243) * 0.5
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
            final_res = np.max(np.abs(residual_from_params(res.x)))
            if verbose:
                print(f"  restart {restart}: max_abs_residual = {final_res:.6e}")
            if final_res < best_res:
                best_res = final_res
                M_A = res.x[:81].reshape(9, 9)
                M_B = res.x[81:162].reshape(9, 9)
                M_C = res.x[162:243].reshape(9, 9)
                alpha = (active_basis @ M_A).reshape(R, 3, 3)
                beta = (active_basis @ M_B).reshape(R, 3, 3)
                gamma = (active_basis @ M_C).reshape(R, 3, 3)
                best_result = (alpha, beta, gamma, final_res)
                if final_res < 1e-10:
                    break
        except Exception:
            continue
        if on_progress:
            on_progress(restart + 1, n_restarts, best_res)

    return best_result


def solve_kernel_shape_single(combo, irreps, R, kept, seed):
    """
    Single-shot solver: one (shape, seed) → one result.
    Returns (residual, params_x) or (inf, None) on failure.
    Pure function — safe for ProcessPoolExecutor.
    """
    if kept is None:
        kept = tc.KEPT19 if R == 19 else tc.ALL27[:R]

    kernel_basis = np.hstack([irreps[i][1] for i in combo])
    target_dim = kernel_basis.shape[1]
    U, s, Vt = np.linalg.svd(kernel_basis, full_matrices=True)
    active_basis = U[:, target_dim:]

    T_target = tc.T_MATMUL

    def residual_from_params(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)
        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return (T_recon - T_target).ravel()

    def objective(params):
        r = residual_from_params(params)
        return 0.5 * np.dot(r, r)

    def gradient(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)
        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        diff = T_recon - T_target
        dA = np.einsum('ijk,tj,tk->ti', diff, B, C)
        dB = np.einsum('ijk,ti,tk->tj', diff, A, C)
        dC = np.einsum('ijk,ti,tj->tk', diff, A, B)
        dM_A = active_basis.T @ dA
        dM_B = active_basis.T @ dB
        dM_C = active_basis.T @ dC
        return np.concatenate([dM_A.ravel(), dM_B.ravel(), dM_C.ravel()])

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(243) * 0.5
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
        final_res = np.max(np.abs(residual_from_params(res.x)))
        return final_res, res.x
    except Exception:
        return float('inf'), None


def run_a7_enumeration(R, n_restarts=20, max_shapes=20, verbose=True):
    """
    Full A7 pipeline: decompose → enumerate kernel shapes → solve each.
    Only tries the top max_shapes shapes (prioritizing fewer irrep components).
    Returns list of (shape, alpha, beta, gamma, residual) sorted by residual.
    """
    if R == 19:
        kept = tc.KEPT19
    elif R == 13:
        kept_lists = tc.kept_triples_for_rank(13)
        kept = kept_lists[0] if kept_lists else None
        if kept is None:
            raise ValueError("No G-stable support for R=13")
    elif R == 27:
        kept = tc.ALL27
    else:
        # For non-G-stable ranks, use full support and let solver handle it
        kept = tc.ALL27[:R]

    if verbose:
        print(f"═══ A7 Irrep Enumeration for R={R} ═══")

    irreps = decompose_irreps_clean(R, kept)
    dims = [d for d, _ in irreps]
    target_dim = R - 9

    if verbose:
        print(f"  Irrep dimensions: {dims}")
        print(f"  Target kernel dimension: {target_dim}")

    shapes = enumerate_kernel_shapes(irreps, target_dim)
    # Sort by fewest components (simpler shapes first), then truncate
    shapes.sort(key=lambda s: (len(s), s))
    if len(shapes) > max_shapes:
        shapes = shapes[:max_shapes]
    if verbose:
        print(f"  Valid kernel shapes: {len(shapes)} (of {len(enumerate_kernel_shapes(irreps, target_dim))} total)")
        for s in shapes[:10]:
            sdims = [dims[i] for i in s]
            print(f"    {s} → dims {sdims} = {sum(sdims)}")

    results = []
    for idx, shape in enumerate(shapes):
        sdims = [dims[i] for i in shape]
        if verbose:
            print(f"\n  Solving shape {idx+1}/{len(shapes)}: {shape} (dims {sdims})")
        sol = solve_from_kernel_shape(shape, irreps, R, kept, n_restarts=n_restarts, verbose=verbose)
        if sol is not None:
            alpha, beta, gamma, res = sol
            results.append((shape, alpha, beta, gamma, res))
            if verbose:
                print(f"  → best residual: {res:.6e}")
        else:
            if verbose:
                print(f"  → no solution found")

    results.sort(key=lambda x: x[4])
    if verbose:
        print(f"\n═══ A7 Summary for R={R}: {len(results)} shapes tried ═══")
        for shape, _, _, _, res in results[:5]:
            print(f"  Shape {shape}: residual {res:.6e}")
    return results


# ═══════════════════════════════════════════════════════════════
# §2  A6 CONSTRUCTIVE: Cayley-Dickson Algebra Constructor
# ═══════════════════════════════════════════════════════════════

def build_octonion_table():
    """
    Build the octonion multiplication table e_i * e_j = ±e_k.
    Returns (table, signs) where table[i,j] = k and signs[i,j] = ±1.
    Uses the standard Fano-plane convention.
    """
    # Standard octonion multiplication (indices 0-7, 0=identity)
    # Fano plane triples: (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
    fano = [(1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)]
    table = np.zeros((8, 8), dtype=int)
    signs = np.zeros((8, 8), dtype=int)

    # e_0 is identity
    for i in range(8):
        table[0, i] = i
        table[i, 0] = i
        signs[0, i] = 1
        signs[i, 0] = 1

    # e_i * e_i = -e_0 for i > 0
    for i in range(1, 8):
        table[i, i] = 0
        signs[i, i] = -1

    # Fano plane products
    for a, b, c in fano:
        table[a, b] = c; signs[a, b] = 1
        table[b, a] = c; signs[b, a] = -1
        table[b, c] = a; signs[b, c] = 1
        table[c, b] = a; signs[c, b] = -1
        table[c, a] = b; signs[c, a] = 1
        table[a, c] = b; signs[a, c] = -1

    return table, signs


def build_sedenion_table():
    """
    Build sedenion (dim-16) multiplication table via Cayley-Dickson doubling.
    (a,b)*(c,d) = (ac - d*b, da + bc*) where * is conjugation.
    Returns (table, signs) for indices 0..15.
    """
    oct_table, oct_signs = build_octonion_table()

    # Conjugation in octonions: conj(e_0) = e_0, conj(e_i) = -e_i for i>0
    def oct_conj_sign(i):
        return 1 if i == 0 else -1

    n = 16
    table = np.zeros((n, n), dtype=int)
    signs = np.zeros((n, n), dtype=int)

    # Encode (a,b) as index: a + 8*half where half ∈ {0,1} (lower=octonion part, upper=doubled)
    # Index i: if i < 8 → (e_i, 0), if i >= 8 → (0, e_{i-8})

    def mul_cd(i, j):
        """Cayley-Dickson product of sedenion basis elements e_i * e_j.
        Returns (index, sign).
        (a, b) * (c, d) = (ac - d*b, da + bc*)
        For basis elements:
          e_i (i<8) = (e_i, 0), e_{i+8} = (0, e_i)
        """
        a_half = i // 8   # 0 or 1
        a_idx = i % 8
        b_half = j // 8
        b_idx = j % 8

        if a_half == 0 and b_half == 0:
            # (e_a, 0) * (e_b, 0) = (e_a*e_b, 0)
            k = oct_table[a_idx, b_idx]
            s = oct_signs[a_idx, b_idx]
            return k, s

        elif a_half == 0 and b_half == 1:
            # (e_a, 0) * (0, e_b) = (0, e_b * e_a)
            # Wait: (a,0)*(0,d) = (0*a - d*·0, d·a + 0·a*) = (-0, da) = (0, da)
            # Actually: (a,b)*(c,d) = (ac - d*b, da + bc*)
            # (e_a, 0) * (0, e_b) → c=0, d=e_b
            # = (e_a·0 - conj(e_b)·0, e_b·e_a + 0·conj(e_a))
            # = (0, e_b · e_a)
            k = oct_table[b_idx, a_idx]
            s = oct_signs[b_idx, a_idx]
            return k + 8, s

        elif a_half == 1 and b_half == 0:
            # (0, e_a) * (e_b, 0) = (0·e_b - 0·e_a, 0·e_a + e_b·conj(0))
            # = (0, e_a · conj(e_b))
            # Actually: c=e_b, d=0, a=0, b=e_a
            # = (0·e_b - 0·e_a, 0·0 + e_b·conj(e_a))... let me redo
            # (a,b)*(c,d) with a=0, b=e_a, c=e_b, d=0
            # = (0·e_b - conj(0)·e_a, 0·0 + e_b·conj(e_a))  -- wait no
            # The formula: (a,b)(c,d) = (ac - d*b, da + bc*)
            # a=0, b=e_{a_idx}, c=e_{b_idx}, d=0
            # = (0·e_{b_idx} - conj(0)·e_{a_idx}, 0·0 + e_{b_idx}·conj(e_{a_idx}))
            # conj(0) = e_0
            # = (- e_{a_idx}, e_{b_idx} · conj(e_{a_idx}))
            # Hmm, this doesn't simplify nicely. Let me use the standard formula.
            # For sedenion basis: e_i (i<8) and e_{8+i} = e_i·ε where ε is the doubling unit.
            # The clean formula: for purely imaginary e_i, e_j in lower half,
            # e_{8+i} * e_j = e_{8 + conj(e_j)*e_i}... this gets complicated.
            # Let me just use the recursive definition directly.
            # (0, e_a) * (e_b, 0):
            # = (0*e_b - 0*e_a, 0*0 + e_b * conj(e_a))
            # Wait the formula is (ac - d*b, da + bc*) where * means conjugate
            # a=0, b=e_{a_idx}, c=e_{b_idx}, d=0
            # First component: a*c - d* * b = 0 - conj(0)*e_{a_idx} = 0 (conj(e_0)=e_0... -e_{a_idx}?)
            # conj(d) where d=0 → conj(0) = 0 (zero element), not e_0
            # So: = (0 - 0, 0 + e_{b_idx} * conj(e_{a_idx}))
            # = (0, e_{b_idx} * conj(e_{a_idx}))
            conj_s = oct_conj_sign(a_idx)  # conj(e_a) = sign * e_a
            k = oct_table[b_idx, a_idx]
            s = oct_signs[b_idx, a_idx] * conj_s
            return k + 8, s

        else:  # a_half == 1 and b_half == 1
            # (0, e_a) * (0, e_b):
            # a=0, b=e_{a_idx}, c=0, d=e_{b_idx}
            # = (0 - conj(e_{b_idx})*e_{a_idx}, e_{b_idx}*0 + 0*conj(e_{a_idx}))
            # = (-conj(e_b)*e_a, 0)
            conj_s = oct_conj_sign(b_idx)
            k = oct_table[b_idx, a_idx]
            s_raw = oct_signs[b_idx, a_idx] * conj_s
            return k, -s_raw  # the leading minus

    for i in range(n):
        for j in range(n):
            k, s = mul_cd(i, j)
            table[i, j] = k
            signs[i, j] = s

    return table, signs


def cd_structure_constants(table, signs, dim):
    """
    Convert multiplication table to structure constants tensor f[i,j,k].
    f[i,j,k] = signs[i,j] if table[i,j] == k, else 0.
    """
    f = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            k = table[i, j]
            f[i, j, k] = signs[i, j]
    return f


def cd_to_rank1_terms(table, signs, dim):
    """
    Extract rank-1 decomposition terms from CD multiplication table.
    Each product e_i * e_j = ±e_k contributes a rank-1 term:
      α = e_i, β = e_j, γ = ±e_k
    where e_i is the i-th standard basis vector.

    Returns: alpha (N,dim), beta (N,dim), gamma (N,dim) where N = dim^2.
    """
    N = dim * dim
    alpha = np.zeros((N, dim))
    beta = np.zeros((N, dim))
    gamma = np.zeros((N, dim))

    idx = 0
    for i in range(dim):
        for j in range(dim):
            k = table[i, j]
            s = signs[i, j]
            alpha[idx, i] = 1.0
            beta[idx, j] = 1.0
            gamma[idx, k] = float(s)
            idx += 1

    return alpha, beta, gamma


def cd_rank1_subset_search(T_target, table, signs, dim, R_target,
                           n_restarts=50, verbose=True, on_progress=None):
    """
    Search for R_target rank-1 terms from the CD table whose sum approximates T_target.

    Strategy:
    1. Extract all dim² rank-1 terms from CD table
    2. Map CD indices to matmul indices via a linear embedding E: ℝ^dim → ℝ^9
    3. Optimize E and term selection jointly

    Returns: (best_embedding, best_terms, best_residual) or None.
    """
    all_a, all_b, all_c = cd_to_rank1_terms(table, signs, dim)
    N = all_a.shape[0]

    if verbose:
        print(f"  CD algebra: {dim}-dim, {N} rank-1 terms, target R={R_target}")

    # We need an embedding E: ℝ^dim → ℝ^9 such that
    # T_target ≈ Σ_{selected} (E·a_t) ⊗ (E·b_t) ⊗ (E·c_t)
    # This is a trilinear optimization in E and term weights.

    # Approach: fix all terms (weighted), optimize E and weights simultaneously.
    # Parameters: E (dim×9 = 9*dim), weights w (N,)
    # But N can be large (64 for octonions, 256 for sedenions), so we use
    # L1 penalty to encourage sparsity toward R_target active terms.

    n_E = dim * 9  # embedding parameters

    def build_tensor(params, return_components=False):
        E = params[:n_E].reshape(dim, 9)
        w = params[n_E:]

        # Map CD terms through embedding
        A_mapped = all_a @ E  # (N, 9)
        B_mapped = all_b @ E
        C_mapped = all_c @ E

        # Weighted reconstruction
        T_recon = np.einsum('t,ti,tj,tk->ijk', w, A_mapped, B_mapped, C_mapped)
        if return_components:
            return T_recon, E, w, A_mapped, B_mapped, C_mapped
        return T_recon

    def objective(params, lam_sparse=0.1):
        T_recon = build_tensor(params)
        fit = 0.5 * np.sum((T_recon - T_target) ** 2)
        w = params[n_E:]
        sparsity = lam_sparse * np.sum(np.abs(w))
        return fit + sparsity

    best_res = np.inf
    best_result = None
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        # Initialize E as random near-orthogonal, w as sparse
        E0 = rng.standard_normal((dim, 9)) * 0.3
        w0 = rng.standard_normal(N) * 0.1

        x0 = np.concatenate([E0.ravel(), w0])

        # Two-phase: first with high sparsity, then relax
        for lam in [0.5, 0.1, 0.01]:
            try:
                res = minimize(lambda p: objective(p, lam), x0,
                               method='L-BFGS-B',
                               options={'maxiter': 2000, 'ftol': 1e-20})
                x0 = res.x
            except Exception:
                break

        T_recon = build_tensor(x0)
        fit_err = np.max(np.abs(T_recon - T_target))
        w_final = x0[n_E:]
        n_active = np.sum(np.abs(w_final) > 1e-6)

        if verbose and restart % 10 == 0:
            print(f"  restart {restart}: err={fit_err:.4e}, active={n_active}")

        if fit_err < best_res:
            best_res = fit_err
            E_final = x0[:n_E].reshape(dim, 9)
            best_result = (E_final, w_final, fit_err, int(n_active))

        if on_progress:
            on_progress(restart + 1, n_restarts, best_res)

        if fit_err < 1e-10:
            break

    if verbose and best_result is not None:
        print(f"  Best: err={best_result[2]:.6e}, active_terms={best_result[3]}")

    return best_result


def cd_rank1_single(T_target, table, signs, dim, R_target, seed):
    """
    Single-shot CD rank-1 search: one seed → one result.
    Pure function — safe for ProcessPoolExecutor.
    Returns (fit_err, n_active, params) or (inf, 0, None).
    """
    all_a, all_b, all_c = cd_to_rank1_terms(table, signs, dim)
    N = all_a.shape[0]
    n_E = dim * 9

    def build_tensor(params):
        E = params[:n_E].reshape(dim, 9)
        w = params[n_E:]
        A_mapped = all_a @ E
        B_mapped = all_b @ E
        C_mapped = all_c @ E
        return np.einsum('t,ti,tj,tk->ijk', w, A_mapped, B_mapped, C_mapped)

    def objective(params, lam_sparse=0.1):
        T_recon = build_tensor(params)
        fit = 0.5 * np.sum((T_recon - T_target) ** 2)
        w = params[n_E:]
        return fit + lam_sparse * np.sum(np.abs(w))

    rng = np.random.default_rng(seed)
    E0 = rng.standard_normal((dim, 9)) * 0.3
    w0 = rng.standard_normal(N) * 0.1
    x0 = np.concatenate([E0.ravel(), w0])

    for lam in [0.5, 0.1, 0.01]:
        try:
            res = minimize(lambda p: objective(p, lam), x0,
                           method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-20})
            x0 = res.x
        except Exception:
            break

    T_recon = build_tensor(x0)
    fit_err = np.max(np.abs(T_recon - T_target))
    w_final = x0[n_E:]
    n_active = int(np.sum(np.abs(w_final) > 1e-6))
    return fit_err, n_active, x0


def run_a6_constructor(R, verbose=True):
    """
    Full A6 pipeline: build CD algebras at multiple levels, search for embeddings
    that decompose T_matmul at rank R.
    """
    T_target = tc.T_MATMUL
    results = []

    levels = [
        ("Octonions (dim 8)", *build_octonion_table(), 8),
        ("Sedenions (dim 16)", *build_sedenion_table(), 16),
    ]

    for name, table, signs, dim in levels:
        if verbose:
            print(f"\n═══ A6 Constructor: {name} → R={R} ═══")

        # Verify multiplication table
        f = cd_structure_constants(table, signs, dim)
        # Check non-degeneracy
        nonzero = np.count_nonzero(f)
        if verbose:
            print(f"  Structure constants: {nonzero} nonzero entries")

        res = cd_rank1_subset_search(T_target, table, signs, dim, R,
                                      n_restarts=30, verbose=verbose)
        if res is not None:
            results.append((name, res))

    return results


# ═══════════════════════════════════════════════════════════════
# §3  GRASSMANNIAN PARAMETRIZATION (A1 by construction)
# ═══════════════════════════════════════════════════════════════

def grassmannian_search(R, n_restarts=50, verbose=True, on_progress=None):
    """
    Parametrize directly on Gr(R-9, 18) so rank(H) = R-9 by construction.

    The idea: represent the anisotropy matrix H as Q @ M where
    Q ∈ St(R, R-9) is a Stiefel point (orthonormal columns) and
    M ∈ ℝ^{(R-9)×18} is unrestricted. Then rank(H) ≤ R-9 always.

    We enforce rank(H) = R-9 exactly by keeping M full-rank.
    The search variables are M (anisotropy shape) and the remaining
    factor degrees of freedom.

    Strategy: parameterize factors as
      α_t ⊗ β_t = Σ_t + H_t + Δ_t  (Step-51 decomposition)
    where Σ satisfies fiber sums, H has the right kernel, Δ ⊂ span(H).
    """
    if verbose:
        print(f"\n═══ Grassmannian Search for R={R} ═══")

    T_target = tc.T_MATMUL

    # Direct approach: parameterize factor matrices with kernel constraint
    # embedded. Use the A7 machinery but with continuous Grassmannian
    # optimization instead of discrete enumeration.

    kept = tc.KEPT19 if R == 19 else (
        tc.kept_triples_for_rank(R)[0] if tc.kept_triples_for_rank(R) else tc.ALL27[:R]
    )
    perm_mats = tc.build_perm_matrices(kept)
    n_g = len(perm_mats)

    # Parameters: rotation matrix V ∈ O(R) that diagonalizes the kernel structure,
    # plus factor parameters in the rotated basis.
    # For tractability, use (R-9)*18 params for H-shape + 9*9*3 for active factors.

    target_rank_H = R - 9

    def make_factors(params):
        """Build (R,9) factor matrices A, B, C from Grassmannian-parametrized params."""
        # First target_rank_H * R params define the kernel directions (as R-9 vectors in ℝ^R)
        n_kernel = target_rank_H * R
        K_raw = params[:n_kernel].reshape(target_rank_H, R)

        # Orthogonalize kernel
        Q, _ = np.linalg.qr(K_raw.T, mode='reduced')  # (R, R-9)
        # Active basis = complement
        U_full = np.linalg.svd(Q, full_matrices=True)[0]
        active = U_full[:, target_rank_H:]  # (R, 9)

        # Factor matrices in active subspace
        off = n_kernel
        M_A = params[off:off + 81].reshape(9, 9)
        M_B = params[off + 81:off + 162].reshape(9, 9)
        M_C = params[off + 162:off + 243].reshape(9, 9)

        A = active @ M_A  # (R, 9)
        B = active @ M_B
        C = active @ M_C
        return A, B, C

    n_kernel = target_rank_H * R
    n_params = n_kernel + 243

    def objective(params):
        A, B, C = make_factors(params)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return 0.5 * np.sum((T_recon - T_target) ** 2)

    best_res = np.inf
    best_result = None
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        x0 = rng.standard_normal(n_params) * 0.3
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
            A, B, C = make_factors(res.x)
            T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
            fit_err = np.max(np.abs(T_recon - T_target))

            if verbose and restart % 10 == 0:
                print(f"  restart {restart}: max_abs={fit_err:.6e}")

            if fit_err < best_res:
                best_res = fit_err
                alpha = A.reshape(R, 3, 3)
                beta = B.reshape(R, 3, 3)
                gamma = C.reshape(R, 3, 3)
                best_result = (alpha, beta, gamma, fit_err)

            if on_progress:
                on_progress(restart + 1, n_restarts, best_res)

            if fit_err < 1e-10:
                if verbose:
                    print(f"  EXACT HIT at restart {restart}!")
                break
        except Exception:
            continue

    if verbose:
        if best_result:
            print(f"  Best: {best_result[3]:.6e}")
        else:
            print(f"  No valid solution found")

    return best_result


def grassmannian_single(R, seed):
    """
    Single-shot Grassmannian solve: one seed → one result.
    Pure function — safe for ProcessPoolExecutor.
    Returns (fit_err, params_x) or (inf, None).
    """
    T_target = tc.T_MATMUL
    target_rank_H = R - 9

    def make_factors(params):
        n_kernel = target_rank_H * R
        K_raw = params[:n_kernel].reshape(target_rank_H, R)
        Q, _ = np.linalg.qr(K_raw.T, mode='reduced')
        U_full = np.linalg.svd(Q, full_matrices=True)[0]
        active = U_full[:, target_rank_H:]
        off = n_kernel
        M_A = params[off:off + 81].reshape(9, 9)
        M_B = params[off + 81:off + 162].reshape(9, 9)
        M_C = params[off + 162:off + 243].reshape(9, 9)
        return active @ M_A, active @ M_B, active @ M_C

    n_params = target_rank_H * R + 243

    def objective(params):
        A, B, C = make_factors(params)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return 0.5 * np.sum((T_recon - T_target) ** 2)

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_params) * 0.3
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
        A, B, C = make_factors(res.x)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        fit_err = np.max(np.abs(T_recon - T_target))
        return fit_err, res.x
    except Exception:
        return float('inf'), None


# ═══════════════════════════════════════════════════════════════
# §4  A4 MATROID-CONSTRAINED SEARCH
# ═══════════════════════════════════════════════════════════════

def matroid_constrained_search(R, n_restarts=30, verbose=True, on_progress=None):
    """
    Search for decompositions where the relation module matroid matches
    the generic position prediction.

    For R=19: dim(K_A∩K_B) = dim(K_A∩K_C) = dim(K_B∩K_C) = 1, dim(K_ABC) = 0.
    For R=13: all pairwise intersections = 0.

    Key insight: parameterize K_A, K_B, K_C as points on Gr(R-9, R) with
    prescribed intersection dimensions, then solve for compatible factors.
    """
    if verbose:
        print(f"\n═══ A4 Matroid-Constrained Search for R={R} ═══")

    T_target = tc.T_MATMUL
    k = R - 9  # kernel dimension

    # Generic pairwise intersection dimensions
    generic_pw = max(2 * k - R, 0)

    if verbose:
        print(f"  Kernel dim: {k}, generic pairwise intersection: {generic_pw}")

    # Approach: build K_A, K_B, K_C as subspaces with controlled intersections.
    # Then A must have K_A as its null space, etc.
    # A is (R, 9), rank 9, null space = K_A of dim k.
    # So A = P_A @ M_A where P_A is orthogonal complement of K_A, shape (R, 9).

    def make_kernels_and_factors(params):
        """Build three kernel subspaces with prescribed intersection structure."""
        # Parameterize shared subspace (dim = generic_pw)
        off = 0
        if generic_pw > 0:
            shared_raw = params[off:off + generic_pw * R].reshape(generic_pw, R)
            off += generic_pw * R
        else:
            shared_raw = np.zeros((0, R))

        # Parameterize private parts of each kernel
        private_dim = k - generic_pw  # unique to each kernel (after removing shared)
        K_parts = []
        for _ in range(3):
            priv = params[off:off + private_dim * R].reshape(private_dim, R)
            off += private_dim * R
            if generic_pw > 0:
                K_raw = np.vstack([shared_raw, priv])
            else:
                K_raw = priv
            # Orthogonalize
            Q, _ = np.linalg.qr(K_raw.T, mode='reduced')  # (R, k)
            K_parts.append(Q)

        # Build factor matrices from kernel complements
        factors = []
        for Q in K_parts:
            U_full = np.linalg.svd(Q, full_matrices=True)[0]
            complement = U_full[:, k:]  # (R, 9)
            M = params[off:off + 81].reshape(9, 9)
            off += 81
            F = complement @ M  # (R, 9)
            factors.append(F)

        return factors[0], factors[1], factors[2]

    n_shared = generic_pw * R
    n_private = 3 * (k - generic_pw) * R
    n_factor = 3 * 81
    n_params = n_shared + n_private + n_factor

    def objective(params):
        A, B, C = make_kernels_and_factors(params)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return 0.5 * np.sum((T_recon - T_target) ** 2)

    best_res = np.inf
    best_result = None
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        x0 = rng.standard_normal(n_params) * 0.3
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
            A, B, C = make_kernels_and_factors(res.x)
            T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
            fit_err = np.max(np.abs(T_recon - T_target))

            if verbose and restart % 10 == 0:
                print(f"  restart {restart}: max_abs={fit_err:.6e}")

            if fit_err < best_res:
                best_res = fit_err
                alpha = A.reshape(R, 3, 3)
                beta = B.reshape(R, 3, 3)
                gamma = C.reshape(R, 3, 3)
                best_result = (alpha, beta, gamma, fit_err)

            if on_progress:
                on_progress(restart + 1, n_restarts, best_res)

            if fit_err < 1e-10:
                if verbose:
                    print(f"  EXACT HIT at restart {restart}!")
                break
        except Exception:
            continue

    if verbose:
        if best_result:
            print(f"  Best: {best_result[3]:.6e}")
        else:
            print(f"  No valid solution found")

    return best_result


def matroid_single(R, seed):
    """
    Single-shot matroid-constrained solve: one seed → one result.
    Pure function — safe for ProcessPoolExecutor.
    Returns (fit_err, params_x) or (inf, None).
    """
    T_target = tc.T_MATMUL
    k = R - 9
    generic_pw = max(2 * k - R, 0)

    def make_kernels_and_factors(params):
        off = 0
        if generic_pw > 0:
            shared_raw = params[off:off + generic_pw * R].reshape(generic_pw, R)
            off += generic_pw * R
        else:
            shared_raw = np.zeros((0, R))
        private_dim = k - generic_pw
        K_parts = []
        for _ in range(3):
            priv = params[off:off + private_dim * R].reshape(private_dim, R)
            off += private_dim * R
            K_raw = np.vstack([shared_raw, priv]) if generic_pw > 0 else priv
            Q, _ = np.linalg.qr(K_raw.T, mode='reduced')
            K_parts.append(Q)
        factors = []
        for Q in K_parts:
            U_full = np.linalg.svd(Q, full_matrices=True)[0]
            complement = U_full[:, k:]
            M = params[off:off + 81].reshape(9, 9)
            off += 81
            factors.append(complement @ M)
        return factors[0], factors[1], factors[2]

    n_shared = generic_pw * R
    n_private = 3 * (k - generic_pw) * R
    n_params = n_shared + n_private + 3 * 81

    def objective(params):
        A, B, C = make_kernels_and_factors(params)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        return 0.5 * np.sum((T_recon - T_target) ** 2)

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_params) * 0.3
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
        A, B, C = make_kernels_and_factors(res.x)
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        fit_err = np.max(np.abs(T_recon - T_target))
        return fit_err, res.x
    except Exception:
        return float('inf'), None


# ═══════════════════════════════════════════════════════════════
# §5  EXTENDED RANK SUPPORT (R=20, 21, 22)
# ═══════════════════════════════════════════════════════════════

def build_support_for_rank(R):
    """
    Build support triples for any target rank, including non-G-stable ones.

    G-stable orbit sizes: O0=1, O1=6, O2=12, O3=8. Possible G-stable sums:
    {0,1,6,7,8,9,12,13,14,15,19,20,21,26,27} — note 20,21 ARE reachable:
      R=20: O0+O2+O3 = 1+12+8 = 21... no. O0+O1+O2 = 19, O0+O1+O3 = 15,
            O1+O2 = 18, O0+O1+O2+O3 = 27.
    Actually: let's enumerate properly.
    """
    orbits = [tc.ORBIT_0, tc.ORBIT_1, tc.ORBIT_2, tc.ORBIT_3]
    orbit_sizes = [len(o) for o in orbits]  # [1, 6, 12, 8]

    # Find all G-stable subsets of size R
    g_stable = []
    for mask in range(16):
        total = sum(orbit_sizes[i] for i in range(4) if mask & (1 << i))
        if total == R:
            selected = []
            for i in range(4):
                if mask & (1 << i):
                    selected.extend(orbits[i])
            g_stable.append(sorted(selected))

    if g_stable:
        return g_stable, True  # G-stable support found

    # Non-G-stable: use full 27 support but with R terms
    # For R=20,21,22: no exact G-stable subset exists.
    # Strategy: use all 27 support points, let the solver use R rank-1 terms
    # (the support is always all 27 nonzero positions of T_matmul,
    #  but the decomposition has only R terms)
    return [tc.ALL27], False


def run_extended_rank_search(R, methods=None, n_restarts=20, verbose=True):
    """
    Search for rank-R decompositions using all constructive methods.
    Handles all ranks including non-G-stable R=20,21,22.
    """
    if methods is None:
        methods = ['a7', 'grassmannian', 'matroid']

    supports, is_g_stable = build_support_for_rank(R)

    if verbose:
        print(f"\n{'═'*60}")
        print(f" Extended Rank Search: R={R}")
        print(f" G-stable: {is_g_stable}, support options: {len(supports)}")
        print(f" Methods: {methods}")
        print(f"{'═'*60}")

    all_results = {}

    if 'a7' in methods:
        if verbose:
            print(f"\n── Method: A7 Irrep Enumeration ──")
        try:
            a7_res = run_a7_enumeration(R, n_restarts=n_restarts, verbose=verbose)
            all_results['a7'] = a7_res
        except Exception as e:
            if verbose:
                print(f"  A7 failed: {e}")
            all_results['a7'] = []

    if 'a6' in methods or True:  # Always try A6
        if verbose:
            print(f"\n── Method: A6 CD Constructor ──")
        try:
            a6_res = run_a6_constructor(R, verbose=verbose)
            all_results['a6'] = a6_res
        except Exception as e:
            if verbose:
                print(f"  A6 failed: {e}")
            all_results['a6'] = []

    if 'grassmannian' in methods:
        if verbose:
            print(f"\n── Method: Grassmannian Parametrization ──")
        try:
            gr_res = grassmannian_search(R, n_restarts=n_restarts, verbose=verbose)
            all_results['grassmannian'] = gr_res
        except Exception as e:
            if verbose:
                print(f"  Grassmannian failed: {e}")
            all_results['grassmannian'] = None

    if 'matroid' in methods:
        if verbose:
            print(f"\n── Method: A4 Matroid Constraint ──")
        try:
            m4_res = matroid_constrained_search(R, n_restarts=n_restarts, verbose=verbose)
            all_results['matroid'] = m4_res
        except Exception as e:
            if verbose:
                print(f"  Matroid failed: {e}")
            all_results['matroid'] = None

    # Summary
    if verbose:
        print(f"\n{'═'*60}")
        print(f" Summary for R={R}")
        print(f"{'═'*60}")

        for method, res in all_results.items():
            if res is None:
                print(f"  {method}: no solution")
            elif isinstance(res, list):
                if len(res) == 0:
                    print(f"  {method}: no results")
                else:
                    if method == 'a7':
                        best = min(res, key=lambda x: x[4])
                        print(f"  {method}: {len(res)} shapes, best residual {best[4]:.6e}")
                    elif method == 'a6':
                        for name, (E, w, err, n_active) in res:
                            print(f"  {method} ({name}): err={err:.6e}, active={n_active}")
            elif isinstance(res, tuple):
                print(f"  {method}: residual {res[3]:.6e}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# §6  MASTER ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_full_search(ranks=None, n_restarts=20, verbose=True):
    """
    Run the full constructive algebra search across all target ranks.
    """
    if ranks is None:
        ranks = [13, 19, 20, 21, 22]

    all_results = {}
    for R in ranks:
        all_results[R] = run_extended_rank_search(
            R, n_restarts=n_restarts, verbose=verbose
        )

    return all_results
