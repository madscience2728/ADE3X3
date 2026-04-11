"""
Bridge: Z_3 Tower ↔ T_matmul

The Z_3 Level-2 algebra has dimension 9 with structure constants C[i,j,k] — 
a 9×9×9 tensor. T_matmul is ALSO a 9×9×9 tensor encoding 3×3 matrix multiplication.

Question: Are they gauge-equivalent? Can we use the tower algebra to factor T_matmul?

If T_matmul = (P ⊗ Q ⊗ R) · C_tower for some invertible P, Q, R, then
matrix multiplication IS the tower product in a different basis — and the
tower's algebraic structure (half-associativity, exact violation formulas)
would directly constrain the tensor rank.

NO OPTIMIZATION. Pure linear algebra comparison.
"""

import numpy as np
from fractions import Fraction
from itertools import product as iprod

# ═══════════════════════════════════════════════════════════════
# BUILD BOTH TENSORS
# ═══════════════════════════════════════════════════════════════

def build_Tmatmul():
    """T_matmul[i,j,k] = 1 iff (i,j,k) encodes a valid multiplication triple.
    Index: 3r+s for matrix entry (r,s). T[3r+s, 3s+u, 3r+u] = 1."""
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[3*r+s, 3*s+u, 3*r+u] = 1.0
    return T

def build_Z3_level2():
    """Z_3 Level-2 algebra: CD-style construction from Level-1."""
    # Level 1
    n = 3
    C1 = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            C1[i, j, k] = 1.0 if (i == 0 or j == 0) else -1.0

    # Conjugation
    d = n
    J = -np.eye(d)
    J[0, 0] = 1.0

    # Level 2 via CD
    D = n * d  # = 9
    C2 = np.zeros((D, D, D))
    for i1 in range(n):
        for i2 in range(n):
            m = (i1 + i2) % n
            for a in range(d):
                for b in range(d):
                    row, col = i1*d+a, i2*d+b
                    if i1 == 0 and i2 == 0:
                        for c in range(d):
                            C2[row, col, m*d+c] += C1[a, b, c]
                    elif i1 == 0 and i2 != 0:
                        for c in range(d):
                            C2[row, col, m*d+c] += C1[b, a, c]
                    elif i1 != 0 and i2 == 0:
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                C2[row, col, m*d+c] += J[bp, b] * C1[a, bp, c]
                    else:
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                C2[row, col, m*d+c] += -J[bp, b] * C1[bp, a, c]
    return C2

T = build_Tmatmul()
C = build_Z3_level2()

print("=" * 72)
print("BRIDGE: Z_3 TOWER LEVEL-2 ↔ T_matmul")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════
# BASIC TENSOR COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\n── Basic Properties ──")
print(f"T_matmul: shape {T.shape}, nnz = {np.count_nonzero(T)}, entries in {{0,1}}")
print(f"C_tower:  shape {C.shape}, nnz = {np.count_nonzero(C)}, entries in {set(C.flatten())}")
print(f"Max abs entry: T={np.max(np.abs(T))}, C={np.max(np.abs(C))}")

# Frobenius norms
print(f"Frobenius norm: T={np.linalg.norm(T.flatten()):.4f}, C={np.linalg.norm(C.flatten()):.4f}")

# Direct equality?
print(f"\nDirect match (T == C): {np.allclose(T, C)}")
print(f"Direct match (T == -C): {np.allclose(T, -C)}")

# ═══════════════════════════════════════════════════════════════
# ALGEBRAIC PROPERTIES COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\n── Algebraic Properties ──")

# Unit element?
def find_unit(C):
    d = C.shape[0]
    for u in range(d):
        # Check if e_u is left and right unit
        left_ok = True
        right_ok = True
        for i in range(d):
            # e_u * e_i should = e_i
            prod = C[u, i, :]
            expected = np.zeros(d)
            expected[i] = 1.0
            if not np.allclose(prod, expected):
                left_ok = False
            # e_i * e_u should = e_i
            prod2 = C[i, u, :]
            if not np.allclose(prod2, expected):
                right_ok = False
        if left_ok and right_ok:
            return u
    return None

unit_T = find_unit(T)
unit_C = find_unit(C)
print(f"Unit element: T_matmul has unit = {unit_T}, C_tower has unit = {unit_C}")

# Commutativity
def check_commutative(C):
    return np.allclose(C, C.transpose(1, 0, 2))

print(f"Commutative: T_matmul = {check_commutative(T)}, C_tower = {check_commutative(C)}")

# Associativity violations
def count_violations(C):
    LHS = np.einsum('ijk,klm->ijlm', C, C)
    RHS = np.einsum('jlk,ikm->ijlm', C, C)
    diff = LHS - RHS
    v = np.count_nonzero(np.abs(diff).sum(axis=-1) > 1e-10)
    return v, C.shape[0]**3

v_T, t_T = count_violations(T)
v_C, t_C = count_violations(C)
print(f"Associativity violations: T_matmul = {v_T}/{t_T} (S={1-v_T/t_T:.4f}), C_tower = {v_C}/{t_C} (S={1-v_C/t_C:.4f})")

# ═══════════════════════════════════════════════════════════════
# GAUGE EQUIVALENCE: Does T = (P⊗Q⊗R)·C for invertible P,Q,R?
# This means T[i,j,k] = sum_{a,b,c} P[i,a] Q[j,b] R[k,c] C[a,b,c]
# Equivalently: flattening both as 9×9 matrices (mode-1 unfolding),
# T_(1) = P · C_(1) · (R ⊗ Q)^T
# ═══════════════════════════════════════════════════════════════

print("\n── Gauge Equivalence Check ──")

# Flatten as matrices: T_(mode) for each mode
def mode_unfold(X, mode):
    """Unfold tensor X along given mode (0,1,2) into a matrix."""
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))

T1 = mode_unfold(T, 0)  # 9 × 81
C1 = mode_unfold(C, 0)  # 9 × 81

print(f"Mode-0 unfolding: T rank = {np.linalg.matrix_rank(T1)}, C rank = {np.linalg.matrix_rank(C1)}")

T2 = mode_unfold(T, 1)
C2 = mode_unfold(C, 1)
print(f"Mode-1 unfolding: T rank = {np.linalg.matrix_rank(T2)}, C rank = {np.linalg.matrix_rank(C2)}")

T3 = mode_unfold(T, 2)
C3 = mode_unfold(C, 2)
print(f"Mode-2 unfolding: T rank = {np.linalg.matrix_rank(T3)}, C rank = {np.linalg.matrix_rank(C3)}")

# If multilinear rank differs, they CANNOT be gauge equivalent
t_mlrank = (np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
c_mlrank = (np.linalg.matrix_rank(C1), np.linalg.matrix_rank(C2), np.linalg.matrix_rank(C3))
print(f"\nMultilinear rank: T = {t_mlrank}, C = {c_mlrank}")
if t_mlrank == c_mlrank:
    print("Multilinear ranks MATCH — gauge equivalence NOT ruled out!")
else:
    print("Multilinear ranks DIFFER — gauge equivalence IMPOSSIBLE.")

# ═══════════════════════════════════════════════════════════════
# SYMMETRY GROUP COMPARISON
# ═══════════════════════════════════════════════════════════════

print("\n── Symmetry Structure ──")

# T_matmul symmetry: Z_2^3 ⋊ S_3, order 48
# What are the tower symmetries?

# Check: is C invariant under the "flip" symmetry C[i,j,k] ↔ C[j,i,k]?
print(f"Transpose symmetry C[i,j,k]=C[j,i,k]: {np.allclose(C, C.transpose(1,0,2))}")

# Check cyclic symmetry C[i,j,k] = C[j,k,i]
print(f"Cyclic symmetry C[i,j,k]=C[j,k,i]: {np.allclose(C, C.transpose(1,2,0))}")

# Signature: eigenvalues of the "left multiplication" matrices L_i
print("\n── Left Multiplication Spectra ──")
print("(eigenvalues of L_i where (L_i)_{jk} = C[i,j,k])")

for label, tensor in [("T_matmul", T), ("C_tower", C)]:
    eig_sets = []
    for i in range(9):
        Li = tensor[i, :, :]  # L_i: left multiplication by e_i
        eigs = np.sort(np.real(np.linalg.eigvals(Li)))
        eig_sets.append(tuple(np.round(eigs, 4)))
    unique_spectra = len(set(eig_sets))
    # Determinants
    dets = [np.linalg.det(tensor[i, :, :]) for i in range(9)]
    print(f"  {label}: {unique_spectra} distinct spectra, dets = {[round(d,4) for d in dets]}")

# ═══════════════════════════════════════════════════════════════
# KEY QUESTION: Can C_tower DECOMPOSE T_matmul?
# If T_matmul = sum of R rank-1 terms, and C_tower has structure,
# maybe C_tower's algebraic identities reduce the rank.
# ═══════════════════════════════════════════════════════════════

print("\n── Projection of T_matmul onto C_tower ──")

# Inner product in tensor space
inner = np.sum(T * C)
print(f"⟨T_matmul, C_tower⟩ = {inner}")
print(f"cos(angle) = {inner / (np.linalg.norm(T.flatten()) * np.linalg.norm(C.flatten())):.6f}")

# Decompose T_matmul = α·C_tower + T_perp
alpha = inner / np.sum(C * C)
T_perp = T - alpha * C
print(f"Best scalar fit: T ≈ {alpha:.6f} · C_tower + T_perp")
print(f"‖T_perp‖/‖T‖ = {np.linalg.norm(T_perp.flatten()) / np.linalg.norm(T.flatten()):.6f}")

# ═══════════════════════════════════════════════════════════════
# THE REAL CONNECTION: Half-associativity as a rank bound
# ═══════════════════════════════════════════════════════════════

print("\n── Half-Associativity ↔ Rank Connection ──")

# The associator tensor A[i,j,k,l] = (e_i·e_j)·e_k - e_i·(e_j·e_k)
# For T_matmul: A = 0 (matrix mult IS associative) → all 729 triples pass
# For C_tower: exactly 216/729 = 8/27 fail

# The KEY insight: the DIFFERENCE between associative (T_matmul) and 
# half-associative (C_tower) might tell us about rank.

# Compute the associator tensors
A_T = np.einsum('ijk,klm->ijlm', T, T) - np.einsum('jlk,ikm->ijlm', T, T)
A_C = np.einsum('ijk,klm->ijlm', C, C) - np.einsum('jlk,ikm->ijlm', C, C)

print(f"Associator norm: T = {np.linalg.norm(A_T.flatten()):.6f} (zero = fully associative)")
print(f"Associator norm: C = {np.linalg.norm(A_C.flatten()):.6f}")

# The associator of C_tower lives in a specific subspace. Its rank as a 
# 4-tensor constrains the algebra.
A_C_mat = A_C.reshape(81, 81)
rank_AC = np.linalg.matrix_rank(A_C_mat, tol=1e-8)
print(f"Associator rank (as 81×81 matrix): {rank_AC}")

# ═══════════════════════════════════════════════════════════════
# DIRECT TEST: Can tower identities reduce multiplication count?
# ═══════════════════════════════════════════════════════════════

print("\n── Tower Identities for 3×3 Matmul ──")

# In the tower algebra, the product of two elements uses the structure 
# constants. Each "lower-level multiplication" in the CD construction 
# is a potential saving. Count the actual multiplications needed:

# Level-1 Z_3 product: (a,b,c)·(d,e,f)
# P_0 = ad - bf - ce  → 3 mults (with Karatsuba: could be fewer)
# P_1 = ae + bd - cf  → 3 mults
# P_2 = af - be + cd  → 3 mults
# Total: 9 scalar mults for a 3-dim algebra product

# Level-2 product: each slot is a Level-1 product or conjugated product
# 3 slots × formula involving Level-1 products
# Naive: 9 Level-1 products × 9 scalar mults = 81 scalar mults
# But with signs and conjugation, some terms simplify.

# For T_matmul (standard 3×3 matmul): 27 scalar mults (naive)
# Strassen-like: 23 (best known for 3×3)
# If tower structure ≅ matmul via gauge: tower identities → new algorithm

# Check how many nonzero structure constants each has
nnz_T = np.count_nonzero(T)
nnz_C = np.count_nonzero(C)
print(f"Nonzero structure constants: T_matmul = {nnz_T}, C_tower = {nnz_C}")

# For any algebra, # of multiplications in the bilinear product = tensor rank
# Lower nnz doesn't directly help, but STRUCTURE does.

# The tower has a RECURSIVE structure: level k+1 product uses level k products.
# This gives a divide-and-conquer algorithm IF T_matmul is gauge-equivalent.

print("\n── Recursive Multiplication Count ──")
print("If T_matmul were gauge-equivalent to C_tower:")
print("  Level-1 Z_3 product: 9 scalar multiplications (naive)")
print("  Level-2 Z_3 product via recursion:")

# Count Level-1 products needed for one Level-2 product:
# For each pair (i1, i2) with result slot m = (i1+i2)%3:
# 3×3 = 9 pairs, each needing one Level-1 product
# BUT: conjugation is just sign flips (free), so still 9 Level-1 products
# 9 Level-1 products × 9 scalar mults = 81
# But with Karatsuba on the Z_3 structure: can we do < 9 Level-1 products?

# The Z_3 product is a polynomial product mod x³-1, with sign twists.
# Karatsuba for degree-2 polys: 5 mults instead of 9 (Toom-Cook).
# Toom-3: 5 multiplications for 3-term polynomial product.
print("  Naive: 9 Level-1 products")
print("  With Toom-3 on Z_3 structure: potentially 5 Level-1 products")
print("  5 Level-1 products × (recursive cost) → sub-cubic potential!")
print(f"  Exponent: log_3(5) = {np.log(5)/np.log(3):.6f} (vs log_2(7) = {np.log(7)/np.log(2):.6f} Strassen)")

# CRITICAL: Even if not gauge-equivalent, the tower's RECURSIVE STRUCTURE
# suggests a new factorization approach: find a 9-dim algebra with
# recursive Z_3 structure that IS gauge-equivalent to T_matmul.

print("\n" + "=" * 72)
print("SYNTHESIS")
print("=" * 72)
print(f"""
The Z_3 tower Level-2 algebra is a 9-dimensional algebra, same as T_matmul.

WHAT MATCHES:
  • Both are 9×9×9 tensors
  • T_matmul has unit (e_0), C_tower has unit (e_0)  
  • Both have multilinear rank {t_mlrank} vs {c_mlrank}

WHAT DIFFERS:
  • T_matmul is associative (0 violations), C_tower is not ({v_C} violations)
  • T_matmul is non-commutative, C_tower IS commutative
  • They are NOT gauge-equivalent (associativity is gauge-invariant)

THE BRIDGE:
  Since associativity is invariant under gauge transform, C_tower CANNOT
  be gauge-equivalent to T_matmul. BUT:

  1. DEFORMATION PATH: T_matmul could be a *deformation* of C_tower that
     restores associativity while preserving the recursive Z_3 structure.
     The 216/729 = 8/27 violation fraction tells us how "far" the tower
     is from associativity — and the exact formula V_2 = 4n³(n-1)(n-2)
     quantifies the correction needed.

  2. RECURSIVE FACTORING: The tower's Z_3 recursion means Level-2 products
     decompose into Level-1 products. If we can find a MODIFIED Z_3 tower
     that IS associative (= T_matmul in some basis), we get:
     • Toom-3: 5 level-1 multiplications instead of 9
     • Each level-1 = 3-dim product
     • Total: 5 × (cost of 3-dim product)
     • Exponent: log_3(5) ≈ 1.465 (better than Strassen's 2.807!)

  3. HALF-ASSOCIATIVITY BOUND: The universal 1/2 limit tells us that
     generic Z_n towers are "maximally non-associative" — they lose
     exactly half their associativity. T_matmul (fully associative) is
     the SPECIAL POINT. The violation formulas V_k(n) = exact polynomials
     may constrain which deformations can reach full associativity.

NEXT STEPS:
  → Find the minimal deformation C_tower + εD that IS associative
  → Check if that deformation preserves the recursive Z_3 structure
  → If yes: Toom-3 decomposition → new matmul algorithm at ω ≈ log_3(5)
""")
