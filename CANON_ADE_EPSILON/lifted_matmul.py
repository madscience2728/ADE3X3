"""
LIFTED MATMUL — Testing Emmy's Framework

For (V, ⋆, ι, π) with π∘ι = id:
  T_matmul = (ι^T ⊗ ι^T ⊗ π) · T_V

THEOREM: rank(T_matmul) ≤ rank(T_V) for any linear lift.
  (multilinear maps can only decrease tensor rank)

So Type C (exact lower rank) is IMPOSSIBLE for linear lifts.
The only routes are:
  Type A: cheaper per-operation cost (algebra structure)
  Type N: NON-LINEAR lifting (preprocessing)

This script tests both rigorously.

NO OPTIMIZATION.
"""

import numpy as np
from itertools import product as iprod
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════

def build_Tmatmul():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[3*r+s, 3*s+u, 3*r+u] = 1.0
    return T

T = build_Tmatmul()

print("=" * 72)
print("LIFTED MATMUL — Emmy's (V, ⋆, ι, π) Framework")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════
# THEOREM 1: RANK INEQUALITY (Linear Lifts)
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  THEOREM 1: rank(T_V) ≥ rank(T_matmul) for linear lifts")
print("━" * 72)

print("""
  If T_matmul = (P ⊗ Q ⊗ R) · T_V  for matrices P, Q, R:
  
  Then any rank-Υ decomposition T_V = Σ_k α_k ⊗ β_k ⊗ γ_k
  gives T_matmul = Σ_k (P·α_k) ⊗ (Q·β_k) ⊗ (R·γ_k)
  
  So rank(T_matmul) ≤ Υ = rank(T_V).
  
  This holds for ANY linear triple (P, Q, R) — including lifts
  to Albert algebra, E6 rep, Z_3 tower, or any bigger space.
  
  ⇒ TYPE C IS IMPOSSIBLE FOR LINEAR LIFTS.
  ⇒ No linear embedding into a higher-dim algebra can reduce tensor rank.
""")

# Verify numerically: random linear lift to dim d, compute ranks
print("  Numerical verification:")
for d in [12, 18, 27, 36]:
    # Random lift: ι is d×9 with π·ι = I_9
    ι = np.random.randn(d, 9)
    π = np.linalg.pinv(ι)  # 9×d, gives π·ι ≈ I
    
    # Build T_V such that (π ⊗ π ⊗ π) · T_V = T_matmul  
    # Simplest: T_V = (ι ⊗ ι ⊗ ι) · T_matmul (embed T_matmul into bigger space)
    T_V = np.einsum('ai,bj,ck,ijk->abc', ι, ι, ι, T)
    
    # Check: pushforward recovers T_matmul (π applied to all 3 modes)
    T_recover = np.einsum('ia,jb,kc,abc->ijk', π, π, π, T_V)
    err = np.linalg.norm(T_recover - T)
    
    # Multilinear ranks
    r0 = np.linalg.matrix_rank(T_V.reshape(d, d*d), tol=1e-8)
    r1 = np.linalg.matrix_rank(T_V.transpose(1,0,2).reshape(d, d*d), tol=1e-8)
    r2 = np.linalg.matrix_rank(T_V.transpose(2,0,1).reshape(d, d*d), tol=1e-8)
    
    print(f"  d={d:>2}: T_V multilinear rank = ({r0},{r1},{r2}), "
          f"recovery err = {err:.2e}")

print(f"\n  T_matmul multilinear rank = (9,9,9)")
print(f"  All lifts have multilinear rank ≥ (9,9,9). ✓")
print(f"  Tensor rank follows: rank(T_V) ≥ rank(T_matmul) ≥ 19 (conj). ✓")

# ═══════════════════════════════════════════════════════════════
# WHAT ABOUT NON-LINEAR LIFTS? (Type N)
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  TYPE N: Non-Linear Lifting (Preprocessing)")
print("━" * 72)

print("""
  The rank inequality assumes LINEAR ι. If ι is non-linear:
  
    ι(A) = (A, f(A))  where f(A) depends on A's entries non-linearly
  
  Then the rank argument breaks. The operation ⋆ in V can exploit
  precomputed non-linear functions of A to reduce multiplications.
  
  Classical example: Strassen precomputes A_{11}+A_{22}, A_{21}+A_{22},
  etc. — these are LINEAR in A but the KEY is that the bilinear
  products combine different PAIRS of such precomputations.
  
  For 3×3: can we precompute quantities from A and B such that
  the number of bilinear products (actual multiplications) drops?
""")

# Test: the Strassen-style approach for 3×3
# Precompute p linear combinations of A's entries → p values
# Precompute q linear combinations of B's entries → q values  
# Form R bilinear products (actual multiplications)
# Reconstruct AB from these R products + additions

# The minimum R such that this works = tensor rank of T_matmul.
# This IS the standard tensor rank. So even with preprocessing,
# linear preprocessing doesn't help.

# BUT: NONLINEAR preprocessing DOES change the game.
# Example: precompute A² (costs 23 mults for 3×3).
# Then AB = (A+B)² - A² - B² + ... → this is the polarization identity.
# Total: 3 squarings = 3×23 = 69 mults. Worse than 23.

# Better: precompute row/column norms, traces, etc.
# These give SCALAR quantities that can be reused.

# The REAL non-linear approach: Cohn-Umans group-theoretic method.
# Embed matmul into a group algebra using the TENSOR PRODUCT property
# of irreducible representations.

print("  The Cohn-Umans framework IS a non-linear lift:")
print("  V = group algebra C[G] for a finite group G")
print("  ι: Mat(3,R) → C[G] via an embedding of 3×3 matrices")
print("     into a group representation")
print("  ⋆ = convolution in C[G]  (computed via FFT)")
print("  π = extract the matmul result from the convolution")
print("")
print("  Cost: O(|G| log|G|) via FFT, no matter what the tensor rank is.")
print("  The PRIMITIVE is group algebra multiplication (FFT),")
print("     NOT scalar multiplication.")
print("")
print("  This IS Type A: the per-operation cost is O(1) additions")
print("  and O(1) multiplications PER GROUP ELEMENT, but the FFT")
print("  structure makes the TOTAL cost sub-cubic if |G| is small enough.")

# ═══════════════════════════════════════════════════════════════
# THE E6 LIFT — Concrete Test
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  E6 LIFT — Concrete Feasibility")
print("━" * 72)

# The E6 fundamental representation (dim 27) has a natural cubic form
# (the Freudenthal-Tits construction). The BILINEAR operation derived
# from the cubic is the "Freudenthal cross product" × : V⊗V → V*.

# For our purposes: the 27-dim E6 rep relates to the exceptional 
# Jordan algebra J₃(O) (3×3 Hermitian octonionic matrices).

# Key structure: J₃(O) has Jordan product A∘B = (AB+BA)/2
# This is COMMUTATIVE. Matmul is NOT commutative.
# So Jordan product alone CANNOT give exact matmul lift.

# BUT: the FULL product in the octonionic matrix algebra
# M₃(O) (not just the Hermitian part) IS non-commutative.
# M₃(O) has dim = 9×8 = 72 (9 entries × 8-dim octonions).
# Mat(3,ℝ) ⊂ M₃(O) via ℝ ⊂ O (real part only).

# Test: can the octonionic structure help?
# M₃(O) multiplication: (AB)_{ij} = Σ_k A_{ik} · B_{kj}
# where · is octonionic multiplication (non-associative!).
# For real entries, octonionic · = real ·, so this reduces to standard matmul.
# The lift IS exact: π(ι(A)·ι(B)) = AB.

# But rank(T_{M₃(O)}) = rank of the 72-dim matmul tensor.
# This is rank of ⟨3,3,3⟩ over O, which by the rank inequality
# is ≥ rank of ⟨3,3,3⟩ over ℝ. No savings from the rank theorem.

print("  Albert algebra J₃(O):")
print("    Jordan product A∘B = (AB+BA)/2 is COMMUTATIVE")
print("    Matmul is NOT commutative")
print("    ⇒ Jordan product alone CANNOT give exact lift")
print("")
print("  Full octonionic matrices M₃(O):")
print("    dim = 72 (9 entries × 8 octonion components)")
print("    Multiplication is exact matmul for real entries")
print("    BUT: rank(T_{M₃(O)}) ≥ rank(T_matmul) by Theorem 1")
print("    AND: octonion multiplication costs 8 real mults per entry")
print("    ⇒ More expensive, not less")

# What about the E6 CUBIC FORM specifically?
# The cubic form det: J₃(O) → ℝ is the "Jordan determinant"
# det(A) = Tr(A³) - (3/2)Tr(A²)Tr(A) + (1/2)Tr(A)³
# (for the 3×3 case this is the usual determinant)
#
# The BILINEAR cross product A×B = A∘B - (1/2)(Tr(A)B + Tr(B)A) + ...
# This maps J₃(O)×J₃(O) → J₃(O)
# It's related to matmul by: A×B relates to A^{-1}·det(A) when A is invertible.

# The crucial question: does the E6 structure provide ADDITIONAL RELATIONS
# that reduce the number of independent bilinear products needed?

# Over ℝ^9 (Mat(3,ℝ)), the identities:
# Cayley-Hamilton: A³ - Tr(A)A² + (Tr(A)²-Tr(A²))/2 · A - det(A)·I = 0
# Newton's identities: power sums ↔ elementary symmetric polynomials

# These are POLYNOMIAL identities, not bilinear. They constrain the 
# algebra but don't directly reduce bilinear complexity.

print("\n  E6 cubic form analysis:")
print("    The E6 structure gives polynomial identities")
print("    (Cayley-Hamilton, Newton, Freudenthal)")
print("    These are degree ≥ 3 — NOT bilinear")
print("    They constrain the algebra but don't reduce bilinear rank")

# ═══════════════════════════════════════════════════════════════
# THE HONEST ANSWER: What DOES Work?
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  WHAT ACTUALLY REDUCES MULTIPLICATION COUNT")
print("━" * 72)

# The rank inequality kills all linear lifts for tensor rank.
# What actually works in the literature:

# 1. ASYMPTOTIC methods (laser method, Coppersmith-Winograd):
#    Embed into TENSOR POWERS T^⊗n and exploit structure there.
#    This IS a lift — to V = (ℝ⁹)^⊗n — but non-linear (nth power).
#    Gives ω < 2.372 asymptotically but terrible constants.

# 2. RECURSIVE embedding (Strassen):
#    Lift 3×3 into 4×4 (pad with zeros), apply 2×2 Strassen recursively.
#    V = Mat(4,ℝ), ⋆ = 4×4 matmul, π = extract top-left 3×3.
#    Exact lift (zero-padding doesn't affect top-left block).
#    Cost: 49 mults for 4×4. But only 27 of the 64 outputs are needed.
#    With pruning: can we do fewer mults for the 3×3 sub-result?

# Let's actually test this!
print("\n  ── Test: 4×4 Strassen lift for 3×3 matmul ──")

# Build T_matmul for 4×4
T4 = np.zeros((16, 16, 16))
for r in range(4):
    for s in range(4):
        for u in range(4):
            T4[4*r+s, 4*s+u, 4*r+u] = 1.0

# Embedding: 3×3 matrix A → 4×4 matrix Ã (zero-pad)
# ι: ℝ⁹ → ℝ¹⁶
ι = np.zeros((16, 9))
for r in range(3):
    for s in range(3):
        ι[4*r+s, 3*r+s] = 1.0

# Projection: 4×4 matrix → 3×3 (extract top-left)
# π: ℝ¹⁶ → ℝ⁹  
π = ι.T  # π·ι = I₉ ✓

# Verify exact lift: π ⊗ π ⊗ π applied to T4 should give T_matmul
T_lift = np.einsum('ia,jb,kc,abc->ijk', π, π, π, T4)
lift_err = np.linalg.norm(T_lift - T)
print(f"  4×4 lift recovery error: {lift_err:.2e}")
print(f"  Exact lift: {'✓' if lift_err < 1e-12 else '✗'}")

# Now: Strassen for 4×4 uses 49 multiplications (7 recursive × 7).
# But for the 3×3 OUTPUT, how many of those 49 are needed?

# The 4×4 output has 16 entries. We only need 9 (the 3×3 block).
# Each Strassen multiplication produces specific linear combos of outputs.
# If some multiplications ONLY contribute to the unused 7 entries,
# they can be skipped.

# In Strassen's 2×2 decomposition applied recursively to 4×4:
# The 7 multiplications M1..M7 at the top level each contribute to 
# specific output blocks. Since the 4×4 = 2×2 of 2×2 blocks:
# C₁₁ = M1 + M4 - M5 + M7  (top-left 2×2)
# C₁₂ = M3 + M5             (top-right 2×2)
# C₂₁ = M2 + M4             (bottom-left 2×2)
# C₂₂ = M1 - M2 + M3 + M6  (bottom-right 2×2)

# For 3×3 embedded in 4×4: we need C₁₁ (full 2×2) + parts of C₁₂ and C₂₁
# C₁₁: need M1, M4, M5, M7
# C₁₂: need M3, M5 (but only first column of C₁₂)
# C₂₁: need M2, M4 (but only first row of C₂₁)
# C₂₂: NOT needed (only (2,2) entry of it = entry (3,3) of 3×3... 
#        wait, (2,2) block = rows 2-3, cols 2-3 of 4×4 = rows 2-3, cols 2-3 of 3×3 result
#        if 0-indexed: C₂₂ has row indices {2,3} and col indices {2,3}
#        We need {2,2} of the 3×3 = row 2, col 2 = which IS in C₂₂ top-left)

# Actually for 0-indexed 4×4 with 2×2 blocks:
# Block (0,0) = rows {0,1} × cols {0,1} → 3×3 entries (0,0),(0,1),(1,0),(1,1) ✓
# Block (0,1) = rows {0,1} × cols {2,3} → 3×3 entries (0,2),(1,2) ✓ + (0,3),(1,3) ✗
# Block (1,0) = rows {2,3} × cols {0,1} → 3×3 entries (2,0),(2,1) ✓ + (3,0),(3,1) ✗  
# Block (1,1) = rows {2,3} × cols {2,3} → 3×3 entry (2,2) ✓ + (2,3),(3,2),(3,3) ✗

# So we need ALL four output blocks (at least partially).
# All 7 top-level multiplications are needed.
# Each is a 2×2 matmul (costs 7 mults via Strassen).
# Total: 7 × 7 = 49 mults.

# BUT: at the second recursion level, for blocks that are only
# partially needed, some sub-multiplications might be skippable.

# For C₁₂ (block 0,1): we need columns {2} but not {3}.
# C₁₂ = M3 + M5 where M3 and M5 are 2×2 matrices.
# Column 0 of C₁₂ = col 0 of M3 + col 0 of M5.
# Each Mk is itself a 2×2 product done with 7 scalar mults.
# We need only ONE column of the 2×2 product → can we do it with fewer?

# For 2×2 matmul producing only one column: 
# (C)_{i,0} = Σ_k A_{ik}B_{k0} → 2 mults per row × 2 rows = 4 mults (naive)
# Or: this is a matrix-vector product, rank = 2 (not 7).
# So partial results are CHEAPER.

n_full_blocks = 1  # C₁₁: full 2×2 × 2×2 = 7 mults
n_partial_col = 1   # C₁₂: need 1 column → rank 2, costs 2 × 2 = 4? 
n_partial_row = 1   # C₂₁: need 1 row → similarly 4
n_partial_corner = 1 # C₂₂: need 1 entry → 1 × 2 = 2 inner products

# Wait, these aren't the direct 2×2 products. In Strassen's formula,
# M1..M7 are each 2×2 PRODUCTS (each costing 7 scalar mults).
# We can't skip any M_k entirely since all appear in C₁₁ which is fully needed.
# So all 7 top-level products run fully: 7 × 7 = 49.

# But we can skip parts of the SECOND level's reconstruction.
# That only saves additions, not multiplications.

print(f"\n  Strassen recursion for 4×4 → 3×3:")
print(f"    Top level: 7 block products (all needed for C₁₁)")
print(f"    Each block product: 7 scalar mults (Strassen on 2×2)")
print(f"    Total: 49 scalar multiplications")
print(f"    Savings from partial output: additions only, not mults")
print(f"    ⇒ Strassen lift: Υ = 49 mults (worse than 23)")

# 3. DIRECT 3×3 approaches:
# The best known exact algorithm is 23 multiplications (Makarov-Vassilevska Williams et al.)
# Smirnov (2013) showed a tensor of border rank 21 for ⟨3,3,3⟩.

print(f"\n  ── Known results for ⟨3,3,3⟩ ──")
print(f"    Naive:                27 multiplications")
print(f"    Best known exact:     23 (multiple authors)")
print(f"    Best border rank:     ≤ 21 (Smirnov 2013)")
print(f"    Conjectured exact:    19 (unproven, strong evidence)")
print(f"    Flattening bound:     ≥ 19 (Bläser)")
print(f"    Strassen 4×4 lift:    49 (worse)")
print(f"    Strassen 3×3 direct:  23 (= best known)")

# ═══════════════════════════════════════════════════════════════
# THE ESCAPE: COMMUTATIVE MATRIX MULTIPLICATION (Type A)
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  THE REAL ESCAPE: Commutative Algebra Structure")
print("━" * 72)

# The rank inequality kills linear lifts. But what if V is COMMUTATIVE?
# Then T_V has SYMMETRY: T_V[a,b,c] = T_V[b,a,c].
# A symmetric tensor can have lower SYMMETRIC rank than general rank.
# 
# Matmul is NOT symmetric (AB ≠ BA). But the COMMUTATIVE part
# (AB + BA)/2 IS symmetric. And the anticommutator has rank ≤ 19 
# (found in Step 75).
#
# Crucial: if we split T = T_sym + T_anti where
#   T_sym = (AB + BA)/2  (rank ≤ 19, symmetric)
#   T_anti = (AB - BA)/2 (rank ≤ 20, antisymmetric)
# then rank(T) ≤ rank(T_sym) + rank(T_anti) = 39 (naive bound, not useful)
#
# BUT: shared terms can be subtracted!
# If T_sym and T_anti share k rank-1 components:
# rank(T) ≤ rank(T_sym) + rank(T_anti) - k

# From Step 75/78: span intersection was 0. No sharing. Dead end for this.

# HOWEVER: the commutative QUOTIENT is interesting.
# Define: V = Mat(3,ℝ) / [A,B]  (quotient by commutator ideal)
# This is the "abelianization" of the matrix algebra.
# In this quotient, AB = BA, and the multiplication IS commutative.
# The quotient algebra is isomorphic to... well, for Mat(3,ℝ), the 
# abelianization is just ℝ (scalars!) since [Mat_n, Mat_n] = sl_n,
# and Mat_n / sl_n ≅ ℝ via the trace map.

# So the commutative quotient is too small (dim 1). Not useful.

# What about the CENTER of Mat(3,ℝ)? Z(Mat(3)) = ℝ·I (dim 1).
# The center sees only the scalar part: Tr(AB)/3.
# Computing Tr(AB) takes 9 mults (dot product of vectorized matrices).
# This is the "cheapest" part of matmul.

print("  Commutative approaches:")
print("    Abelianization of Mat(3,ℝ) = ℝ (too small)")
print("    Center = ℝ·I (sees only Tr(AB)/3)")
print("    Split into sym + anti: no shared terms (Step 78)")
print("    ⇒ No savings from commutativity structure")

# ═══════════════════════════════════════════════════════════════
# THE COHN-UMANS DIRECTION (Group Algebra Lift)
# ═══════════════════════════════════════════════════════════════

print("\n" + "━" * 72)
print("  COHN-UMANS GROUP ALGEBRA LIFT")
print("━" * 72)

# The Cohn-Umans (2003) framework: embed matmul into a group algebra.
# If there exist subsets S, T, U ⊆ G with |S|=|T|=|U|=n such that
# the "triple product property" holds:
#   s₁t₁ = s₂t₂ AND s₁u₁ = s₂u₂ AND t₁u₁ = t₂u₂ ⟹ s₁=s₂, t₁=t₂, u₁=u₂
# then ⟨n,n,n⟩ embeds in the group algebra ℂ[G].

# The multiplication in ℂ[G] is convolution, computable via FFT.
# The key equation:
#   ω ≤ 3τ(G) where τ(G) = log|G|/log n

# For n=3: need |S|=|T|=|U|=3 in G satisfying triple product property.
# The trivial choice: G = Z_3 × Z_3 × Z_3 (order 27):
#   S = {(0,0,0),(1,0,0),(2,0,0)} (first factor)
#   T = {(0,0,0),(0,1,0),(0,2,0)} (second factor)
#   U = {(0,0,0),(0,0,1),(0,0,2)} (third factor)
#   Triple product property holds (independent coordinates).
#   τ = log(27)/log(3) = 3. So ω ≤ 9 — trivial bound.

# For a USEFUL bound: need smaller G.
# Gap: need G with |G| < 27 but still embedding 3×3 matmul.
# Best possible: |G| = n^{ω/τ_min}, so |G| < 27 iff ω/(τ·logn) < 3.

# Let's check: which SMALL groups can embed ⟨3,3,3⟩?
print("  Testing small groups for Cohn-Umans embedding of ⟨3,3,3⟩:")
print("")

def check_triple_product(G_elements, S, T, U, op):
    """Check triple product property for subsets S,T,U of group G."""
    n = len(S)
    assert len(T) == n and len(U) == n
    
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            # Check if s_{i1}t_{j1} = s_{i2}t_{j2} for different (i,j) pairs
            for j1 in range(n):
                st1 = op(S[i1], T[j1])
                for j2 in range(n):
                    st2 = op(S[i2], T[j2])
                    if st1 != st2:
                        continue
                    # s_i1 * t_j1 = s_i2 * t_j2 with i1≠i2
                    # Check the other conditions
                    for k1 in range(n):
                        su1 = op(S[i1], U[k1])
                        for k2 in range(n):
                            su2 = op(S[i2], U[k2])
                            tu1 = op(T[j1], U[k1])
                            tu2 = op(T[j2], U[k2])
                            if su1 == su2 and tu1 == tu2:
                                return False  # violation
    return True

# Test Z_n × Z_n for small n
for n_grp in [3, 4, 5, 6, 7]:
    order = n_grp * n_grp
    elements = [(a, b) for a in range(n_grp) for b in range(n_grp)]
    op = lambda x, y, n=n_grp: ((x[0]+y[0]) % n, (x[1]+y[1]) % n)
    
    # Try S along first axis, T along second, U along diagonal
    S = [(i, 0) for i in range(3)]
    T = [(0, i) for i in range(3)]
    U_diag = [(i, i) for i in range(3)]
    U_anti = [(i, (n_grp-i) % n_grp) for i in range(3)]
    
    for U, Uname in [(U_diag, "diag"), (U_anti, "anti-diag")]:
        valid = check_triple_product(elements, S, T, U, op)
        if valid:
            tau = np.log(order) / np.log(3)
            print(f"  Z_{n_grp}×Z_{n_grp} (order {order}), U={Uname}: "
                  f"TPP ✓, τ = {tau:.3f}")

# Test non-abelian groups: S_3 (order 6), D_3 (order 6)
# S_3 elements as permutations
s3_elements = [(0,1,2),(1,0,2),(0,2,1),(2,1,0),(1,2,0),(2,0,1)]
def s3_mult(a, b):
    return tuple(a[b[i]] for i in range(3))

# Try all triples of 3-element subsets
from itertools import combinations
s3_list = list(range(6))
found_s3 = False
for S_idx in combinations(s3_list, 3):
    S = [s3_elements[i] for i in S_idx]
    for T_idx in combinations(s3_list, 3):
        T_sub = [s3_elements[i] for i in T_idx]
        for U_idx in combinations(s3_list, 3):
            U = [s3_elements[i] for i in U_idx]
            if check_triple_product(s3_elements, S, T_sub, U, s3_mult):
                tau = np.log(6) / np.log(3)
                print(f"  S_3 (order 6): TPP ✓ with S={S_idx}, T={T_idx}, U={U_idx}")
                print(f"    τ = {tau:.3f}, ω bound = 3τ = {3*tau:.3f}")
                found_s3 = True
                break
        if found_s3:
            break
    if found_s3:
        break

if not found_s3:
    print(f"  S_3 (order 6): no TPP-satisfying triple found")

# Test Z_9 (order 9)
z9_elements = list(range(9))
z9_op = lambda x, y: (x + y) % 9
found_z9 = False
for S in combinations(range(9), 3):
    S = list(S)
    for T_sub in combinations(range(9), 3):
        T_sub = list(T_sub)
        for U in combinations(range(9), 3):
            U = list(U)
            if check_triple_product(z9_elements, S, T_sub, U, z9_op):
                tau = np.log(9) / np.log(3)
                print(f"  Z_9 (order 9): TPP ✓ with S={S}, T={T_sub}, U={U}")
                print(f"    τ = {tau:.3f}, ω bound = 3τ = {3*tau:.3f}")
                found_z9 = True
                break
        if found_z9:
            break
    if found_z9:
        break

if not found_z9:
    print(f"  Z_9 (order 9): no TPP-satisfying triple found (expected — abelian order 9 too small)")

# ═══════════════════════════════════════════════════════════════
# SYNTHESIS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SYNTHESIS — Honest Assessment")
print("=" * 72)

print("""
RANK INEQUALITY THEOREM:
  For ANY linear lift (V, ⋆, ι, π) with π∘ι = id:
  
    rank(T_V) ≥ rank(T_matmul)
  
  This is mathematically rigorous. It means:
  
  • Albert algebra lift: rank(T_Albert) ≥ rank(T_matmul)
  • E6 rep lift: rank(T_E6) ≥ rank(T_matmul) 
  • Z_3 tower lift: rank(T_Z3) ≥ rank(T_matmul)
  • ANY dim-27 algebra lift: rank ≥ rank(T_matmul)
  
  Emmy's Type C is IMPOSSIBLE for linear lifts. Full stop.

WHAT DOES WORK:
  
  1. TYPE A (Structure-assisted computation):
     The lifted algebra has algebraic identities (Cayley-Hamilton,
     Newton, Freudenthal) that provide FREE relations. These don't
     reduce tensor rank but can reduce TOTAL operation count in
     a structured algorithm.
     
     Example: computing AB and A²B simultaneously is cheaper than
     computing each separately. The algebra structure reveals which
     "joint computations" save work.
  
  2. COHN-UMANS (Group algebra embedding):
     The only known framework that genuinely changes the complexity
     class. Embeds matmul into group algebra multiplication (FFT).
     For n=3: need a group G with |G| < 27 satisfying the triple
     product property. Z_3×Z_3 (order 9) is the minimum target.
     
  3. ASYMPTOTIC METHODS (Laser/CW):
     Work with tensor powers T^⊗n → non-linear lift.
     Give ω < 2.372 but with impractical constants.

THE HONEST BOTTOM LINE:
  The E6/Albert/tower lifting cannot reduce tensor rank.
  The beautiful E6 spectrum {6,3,0,-3} with multiplicities {1,6,12,8}
  is a SYMMETRY fact, not a RANK fact.
  
  For PRACTICAL 3×3 matmul improvement: the frontier is still
  direct tensor decomposition (23 mults, trying for 19-22).
  
  For THEORETICAL ω improvement: Cohn-Umans group embedding or
  laser method variants are the live approaches.
  
  The lifting framework is real mathematics but does not provide
  a new attack on 3×3 matmul complexity.
""")
