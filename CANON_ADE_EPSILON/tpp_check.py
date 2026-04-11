"""
TPP CHECK — Does G = Z_2 ≀ S_3 satisfy the Triple Product Property?
====================================================================

The Triple Product Property (Cohn-Umans 2003):
  Subsets S, T, U ⊂ G satisfy TPP if:
    For all s∈S, t∈T, u∈U:  s·t·u = e  ⟹  s = t = u = e

  For ⟨n,n,n⟩ matmul embedding, we need |S|=|T|=|U|=n² 
  and the subsets must "realize" the matmul tensor.

  Specifically: S, T, U must be such that the group algebra
  multiplication restricted to C[S]·C[T] projected onto C[U]
  recovers the matmul tensor.

The SIMULTANEOUS Triple Product Property (STPP) is weaker:
  Needs S_i, T_i, U_i for each irrep i, with the union satisfying
  a combined condition.

We check:
  1. Construct G = Z_2 ≀ S_3 explicitly as permutation group
  2. Enumerate all (S,T,U) with |S|=|T|=|U|=9, check TPP
  3. If TPP holds, compute τ and ω bound
  4. Also check relaxed conditions
"""

import numpy as np
from itertools import combinations, product
from collections import defaultdict
import time

print("=" * 72)
print("TPP CHECK — G = Z₂ ≀ S₃, order 48")
print("=" * 72)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Construct G = Z_2 ≀ S_3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Z_2 ≀ S_3 = Z_2³ ⋊ S_3
# Elements: (ε, σ) where ε ∈ Z_2³ = {0,1}³, σ ∈ S_3
# Multiplication: (ε₁, σ₁) · (ε₂, σ₂) = (ε₁ + σ₁(ε₂), σ₁∘σ₂)
# where σ₁(ε₂) permutes the components of ε₂

from itertools import permutations

# S_3 as permutations of {0,1,2}
S3 = list(permutations(range(3)))  # 6 elements
Z2_3 = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]  # 8 elements

# Group elements as (eps, sigma) pairs
G_elements = []
for eps in Z2_3:
    for sigma in S3:
        G_elements.append((eps, sigma))

N = len(G_elements)
print(f"\n|G| = {N}")
assert N == 48

# Index lookup
elem_to_idx = {g: i for i, g in enumerate(G_elements)}

def g_mul(g1, g2):
    """Multiply (ε₁,σ₁)·(ε₂,σ₂) = (ε₁ + σ₁(ε₂), σ₁∘σ₂)"""
    eps1, sig1 = g1
    eps2, sig2 = g2
    # σ₁ acts on ε₂ by permuting components
    sig1_eps2 = tuple(eps2[sig1[i]] for i in range(3))
    new_eps = tuple((eps1[i] + sig1_eps2[i]) % 2 for i in range(3))
    new_sig = tuple(sig1[sig2[i]] for i in range(3))
    return (new_eps, new_sig)

def g_inv(g):
    """Inverse: (ε,σ)⁻¹ = (σ⁻¹(ε), σ⁻¹)"""
    eps, sig = g
    # σ⁻¹
    sig_inv = [0, 0, 0]
    for i in range(3):
        sig_inv[sig[i]] = i
    sig_inv = tuple(sig_inv)
    # σ⁻¹(ε)
    new_eps = tuple(eps[sig_inv[i]] for i in range(3))
    # But we need -σ⁻¹(ε) = σ⁻¹(ε) in Z_2
    return (new_eps, sig_inv)

# Verify group axioms on a sample
e = ((0, 0, 0), (0, 1, 2))  # identity
assert elem_to_idx[e] == elem_to_idx[g_mul(e, e)]

# Build multiplication table
mul_table = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(N):
        prod = g_mul(G_elements[i], G_elements[j])
        mul_table[i, j] = elem_to_idx[prod]

# Build inverse table
inv_table = np.zeros(N, dtype=int)
for i in range(N):
    inv_table[i] = elem_to_idx[g_inv(G_elements[i])]

# Verify
e_idx = elem_to_idx[e]
for i in range(N):
    assert mul_table[i, inv_table[i]] == e_idx
    assert mul_table[inv_table[i], i] == e_idx

print("Group multiplication table verified ✓")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Irreducible representations of Z_2 ≀ S_3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Z_2 ≀ S_3 has irreps indexed by partitions of 3 into parts,
# each labeled by a Z_2 character.
# 
# The irreps of Z_2 ≀ S_3 are:
# dim 1: trivial, sign, det, sign⊗det (4 one-dimensional)
# dim 2: standard of S_3 ⊗ various Z_2 chars (several 2-dim)
# dim 3: induced reps (several 3-dim)
#
# Actually, for Z_2 ≀ S_3: the irreps come from pairs of partitions
# (λ⁺, λ⁻) where |λ⁺| + |λ⁻| = 3.
# 
# Pairs: (3,∅), (2,1), (1,2), (∅,3), (21,∅), (∅,21), (1,1+1), (1+1,1), (111,∅), (∅,111)
# Wait, let me be more careful.
#
# For Z_2 ≀ S_n, irreps are parametrized by pairs of partitions (α,β)
# with |α|+|β|=n. For n=3:
# (3,∅): dim = 1
# (21,∅): dim = 2  
# (111,∅): dim = 1
# (∅,3): dim = 1
# (∅,21): dim = 2
# (∅,111): dim = 1
# (2,1): dim = 3
# (1,2): dim = 3
# (1,11): dim = 3  wait...
# (11,1): dim = 3
#
# Sum of squares: 1+4+1+1+4+1+9+9+9+9 = 48? No that's too many.
# 1²+2²+1²+1²+2²+1²+3²+3² = 1+4+1+1+4+1+9+9 = 30 ≠ 48
# Missing some.
# 
# Let me just compute irreps numerically via the regular representation.

print("\n━━━ Irreducible Representations ━━━")

# Regular representation
reg_rep = np.zeros((N, N, N))
for g_idx in range(N):
    for h_idx in range(N):
        gh_idx = mul_table[g_idx, h_idx]
        reg_rep[g_idx, gh_idx, h_idx] = 1.0

# Character of each element = trace of regular rep matrix
# In the regular rep, only identity has nonzero trace = |G|
# Instead, let's get the character table via class functions.

# Conjugacy classes
visited = set()
classes = []
for i in range(N):
    if i in visited:
        continue
    cls = set()
    for j in range(N):
        # j·i·j⁻¹
        conj = mul_table[j, mul_table[i, inv_table[j]]]
        cls.add(conj)
    classes.append(frozenset(cls))
    visited.update(cls)

print(f"Number of conjugacy classes: {len(classes)}")
print(f"Class sizes: {sorted(len(c) for c in classes)}")

# Number of irreps = number of conjugacy classes
n_irreps = len(classes)

# To find irreps, we diagonalize the center of the group algebra.
# Build class sums as matrices in the regular representation.
class_sums = []
for cls in classes:
    M = np.zeros((N, N))
    for i in cls:
        # Left multiplication by g_i in regular rep
        for j in range(N):
            M[mul_table[i, j], j] += 1.0
    class_sums.append(M)

# These commute, so we can simultaneously diagonalize.
# Take a random combination and find eigenspaces.
rng = np.random.default_rng(42)
combo = sum(rng.standard_normal() * M for M in class_sums)
eigenvalues, eigenvectors = np.linalg.eigh(combo + combo.T)  # symmetrize for stability

# Cluster eigenvalues to find irrep dimensions
from collections import Counter
rounded = np.round(eigenvalues, 6)
dims_counter = Counter(rounded)
irrep_dims = sorted(dims_counter.values(), reverse=True)
print(f"Irrep dimensions (from eigenvalue clustering): {sorted(irrep_dims)}")
print(f"Sum of dim²: {sum(d**2 for d in irrep_dims)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: TPP Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# For ⟨3,3,3⟩ matmul, we need subsets S, T, U ⊂ G with |S|=|T|=|U|=9
# such that:
#   (1) TPP: s·t·u = e ⟹ s=t=u=e  (for s∈S, t∈T, u∈U)
#   (2) The matmul tensor is realized by the embedding
#
# Condition (1) is equivalent to:
#   S ∩ (t·U⁻¹)⁻¹ = {e} for all t∈T, or more precisely:
#   |{(s,t,u) ∈ S×T×U : s·t·u = e}| = 1 (only s=t=u=e)
#   Wait, no. TPP says: s·t·u = e ⟹ s=t=u=e for ALL s∈S,t∈T,u∈U.
#   But if e∈S∩T∩U, then e·e·e=e works. So we need e∈S∩T∩U.
#   And for any OTHER triple, s·t·u ≠ e.
#
# Actually the standard TPP definition:
#   S, T, U satisfy TPP if: s·t = u ⟹ unique solution.
#   More precisely: the map S×T → G given by (s,t) ↦ s·t
#   when restricted to hit U, has the property that each u∈U
#   is hit at most once. 
#
# Let me use the correct Cohn-Umans definition:
#   S, T, U ⊂ G satisfy TPP if:
#   For each pair (s,t) ∈ S×T, if s·t ∈ U, then this pair
#   is the ONLY pair mapping to that element of U.
#   Equivalently: the sets {s·T : s∈S} are "disjoint enough."
#
# Actually the precise definition from Cohn-Umans 2003:
#   Subgroups H₁, H₂, H₃ ≤ G satisfy TPP if
#   H₁·H₂ ∩ H₃ = {e} (after appropriate translating)
#
# For the general (non-subgroup) version:
#   S, T, U ⊂ G with |S|=|T|=|U|=n satisfy the
#   "simultaneous triple product property" if
#   for each irrep ρ of G:
#     dim(ρ(S)V) + dim(ρ(T)V) + dim(ρ(U)V) ≤ 2·dim(ρ)
#   (no that's the "support" version)
#
# Let me use the simplest formulation for a direct TPP check.
# 
# For embedding ⟨n,n,n⟩ into C[G]:
# We need injections φ_A: M_n → C[G], φ_B: M_n → C[G]
# and extraction map φ_C: C[G] → M_n such that
# φ_C(φ_A(A) * φ_B(B)) = A·B
# where * is convolution in C[G].
#
# The simplest embedding: identify matrix entries with group elements.
# A = (a_{ij}), map to f_A = Σ a_{ij} · s_{ij} where s_{ij} ∈ G.
# Similarly B → f_B using t_{kl}, and extract C using u_{ij}.
# Then (f_A * f_B)(g) = Σ_{h} f_A(h) f_B(h⁻¹g)
# For the matmul entry c_{ij} = Σ_k a_{ik}b_{kj}, we need:
#   s_{ik} · t_{kj} = u_{ij} for all i,k,j
# AND: no other products s_{ab}·t_{cd} = u_{ij} unless a=i, b=c=k, d=j.
# 
# This is the "uniqueness of products" condition:
#   The map S×T → G: (s_{ik}, t_{kj}) ↦ s_{ik}·t_{kj} is
#   injective when restricted to pairs producing elements of U.
#
# AND: u_{ij} = s_{ik}·t_{kj} for each valid (i,k,j).
#
# Since we have 9 entries in each matrix, |S|=|T|=|U|=9.
# We need 9 elements s_{ij}, 9 elements t_{ij}, 9 elements u_{ij}
# from G (with |G|=48) such that:
#   s_{ik}·t_{kj} = u_{ij}  for all i,j,k ∈ {0,1,2}
# AND: no other pair (s_{ab}, t_{cd}) with b≠c gives a product in U.

print("\n━━━ TPP Check for ⟨3,3,3⟩ embedding ━━━")
print(f"Need: S, T, U ⊂ G with |S|=|T|=|U|=9")
print(f"Condition: s_{{ik}}·t_{{kj}} = u_{{ij}} and no collisions\n")

# Strategy: S and T are 3×3 arrays of group elements.
# For each (i,k), s_{ik} is a group element.
# For each (k,j), t_{kj} is a group element.
# u_{ij} = s_{i0}·t_{0j} = s_{i1}·t_{1j} = s_{i2}·t_{2j}
# This forces: s_{i0}·t_{0j} = s_{i1}·t_{1j} for all i,j.
# i.e., s_{i1}⁻¹·s_{i0} = t_{1j}·t_{0j}⁻¹ for all i,j.
# The LHS depends only on i, the RHS only on j.
# So both must be constant = some element g.
# This means: s_{i1}⁻¹·s_{i0} = g for all i, AND t_{1j}·t_{0j}⁻¹ = g for all j.
#
# Similarly s_{i2}⁻¹·s_{i0} = h for all i, etc.
#
# This is very restrictive. Let me think differently.
#
# Actually, the standard Cohn-Umans embedding uses COSETS of subgroups.
# For S_3 ≤ G = Z_2 ≀ S_3:
# Can we embed Mat(3) into cosets of S_3?
#
# Let me try a direct computational approach.
# First, check if the SUBGROUP version works.

# Subgroup TPP: H₁, H₂, H₃ ≤ G satisfy TPP if
#   H₁ ∩ H₂·H₃ = {e}  (and cyclic permutations)
# For ⟨n,n,n⟩: need |H_i| ≥ n = 3.

# Find all subgroups of order ≥ 3
print("Finding subgroups...")

# Subgroups of order 3: Z_3 subgroups
# These are generated by 3-cycles in S_3 (with trivial Z_2 part)
# or by elements of order 3.

# Find elements by order
orders = np.zeros(N, dtype=int)
for i in range(N):
    power = i
    for k in range(1, N + 1):
        if power == e_idx:
            orders[i] = k
            break
        power = mul_table[power, i]

order_counts = Counter(orders)
print(f"Element orders: {dict(sorted(order_counts.items()))}")

# For the direct computation: try to find S, T, U as 3×3 arrays
# of group elements satisfying the matmul embedding condition.
#
# Key insight: We can try S = {s_{ij} = εᵢ · σⱼ} for some 
# choice of ε's and σ's. This is a "grid" structure.
#
# Simpler: try S = coset of a subgroup.

# Let me check the DIRECT TPP condition computationally.
# For each triple of 9-element subsets, this is C(48,9)³ ≈ 10²¹.
# Way too many. Need structure.

# Instead: try the STRUCTURED embedding.
# Use the natural action of G on {0,...,8} (the 9 matrix entries).

# The group G = Z_2³ ⋊ S_3 acts on 3×3 matrices.
# The action on indices: σ permutes rows/cols, ε flips signs.
# On matrix entry (i,j): σ maps it to (σ(i), σ(j)), ε maps entry to ±1 times entry.

# For Cohn-Umans: the key is the REGULAR representation.
# In the regular rep, group algebra multiplication IS convolution.
# We embed A into C[G] by: f_A(g) = trace(ρ(g)·A) for a suitable rep ρ.

# Let's try the simplest approach:
# Use the 3-dim permutation representation of S_3 inside G.
# ρ: G → GL(3,R): (ε,σ) ↦ diag((-1)^ε₀, (-1)^ε₁, (-1)^ε₂) · P_σ
# where P_σ is the permutation matrix.

# Build the 3×3 representation
def rep_3(g_idx):
    """3-dimensional representation: signed permutation matrices"""
    eps, sig = G_elements[g_idx]
    M = np.zeros((3, 3))
    for i in range(3):
        M[i, sig[i]] = (-1) ** eps[i]
    return M

# This gives a 3-dim rep. For matmul embedding we need the 
# tensor product rep: ρ⊗ρ = 9-dim rep on Mat(3).
# (ρ⊗ρ)(g)(A) = ρ(g) · A · ρ(g)⁻¹  ... no, that's the adjoint.
# For matmul: we need ρ(g)·A·ρ(g)ᵀ or similar.

# Actually for Cohn-Umans via representations:
# The key formula is: 
#   (A·B)_{ij} = Σ_k A_{ik} B_{kj}
# We want to embed this into convolution:
#   (f * h)(g) = Σ_x f(x) h(x⁻¹g)
# So we need:
#   f_A(x) = Σ_{ik} A_{ik} · δ(x = s_{ik})
#   f_B(x) = Σ_{kj} B_{kj} · δ(x = t_{kj})
# Then: (f_A * f_B)(g) = Σ_{ik,kj} A_{ik}·B_{kj}·δ(s_{ik}·t_{kj} = g)
#        ... but the k in the A-index and B-index must match!
#
# For this to give (AB)_{ij} at g = u_{ij}:
# We need s_{ik}·t_{kj} = u_{ij} for all k.
# AND no other term s_{ab}·t_{cd} = u_{ij} with (a,b,c,d) ≠ (i,k,k,j).

# So the condition is:
# 1) s_{ik}·t_{kj} = u_{ij}  (product rule)
# 2) s_{ab}·t_{cd} ∉ U  whenever b ≠ c  (no cross-contamination)

# Condition 1 means u_{ij} is determined by any k:
#   u_{ij} = s_{i0}·t_{0j} = s_{i1}·t_{1j} = s_{i2}·t_{2j}

# Let's search for valid (S, T) and derive U.
# Use structure: let S = {s_{ik}} be a 3×3 grid of distinct elements,
# similarly T = {t_{kj}}.

# Given the group structure, try embedding via the Z_2³ and S_3 parts.
# Attempt 1: Use S_3 part for the permutation index and Z_2³ for the other.

# S_3 elements (with trivial Z_2):
s3_in_G = [elem_to_idx[((0,0,0), sigma)] for sigma in S3]
print(f"\nS_3 subgroup indices: {s3_in_G}")

# Z_2³ elements (with identity permutation):
z2_in_G = [elem_to_idx[(eps, (0,1,2))] for eps in Z2_3]
print(f"Z_2³ subgroup indices: {z2_in_G}")

# Try: s_{ik} labeled by (i, k) using a coset structure.
# The natural approach: s_{ik} = g_i · h_k where g_i, h_k ∈ G.
# Then s_{ik}·t_{kj} = g_i · h_k · t_{kj}.
# For this to equal u_{ij} (independent of k):
#   h_k · t_{kj} must be independent of k, i.e., h_k · t_{kj} = v_j for all k.
#   So t_{kj} = h_k⁻¹ · v_j.
# Then u_{ij} = g_i · v_j.
# And S = {g_i · h_k}, T = {h_k⁻¹ · v_j}, U = {g_i · v_j}.
# 
# For |S|=9: need 3 g_i's and 3 h_k's giving 9 distinct products.
# For |T|=9: need 3 h_k⁻¹'s and 3 v_j's giving 9 distinct products.
# For no cross-contamination: 
#   s_{ab}·t_{cd} = g_a · h_b · h_c⁻¹ · v_d
#   This is in U = {g_i · v_j} iff h_b · h_c⁻¹ = e iff b = c.
#   WAIT — that's not quite right. We need:
#   g_a · h_b · h_c⁻¹ · v_d ∈ {g_i · v_j} ⟹ b = c
#   i.e., g_a · (h_b · h_c⁻¹) · v_d = g_i · v_j 
#   i.e., g_i⁻¹ · g_a · (h_b · h_c⁻¹) · v_d · v_j⁻¹ = e
#   i.e., g_i⁻¹ · g_a = (h_b · h_c⁻¹)⁻¹ · v_j · v_d⁻¹
#
# Hmm. The clean condition is:
#   H = {h_0, h_1, h_2} generates a subgroup, and the COSETS
#   {g_i · H}, {H⁻¹ · v_j}, {g_i · v_j} are disjoint in the right way.
#
# SIMPLER FORMULATION:
# Let A = {g_0, g_1, g_2}, B = {h_0, h_1, h_2}, C = {v_0, v_1, v_2}.
# S = A · B = {a·b : a∈A, b∈B}  (9 elements if products distinct)
# T = B⁻¹ · C = {b⁻¹·c : b∈B, c∈C}  (9 elements if products distinct)
# U = A · C = {a·c : a∈A, c∈C}  (9 elements if products distinct)
#
# Product rule: s_{ik}·t_{kj} = (g_i·h_k)·(h_k⁻¹·v_j) = g_i·v_j = u_{ij} ✓
# No cross-contamination: s_{ab}·t_{cd} = g_a·h_b·h_c⁻¹·v_d.
#   This is in U iff g_a·h_b·h_c⁻¹·v_d = g_i·v_j for some i,j.
#   iff h_b·h_c⁻¹ = g_a⁻¹·g_i · v_j·v_d⁻¹
#   For b≠c, h_b·h_c⁻¹ ≠ e. We need this NOT to be in (A⁻¹·A)·(C·C⁻¹) \ {e}.
#
# CONDITION: (B·B⁻¹ \ {e}) ∩ (A⁻¹·A · C·C⁻¹) = ∅
#
# This is the standard "disjointness of product sets" condition.
# It can be stated as: the sets A, B, C satisfy
#   A⁻¹·A ∩ B·(C·C⁻¹)⁻¹ = {e}   (or some variation)
#
# Let's just search directly for triples (A, B, C) of 3-element subsets.

print("\n━━━ Searching for TPP embedding (A·B, B⁻¹·C, A·C structure) ━━━")
print(f"Searching over 3-element subsets A, B, C ⊂ G...")

t0 = time.time()

# Precompute: for each triple (a, b, c) of group elements,
# check if the 9 products a_i·b_k are distinct, etc.

# C(48, 3) = 17296 — manageable but 17296³ is too many.
# Need to be smarter.

# First: enumerate all 3-element subsets containing the identity.
# (WLOG we can translate so e ∈ A, then compensate.)
# Actually WLOG fix g_0 = e (translate A). Then A = {e, a₁, a₂}.

three_subsets_with_e = []
for i in range(N):
    if i == e_idx:
        continue
    for j in range(i + 1, N):
        if j == e_idx:
            continue
        three_subsets_with_e.append((e_idx, i, j))

n_sub_e = len(three_subsets_with_e)
print(f"3-element subsets containing e: {n_sub_e}")

# For all 3-element subsets (not necessarily containing e)
all_three_subsets = list(combinations(range(N), 3))
n_sub = len(all_three_subsets)
print(f"Total 3-element subsets: {n_sub}")

# For each subset, precompute its "product set" and "inverse product set"
def product_set_9(A, B):
    """Return set of all a·b for a∈A, b∈B"""
    prods = set()
    for a in A:
        for b in B:
            prods.add(mul_table[a, b])
    return prods

def all_products_distinct(A, B):
    """Check if all 9 products a·b are distinct"""
    return len(product_set_9(A, B)) == len(A) * len(B)

# The search: fix A containing e, enumerate B and C (any 3-element subsets)
# Check:
# 1) |A·B| = 9 (all products distinct)
# 2) |B⁻¹·C| = 9 
# 3) |A·C| = 9
# 4) No cross-contamination: for b≠c in B, a·b·c'⁻¹·d ∉ A·C

# Simplification: translate so e ∈ A.
# Then work over all B, C.
# But 17296² is ~3×10⁸ — too slow for full enumeration.

# Further fix: e ∈ B too (translate B). Then B = {e, b₁, b₂}.
# And C is free.

# With A = {e, a₁, a₂} and B = {e, b₁, b₂}:
# A·B = {e, b₁, b₂, a₁, a₁b₁, a₁b₂, a₂, a₂b₁, a₂b₂} — need 9 distinct
# B⁻¹·C = {c₀, c₁, c₂, b₁⁻¹c₀, b₁⁻¹c₁, b₁⁻¹c₂, b₂⁻¹c₀, b₂⁻¹c₁, b₂⁻¹c₂}
# A·C = {c₀, c₁, c₂, a₁c₀, a₁c₁, a₁c₂, a₂c₀, a₂c₁, a₂c₂}

# Cross-contamination check: s_{ab}·t_{cd} = a·b·c⁻¹·d ∈ U iff
# a·(b·c⁻¹)·d = a'·d' for some a'∈A, d'∈C.
# With A, B containing e: 
# For b ≠ c: b·c⁻¹ ≠ e. Need: a·(b·c⁻¹)·d ∉ A·C.
# i.e., (b·c⁻¹) ∉ a⁻¹·A·C·d⁻¹ = A⁻¹·A · C·C⁻¹ (using our special forms)
# Actually: a·(b·c⁻¹)·d = a'·d' ⟺ (b·c⁻¹) = a⁻¹·a'·d'·d⁻¹
# So we need: B·B⁻¹ \ {e} is disjoint from A⁻¹·A · C·C⁻¹.
# But A⁻¹·A contains e so A⁻¹·A · C·C⁻¹ ⊃ C·C⁻¹ ⊃ {e}.
# Wait, we need (B·B⁻¹ \ {e}) ∩ (A⁻¹·A · C·C⁻¹) = ∅.
# Hmm but A⁻¹·A · C·C⁻¹ contains e·e = e and also a₁⁻¹·a₂ · c₀·c₁⁻¹ etc.

# This is getting complicated. Let me just do a direct brute-force check
# with some pruning.

# APPROACH: Fix A = {e, a1, a2} (loop over a1 < a2, both ≠ e).
# Fix B = {e, b1, b2} (loop over b1 < b2, both ≠ e).
# Check |A·B| = 9.
# Then for each C = {c0, c1, c2}:
#   Check |B⁻¹·C| = 9 and |A·C| = 9.
#   Check cross-contamination.

# With A, B fixed and containing e: C(47,2) ≈ 1081 choices for each.
# Total: 1081 × 1081 × 17296 ≈ 2×10¹⁰ — still too large.

# Let me reduce further. Fix A = {e, a₁, a₂}.
# For each B = {e, b₁, b₂} with |A·B|=9:
#   Compute the "forbidden set" F = (B·B⁻¹) \ {e} (at most 6 elements)
#   For each C = {c₀, c₁, c₂}:
#     Check |A·C|=9, |B⁻¹·C|=9
#     Check (A⁻¹·A · C·C⁻¹) ∩ F = ∅

# Cost: 1081 A's × (some B's) × 17296 C's with fast checks.
# This might be feasible if we prune aggressively.

# Actually let me think about the size more carefully.
# A choices (containing e): C(47,2) = 1081
# B choices (containing e): C(47,2) = 1081  
# For each (A,B): check |A·B|=9 — this filters heavily. ~1081² / something.
# C choices: C(48,3) = 17296

# Let me just time a single (A,B) check to calibrate.

found_tpp = []
n_checked = 0
t_start = time.time()

# Precompute inverse table (already done above)
# Precompute all products for fast lookup
# mul_table[i,j] already gives product.

# Enumerate A = {e, a1, a2}
for a1 in range(N):
    if a1 == e_idx:
        continue
    for a2 in range(a1 + 1, N):
        if a2 == e_idx:
            continue
        A = (e_idx, a1, a2)
        
        # Precompute A⁻¹·A \ {e}
        AinvA = set()
        for x in A:
            for y in A:
                p = mul_table[inv_table[x], y]
                if p != e_idx:
                    AinvA.add(p)
        
        for b1 in range(N):
            if b1 == e_idx:
                continue
            for b2 in range(b1 + 1, N):
                if b2 == e_idx:
                    continue
                B = (e_idx, b1, b2)
                
                # Check |A·B| = 9
                AB = set()
                for a in A:
                    for b in B:
                        AB.add(mul_table[a, b])
                if len(AB) != 9:
                    continue
                
                # Compute F = B·B⁻¹ \ {e}
                F = set()
                for x in B:
                    for y in B:
                        p = mul_table[x, inv_table[y]]
                        if p != e_idx:
                            F.add(p)
                
                # Compute B⁻¹ elements
                Binv = [inv_table[b] for b in B]
                
                # Now search C
                for c0 in range(N):
                    for c1 in range(c0 + 1, N):
                        for c2 in range(c1 + 1, N):
                            C = (c0, c1, c2)
                            
                            # Check |A·C| = 9
                            AC = set()
                            ok = True
                            for a in A:
                                for c in C:
                                    p = mul_table[a, c]
                                    if p in AC:
                                        ok = False
                                        break
                                    AC.add(p)
                                if not ok:
                                    break
                            if not ok:
                                continue
                            
                            # Check |B⁻¹·C| = 9
                            BinvC = set()
                            for bi in Binv:
                                for c in C:
                                    p = mul_table[bi, c]
                                    if p in BinvC:
                                        ok = False
                                        break
                                    BinvC.add(p)
                                if not ok:
                                    break
                            if not ok:
                                continue
                            
                            # Check cross-contamination:
                            # Need (A⁻¹·A · C·C⁻¹) ∩ F = ∅
                            # Compute C·C⁻¹ \ {e}
                            CCinv = set()
                            for x in C:
                                for y in C:
                                    p = mul_table[x, inv_table[y]]
                                    if p != e_idx:
                                        CCinv.add(p)
                            
                            # Check: for each f ∈ F, is f ∈ AinvA · CCinv ∪ AinvA ∪ CCinv?
                            # Actually need full check: f ∈ AinvA_full · CCinv_full
                            # where AinvA_full includes e and CCinv_full includes e.
                            AinvA_full = AinvA | {e_idx}
                            CCinv_full = CCinv | {e_idx}
                            
                            cross = set()
                            for x in AinvA_full:
                                for y in CCinv_full:
                                    p = mul_table[x, y]
                                    cross.add(p)
                            cross.discard(e_idx)
                            
                            if F & cross:
                                continue
                            
                            # FOUND TPP TRIPLE!
                            found_tpp.append((A, B, C))
                            elapsed = time.time() - t_start
                            print(f"  ✓ FOUND TPP! A={A}, B={B}, C={C}")
                            print(f"    |A·B|=9, |B⁻¹·C|=9, |A·C|=9, no cross-contamination")
                            print(f"    Time: {elapsed:.1f}s")
                            
                            if len(found_tpp) >= 3:
                                break
                        if len(found_tpp) >= 3:
                            break
                    if len(found_tpp) >= 3:
                        break
                
                n_checked += 1
                if n_checked % 1000 == 0:
                    elapsed = time.time() - t_start
                    print(f"  Checked {n_checked} (A,B) pairs, "
                          f"found {len(found_tpp)} TPP triples, "
                          f"time: {elapsed:.1f}s")
                
                if len(found_tpp) >= 3:
                    break
            if len(found_tpp) >= 3:
                break
        if len(found_tpp) >= 3:
            break
    if len(found_tpp) >= 3:
        break

elapsed = time.time() - t_start
print(f"\nSearch complete. Found {len(found_tpp)} TPP triples in {elapsed:.1f}s")
print(f"Checked {n_checked} (A,B) pairs")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Also check using subgroups
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n━━━ Subgroup TPP check ━━━")

# Find all subgroups of order 3 (cyclic, generated by order-3 elements)
order3_elts = [i for i in range(N) if orders[i] == 3]
print(f"Elements of order 3: {len(order3_elts)}")

subgroups_3 = set()
for g in order3_elts:
    g2 = mul_table[g, g]
    sg = frozenset([e_idx, g, g2])
    subgroups_3.add(sg)

print(f"Subgroups of order 3: {len(subgroups_3)}")
subgroups_3 = [tuple(sorted(sg)) for sg in subgroups_3]

# Check all triples of order-3 subgroups for the "subgroup TPP"
# Condition: H₁ ∩ H₂·H₃ = {e} for all cyclic permutations
print(f"Checking {len(subgroups_3)}³ = {len(subgroups_3)**3} subgroup triples...")

n_subgroup_tpp = 0
for H1 in subgroups_3:
    for H2 in subgroups_3:
        H2H3_cache = {}
        for H3 in subgroups_3:
            # H₁ ∩ H₂·H₃ = {e}
            H2H3 = product_set_9(H2, H3) if (H2, H3) not in H2H3_cache else H2H3_cache[(H2, H3)]
            H2H3_cache[(H2, H3)] = H2H3
            
            if set(H1) & H2H3 != {e_idx}:
                continue
            
            # H₂ ∩ H₃·H₁
            H3H1 = product_set_9(H3, H1)
            if set(H2) & H3H1 != {e_idx}:
                continue
            
            # H₃ ∩ H₁·H₂
            H1H2 = product_set_9(H1, H2)
            if set(H3) & H1H2 != {e_idx}:
                continue
            
            n_subgroup_tpp += 1
            if n_subgroup_tpp <= 3:
                print(f"  ✓ Subgroup TPP: H1={H1}, H2={H2}, H3={H3}")

print(f"\nSubgroup TPP triples found: {n_subgroup_tpp}")

if n_subgroup_tpp > 0:
    # For ⟨n,n,n⟩ with subgroups of order n:
    # ω ≤ 3·log_{|G|}(|G|/n) ... actually the formula is different.
    # For the subgroup version: if subgroups of order n satisfy TPP in G of order |G|,
    # then ω(n) ≤ log_n |G|.
    # For n=3, |G|=48: ω ≤ log_3(48) = ln(48)/ln(3) ≈ 3.526
    print(f"Subgroup TPP → ω ≤ log_3(48) = {np.log(48)/np.log(3):.3f}")
    print("(Worse than cubic — subgroups too small relative to group)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Check the SIMULTANEOUS TPP (weaker condition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n━━━ Character-theoretic analysis ━━━")

# Compute character table using class sums
# The character of the regular representation gives all irreps.
# Let's use the class matrices to find character table.

# Character values: for each class C_j and irrep ρ_i,
# χ_i(C_j) = trace(ρ_i(g)) for any g ∈ C_j.

# From the regular representation, the character table can be extracted
# by finding joint eigenvalues of the class sum matrices.

# Let's compute eigenvalues of each class sum restricted to each eigenspace.
# First, find eigenspaces of our random combination matrix.

# Use a cleaner approach: find irreps via character orthogonality.
# The characters span a |classes|-dim space.
# We get them from the class multiplication coefficients.

# Actually, let me just compute the irreps directly by finding
# representations via the known structure.

# Z_2 ≀ S_3 = (Z_2)³ ⋊ S_3
# The irreps are parametrized by pairs of partitions (α, β) with |α|+|β|=3.
# Formula: dim = (3! / (|α|!·|β|!)) · dim(S_α) · dim(S_β)
#   ... actually it's more nuanced.

# For Z_2 ≀ S_3, the irreps are:
# Partition pairs (α,β) with |α|+|β|=3:
# (3,∅), (21,∅), (111,∅), (∅,3), (∅,21), (∅,111)  — |α|+|β|=3, one empty
# (2,1), (11,1), (1,2), (1,11)  — |α|+|β|=3, both non-empty

# Dimensions:
# (3,∅): C(3,3)·1·1 = 1
# (21,∅): C(3,3)·2·1 = 2   wait...
# 
# The dimension formula for Z_2 ≀ S_n irreps labeled by (α,β):
# dim = n! / (|α|!·|β|!) · dim(S^α) · dim(S^β)
# where |α|+|β|=n, S^α is the Specht module of S_{|α|} indexed by α.

# For n=3:
# (3,∅): 3!/(3!·0!) · 1 · 1 = 1
# (21,∅): 3!/(3!·0!) · 2 · 1 = 2  → but 3!/(3!·0!) = 1, so dim = 2
# oh wait, the formula uses multinomial, not C(n,k).
# dim(Z_2 ≀ S_n, (α,β)) = (n choose |α|) · dim(S^α) · dim(S^β)
# For (3,∅): C(3,3) · dim(S^{(3)}) = 1·1 = 1
# For (21,∅): C(3,3) · dim(S^{(21)}) = 1·2 = 2
# For (111,∅): C(3,3) · dim(S^{(111)}) = 1·1 = 1
# For (∅,3): C(3,0) · dim(S^{(3)}) = 1·1 = 1
# For (∅,21): C(3,0) · dim(S^{(21)}) = 1·2 = 2
# For (∅,111): C(3,0) · dim(S^{(111)}) = 1·1 = 1
# For (2,1): C(3,2) · dim(S^{(2)}) · dim(S^{(1)}) = 3·1·1 = 3
# For (11,1): C(3,2) · dim(S^{(11)}) · dim(S^{(1)}) = 3·1·1 = 3
# For (1,2): C(3,1) · dim(S^{(1)}) · dim(S^{(2)}) = 3·1·1 = 3
# For (1,11): C(3,1) · dim(S^{(1)}) · dim(S^{(11)}) = 3·1·1 = 3

# Sum of squares: 1+4+1+1+4+1+9+9+9+9 = 48 ✓

print("Irreps of Z_2 ≀ S_3 (hyperoctahedral group B_3):")
irreps_theory = {
    '(3,∅)': 1, '(21,∅)': 2, '(111,∅)': 1,
    '(∅,3)': 1, '(∅,21)': 2, '(∅,111)': 1,
    '(2,1)': 3, '(11,1)': 3, '(1,2)': 3, '(1,11)': 3
}
for name, dim in sorted(irreps_theory.items(), key=lambda x: x[1]):
    print(f"  {name}: dim = {dim}")
print(f"Sum of dim²: {sum(d**2 for d in irreps_theory.values())}")

# The largest irrep has dim 3. 
# For Cohn-Umans with n=3: we need the matmul tensor to embed
# into the irreps of dimension ≥ 3.
# There are four 3-dimensional irreps.

# The Cohn-Umans ω bound uses:
# ω ≤ inf { τ : Σ_ρ (dim ρ)^τ ≥ n^τ · (stuff) }
# More precisely, for the general embedding:
# Find S, T, U ⊂ G with the matmul structure, then
# ω = inf {τ : possible with these subsets}

# Key insight: the 4 three-dimensional irreps of G decompose:
# C[G] ≅ 1⁴ ⊕ 2² ⊕ 3⁴
# The 3-dim irreps are the natural signed permutation rep and variants.

# For matmul embedding: we need the 3-dim irreps to contain
# the structure of 3×3 matmul. Since G = symmetries of T_matmul,
# the natural rep IS the permutation action on {0,1,2}.

print(f"\nLargest irrep dimension: 3")
print(f"For ⟨3,3,3⟩ matmul: n = 3")
print(f"Need n ≤ max irrep dim? {3 <= 3}: ✓")

# The Cohn-Umans exponent bound via character theory:
# ω(n) ≤ min over groups G with TPP { 3·log_{|G|}(Σ_ρ d_ρ^{ω/3}) }
# This is circular. The non-circular version:
# If G has TPP subsets of sizes q₁, q₂, q₃ for ⟨q₁,q₂,q₃⟩tensor:
# then ω(q₁,q₂,q₃) ≤ log_{|G|}(Σ_ρ d_ρ^{s}) where s satisfies
# q₁^s·q₂^s·q₃^s = |G|^s somehow... 

# Actually the CLEAN Cohn-Umans bound is:
# If ⟨m,n,p⟩ embeds into C[G], then
# ω(m,n,p) ≤ 3·log_{mnp}(|G|)  ... no that's not right either.

# The correct statement:
# If ⟨n,n,n⟩ can be realized in C[G] (via TPP subsets of size n²),
# then exponent ω satisfies: n^ω ≤ Σ_ρ (dim ρ)^{ω/3}
# which gives a bound on ω.

# For G of order 48 with dims {1,1,1,1,2,2,3,3,3,3}:
# n^ω ≤ 4·1^(ω/3) + 2·2^(ω/3) + 4·3^(ω/3)
# 3^ω ≤ 4 + 2·2^(ω/3) + 4·3^(ω/3)

# Let's find the ω that satisfies this (the "support bound")
print("\n━━━ Cohn-Umans support bound for G = Z₂ ≀ S₃ ━━━")

dims = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]

def cu_bound(omega):
    """3^omega vs sum of d_rho^(omega/3)"""
    lhs = 3 ** omega
    rhs = sum(d ** (omega / 3) for d in dims)  # Not right...
    # Actually the bound is: n^omega ≤ sum_rho d_rho^(2*omega/3)
    # No... Let me use the correct formula.
    # From Cohn-Umans 2003, Theorem 4.1:
    # If there exist S,T,U in G satisfying TPP with |S|=|T|=|U|=q,
    # then ω(q,q,q) ≤ log_q(|G|) · (something)
    # 
    # Actually the simplest result is:
    # R(n,n,n) ≤ |G| if ⟨n,n,n⟩ embeds via TPP into C[G]
    # So ω ≤ log_n(|G|)
    #
    # For n=3, |G|=48: ω ≤ log_3(48) = 3.526
    # That's worse than 3 (cubic). Not useful directly.
    #
    # BUT: the real power comes from taking TENSOR POWERS.
    # ⟨n,n,n⟩^⊗k embeds into C[G^k] if ⟨n,n,n⟩ embeds into C[G].
    # Then ω ≤ log_{n^k}(|G|^k) = log_n(|G|).
    # Same bound. Useless for asymptotic improvement this way.
    #
    # The REAL Cohn-Umans gain comes from the SIMULTANEOUS version:
    # Using multiple irreps simultaneously, giving:
    # n^ω ≤ Σ_ρ (multiplicity of n-dim blocks in ρ⊗ρ⊗ρ)
    # ... this is very specific.
    return lhs - rhs

# Direct bound:
omega_direct = np.log(48) / np.log(3)
print(f"Direct bound: ω ≤ log₃(48) = {omega_direct:.3f}")
print(f"  (Only useful if < 3.0; {omega_direct:.3f} > 3.0 → USELESS for direct approach)")

# The simultaneous version (Theorem 5.4 of Cohn-Umans):
# ω < 3τ where τ = log_{|G|}(m) and m comes from simultaneous disjointness.
# If we use ALL four 3-dim irreps:
# Each contributes a 3×3 "block." The capacity is 4 blocks of size 3.
# The CU bound with simultaneous embedding gives:
# n^{ω/3} ≤ Σ (d_i)^{s_i} for some exponents s_i... this is getting circular.

# Let me just report the clean results.

print(f"\n━━━ Final Analysis ━━━")
print(f"G = Z₂ ≀ S₃, order 48")
print(f"Irrep dimensions: {sorted(dims)}")
print(f"Conjugacy classes: {len(classes)}")
print()
print(f"Direct Cohn-Umans bound: ω ≤ log₃(48) = {omega_direct:.3f}")
print(f"  → DOES NOT beat cubic (need < 3.0)")
print()
print(f"The issue: |G| = 48 > 3³ = 27.")
print(f"For ω < 3, need |G| < n³ = 27 for direct approach.")
print(f"G is TOO BIG for direct Cohn-Umans on ⟨3,3,3⟩.")
print()
print(f"Emmy's calculation ω ≤ 1.702 was WRONG because:")
print(f"  She used τ = log_{{|G|}}(n²) = log_48(9) ≈ 0.567")
print(f"  But the correct bound is ω ≤ log_n(|G|) = log_3(48) ≈ 3.53")
print(f"  The τ formula applies to TENSOR POWER asymptotics,")
print(f"  not to direct embedding of ⟨3,3,3⟩.")
print()
print(f"For ω < 3 via Cohn-Umans for ⟨3,3,3⟩ directly:")
print(f"  Need a group G with |G| < 27 satisfying TPP.")
print(f"  Candidates: Z₃×Z₃ (order 9), Z₉ (order 9),")
print(f"              various order-18 groups, etc.")
print()
print(f"Z₃×Z₃ (order 9): ω ≤ log_3(9) = 2.000")
print(f"  IF TPP is satisfied → ω ≤ 2.0 (extraordinary)")
print(f"  But: |Z₃×Z₃| = 9 = n² = minimal possible.")
print(f"  This means S=T=U=G (the whole group). TPP becomes:")
print(f"  s·t·u = e for s,t,u ∈ G ⟹ s=t=u=e.")
print(f"  But: (1,0)·(0,1)·(2,2) = (0,0) in Z₃×Z₃. TPP FAILS.")
print()

# Verify: does Z_3×Z_3 actually fail TPP when S=T=U=G?
print("━━━ Verifying Z₃×Z₃ TPP failure ━━━")
# Z_3×Z_3 elements
Z3Z3 = [(a, b) for a in range(3) for b in range(3)]
n_violations = 0
for s in Z3Z3:
    for t in Z3Z3:
        for u in Z3Z3:
            # s + t + u = (0,0) mod 3
            if (s[0]+t[0]+u[0]) % 3 == 0 and (s[1]+t[1]+u[1]) % 3 == 0:
                if s != (0,0) or t != (0,0) or u != (0,0):
                    n_violations += 1
                    if n_violations <= 3:
                        print(f"  TPP violation: {s}+{t}+{u} = (0,0)")

print(f"  Total TPP violations: {n_violations}")
print(f"  Z₃×Z₃ with S=T=U=G: TPP FAILS ✗")
print()

# For Z_3×Z_3 with SUBSETS (not the whole group):
# Need |S|=|T|=|U|=9 but |G|=9, so S=T=U=G. No room for subsets.
# For matmul ⟨3,3,3⟩ we need |S|·|T|·|U| ≤ |G|^ω/3... 
# Actually we just need |S|=|T|=|U|= n² = 9. But |G|=9.
# So S=T=U=G and TPP fails. Done.

# What about |S|=|T|=|U|=3 (just n, not n²)?
# That would embed ⟨3,3,3⟩ as a DIRECT product, not matmul.
# No, for matmul ⟨n,n,n⟩ the Cohn-Umans framework needs
# |S|·|T|·|U| ≥ n³ to have enough room for the tensor.
# Actually... the original framework embeds M(n) into G
# by choosing S_n ⊂ G. For matrix multiplication ⟨n,n,n⟩,
# one needs n elements for rows and n for columns in each factor.

# Actually I need to re-examine. The Cohn-Umans embedding for
# ⟨n,n,n⟩ needs subsets of size n (NOT n²). Let me re-read.

print("━━━ CORRECTED: Cohn-Umans framework for ⟨n,n,n⟩ ━━━")
print()
print("The Cohn-Umans framework embeds ⟨n,n,n⟩ into C[G] using")
print("three subsets S, T, U ⊂ G with |S|=|T|=|U|=n where:")
print("  - S indexes rows of first matrix")
print("  - T indexes shared dimension")  
print("  - U indexes columns of second matrix")
print("  - (A)_{st} for s∈S, t∈T → f_A = Σ A_{st} · (s·t⁻¹)")
print("  - (B)_{tu} for t∈T, u∈U → f_B = Σ B_{tu} · (t·u⁻¹)")
print("  - (AB)_{su} extracted from f_A * f_B")
print()
print("TPP condition (Cohn-Umans 2003, Def 3.1):")
print("  S·T⁻¹, T·U⁻¹, S·U⁻¹ are 'triangle-free'")
print("  i.e., for s∈S, t∈T, u∈U: s·t⁻¹·t'·u⁻¹ = s'·u'⁻¹")
print("  implies s=s', t=t', u=u'")
print()
print("For n=3, need |S|=|T|=|U|=3.")
print("Then |G| ≥ n = 3 (trivially satisfied).")
print("ω bound: ω ≤ 3·log_n(|G|)/2 ... depends on formulation.")
print()

# OK the Cohn-Umans bound for the DIRECT approach is:
# R(⟨n,n,n⟩) ≤ |G| if ⟨n,n,n⟩ embeds into C[G].
# So for G order 48: R ≤ 48, which is trivially true (R=23).
# 
# The REAL power is asymptotic: using tensor powers.
# ⟨n,n,n⟩^⊗N embeds into C[G^N] or C[G ≀ S_N]
# But: ω ≤ lim_{N→∞} log_n R(⟨n,n,n⟩^⊗N)^{1/N}
# And: R(⟨n,n,n⟩^⊗N) ≤ |G|^N if the embedding works for all N
# So: ω ≤ log_n |G| = 3.53. Still > 3.
#
# The simultaneous version gives improvements by using the
# decomposition into irreps more efficiently.

# For the SIMULTANEOUS approach:
# Given irrep dims d_1,...,d_k:
# ω(n,n,n) ≤ min over valid embeddings of { value depending on dims }

# The key insight: if we can embed ⟨n,n,n⟩ into EACH irrep block
# simultaneously, the complexity is bounded by the sum:
# Σ_ρ d_ρ^{ω/3} ≥ n^{2ω/3}
# For our group: 4·1^{ω/3} + 2·2^{ω/3} + 4·3^{ω/3} ≥ 3^{2ω/3}
# 4 + 2·2^{ω/3} + 4·3^{ω/3} ≥ 9^{ω/3}

from scipy.optimize import brentq

def cu_simultaneous(omega):
    return 4 + 2 * 2**(omega/3) + 4 * 3**(omega/3) - 9**(omega/3)

# Find where this crosses zero
try:
    omega_sim = brentq(cu_simultaneous, 2.0, 10.0)
    print(f"Simultaneous CU bound: 4 + 2·2^(ω/3) + 4·3^(ω/3) = 9^(ω/3)")
    print(f"  Solution: ω ≤ {omega_sim:.3f}")
    if omega_sim < 3:
        print(f"  ← BETTER THAN CUBIC!")
    else:
        print(f"  ← Still worse than cubic")
except:
    print("  Simultaneous bound: no valid solution found")
    # Check values
    for w in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        val = cu_simultaneous(w)
        print(f"    ω={w}: LHS-RHS = {val:.3f}")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print()
print("G = Z₂ ≀ S₃ (order 48) is the symmetry group of T_matmul.")
print()
if found_tpp:
    print("TPP: ✓ SATISFIED")
    print(f"Found {len(found_tpp)} TPP embeddings")
    print(f"Direct bound: R(⟨3,3,3⟩) ≤ 48 (trivially known, R=23)")
    print("Interest: structural (symmetry group realizes matmul)")
else:
    print("TPP: search completed (see results above)")

print()
print("For practical 3×3 improvement via Cohn-Umans:")
print("  Need group with |G| < 23 satisfying TPP for ⟨3,3,3⟩")
print("  Minimum possible: |G| = 19 (if R=19 conjecture is true)")
print()
print("For asymptotic ω improvement:")  
print("  G = Z₂ ≀ S₃ gives ω ≤ 3.53 (worse than cubic)")
print("  Need either: smaller group, or simultaneous embedding")
print("  using irrep decomposition to beat the direct bound.")
