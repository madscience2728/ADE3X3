"""
Explicit rank-21 decomposition of T_matmul(3) from Z_7 ⋊ Z_3.

The Cohn-Umans framework: for <n,n,n> matmul, find S, S', S'' ⊂ G
with |S|=|S'|=|S''|=n satisfying TPP. Then R(<n,n,n>) ≤ |G|.

For n=3, |G|=21: need |S|=|S'|=|S''|=3.
"""

import numpy as np
from itertools import product, combinations

# ================================================================
# Build Z_7 ⋊ Z_3
# ================================================================
def mul(e1, e2):
    a1, b1 = e1; a2, b2 = e2
    return ((a1 + pow(2, b1, 7) * a2) % 7, (b1 + b2) % 3)

def inv(e):
    a, b = e
    b_inv = (-b) % 3
    return ((-pow(2, b_inv, 7) * a) % 7, b_inv)

G = [(a, b) for a in range(7) for b in range(3)]
assert all(mul(g, inv(g)) == (0,0) for g in G)

# ================================================================
# Section 1: Find TPP subsets with |S|=|S'|=|S''|=3
# ================================================================
print("=" * 70)
print("SEARCHING FOR TPP-3 SUBSETS IN Z_7 ⋊ Z_3")
print("=" * 70)

# TPP condition (Cohn-Umans 2003, Definition 2.2):
# For S, S', S'' ⊂ G with |S|=|S'|=|S''|=n:
# If s1·s1'·s1'' = s2·s2'·s2'' (si∈S, si'∈S', si''∈S'')
# then s1=s2, s1'=s2', s1''=s2''.
# Equivalently: the map S × S' × S'' → G is injective.
# This requires |S|·|S'|·|S''| ≤ |G|, i.e., 27 ≤ 21. FAILS!

# So DIRECT TPP with |S|=3 is IMPOSSIBLE in a group of order 21.
# 3³ = 27 > 21.

print("\nDirect TPP with |S|=|S'|=|S''|=3:")
print(f"  Need 3·3·3 = 27 distinct products in |G| = 21. IMPOSSIBLE.")
print(f"  Direct TPP requires |G| ≥ n³ = 27.")

# ================================================================
# Section 2: What does "TPP-9" actually mean?
# ================================================================
print("\n" + "=" * 70)
print("WHAT IS TPP-9?")
print("=" * 70)

# The SIMULTANEOUS triple product property (STPP) is different.
# Cohn-Umans (2005): use CHARACTERS, not triple products.
# The bound comes from embedding M_n into ⊕_ρ M_{d_ρ}(C)
# (the Wedderburn decomposition of C[G]).
# 
# For Z_7⋊Z_3: C[G] ≅ C ⊕ C ⊕ C ⊕ M_3(C) ⊕ M_3(C)
#
# To embed <n,n,n> = <3,3,3> matmul:
# Need to embed M_3 into C[G] as a SUBALGEBRA.
# The M_3(C) block has dimension 9 = 3², which is exactly M_3!
# So embed M_3 into ONE of the two M_3(C) Wedderburn blocks.
# 
# Then R(<3,3,3>) ≤ Σ_ρ d_ρ · (multiplicity of ρ in the support)
# But the simple bound is R ≤ |G| = 21.
# The better bound uses the "support rank" in each block.

print("The Cohn-Umans framework for Z_7⋊Z_3:")
print("  C[G] ≅ C ⊕ C ⊕ C ⊕ M_3(C) ⊕ M_3(C)")
print()
print("  Embed M_3 into one M_3(C) Wedderburn block.")
print("  This gives R(<3,3,3>) ≤ |G| = 21.")
print()
print("  BUT: the decomposition lives in C[G], not directly in C^9⊗C^9⊗C^9.")
print("  Extracting an explicit 21-term decomposition requires:")
print("  1. Build the Wedderburn isomorphism Φ: C[G] → ⊕ M_{d_ρ}")
print("  2. Identify the M_3 block")
print("  3. Project each group element g onto its contribution")
print("  4. Extract the 9×9×9 tensor from the 21-dim group algebra")

# ================================================================
# Section 3: Build Wedderburn isomorphism explicitly
# ================================================================
print("\n" + "=" * 70)
print("SECTION 3: EXPLICIT WEDDERBURN ISOMORPHISM")
print("=" * 70)

zeta7 = np.exp(2j * np.pi / 7)
omega3 = np.exp(2j * np.pi / 3)

# Irreps of Z_7 ⋊ Z_3:
# 1-dim: χ_k(a^i b^j) = ω_3^{kj} for k=0,1,2
# 3-dim ρ: induced from χ_1 of Z_7
#   ρ(a) = diag(ζ_7, ζ_7^2, ζ_7^4)
#   ρ(b) = cyclic permutation (0→1→2→0)
# 3-dim ρ̄: complex conjugate (induced from χ_6 = χ_1^{-1})

rho_a = np.diag([zeta7, zeta7**2, zeta7**4])
rho_b = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)

def rho_elem(g):
    a, b = g
    return np.linalg.matrix_power(rho_a, a) @ np.linalg.matrix_power(rho_b, b)

def rho_bar_elem(g):
    return np.conj(rho_elem(g))

# The Wedderburn isomorphism Φ: C[G] → C³ ⊕ M_3(C) ⊕ M_3(C)
# For f = Σ c_g · g ∈ C[G]:
#   Φ(f)_χk = Σ c_g · χ_k(g)     (1-dim component)
#   Φ(f)_ρ  = Σ c_g · ρ(g)       (3-dim component, a 3×3 matrix)
#   Φ(f)_ρ̄  = Σ c_g · ρ̄(g)       (conjugate 3-dim component)

# The INVERSE: given target A ∈ M_3(C) in the ρ-block:
#   f = (d/|G|) Σ_g tr(A · ρ(g)^{-1}) · g = (3/21) Σ_g tr(A · ρ(g^{-1})) · g
# This puts A in the ρ-block and 0 in all other blocks.

def embed_in_rho_block(A):
    """Embed A ∈ M_3(C) into C[G] via the ρ-Wedderburn block.
    Returns coefficients c_g for g ∈ G."""
    coeffs = {}
    for g in G:
        rg_inv = rho_elem(inv(g))
        coeffs[g] = (3.0 / 21.0) * np.trace(A @ rg_inv)
    return coeffs

def group_algebra_multiply(f, h):
    """Multiply two elements f, h in C[G]. Returns coefficients."""
    result = {g: 0.0 + 0j for g in G}
    for g1, c1 in f.items():
        if abs(c1) < 1e-15:
            continue
        for g2, c2 in h.items():
            if abs(c2) < 1e-15:
                continue
            prod = mul(g1, g2)
            result[prod] += c1 * c2
    return result

def extract_rho_block(f):
    """Extract the ρ-block from f ∈ C[G]. Returns a 3×3 matrix."""
    A = np.zeros((3, 3), dtype=complex)
    for g, c in f.items():
        A += c * rho_elem(g)
    return A

# Verify the embedding works: embed e_{ij}, multiply, extract
print("Verifying Wedderburn embedding of M_3 into C[Z_7⋊Z_3]...")

e = np.zeros((3, 3))
errors = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                E_ij = np.zeros((3, 3)); E_ij[i, j] = 1
                E_kl = np.zeros((3, 3)); E_kl[k, l] = 1
                
                f_ij = embed_in_rho_block(E_ij)
                f_kl = embed_in_rho_block(E_kl)
                prod = group_algebra_multiply(f_ij, f_kl)
                result = extract_rho_block(prod)
                
                # Should equal δ_{jk} E_{il}
                expected = np.zeros((3, 3), dtype=complex)
                if j == k:
                    expected[i, l] = 1.0
                
                errors.append(np.max(np.abs(result - expected)))

max_err = max(errors)
print(f"  Max error in M_3 multiplication: {max_err:.2e}")
if max_err < 1e-10:
    print("  *** M_3 EMBEDS CORRECTLY into ρ-block of C[G] ***")

# ================================================================
# Section 4: Extract the rank-21 decomposition
# ================================================================
print("\n" + "=" * 70)
print("SECTION 4: EXTRACTING RANK-21 DECOMPOSITION")
print("=" * 70)

# The multiplication tensor of M_3 through the group algebra:
# T_matmul[ij, kl, mn] = Σ_g u_g[ij] · v_g[kl] · w_g[mn]
#
# where for each g ∈ G:
#   u_g[ij] = (3/21) · ρ(g^{-1})_{ji}  (coefficient of g in embed(E_ij))
#   v_g[kl] = (3/21) · ρ(g^{-1})_{lk}  (coefficient of g in embed(E_kl))
#   w_g[mn] = from the product extraction
#
# The product f_ij · f_kl has coefficient at group element h:
#   (f_ij * f_kl)_h = Σ_{g1·g2=h} (f_ij)_{g1} · (f_kl)_{g2}
# And we extract the ρ-block:
#   result_{mn} = Σ_h (f_ij * f_kl)_h · ρ(h)_{mn}
#
# The DIRECT approach: each g ∈ G gives a rank-1 contribution.
# The group algebra multiplication T_G has rank |G|.
# T_matmul is obtained by COMPOSING:
#   embed → multiply in C[G] → extract
# This composition preserves rank bound: R(T_matmul) ≤ R(T_G) = |G| = 21.
#
# Explicitly: T_matmul[ij, kl, mn] = Σ_g Σ_{g1·g2=g} u_{g1}[ij] · v_{g2}[kl] · w_g[mn]
# Regrouping: = Σ_{g1} Σ_{g2} u_{g1}[ij] · v_{g2}[kl] · w_{g1·g2}[mn]
# This is NOT rank 21 yet — it's rank |G|² in this form.
#
# The correct rank-|G| form uses the REGULAR REPRESENTATION multiplication:
# In C[G], the multiplication by a fixed element g is a linear map.
# T_{C[G]}[g1, g2, g3] = δ(g1·g2 = g3)
# This tensor has rank |G| (each g contributes one rank-1 term).
#
# T_matmul = (embed ⊗ embed ⊗ extract) ∘ T_{C[G]}
# So T_matmul[ij, kl, mn] = Σ_g L[ij,g] · R[kl,g] · E[mn, g^{-1}...] ... 
#
# Let me just compute it directly.

# For each g ∈ G, the rank-1 term is:
# u_g ∈ C^9, v_g ∈ C^9, w_g ∈ C^9
# 
# The group algebra multiplication tensor: T_G[(g1), (g2), (g3)] = δ(g1·g2 = g3)
# = δ(g3^{-1}·g1·g2 = e) = δ(g2 = g1^{-1}·g3)
#
# As a rank-|G| decomposition: T_G = Σ_g e_g ⊗ e_g ⊗ e_g 
# No, that's wrong. T_G[g1,g2,g3] = δ(g1·g2·g3^{-1}=e).
# Using left regular rep: T_G = Σ_g (e_g) ⊗ (left-mult-by-g^{-1}) ⊗ (e_g)
# Actually for the "structure constants" C[G]:
# The bilinear map μ: C[G] × C[G] → C[G], μ(g1, g2) = g1·g2
# In basis {e_g}: μ(e_{g1}, e_{g2}) = e_{g1·g2}
# Tensor: T_μ[g1, g2, g3] = δ(g1·g2 = g3)
#
# Rank-|G| decomposition of T_μ:
# T_μ = Σ_{h ∈ G} u_h ⊗ v_h ⊗ w_h
# where u_h[g1] = δ(g1=h), v_h[g2] = 1 for all g2 ... no.
#
# Actually T_μ[g1,g2,g3] = δ(g1·g2=g3) has rank |G| because:
# Fix any g3. The "slice" T_μ[:,:,g3] is the indicator of {(g1,g2): g1·g2=g3},
# which is a permutation matrix. Each permutation matrix has rank 1? No, rank |G|.
# Hmm, no. A permutation matrix is full rank.
#
# Wait: T_μ is a 3-tensor, |G|×|G|×|G|. Its rank is the minimum R such that
# T = Σ_{r=1}^R u_r ⊗ v_r ⊗ w_r.
# For a group of order n, the rank of the group algebra multiplication tensor
# equals the rank of <n,n,n>? No, it equals n (trivially, since we can write
# it as T = Σ_g e_g ⊗ ... actually let me think again).
#
# T_μ[g1, g2, g3] = δ(g1·g2 = g3)
# Rewrite: = Σ_h δ(g1=h) · δ(g2=h^{-1}·g3)  ... sum over h doesn't help.
# Actually: T_μ[g1,g2,g3] = Σ_h δ(g1=h) · δ(g2 = h^{-1}g3) · 1_{g3}
# This doesn't factor.
#
# The rank of T_μ for G abelian of order n is n (by DFT diagonalization).
# For non-abelian... it's related to Σ d_ρ^3 or Σ R(M_{d_ρ}).
#
# For the Cohn-Umans approach, the key insight is different:
# We DON'T decompose T_G directly. Instead we use the EMBEDDING.

# Let me try the most direct approach: brute-force compute T_matmul
# through the group algebra and express it as rank-1 terms.

# The embedding maps: 
# φ_L: C^{3×3} → C^{|G|}: φ_L(E_{ij}) has coefficient (3/21)·ρ(g^{-1})_{ji} at g
# φ_R: C^{3×3} → C^{|G|}: φ_R(E_{kl}) has coefficient (3/21)·ρ(g^{-1})_{lk} at g
# π: C^{|G|} → C^{3×3}: π(Σ c_g g) = Σ c_g ρ(g)
#
# T_matmul[ij, kl, mn] = π(φ_L(E_{ij}) · φ_R(E_{kl}))_{mn}
# = Σ_h [Σ_{g1·g2=h} φ_L(E_ij)_{g1} · φ_R(E_kl)_{g2}] · ρ(h)_{mn}
# = Σ_{g1,g2} φ_L(E_ij)_{g1} · φ_R(E_kl)_{g2} · ρ(g1·g2)_{mn}
# = Σ_{g1,g2} (3/21)²  ρ(g1^{-1})_{ji} · ρ(g2^{-1})_{lk} · ρ(g1·g2)_{mn}
#
# Now use: ρ(g1·g2) = ρ(g1)·ρ(g2), so ρ(g1·g2)_{mn} = Σ_p ρ(g1)_{mp} ρ(g2)_{pn}
# = (3/21)² Σ_{g1,g2} Σ_p ρ(g1^{-1})_{ji} ρ(g1)_{mp} · ρ(g2^{-1})_{lk} ρ(g2)_{pn}
#
# By Schur orthogonality: Σ_g ρ(g^{-1})_{ji} ρ(g)_{mp} = (|G|/d) δ_{jm} δ_{ip}
# where d=3, |G|=21.
# So = (3/21)² · Σ_p (21/3)δ_{jm}δ_{ip} · (21/3)δ_{lp}δ_{kn}
# = (9/441) · (49) · Σ_p δ_{jm}δ_{ip}δ_{lp}δ_{kn}
# = (9·49/441) · δ_{jm} δ_{il} δ_{kn}
# = (441/441) · δ_{jm} δ_{il} δ_{kn}
# = δ_{jm} δ_{il} δ_{kn}
#
# Wait — that gives T[ij, kl, mn] = δ_{jm} δ_{il} δ_{kn}
# With indexing: ij = 3i+j, kl = 3k+l, mn = 3m+n:
# T[3i+j, 3k+l, 3m+n] = δ_{jm} δ_{il} δ_{kn}
#
# But T_matmul[3r+s, 3s+u, 3r+u] = 1.
# So T[3i+j, 3k+l, 3m+n] = 1 iff (i,j,k,l,m,n) satisfies:
#   r=i, s=j, 3s+u = 3k+l → s=k, u=l, 3r+u = 3m+n → r=m, u=n
# So T_matmul[3i+j, 3k+l, 3m+n] = δ_{jk} δ_{im} δ_{ln} 
#
# And from Schur: T[3i+j, 3k+l, 3m+n] = δ_{jm} δ_{il} δ_{kn}
#
# These are DIFFERENT TENSORS! 
# matmul: δ_{jk} δ_{im} δ_{ln}
# Schur:  δ_{jm} δ_{il} δ_{kn}
#
# They differ by a permutation of indices! Both have the same rank.
# matmul has (j=k, i=m, l=n) → T[3i+j, 3j+l, 3i+l] = 1
# Schur has  (j=m, i=l, k=n) → T[3i+j, 3k+i, 3j+k] = 1
# These are related by permuting (r,s,u) → some permutation.
# Let me check: in matmul, the nonzeros are at (3r+s, 3s+u, 3r+u).
# In Schur: (3i+j, 3k+i, 3j+k) - set r=i, s=j, then third index = 3s+k = 3j+k
# and second = 3k+r = 3k+i. Hmm, this is T[3r+s, 3k+r, 3s+k].
# Compare with matmul: T[3r+s, 3s+u, 3r+u].
# Schur: first idx 3r+s ✓, second 3k+r (should be 3s+u → u=r, k=s? NO)
# They're related by a specific permutation of the matrix indices.
# Both tensors encode matrix multiplication, just with different
# ordering conventions. Same rank.

print("SCHUR ORTHOGONALITY CALCULATION:")
print("-" * 40)
print()
print("T_embed[ij, kl, mn]")
print("  = (3/21)² Σ_{g1,g2} ρ(g1⁻¹)_{ji} · ρ(g2⁻¹)_{lk} · ρ(g1·g2)_{mn}")
print()
print("Using Schur orthogonality Σ_g ρ(g⁻¹)_{αβ} ρ(g)_{γδ} = (|G|/d)·δ_{αγ}δ_{βδ}:")
print()
print("  = δ_{jm} · δ_{il} · δ_{kn}")
print()
print("This is the matrix multiplication tensor (up to index relabeling)!")
print("  Standard: T[3r+s, 3s+u, 3r+u] = 1  (C=AB convention)")
print("  Schur:    T[3i+j, 3k+i, 3j+k] = 1  (equivalent under S_3 action)")
print()

# Verify numerically
T_matmul = np.zeros((9, 9, 9), dtype=int)
T_schur = np.zeros((9, 9, 9), dtype=int)
for r, s, u in product(range(3), repeat=3):
    T_matmul[3*r+s, 3*s+u, 3*r+u] = 1
    T_schur[3*r+s, 3*u+r, 3*s+u] = 1  # the Schur version

print(f"T_matmul nnz = {T_matmul.sum()}, T_schur nnz = {T_schur.sum()}")
print(f"Same tensor? {np.array_equal(T_matmul, T_schur)}")

# They're isomorphic (same rank) but not identical.
# Find the permutation: T_schur[a,b,c] = T_matmul[a, perm(b), perm(c)]

# T_matmul[3i+j, 3j+l, 3i+l]: (i,j) → (j,l) → (i,l)
# T_schur[3i+j, 3k+i, 3j+k]:  (i,j) → (k,i) → (j,k)
# In second slot: matmul has 3s+u (s=j, u=l), schur has 3k+i (k,i)
# In third slot: matmul has 3r+u (r=i, u=l), schur has 3s+k (s=j, k)
# Mapping: second slot: (s,u) → (k,i) means swap+rename
# Schur's second slot index is 3k+r where r=i, and matmul's is 3s+u.
# With s=j, u=l: schur has (k,r=i) in slot 2 vs matmul (j,l).

# Actually let's just check: is there a permutation matrix P such that
# T_schur[:,P,:] = T_matmul or similar?
# Slot permutation: T_schur[a, b, c] = T_matmul[a, c, b] ?
perm_check = np.array_equal(T_schur, T_matmul.transpose(0, 2, 1))
print(f"T_schur = T_matmul transposed in slots 2,3? {perm_check}")

# Try all 6 permutations of the 3 modes
for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
    if np.array_equal(T_schur, T_matmul.transpose(p)):
        print(f"T_schur = T_matmul.transpose{p}")
        break

# Also try with index permutations within each mode
# The 9-dim index 3a+b can be permuted to 3b+a via a permutation P_swap
P_swap = np.zeros((9, 9), dtype=int)
for a in range(3):
    for b in range(3):
        P_swap[3*a+b, 3*b+a] = 1

# T_schur[i,j,k] = T_matmul[P@i, Q@j, R@k] for some permutation matrices
# Try various combos
found = False
for transpose_perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
    for swaps in product([False, True], repeat=3):
        T_test = T_matmul.transpose(transpose_perm).copy()
        # Apply index swaps within modes
        if swaps[0]:
            T_test = np.einsum('ij,jkl->ikl', P_swap, T_test)
        if swaps[1]:
            T_test = np.einsum('ijk,lj->ilk', T_test, P_swap)  
        if swaps[2]:
            T_test = np.einsum('ijk,lk->ijl', T_test, P_swap)
        if np.array_equal(T_schur, T_test):
            print(f"  Found: transpose{transpose_perm}, swaps={swaps}")
            found = True
            break
    if found:
        break

# ================================================================
# Section 5: The ACTUAL rank-21 decomposition
# ================================================================
print("\n" + "=" * 70)
print("SECTION 5: THE RANK-21 DECOMPOSITION")
print("=" * 70)

# From the Schur orthogonality calculation:
# T_schur[3i+j, 3k+i, 3j+k] = (3/21)² Σ_{g1,g2} ρ(g1⁻¹)_{ji}·ρ(g2⁻¹)_{ik}·ρ(g1g2)_{jk}
#
# Wait, I need to redo this more carefully.
# T[ij, kl, mn] = Σ_h [Σ_{g1g2=h} embed_L(E_ij)_{g1} · embed_R(E_kl)_{g2}] · ρ(h)_{mn}
#
# But there's a subtlety: the LEFT and RIGHT embeddings use the SAME ρ.
# embed_L(A) = (d/|G|) Σ_g tr(A·ρ(g⁻¹))·g = (3/21) Σ_g Σ_{αβ} A_{αβ}ρ(g⁻¹)_{βα} · g
# So embed_L(E_{ij})_g = (3/21) ρ(g⁻¹)_{ji}
# Similarly embed_R(E_{kl})_g = (3/21) ρ(g⁻¹)_{lk}
#
# Now: the rank-21 decomposition comes from writing the C[G] multiplication
# NOT as a double sum, but as a SINGLE sum over group elements using the 
# regular representation.
#
# Key: In C[G], multiplication by a basis element e_h from the LEFT is:
# L_h: C[G] → C[G], (L_h f)_g = f_{h⁻¹g}
# So (f · f')_g = Σ_h f_h · f'_{h⁻¹g}
#
# The bilinear map μ(f, f')_g = Σ_h f_h · f'_{h⁻¹g} can be written as:
# μ(f, f') = Σ_h f_h · (L_h f')
# This expresses f·f' as a sum of |G| rank-1 terms in C[G]:
#   μ = Σ_{h ∈ G} δ_h ⊗ L_h
# where δ_h picks out the h-coefficient.
#
# Composing with embed and extract:
# T_matmul = extract ∘ μ ∘ (embed_L ⊗ embed_R)
# = extract ∘ [Σ_h δ_h ⊗ L_h] ∘ (embed_L ⊗ embed_R)
# = Σ_h [δ_h ∘ embed_L] ⊗ [extract ∘ L_h ∘ embed_R]
#
# Hmm, this gives a rank-21 decomposition of a 9 × (9→9) map, 
# i.e., effectively a matrix, not a 3-tensor.
#
# For the 3-tensor: T_matmul[ij, kl, mn] = Σ_h u_h[ij] · M_h[kl, mn]
# This is a rank-21 MATRIX decomposition of the "unfolded" tensor.
# But tensor rank and matrix rank are different.
#
# THE CATCH: The rank-|G| decomposition of the GROUP ALGEBRA multiplication
# tensor T_{C[G]} gives R(T_{C[G]}) = |G|. But the COMPOSITION
# T_matmul = Φ ∘ T_{C[G]} ∘ Ψ can only INCREASE rank, not decrease it!
# So R(T_matmul) ≤ R(T_{C[G]}) IS WRONG in general.
#
# What IS true: R(T_matmul) ≤ Σ_ρ d_ρ · R(<d_ρ, d_ρ, d_ρ>)
# For Z_7⋊Z_3: Σ = 1·1 + 1·1 + 1·1 + 3·R(<3,3,3>) + 3·R(<3,3,3>)
# = 3 + 6·R(<3,3,3>)
# This is CIRCULAR for bounding R(<3,3,3>).
#
# The CORRECT Cohn-Umans bound uses the SUPPORT RANK:
# R(<n,n,n>) ≤ |support of the embedding| ≤ |G|
# IF there exist subsets S,S',S'' with TPP and |S|=|S'|=|S''|=n.
# Since TPP requires n³ ≤ |G| and 27 > 21, this FAILS.

print("CRITICAL REALIZATION:")
print("-" * 40)
print()
print("Direct TPP for <3,3,3> requires |S|=3 and n³=27 ≤ |G|.")
print("But |Z_7⋊Z_3| = 21 < 27. So DIRECT R ≤ 21 via Cohn-Umans is IMPOSSIBLE.")
print()
print("What 'TPP-9 verified' actually means:")
print("  The SIMULTANEOUS TPP (Cohn-Umans-Umans 2005) gives an ω bound")
print("  via the laser method / asymptotic sum inequality.")
print("  It does NOT give R(T_matmul(3)) ≤ 21.")
print()
print("The Z_7⋊Z_3 result gives: ω ≤ 2.771 (via laser method)")
print("This is an ASYMPTOTIC bound, not a finite rank bound.")
print()
print("CORRECT STATUS:")
print("  R(T_matmul(3)) ≤ 23  (Smirnov 2013, explicit construction)")
print("  R(T_matmul(3)) ≥ 19  (lower bound)")
print("  ω ≤ 2.771            (from Z_7⋊Z_3 STPP, asymptotic)")
print("  ω ≤ 2.373            (best known, from other techniques)")
print()
print("There IS no rank-21 decomposition of T_matmul(3)!")
print("Or rather: TPP does not prove one exists.")

# ================================================================
# Section 6: But CAN we still extract a decomposition from the irrep?
# ================================================================
print("\n" + "=" * 70)
print("SECTION 6: WHAT THE IRREP DOES GIVE")
print("=" * 70)

# The Schur orthogonality argument showed:
# (3/21)² Σ_{g1,g2} ρ(g1⁻¹)_{ji}·ρ(g2⁻¹)_{lk}·[ρ(g1)ρ(g2)]_{mn}
# = (9/441)·Σ_p (|G|/d)²·δ_{jm}δ_{ip}·δ_{lp}δ_{kn}
# Wait let me redo from scratch.
#
# Σ_{g1} ρ(g1⁻¹)_{ji} · ρ(g1)_{mp} = (|G|/d)·δ_{jm}δ_{ip}
# Σ_{g2} ρ(g2⁻¹)_{lk} · ρ(g2)_{pn} = (|G|/d)·δ_{lp}δ_{kn}  ... wait, wrong indices
# No: Σ_g ρ(g⁻¹)_{αβ}·ρ(g)_{γδ} = Σ_g ρ(g)^*_{βα}·ρ(g)_{γδ} 
# By Schur orthogonality for unitary irreps:
# Σ_g ρ(g)^*_{βα}·ρ(g)_{γδ} = (|G|/d)·δ_{αδ}δ_{βγ}  ... standard form
#
# Hmm, but ρ(g⁻¹) = ρ(g)^{-1} = ρ(g)^* (for unitary rep).
# So ρ(g⁻¹)_{ji} = ρ(g)^*_{ji} = conj(ρ(g)_{ji}) ... 
# But our ρ is unitary? Let's check.

# Check unitarity
for g in G[:5]:
    rg = rho_elem(g)
    should_be_I = rg @ rg.conj().T
    if not np.allclose(should_be_I, np.eye(3)):
        print(f"  ρ({g}) is NOT unitary! ρρ* = \n{should_be_I}")
        break
else:
    print("ρ is unitary (checked 5 elements)")

# For unitary ρ: ρ(g⁻¹) = ρ(g)^†
# Schur: Σ_g ρ(g)^†_{ji} · ρ(g)_{mp} = Σ_g conj(ρ(g)_{ij}) · ρ(g)_{mp}
# = (|G|/d) δ_{im} δ_{jp}  (standard Schur orthogonality)
#
# The double sum becomes: 
# Σ_{g1,g2} ρ(g1⁻¹)_{ji}·ρ(g2⁻¹)_{lk}·ρ(g1g2)_{mn}
# = Σ_{g1,g2} ρ(g1⁻¹)_{ji}·ρ(g2⁻¹)_{lk}·Σ_p ρ(g1)_{mp}·ρ(g2)_{pn}
# = Σ_p [Σ_{g1} ρ(g1⁻¹)_{ji}·ρ(g1)_{mp}] · [Σ_{g2} ρ(g2⁻¹)_{lk}·ρ(g2)_{pn}]
# = Σ_p [(|G|/d)δ_{im}δ_{jp}] · [(|G|/d)δ_{kn}δ_{lp}]
# = (|G|/d)² · δ_{im}·δ_{kn}·δ_{jl}  (summing over p: p=j and p=l, so j=l)
#
# So: T[ij,kl,mn] = (d/|G|)² · (|G|/d)² · δ_{im}δ_{kn}δ_{jl} = δ_{im}δ_{jl}δ_{kn}
#
# Mapping: (i,j,k,l,m,n) → T[3i+j, 3k+l, 3m+n] = 1 iff i=m, j=l, k=n
# Nonzeros: (3i+j, 3k+j, 3i+k) for all i,j,k ∈ {0,1,2}
# Compare T_matmul: (3r+s, 3s+u, 3r+u)
# Set i=r, j=s, k=u: get (3r+s, 3u+s, 3r+u) vs (3r+s, 3s+u, 3r+u)
# Second slot differs: 3u+s vs 3s+u. These are equal iff s=u (not generally).
# So the Schur reconstruction gives a DIFFERENT (but isomorphic) tensor.

# Let's verify the Schur identity numerically
T_schur2 = np.zeros((9, 9, 9), dtype=complex)
scale = (3.0 / 21.0) ** 2
for g1 in G:
    for g2 in G:
        rg1_inv = rho_elem(inv(g1))
        rg2_inv = rho_elem(inv(g2))
        rg12 = rho_elem(mul(g1, g2))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                T_schur2[3*i+j, 3*k+l, 3*m+n] += (
                                    scale * rg1_inv[j,i] * rg2_inv[l,k] * rg12[m,n]
                                )

T_expected = np.zeros((9, 9, 9))
for i,j,k in product(range(3), repeat=3):
    # δ_{im}δ_{jl}δ_{kn}
    T_expected[3*i+j, 3*k+j, 3*i+k] = 1  # m=i, l=j, n=k

err = np.max(np.abs(T_schur2 - T_expected))
print(f"\nSchur orthogonality verification:")
print(f"  max|T_computed - δ_{{im}}δ_{{jl}}δ_{{kn}}| = {err:.2e}")
if err < 1e-10:
    print("  *** VERIFIED: double sum gives δ_{im}δ_{jl}δ_{kn} ***")

# Now: can we convert this 21² -term sum into a 21-term sum?
# The double sum Σ_{g1,g2} f(g1)·f(g2)·f(g1g2) is NOT a 21-term tensor decomposition.
# It's a 21²=441 term sum that COLLAPSES via Schur.
# To get a 21-term decomposition, we'd need to sum over ONE variable.
#
# Substituting h = g1·g2 (so g2 = g1⁻¹h):
# = Σ_{g1,h} ρ(g1⁻¹)_{ji}·ρ(h⁻¹g1)_{lk}·ρ(h)_{mn} · scale
# = Σ_h ρ(h)_{mn} · [Σ_{g1} ρ(g1⁻¹)_{ji}·ρ(h⁻¹g1)_{lk}] · scale
#
# Inner sum: Σ_{g1} ρ(g1⁻¹)_{ji}·ρ(h⁻¹g1)_{lk}
# = Σ_{g1} ρ(g1)^†_{ji}·[ρ(h⁻¹)ρ(g1)]_{lk}
# = Σ_{g1} conj(ρ(g1)_{ij})·Σ_p ρ(h⁻¹)_{lp}·ρ(g1)_{pk}
# = Σ_p ρ(h⁻¹)_{lp}·[Σ_{g1} conj(ρ(g1)_{ij})·ρ(g1)_{pk}]
# = Σ_p ρ(h⁻¹)_{lp}·(|G|/d)·δ_{ip}δ_{jk}   ... wait, that's not right
# Schur: Σ_g conj(ρ(g)_{ij})·ρ(g)_{pk} = (|G|/d)·δ_{ip}δ_{jk}
# So = Σ_p ρ(h⁻¹)_{lp}·(|G|/d)·δ_{ip}δ_{jk} = (|G|/d)·δ_{jk}·ρ(h⁻¹)_{li}
#
# Putting back:
# T[ij,kl,mn] = scale · Σ_h ρ(h)_{mn} · (|G|/d)·δ_{jk}·ρ(h⁻¹)_{li}
# = (d/|G|)² · (|G|/d) · δ_{jk} · Σ_h ρ(h)_{mn}·ρ(h⁻¹)_{li}
# = (d/|G|) · δ_{jk} · Σ_h ρ(h)_{mn}·ρ(h)^†_{li}
# = (d/|G|) · δ_{jk} · Σ_h ρ(h)_{mn}·conj(ρ(h)_{il})
# = (d/|G|) · δ_{jk} · (|G|/d)·δ_{mi}δ_{nl}  (Schur again!)
# = δ_{jk}·δ_{mi}·δ_{nl}
# Which is T_matmul[3i+j, 3k+l, 3m+n] = δ_{jk}δ_{im}δ_{ln} ✓
#
# BUT: the 21-term form is: 
# T[ij,kl,mn] = (d/|G|) · δ_{jk} · Σ_h ρ(h)_{mn} · conj(ρ(h)_{il})
#
# This is (d/|G|) Σ_h [δ_{jk} · conj(ρ(h)_{il})] · ρ(h)_{mn}
# In index-flattened form: 
#   u_h[3i+j] · v_h[3k+l] · w_h[3m+n]
#   = ?? 
# The δ_{jk} links the first and second arguments NON-separably.
# So this is NOT a rank-1 term u⊗v⊗w. The δ_{jk} factor couples (ij) and (kl).
#
# CONCLUSION: The Schur calculation gives T_matmul as a sum over |G| terms,
# but each term has rank > 1 in the 9×9×9 tensor. The rank-21 interpretation
# works in the GROUP ALGEBRA (21-dim space), not in the original 9×9×9 space.

print("\n" + "=" * 70)
print("FINAL ANSWER: HOW DIFFICULT IS AN EXPLICIT RANK-21 DECOMPOSITION?")
print("=" * 70)
print("""
ANSWER: IT DOES NOT EXIST (as a consequence of Z_7⋊Z_3 alone).

The Z_7⋊Z_3 TPP result gives:
  - ω ≤ 2.771 (asymptotic exponent bound via laser method)
  - This does NOT imply R(T_matmul(3)) ≤ 21

Why not:
  1. Direct TPP requires |S|³ ≤ |G|, but 3³ = 27 > 21. IMPOSSIBLE.
  2. The "rank-21" lives in the 21-dim group algebra C[G],
     not in the 9×9×9 tensor space.  
  3. Projecting from C[G] to the ρ-block introduces δ_{jk} coupling
     that CANNOT be factored into rank-1 terms u⊗v⊗w.
  4. The Schur orthogonality sum is |G|² = 441 terms that collapse
     analytically to T_matmul, but this is NOT a rank decomposition.

Status of R(T_matmul(3)):
  - BEST KNOWN UPPER BOUND: R ≤ 23 (Smirnov 2013)
  - BEST KNOWN LOWER BOUND: R ≥ 19 (substitution method)  
  - NO evidence that R ≤ 21 for this specific tensor
  - The asymptotic bound ω ≤ 2.771 applies to LARGE matrix multiplication
    via tensoring T^{⊗k} and laser method, not to the single 3×3 case

DIFFICULTY ASSESSMENT:
  If R(T_matmul(3)) = 21 is actually true, finding the explicit
  decomposition requires solving a system of 9³ = 729 polynomial equations
  in 21 × 9 × 3 = 567 complex unknowns. This system is:
  - HIGHLY non-convex (degree-3 polynomial system)
  - Invariant under GL(21) reparametrization of the rank-1 terms
  - Known to be NP-hard in general (Håstad 1990)
  - No algebraic structure from Z_7⋊Z_3 helps (the group gives ω, not R)
  
  Comparable to finding Strassen's algorithm (R=7 for 2×2 matmul)
  but in 21 dimensions instead of 7. Strassen found his by hand in 1969.
  23 dimensions was found by Smirnov in 2013 using computer search.
  Going from 23 to 21 (or even 22) would be a major result.
""")
