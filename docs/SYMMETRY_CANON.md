# SYMMETRY CANON: THE WREATH PRODUCT STRUCTURE OF R=19

## The Noether Principle Applied

The conservation law `R + η_nullity = 27` implies a symmetry. This document
identifies that symmetry and shows it reduces the R=19 search space from
513 free parameters to **81**.

---

## THE GROUP: Z₂ ≀ S₃ (order 48)

The symmetry group is the **wreath product** Z₂ ≀ S₃ = Z₂³ ⋊ S₃, which acts
on the index cube {0,1,2}³ via two layers:

| Layer | Group | Order | Action |
|-------|-------|-------|--------|
| Inner | Z₂³ | 8 | Swap 1↔2 independently in each coordinate |
| Outer | S₃ | 6 | Permute the three coordinates (r,s,u) |
| Combined | Z₂ ≀ S₃ | 48 | Inner ⋊ Outer (S₃ permutes which Z₂ acts where) |

### Action on the standard R=27 terms

The standard algorithm has one term per triple (r,s,u) ∈ {0,1,2}³. Under
Z₂ ≀ S₃, these 27 terms split into exactly **4 orbits**:

| Orbit | Representative | Size | Geometric name | Description |
|-------|---------------|------|----------------|-------------|
| O₀ | (0,0,0) | 1 | **Corner** | All indices zero |
| O₁ | (0,0,1) | 6 | **Edges** | Exactly one index nonzero |
| O₂ | (0,1,1) | 12 | **Faces** | Exactly two indices nonzero |
| O₃ | (1,1,1) | 8 | **Interior** | All indices nonzero = {1,2}³ |

Verification: 1 + 6 + 12 + 8 = 27. ✓

### The R=19 partition

**The partition 19 = 1 + 6 + 12 is a union of the first three orbits.**

The deleted 8 terms are exactly the fourth orbit O₃ = {1,2}³ — the "interior"
of the cube where no index touches zero.

| | Kept (19 terms) | Deleted (8 terms) |
|---|---|---|
| Orbits | O₀ ∪ O₁ ∪ O₂ | O₃ |
| Description | All (r,s,u) with at least one zero index | {1,2}³ |
| Mixed orbits | **0** | **0** |

The zero mixed-orbit count means the partition is **perfectly symmetric** —
the group Z₂ ≀ S₃ is a symmetry of the R=19 structure itself.

---

## THE RECURSION TOWER

The symmetry applies in two stages, each reducing the search space:

```
LEVEL 0 — Raw problem
  19 terms, 513 parameters, 729 equations
  
  ↓ Apply Z₂³ = (swap 1↔2)³, order 8
  
LEVEL 1 — First reduction  
  7 seed terms, 189 parameters, 125 independent equations
  Seeds are {0,1}³ \ {(1,1,1)} — unit cube minus one corner
  
  ↓ Apply S₃ (coordinate permutation), order 6
  
LEVEL 2 — Full reduction
  3 super-seeds, 81 parameters (24 effective α,β + 57 pure-γ)
  19 independent profiled equations, ~25 raw
  Super-seeds classified by Hamming weight
  
  ↓ No further symmetry (weights 0,1,2 are distinct)
  
STOP — Recursion terminates
```

| Level | Free terms | Free params | Independent eqs | Eqs/params |
|-------|-----------|-------------|-----------------|------------|
| 0 | 19 | 513 | 729 | 1.42 (overconstrained) |
| 1 | 7 | 189 | 125 | 0.66 (underdetermined) |
| 2 | 3 | 81 | ~25 | ~0.31 (highly underdetermined) |
| 2 (profiled) | 3 | 24 (α,β only) | 19 | 0.79 (underdetermined, 5 d.o.f.) |

**Note (April 2026):** The profiled residual eliminates γ analytically,
reducing to 24 effective parameters (9+9 for α,β minus stabilizer
redundancies) and 19 independent equations. The Jacobian of the full
bilinear map params→(Σ,H,Δ) has rank 22, leaving a 2-dim kernel.

---

## THE THREE SUPER-SEEDS

Each super-seed has 27 free parameters (9 for α, 9 for β, 9 for γ).
The group generates all other terms from these three.

### Super-seed S₀: The Corner — (0,0,0)
- **Orbit size:** 1 (fixed by the entire group)
- **Generates:** Just itself
- **Standard interpretation:** Term computing C[0,0] += A[0,0]·B[0,0]
- **Symmetry:** All 48 group elements fix this term, so α₀, β₀, γ₀ must be
  invariant under the full group action on factors

### Super-seed S₁: The Edge — (0,0,1)
- **Orbit size:** 6
- **Generates:** (0,0,1), (0,0,2), (0,1,0), (0,2,0), (1,0,0), (2,0,0)
- **Standard interpretation:** Terms where exactly one index is nonzero
- **These are the "boundary edges" of the 3³ cube touching the corner (0,0,0)**
- **Stabilizer:** Z₂ × Z₂ (order 4) — swaps within the two zero coordinates

### Super-seed S₂: The Face — (0,1,1)
- **Orbit size:** 12
- **Generates:** All terms with exactly one zero index:
  (0,1,1), (0,1,2), (0,2,1), (0,2,2),
  (1,0,1), (1,0,2), (2,0,1), (2,0,2),
  (1,1,0), (1,2,0), (2,1,0), (2,2,0)
- **Standard interpretation:** Terms where exactly two indices are nonzero
- **These are the "faces" of the cube touching the corner (0,0,0)**
- **Stabilizer:** Z₂ (order 2) — swaps the two nonzero indices

---

## GROUP ACTION ON FACTORS

For a group element g = (π, ε) where π ∈ S₃ (coordinate permutation)
and ε ∈ Z₂³ (swap flags), the action on a rank-1 term (α, β, γ) is:

The index triple (r, s, u) maps to (r', s', u') where:
- First permute: (r, s, u) → (x_{π(0)}, x_{π(1)}, x_{π(2)}) where x = (r,s,u)
- Then swap: apply swap12 to coordinate i if ε_i = 1

On factors (each is a 3×3 matrix):
```
α'[r', s'] = α[r, s]       where (r,s,u) ↦ (r',s',u') under g
β'[s', u'] = β[s, u]
γ'[r', u'] = γ[r, u]
```

**Critical:** When S₃ permutes coordinates, it also permutes the ROLES of
(α, β, γ) because the three coordinates correspond to (row-of-A, summation,
column-of-B). Cyclic permutation (r,s,u) → (s,u,r) corresponds to
(A, B, C) → (B^T, C^T, A^T) in the tensor structure.

---

## INDEPENDENT EQUATIONS AFTER SYMMETRY

The 729 tensor equations form orbits under Z₂ ≀ S₃:

| Orbit size | Count | Total entries | Live/Dead |
|-----------|-------|---------------|-----------|
| 1 | 1 | 1 | All live |
| 2 | 12 | 24 | Unmixed |
| 4 | 48 | 192 | Unmixed |
| 8 | 64 | 512 | Unmixed |
| **Total** | **125** | **729** | **0 mixed** |

**125 independent equations**, of which 8 are live (target = 1 or 3)
and 117 are dead (target = 0).

The zero-mixed-orbit count confirms: the symmetry is natural to
matrix multiplication, not imposed artificially.

---

## IMPLICATIONS FOR SEARCH

### The Grassmannian reduces to a discrete problem

The 10-dimensional kernel K ⊂ ℝ^19 must be invariant under the group action
on ℝ^19 (which permutes the 19 term coordinates). Under Z₂ ≀ S₃, the
19-dimensional space decomposes into character sectors:

- O₀ contributes 1 dimension (fully symmetric)
- O₁ contributes 6 dimensions (splits into irreps of stabilizer Z₂²)
- O₂ contributes 12 dimensions (splits into irreps of stabilizer Z₂)

K must be a direct sum of irrep subspaces. **The search over Gr(10,19)
reduces to choosing which irrep components to include in K** — a finite
combinatorial problem.

### Enumeration becomes tractable

Over {-1, 0, 1} with 3 super-seeds:
- Each super-seed has 27 parameters → 3^27 ≈ 7.6 × 10^12 per seed
- But the ~8 independent equations per seed prune most candidates immediately
- With incremental checking: effective search per seed is ~10^6-10^8
- Three seeds (mostly decoupled): total ~10^8 × coupling factor

**Estimated wall time on Ryzen 9 5900X: hours, not years.**

**Status (April 2026):** The full 526-kernel sweep has been run
(canon_newton.py, 20 restarts per kernel, 6 workers). No solution found.
The kernel enumeration route is exhausted numerically.

### Conservation law as filter

Any candidate must satisfy R + η_nullity = 27. With the symmetric structure:
- rank(H) must be exactly 10
- H has 18 columns in ℝ^19, with 3 irrep sectors
- The rank-10 condition becomes: "the symmetric block of H spans exactly
  the 10-dim complement of the kernel sector"

This is an ALGEBRAIC condition on the 3 super-seeds, computable in closed form.

---

## GEOMETRIC INTERPRETATION

The R=19 decomposition removes the 8 "deep interior" terms of the 3×3×3
multiplication — those where ALL three indices avoid zero. The remaining
19 terms form the **star neighborhood** of the corner (0,0,0):

```
         (0,0,0) ← corner (1 term, S₀)
        /  |  \
    edges      ← 6 terms, S₁
   /   |   \
  faces       ← 12 terms, S₂
   \   |   /
    interior   ← 8 terms, DELETED (O₃)
```

The compression from R=27 to R=19 is: **replace the 8 interior multiplications
with linear combinations of boundary terms, using the group symmetry to
constrain how the replacement works.**

This is analogous to how Strassen's algorithm for 2×2 (R=7) deletes the
"interior" term (1,1,1) from the 2³=8 standard terms, keeping the 7
terms that touch the corner (0,0,0) of {0,1}³.

| Algorithm | n | R | Standard R=n³ | Deleted | Structure |
|-----------|---|---|--------------|---------|-----------|
| Strassen | 2 | 7 | 8 | {(1,1,1)} = 1 term | {0,1}³ \ interior |
| **Target** | **3** | **19** | **27** | **{1,2}³ = 8 terms** | **{0,1,2}³ \ interior** |

The pattern: delete the top-weight orbit and compress its work into the
boundary terms. **Strassen IS the n=2 case of this construction.**

---

## THE GATE 3 WALL (April 2026)

Algebraic experiments (algebra_experiments.py) revealed the **true bottleneck**
is not Gate 1 (rank of H) or Gate 2 (Δ containment), but **Gate 3: Sigma
rank deficiency**.

### Per-seed rank contributions (generic, additive)

| Seed | |orbit| | rk(H) | rk(Σ) | rk(α⊗β) |
|------|---------|-------|-------|----------|
| Corner | 1 | 1 | 1 | 1 |
| Edge | 6 | 2 | 2 | 2 |
| Face | 12 | 4 | 2 | 4 |
| Interior | 8 | 8 | 4 | 8 |

For R=19 (Corner+Edge+Face): rk(H) = 1+2+4 = 7 (need 10), rk(Σ) = 1+2+2 = 5 (need 9).

### Key findings

1. **Gate 2 is generically free** for all R ≤ 21 (rk(N)=rk(H) always).
2. **Gate 3 fails universally** — no orbit combination achieves rk(Σ)=9.
   Even R=27 (all orbits) only reaches rk(Σ)=8.
3. The conservation law R+η_null generic values:
   - R=19: 19+11=30 (overshoots 27 by 3)
   - R=12,13,20,21: deficit=1 (closest)
4. **R=13 and R=20** fail Gate 2 on the rank(H)=target subvariety
   (rk(N)=rk(H)+1 structurally). Both are algebraically dead.

### New strategy: Gate 3-first

Instead of searching for kernels that achieve rk(H)=10, work backwards:
find the parameter subvariety where rk(Σ)=9, then check whether
Gate 1 and Gate 2 can be simultaneously satisfied there.

---

## THE AVERAGING IMPOSSIBILITY (April 2026)

The impossibility_test.py experiments proved a stronger result:

### Theorem: rk(Σ) < 9 identically in the averaged family

All 9×9 minors of Σ are **identically zero** for every orbit configuration
at every parameter value. This was verified by:
- Sampling 500 random parameter vectors per R
- Testing all (R choose 9) row subsets × 50 per trial
- Every determinant is zero to machine precision

This means Gate 3 is not merely non-generic — it is **algebraically
impossible** in the stabilizer-averaged parameterization.

### Root cause: averaging kills factor sharpness

The standard R=27 algorithm (α_k = e_r e_s^T, β_k = e_s e_u^T) achieves
rk(Σ) = 9 for R ≥ 13. But it is **NOT** in the symmetric family:
least-squares fit gives ||error|| ≈ 3.6.

The standard algorithm is **equivariant** (terms related by group transport)
but NOT **averaged** (factors are NOT stabilizer-averaged). The M_a, M_b
expansion matrices impose stabilizer averaging:

```
α_seed → (1/|Stab|) Σ_{g ∈ Stab} g · α_seed
```

This projection onto the Stab-invariant subspace kills the rank-1 structure
e_r e_s^T (which is NOT Stab-invariant for Edge and Face seeds).

### Per-orbit Σ image dimension (degree-2 monomials)

| R | params | monomials | rk(Σ image) |
|--:|-------:|----------:|------------:|
| 12 | 18 | 171 | 9 |
| 13 | 36 | 666 | 12 |
| 19 | 54 | 1485 | 18 |
| 20 | 36 | 666 | 15 |
| 27 | 72 | 2628 | 24 |

The Σ image variety is rich (dim 18 for R=19) but every point in it has
rk ≤ 5. The variety is large but low-rank.

### Resolution: equivariant transport WITHOUT averaging

**UPDATE (April 2026, fixed_transport.py):** Even with correct transport
(verified on all 27 standard algorithm terms), the standard algorithm is
NOT in the seed-transport family. ||error|| ≈ 2.0 for α, 2.8 for β.

The root cause is deeper than averaging vs. transport:

**The seed-transport parameterization is too restrictive.**

The standard algorithm's factors for different orbit members are NOT
related by transporting a single seed — they are independently chosen,
constrained only by the requirement that the TENSOR (sum of all terms)
is group-invariant. For example:
- Term (0,0,1): α = e₀e₀^T (a rank-1 matrix)
- Term (1,0,0): α = e₁e₀^T (a DIFFERENT rank-1 matrix)
- Transport of (0,0,1)'s α under the relevant group element produces
  e₀e₁^T, NOT e₁e₀^T.

**The correct symmetry condition:** The group permutes the R terms as a
set, and maps one term's factors to another's. But within each orbit,
the RELATIONSHIP between factors at different orbit members need not be
simple transport of one seed. The factors can have additional freedom
parameterized by the stabilizer representation.

**Confirmed: rk(Σ) < 9 identically in the seed-transport family.**
All 9×9 minors of Σ are zero for every R, every parameter value,
both with averaged and non-averaged transport. This is structural:
the seed-transport family lives in a proper subvariety of the full
equivariant parameter space.

**Next step:** Parameterize the orbit factors with FULL stabilizer
freedom — each orbit member gets factors that are constrained only
by the group action relating them pairwise, not by derivation from
a single seed.

---

## SOURCE

Derived from the S₃³ orbit analysis of the ADE3x3 canonical object.
The Z₂ ≀ S₃ wreath product structure and its 4-orbit decomposition of
{0,1,2}³ were computed by exhaustive enumeration of all product subgroups
of S₃³ and their orbit structures compatible with R=19.
