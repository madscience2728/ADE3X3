# ADE3×3 Novel Axiom System v2

## Design Principles

1. Must work at ALL target ranks (13, 19, 20, 21, 22) — not hardcoded to R=19
2. Must respect R_rank=1 universality (composition algebra is always diagonal)
3. Must avoid the Step-51 coordinate trap (Gates 2+3 mutually exclusive there)
4. Axioms are LENSES — each one views the decomposition from a different angle
5. A valid decomposition satisfies ALL axioms simultaneously

---

## Known Hard Facts (not axioms — these are theorems)

- T_matmul IS the structure constant tensor of Mat(3,ℝ), 27 nonzero entries
- Flattening bound: rank ≥ 9
- Conservation: R + η_nullity = 27 (verified for Strassen, AlphaTensor, standard)
- Symmetry group G = Z₂ ≀ S₃, order 48
- Support splits 8 hull vertices + 19 interior lattice points (parity partition)
- r-coordinate drops out of Fourier constraints (3r ≡ 0 mod 3)
- Fiber sums: Γ(s,u) = 3 for all 9 (s,u) pairs (complete Fourier content)
- Coupling Λ = W^T·U always rank ≤ 9; composition f[t1,t2,t3] = Λ[t1,t2]·δ(t2,t3)
- Aug_gap = 0 on integer lattice is a discrete accident (ε-perturbation breaks it)

---

## AXIOM CANDIDATES

### A1 — Conservation Envelope

**Statement:** For rank R, the anisotropy matrix H (shape R×18) must have
rank(H) = R − (27 − R) = 2R − 27.

Wait — conservation says R + η_null = 27, η_null = 18 − rank(H), so rank(H) = R − 9.

**Actionable form:** rank(H) = R − 9. This is:
- R=13 → rank(H) = 4
- R=19 → rank(H) = 10
- R=20 → rank(H) = 11
- R=21 → rank(H) = 12
- R=22 → rank(H) = 13
- R=23 → rank(H) = 14 ✓ (AlphaTensor verified)

**Use:** Hard pruning filter. Any candidate with wrong rank(H) is immediately dead.
Reduces parameter search to a Grassmannian Gr(R−9, 18).

**Novel angle:** Don't optimize rank(H) toward target — PARAMETERIZE on Gr(R−9, 18)
directly. The subspace IS the search variable, not the factor vectors.

---

### A2 — Parity Block Partition

**Statement:** The 27 support points split into parity classes under (r%2, s%2, u%2).
The 8 hull vertices are the all-even class {0,2}³. The 19 interior points have at
least one odd coordinate.

For rank R < 27, the deleted terms must form a G-stable subset. The orbit
decomposition constrains which terms can be dropped:
- R=19: delete O₃ (8 interior, all-odd {1,2}³)
- R=20: delete from {O₃} keeping 1 → orbits [8,12] or [1,7,12]
- R=21: delete from {O₃} keeping 2 → [1,8,12]
- R=13: delete O₂ + O₃ (12+8=20), keep [1,12] — wait, 1+12=13 ✓

**Actionable form:** Enumerate G-stable subsets of {0,1,2}³ of size R.
For each, the stabilizer quotient gives the effective parameter count.
This is a DISCRETE enumeration — finitely many orbit configurations per rank.

**Novel angle:** The parity partition isn't just bookkeeping — it determines which
fibers are "pinned" (forced γ=3, single element) vs "free" (sum=3 constraint,
multiple elements). The ratio of pinned-to-free fibers is a discrete invariant
of the decomposition family.

---

### A3 — Fiber Mode Decomposition (the 2D Lens)

**Statement:** Since 3r ≡ 0 mod 3, the 729 tensor equations collapse to 81
independent constraints on the 9 fiber sums Γ(s,u) = Σ_r γ(r,s,u).

The decomposition problem in fiber coordinates becomes:
- 9 target equations: Γ(s,u) = 3
- Each fiber has |fiber| terms contributing, with individual γ values summing to 3
- The within-fiber distribution of γ is FREE (any split summing to 3 works)

**Actionable form:** Parameterize the search as:
1. Choose fiber-sum allocation (how many terms per fiber)
2. For each fiber, choose within-fiber γ distribution
3. Check that the resulting (α,β,γ) factors are mutually consistent rank-1 tensors

**Novel angle:** This decouples the "macro" problem (fiber allocation) from the
"micro" problem (within-fiber factor design). The macro problem is combinatorial
and finite. The micro problem is algebraic but LOCAL to each fiber.

---

### A4 — Relation Module Spectrum

**Statement:** For R terms in ℝ⁹, the three factor families {aₜ}, {bₜ}, {cₜ}
each have an (R−9)-dimensional kernel (relation module). Define:

  K_A = {λ ∈ ℝ^R : Σ λₜ aₜ = 0}  (dim R−9)
  K_B = {λ ∈ ℝ^R : Σ λₜ bₜ = 0}  (dim R−9)
  K_C = {λ ∈ ℝ^R : Σ λₜ cₜ = 0}  (dim R−9)

The TRIPLE KERNEL K_ABC = K_A ∩ K_B ∩ K_C consists of λ that simultaneously
annihilate all three factor families. This is the "completely useless" subspace —
terms that cancel across all modes.

**Actionable form:**
- dim(K_ABC) = 0 is NECESSARY (otherwise one term is redundant → not minimal rank)
- For minimal rank: K_A, K_B, K_C must be in "general position" in ℝ^R
  (pairwise intersections as small as possible)
- The matroid of (K_A, K_B, K_C) is a discrete invariant

**Novel angle:** This is NOT about the factor vectors themselves — it's about the
DEPENDENCIES among them. Two decompositions with the same relation matroid are
"algebraically equivalent" even if their factor vectors differ. The axiom system
should classify relation matroids, not factor vectors.

**Spectrum:** For each pair (K_A, K_B), compute dim(K_A ∩ K_B). The triple
(dim K_A∩K_B, dim K_A∩K_C, dim K_B∩K_C) is the "relation spectrum."
Conservation + general position gives bounds on each component.

For R=19: each K has dim 10, and generic intersection of two 10-dim subspaces
in ℝ¹⁹ has dim 10+10−19 = 1. So generically:
  dim(K_A∩K_B) = dim(K_A∩K_C) = dim(K_B∩K_C) = 1
  dim(K_ABC) = 0

For R=13: each K has dim 4, generic pairwise = max(4+4−13, 0) = 0.
All three kernels are generically disjoint! Clean separation.

For R=23: each K has dim 14, generic pairwise = 14+14−23 = 5.
Heavy overlap — lots of "shared cancellation structure."

This gives a QUALITATIVE signature per rank that constrains the search.

---

### A5 — Coordinate Liberation (Anti-Step-51)

**Statement:** Gates 2+3 are provably mutually exclusive in Step-51 coordinates
when null_dim = 9. Therefore: any valid decomposition must be expressed in
coordinates where the Σ/H/Δ block structure doesn't trap aug_gap at 0.

**Actionable form:** Instead of the Step-51 split (fiber-sum / anisotropy / dead-X),
use a CHANGE OF BASIS on ℝ^R that diagonalizes the relation module structure.
Specifically:

Choose basis for ℝ^R aligned with K_A, K_B, K_C:
- First 9 basis vectors span a complement of K_A (the "A-active" subspace)
- The remaining R−9 span K_A
- Express B and C constraints in this A-adapted basis

In this basis, the solvability condition becomes: the projection of the B-constraint
and C-constraint onto K_A must have specific rank properties.

**Novel angle:** The Step-51 coordinates privileged one mode's fiber structure.
The relation module basis treats all three modes democratically. This may break
the Gate 2/3 deadlock by making the aug_gap visible in a different block structure.

---

### A6 — Cayley-Dickson Parity Bridge

**Statement:** The Cayley-Dickson doubling construction at level 4 (sedenions, dim 16)
produces 168 associativity-violating triples whose block-type signature matches the
parity partition of the 19 interior points.

Specifically, violating block types have odd XOR-parity: {001, 010, 100, 111}.
Non-violating have even parity: {000, 011, 101, 110}.

The 19 interior points (those with at least one odd coordinate in {0,1,2}³) can be
mapped to violating-type sedenion triples via (r,s,u) → (r%2, s%2, u%2).

**Actionable form:** Use the Cayley-Dickson structure to generate CANDIDATE
factor vectors. Each sedenion multiplication rule e_i · e_j = ±e_k gives a
rank-1 tensor. Selecting 19 such products (the violating subset) gives a
candidate decomposition skeleton.

**Novel angle:** The sedenion multiplication table provides FREE cancellation
patterns — the alternator (associator) identities automatically zero out certain
tensor components. If T_matmul's support structure matches the non-zero pattern
of a sedenion sub-algebra's structure constants, we get a decomposition "for free"
from the Cayley-Dickson recursion.

**Status:** Most speculative axiom. Needs: explicit map from {0,1,2}³ to
sedenion index pairs, verification that the resulting rank-1 tensors span T.

---

### A7 — Grassmannian Quantization

**Statement:** The solution variety for rank-R decompositions of T_matmul is a
finite set of points (modulo gauge) on the Grassmannian Gr(R−9, R) × (factor spaces).
The G-symmetry (order 48) acts on this variety.

By conservation, rank(H) = R−9 is fixed, so the 18-dim anisotropy modes
select a (R−9)-dimensional subspace of ℝ^R. This subspace lives on Gr(R−9, R).

The G-action on Gr(R−9, R) has finitely many fixed-point types (by representation
theory of Z₂ ≀ S₃). Each type corresponds to a specific irrep decomposition of the
(R−9)-dim subspace.

**Actionable form:** Decompose ℝ^R into G-irreps. The target subspace (dim R−9)
must be a specific direct sum of irrep components. Enumerate all such sums.
For each candidate irrep decomposition:
1. The subspace is determined up to finitely many parameters
2. Check conservation, fiber constraints, and relation module spectrum
3. If all pass: solve the remaining factor-vector equations (now a SMALL system)

**Novel angle:** This converts the continuous Grassmannian search into a DISCRETE
irrep enumeration + small algebraic system per candidate. The group theory does
the heavy lifting.

For R=19: ℝ¹⁹ under G has specific irrep multiplicities (from the orbit structure
O₀(1) + O₁(6) + O₂(12)). The 10-dim kernel must be a sub-representation.
The number of valid sub-representations is FINITE and small.

---

## AXIOM INTERACTION MAP

```
A1 (Conservation)  ──────→  fixes rank(H), constrains A4 spectrum
       │
       ▼
A2 (Parity)  ────────────→  determines fiber structure for A3
       │
       ▼
A3 (Fiber 2D)  ──────────→  macro/micro split, reduces to A4 per fiber
       │
       ▼
A4 (Relation Module)  ───→  matroid structure, feeds A5 basis choice
       │
       ▼
A5 (Coord Liberation)  ──→  breaks Gate 2/3 deadlock, enables solvability
       │
       ▼
A7 (Grassmannian Quant) ─→  irrep enumeration on the liberated coordinates
       │
A6 (Cayley-Dickson)  ─────→  generates candidate factor vectors for A4/A7
```

## PRIORITY ORDER FOR IMPLEMENTATION

1. **A1 + A4** — Conservation + relation module spectrum. Purely algebraic,
   computable from AlphaTensor data as validation. Gives qualitative signatures
   per rank. START HERE.

2. **A7** — Grassmannian quantization via irrep decomposition. Converts search
   to enumeration. The key algorithmic acceleration.

3. **A2 + A3** — Parity + fiber. Combinatorial structure, reduces problem size.

4. **A5** — Coordinate liberation. Needed only if A7 enumeration hits the
   same Gate 2/3 wall in all candidates.

5. **A6** — Cayley-Dickson bridge. Speculative but high-reward if it works.