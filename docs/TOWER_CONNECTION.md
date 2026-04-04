# Cayley-Dickson Tower Connection to R=19

*Working document — updated as theory develops.*

---

## Core Hypothesis

The 24 dead Fourier modes (cancellation-only equations) in the R=19 decomposition
of T_matmul are not arbitrary cancellation requirements. They correspond to the
**zero divisor / associativity violation structure** of the Cayley-Dickson tower
at the right level — and that structure provides the cancellation **for free**,
algebraically, without engineering.

---

## The Matching

### Our problem's partition: 8 + 19 = 27

| Layer | Count | Description | Parity |
|-------|-------|-------------|--------|
| Hull vertices (O₃ in canon) | 8 | (r,s,u) ∈ {0,2}³ — all even | All-even = L-sector |
| Interior points (O₀∪O₁∪O₂) | 19 | at least one coordinate = 1 | Odd-touching = U-sector |

### CD tower block types: 4 violating + 4 non-violating out of 8

Violating block types (odd XOR-parity): `001, 010, 100, 111`
Non-violating (even XOR-parity): `000, 011, 101, 110`

The violation structure is a **parity filter** on {L,U}³ labeling of triples.

### The identification

```
{0,2}³  =  L-sector  =  non-violating hull  =  8 points
odd-touching  =  U-sector  =  violation locus  =  19 points
```

The 19 interior points need cancellation in 24 Fourier modes.
The CD tower provides zero divisors (a·b=0, a≠0, b≠0) as free cancellation slots.
The violation locus of the tower IS the cancellation budget.

---

## Dimension Problem (Resolved)

~~Sedenions (level 4) = 16-dimensional. We need 19 points covered.~~

**Resolution**: The CD tower doesn't embed as "19 = CD basis elements."
Instead, the structure acts through the **fiber parity** of (r,s,u) indices.
The r-direction drops out of the Fourier picture entirely (see Key Result below),
so the 19 interior points live over a 9-point base (Z₃² in (s,u)). The sedenion
dimension is not the relevant count.

The relevant structure is:
- 9 fibers in (s,u) ∈ Z₃² (sedenion-like Z₂³ structure in the parity projection)
- 4 forced fibers (type-(1,0,0) violating blocks)
- 5 free-sum fibers with 2 γ d.o.f. each

---

## Key Numbers from the Tower

| Level n | Dim | V[n] (violations) | S_assoc | Violation fraction |
|---------|-----|-------------------|---------|-------------------|
| 3 (𝕆) | 8 | 168 | 0.672 | 0.328 |
| 4 (𝕊) | 16 | 1848 | 0.549 | 0.451 |
| 5 | 32 | 15960 | 0.513 | 0.487 |
| ∞ | — | — | 0.5 | 0.5 |

The 50/50 limit is the asymptotic cancellation budget.
At level 4 (sedenions): 45.1% of triples are violation pairs = ~45% free cancellation.

---

## The 4 Violating Block Types

From the Limit Theorem doc, exactly 4 of 8 block types produce violations
for every all-imaginary triple:

| Block | Type | Violation? | Parity |
|-------|------|------------|--------|
| 000 | LLL | No (inherited from level below) | even |
| 001 | LLU | **Yes** | odd |
| 010 | LUL | **Yes** | odd |
| 011 | LUU | No | even |
| 100 | ULL | **Yes** | odd |
| 101 | ULU | No | even |
| 110 | UUL | No | even |
| 111 | UUU | **Yes** | odd |

The violating types have odd number of U-components in positions {1,3} — exactly
the XOR-parity condition matching our Fourier mode structure.

---

## The 3 Nonzero Fourier Modes (Key Result)

T_matmul has nonzero DFT only at:

```
T̂(p,q,w) = 27·δ_{p,0}·δ_{w,−q mod 3}
```

Nonzero modes: (0,0,0), (0,1,2), (0,2,1) — exactly **3 modes**.
24 dead modes require pure cancellation.

### r DROPS OUT ENTIRELY

The Fourier character on support (r,s,u):

```
χ_{p,q,w}(r,s,u) = e^{2πi[p(3r+s) + q(3s+u) + w(3r+u)]/3}
                  = e^{2πi(ps + qu + wu)/3}   [since 3·anything ≡ 0 mod 3]
```

**The r-index vanishes from the character completely.**

Consequence: the full 27-mode Fourier constraint on the rank-1 coefficients γ
collapses to a **9-mode 2D DFT** in (s,u) ∈ Z₃².

Define the **fiber sum**: Γ(s,u) = Σ_{r: (r,s,u) interior} γ(r,s,u)

Then T̂(p,q,w) = 3·Γ̂(p, q+w mod 3), where Γ̂ is the Z₃² DFT of Γ.

For this to equal 27·δ_{p,0}·δ_{q+w,0}, we need:

```
Γ(s,u) = 3   for ALL (s,u) ∈ Z₃²
```

**This is the complete Fourier constraint: every (s,u)-fiber must sum to 3.**
The 24 dead modes and 3 active modes alike reduce to this single statement.

### Fiber Structure of the 19 Interior Points

The 19 interior points form 9 fibers over (s,u) ∈ Z₃²:

| (s,u) | s%2 | u%2 | Interior r-values | Fiber size | γ constraint |
|-------|-----|-----|-------------------|------------|--------------|
| (0,0) | 0 | 0 | {1} | 1 | γ(1,0,0) = 3 FORCED |
| (0,2) | 0 | 0 | {1} | 1 | γ(1,0,2) = 3 FORCED |
| (2,0) | 0 | 0 | {1} | 1 | γ(1,2,0) = 3 FORCED |
| (2,2) | 0 | 0 | {1} | 1 | γ(1,2,2) = 3 FORCED |
| (0,1) | 0 | 1 | {0,1,2} | 3 | sum = 3, 2 d.o.f. |
| (2,1) | 0 | 1 | {0,1,2} | 3 | sum = 3, 2 d.o.f. |
| (1,0) | 1 | 0 | {0,1,2} | 3 | sum = 3, 2 d.o.f. |
| (1,2) | 1 | 0 | {0,1,2} | 3 | sum = 3, 2 d.o.f. |
| (1,1) | 1 | 1 | {0,1,2} | 3 | sum = 3, 2 d.o.f. |

**4 forced + 5×2 free = 10 γ degrees of freedom** (before factor vector constraints).

### CD Type Interpretation of the Fibers

```
Type (0,0,0) = O₃ = {r,s,u all even} → DELETED  
Type (1,0,0) = fibers (s,u) ∈ {0,2}², r=1 only → γ PINNED = 3 (violating type, forced)
Type (*,0,1) etc. = fibers with odd u or s → 3-element fibers with 2 d.o.f. each
```

The 4 forced fibers correspond exactly to block type (1,0,0) — the U×L×L violating type
where only r is in the U-sector. These are pinned because there is NO cancellation partner:
r=0 and r=2 are deleted (O₃), leaving exactly one term per (s,u) fiber when s,u are even.

This gives the CD explanation of why **the violating type-(1,0,0) terms are structurally
constrained to carry the full weight (γ=3)**: they have no siblings to share the load.

---

## Connection to the Face Lattice / Chain Complex

The 27 support points = face lattice of the 3-cube:
- 1-cell (interior): O₀
- 6 faces: O₁  
- 12 edges: O₂
- 8 vertices: O₃ (deleted in R=19)

Cellular chain complex:
```
0 → C₃ → C₂ → C₁ → C₀ → 0
    1     6    12    8
```

χ(S²) = 8 - 12 + 6 = 2  (full cube boundary)
χ(R=19 complex) = 0 - 12 + 6 + 1 = -5  (vertices deleted)

Euler defect = -7 relative to S². Matches: canon Level 1 gives **7 seed terms**.

The R=19 problem = capping off a chain complex with Euler defect -7.
The CD tower's violation pairs may be the "virtual cap" that restores ∂²=0
without adding actual rank-1 terms.

---

## Sedenion Zero Divisor Graph (Computed)

Sedenions (level 4, dim=16) have zero divisors only involving **cross-sector elements**:
combinations of one L-imaginary (e₁..e₇) with one U-sector element (e₉..e₁₅).

Key facts (computed):
- **42 zero-divisor a-vertices**: pairs (eᵢ + eⱼ), i∈{1..7}, j∈{9..15}
  - 7 L-imaginary directions × 6 eligible U-directions (j=i+8 excluded: CD pair)
  - = 42 total
- **Each vertex has exactly 8 zero-annihilators** (4 modulo overall sign)
- **84 undirected zero-divisor pairs** total

The j=i+8 exclusion is structural: eᵢ and eᵢ₊₈ are a CD pair and their combination
eᵢ + eᵢ₊₈ cannot be a zero divisor (they're "in proper relation").

### Does ZD map to 24 dead modes?

84 undirected ZD pairs / 24 dead modes = 3.5 — **not a clean ratio**.
The sedenion dim=16 is also too small for a direct 19-point embedding.

**Current interpretation**: The sedenion ZD graph structure does not map
literally to the 24 dead modes. The CD connection operates at a more
abstract level — through the **fiber constraint** derivation above.

The correct CD-to-tensor correspondence is:

```
Block type (r%2, s%2, u%2) ↔ CD sector label of support point
Fiber sum Γ(s,u) = 3 ↔ "balance condition" between L and U contributions
r dropping out ↔ r-direction is a "CD level" decoration, not a Fourier direction
```

The sedenion ZD pairs are relevant for a **different** (more indirect) channel:
they constrain which pairs of rank-1 factors can simultaneously be nonzero
if we demand the product structure to have the right form.

---

## Open Questions

1. **Why does r drop out?** — the index r appears symmetrically in T[3r+s, 3s+u, **3r**+u]
   and T[**3r**+s, ...], but the character p·(3r+s) = p·s mod 3 since 3r≡0. This
   is a direct consequence of base-3 representation, not a choice. It means the
   decomposition problem is fundamentally 2D in (s,u). **Does this suggest a
   Z₃-graded algebra as the natural home?**

2. **The 10 γ d.o.f.:** After fiber sum constraints (Γ=3), 10 free parameters remain
   in the γ values alone. The factor vectors a,b,c add more. How does this square with
   the symmetry canon's "5 d.o.f. after Z₂≀S₃ and γ elimination"?
   Reconciliation: the symmetry group further reduces 10 → 5 by quotienting.

3. **The 4 forced fibers:** γ(1,s,u)=3 for s,u∈{0,2}. These are 4 specific rank-1 terms
   with NO freedom. Their factor vectors (a,b,c) must be determined purely from the
   tensor fitting equations. **Are these the "seed terms" of the Z₂≀S₃ super-seeds?**

4. **Associativity = Gate-2?** The gate system's Gate-2 (off-diagonal cancellation)
   may be the algebraic encoding of the associativity violation condition. Verify.

5. **The 168:** V[3] = 168 = |PSL(2,7)|. Our Aut(T) = 432. Any relation?
   432/168 not integer. But 432 = 16·27 = 2⁴·3³. V[4]=1848 = 432·4.28... not clean.

---

## Next Steps

1. Compute zero divisor pairs of sedenions explicitly
2. Map pairs to our (r,s,u) support labeling
3. Check if the 24 dead Fourier modes embed into the sedenion zero divisor graph
4. If yes: construct rank-1 terms whose factor matrices are sedenion basis elements
5. Check if the 3 DC modes are the "real part" (L-sector) of the sedenion products
