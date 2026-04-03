# CANON CONSTRAINTS FOR R=19 SEARCH
## What every new optimizer/strategy MUST enforce

This file is the single authoritative summary of constraints derived from the canonical
dossier (`ADE3x3_CANONICAL_OBJECT.md`). Any new tool that ignores these constraints is
searching the wrong space and will perpetually miss the variety.

---

## THE PROBLEM STATEMENT

Find (α, β, γ) ∈ ℝ^{R×9} each, with R=19, such that the CP decomposition
`T = Σ_k γ_k ⊗ α_k ⊗ β_k` exactly reconstructs the 3×3 matrix multiplication tensor.

**Fitness proxy (current working metric):** ‖T - decomposition‖_∞, minimized toward 0.

---

## COORDINATE SYSTEM (§40 / Step 51)

Every rank-1 term (α_k, β_k) generates four coordinate blocks. These are the **exact** formulas:

| Block | Shape | Definition | Role |
|-------|-------|------------|------|
| `Sigma` | R×9 | `σ_k[r,u] = Σ_s α_k[r,s]·β_k[s,u]` | Fiber-sum (carries the target) |
| `Eta1` | R×9 | `η1_k[r,u] = α_k[r,0]·β_k[0,u] − α_k[r,1]·β_k[1,u]` | First anisotropy |
| `Eta2` | R×9 | `η2_k[r,u] = α_k[r,1]·β_k[1,u] − α_k[r,2]·β_k[2,u]` | Second anisotropy |
| `Delta` | R×54 | `δ_k[r,s,t,u]` for s≠t | Dead-X coordinates |
| `H` | R×18 | `[Eta1 \| Eta2]` | Combined anisotropy (nuisance part A) |
| `Nuisance` | R×72 | `[H \| Delta]` | Full nuisance block |

---

## MASTER EQUATION (§40 / Step 51)

The 729 tensor equations decompose exactly into three blocks:

```
Γ · Σ  = 3 I₉      ← 81 fiber-sum equations   (target lives here)
Γ · Eta1 = 0        ← 81 live-anisotropy equations
Γ · Eta2 = 0        ← 81 live-anisotropy equations
Γ · Delta = 0       ← 486 dead-X equations
```

Compact form: **`Γ · Σ = 3 I₉`** and **`Γ · Nuisance = 0`**.

A valid decomposition exists iff there exists Γ (9×R) satisfying these simultaneously.

---

## NECESSARY AND SUFFICIENT CONDITIONS (§41 / Step 52)

### The Solvability Criterion (T2–T3, §41)

For a fixed (α, β), a valid Γ exists **if and only if**:

> The 9 columns of `Σ` are linearly independent modulo the nuisance span.
> Equivalently: `rank([Σ | Nuisance]) = rank(Nuisance) + 9`

This gives the per-algorithm bound: **`R ≥ 9 + rank(Nuisance)`**.

### For R = 19 specifically:

| Constraint | Exact Form | Status in current code |
|-----------|------------|----------------------|
| **C1** | `rank(Nuisance) ≤ 10` | ❌ not enforced |
| **C2** | `rank(H) ≤ 10` | ✅ hard LP equality |
| **C3** | `Δ ⊂ span(H)` → rank(Nuisance)=rank(H) | ⚠️ soft merit only |
| **C4** | `rank([Σ \| Nuisance]) = rank(Nuisance) + 9` | ❌ not enforced |

C3 implies C1 implies C2. Enforcing C2 alone (current state) is a necessary-but-insufficient
subset. **The floor at fitness ~0.27 is the consequence of not enforcing C1/C3.**

---

## THE CONSERVATION LAW (§54, §71, §83 / Phases 17, 33, Synthesis AB)

For any **exact** matrix-multiplication decomposition:

> **`R + η_nullity = 27`** where `η_nullity = 18 − rank(H)`

This follows from `rank(H) = R − 9` (i.e., H spans all of ker(Γ)), which is equivalent to
`Δ ⊂ span(H)`.

| Known exact decomposition | R | rank(H) | η_nullity | R + η_nullity |
|--------------------------|---|---------|-----------|---------------|
| Standard 3×3 (R=27) | 27 | 18 | 0 | 27 ✓ |
| AlphaTensor (R=23) | 23 | 14 | 4 | 27 ✓ |
| **Target: R=19** | **19** | **10** | **8** | **27 ✓** |
| Best near-miss (fitness 0.073) | 19 | **18** | **0** | **19 ✗** (invalid) |

**Key diagnostic:** The 0.073 basin has `rank(H) = 18` — it is **not on the variety**. It
cannot be continuously deformed to an exact decomposition without crossing through the
rank-10 variety. The conservation law gives R + η_nullity = 19 + 0 = 19 ≠ 27, confirming
the near-miss is structurally incompatible with exact multiplication.

---

## DELTA CONTAINMENT (§54, §83 / Phase 17, Synthesis AB)

The condition `Δ ⊂ span(H)` is the **sole remaining hypothesis** in the conservation law.

**Verified exactly (with integer coefficients):**
- AlphaTensor R=23: all 54 Delta columns solved exactly, max denominator = 1
- Strassen 2×2 R=7: all 8 Delta columns solved exactly
- All 23 single-deletions of AlphaTensor: exact containment

**Known failures of containment:**
- Random non-multiplication rank-R collections: 0/1000 satisfy it
- Step 75 anticommutator rank-19: fails (not a matrix multiplication decomposition)
- All rank-19 near-misses (fitness 0.073): Δ leaks outside ker(Γ) — **this leakage IS
  the fitness floor**. The 0.073 residual is exactly `‖Γ · Δ‖`.

**Geometric interpretation:** `Δ ⊂ span(H)` is generically true on the variety (§83):
for R ≤ 27, H has 18 columns and ker(Γ) has dimension R−9 ≤ 18, so 18 vectors generically
span the (R−9)-dimensional space.

---

## CONSTRAINT HIERARCHY FOR SEARCH

Enforce in this order (each gate subsumes the previous):

### Gate 1: rank(H) ≤ 10
`rank([Eta1|Eta2]) ≤ 10`

- **How:** For fixed α_k, the condition `H[k,:] ∈ V` (V = 10-dim subspace) is
  **linear** in β_k. Exact LP enforcement via nullspace projection. *(Currently implemented.)*
- **What it gives:** 18-column matrix with rank ≤ 10. Fitness floor ~0.27.
- **Warning:** Not sufficient by itself. H can have rank 10 while Δ still leaks.

### Gate 2: Δ ⊂ span(H), i.e., rank(Nuisance) = rank(H) = 10
`rank([H|Delta]) = rank(H)`

- **How:** For fixed H (i.e., fixed α, β), the condition `Delta[k,:] ∈ span(H)` is
  **bilinear** in (α_k, β_k). For fixed α_k or β_k it becomes linear.
  - In the γ-LP step: `Γ · Δ = 0` is **linear** in Γ (add rows to the LP constraint matrix).
  - Equivalently: add `Delta[k,:]^T` as extra equality constraints alongside `H[k,:]^T`
    in the V-update SVD step.
- **What it gives:** Exact containment → rank(Nuisance) = 10. Fitness floor unknown (likely < 0.1).

### Gate 3: Γ solvability — solve `Γ·Σ = 3I₉`, `Γ·Nuisance = 0`
Given (α, β) satisfying Gates 1–2, this is a **linear system** in Γ.

- **How:** Build the system `[Σ | Nuisance]^T Γ^T = [3I₉ | 0]^T` and solve by least squares.
- **Test:** Check `rank([Σ | Nuisance]) = rank(Nuisance) + 9 = 19`. If yes, Γ exists exactly.
  Check `‖T − Σ_k γ_k ⊗ α_k ⊗ β_k‖_∞`.
- **What it gives:** If solvable → exact R=19 decomposition. Done.

---

## DIAGNOSTICS TO REPORT ON EVERY ITERATE

Every optimizer output should report these numbers to know which gate is active:

| Metric | Target | Interpretation if off |
|--------|--------|----------------------|
| `rank(H)` | 10 | Gate 1 not met |
| `rank(Nuisance)` = `rank([H|Delta])` | 10 | Gate 2 not met (Δ leaks) |
| `delta_residual` = ‖Δ − H·M‖_F for best M | 0 | Δ containment residual |
| `gamma_residual` = ‖Γ·Δ‖_F | 0 | How much Δ leaks outside ker(Γ) |
| `augmented_rank` = `rank([Σ|Nuisance])` | 19 = 10+9 | Gate 3 solvability test |
| `fitness` = ‖T − decomp‖_∞ | 0 | Main objective |

---

## WHAT CURRENT TOOLS DO WRONG

| Tool/Phase | What it enforces | Missing constraints | Effect |
|------------|-----------------|---------------------|--------|
| slp_turbo (best: 0.073) | Nothing — pure gradient descent | C1, C2, C3, C4 | rank(H)=18, searches wrong space |
| phase2_grassmannian.py | rank(H)=10 via LP | C1, C3, C4 | Floor ~0.27, Δ still leaks |
| phase2_continuation_search.py | rank(H)=10 (continuation) | C1, C3, C4 | Floor ~0.27, Δ merit only |
| db_optimizer | None (minimax LP) | C1, C2, C3, C4 | Same as slp_turbo |

**None of the current tools enforce Gate 2 (Δ ⊂ span(H)) as a hard constraint.**

---

## SPECIAL STRUCTURES TO EXPLOIT

### The Omega Operator (§76 / Phase 37; Gram form discovered in §83 / Synthesis AB)
On the sigma-silent sector `Z = ker(H^T) ∩ ker(Δ^T)`:
- `Omega = 3·(Σ_Z · Σ_Z^T)^{-1}` is automatically symmetric positive definite (Gram structure)
- For valid decompositions, `dim(Z) = 9` and Omega is exactly invertible
- This gives an exact certificate: if `dim(Z) < 9`, the decomposition cannot be valid

### The Spectral Gap (§72 / Phase 34)
The channel-separation quadratic form `Q_chan(w) = Σ_{s<t} ‖W_s − W_t‖_F²` has positive
minimum eigenvalue on ker(Γ) for all known exact decompositions. Spectral collapse to 0 is
the exact defect condition for rank(H) < R−9.

### Conservation Law as Diagnostic
For any near-miss candidate, compute:
```
conservation_violation = |R + eta_nullity - 27|  where eta_nullity = 18 - rank(H)
```
For R=19 and rank(H)=10: `19 + 8 = 27` ✓. Any violation means Gate 1 is not met.

---

## REFERENCE TABLE: R=19 TARGETS

| Quantity | Value | Formula |
|---------|-------|---------|
| R | 19 | target rank |
| dim(ker Γ) | 10 | R − 9 |
| rank(H) target | 10 | = dim(ker Γ) |
| rank(Nuisance) target | 10 | = rank(H) when Δ ⊂ span(H) |
| η_nullity target | 8 | 18 − rank(H) = 18 − 10 |
| Conservation check | 27 | R + η_nullity = 19 + 8 = 27 ✓ |
| H columns | 18 | always (fixed by encoding) |
| Sigma columns | 9 | always (9 output fibers) |
| Delta columns | 54 | always (54 dead-X coordinates) |
| augmented_rank target | 19 | rank([Σ\|Nuisance]) = 10 + 9 |

---

## SOURCE SECTIONS IN CANON DOC

| Section | Lines | Content |
|---------|-------|---------|
| §40 (Step 51) | ~13284 | Exact matrix form, Sigma/Eta/Delta definitions |
| §41 (Step 52) | ~13366 | Quotient-space rank criterion, nuisance targets |
| §43 (Step 54) | ~13462 | Analytical low-nuisance construction |
| §54 (Phase 17) | ~14542 | Conservation law: Part 1 proved, Part 2 (Δ⊂span(H)) open |
| §55 (Phase 18) | ~14617 | Right-inverse kernel saturation |
| §71 (Phase 33) | ~15154 | Conservation law hand derivation and defect condition |
| §72 (Phase 34) | ~15212 | Spectral gap of channel-separation form |
| §76 (Phase 37) | ~15457 | Omega operator obstruction |
| §83 (Synthesis AB) | ~16043 | Full unification: sole remaining hypothesis is Δ⊂span(H) |
