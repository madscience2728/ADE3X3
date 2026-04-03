# Variety Search: Incremental Constraint Descent

**Goal:** Apply canon-derived constraints one at a time.
If the theory is right, each constraint lowers the floor.

**Canon reference:** `docs/ADE3x3_CANONICAL_OBJECT.md`

---

## Phase 0: Baseline Measurement
- [x] Measure current best (slp_turbo_best.json) fiber-mode profile
- [x] Record: nuisance_rank, rank(H), Delta containment residual, fitness
- [ ] Measure AlphaTensor R=23 profile as ground-truth comparison
- [x] Measure several cold-start alternating-L∞ fixed points

**Script:** `variety_search/phase0_baseline.py`
**Key question:** How far is the 0.073 best from satisfying the variety conditions?

**Results (2026-04-02):**
- ALL candidates: H_rank=18 (formal), nuis_rank=19, η_null=0, R+η=19 (target 27)
- SVD of H reveals natural gap: σ₀–σ₇ big, σ₈–σ₁₀ small, σ₁₂–σ₁₇ ≈ 0
- Tail energy (i>10) is only 0.0014% of total — effective rank ≈ 11-12
- Barrier: σ₁₀=0.35, σ₁₁=0.13 need to reach zero for rank(H)=10

---

## Phase 1: Constrain rank(H) ≤ 10
**Canon ref:** §41 (Quotient-Space Rank Criterion), §49 (Nuisance Budget Table)

For R=19: `rank(Nuisance) ≤ 10`, so `rank(H) = rank([Eta1|Eta2]) ≤ 10`.
H is (R×18) built quadratically from α,β. Constrain via penalty or projection.

- [x] Attempt 1: Alternating LP + gradient steps on rank penalty → LP undoes all rank progress
- [x] Attempt 2: Gradient on (α,β) + LP on γ only → still H_rank=18
- [x] Attempt 3: L-BFGS-B on ||R||²_F + λ·tail_energy → fitness 0.086, H_rank=18
- [ ] Attempt 4: Hard SVD projection — truncate H to rank 10, solve for nearest (α,β)
- [ ] Record best fitness and rank(H) achieved

**Key finding:** rank(H)=18 is a formal artifact. The actual SVD shows σ₁₂–σ₁₇ ≈ 0.
The effective rank is 11-12.  Only σ₁₀=0.35 and σ₁₁=0.13 separate us from rank 10.
Smooth optimizers can't cross the rank boundary (codimension-1 algebraic stratum).

**Script:** `variety_search/phase1_rank_H.py`
**Expected outcome:** If correct, penalizing high rank(H) guides into deeper basins.

---

## Phase 2: Constrain rank(Nuisance) ≤ 10
**Canon ref:** §41, §49

Full nuisance = [Eta1 | Eta2 | Delta], shape (R×72). Must have rank ≤ 10.
Stronger than Phase 1 (includes Delta).

- [ ] Add rank(Nuisance) penalty
- [ ] Multi-start with combined fitness + nuisance penalty
- [ ] Compare with Phase 1 results

**Script:** `variety_search/phase2_nuisance_rank.py`
**Expected outcome:** Tighter constraint → either deeper basin or proof of incompatibility.

---

## Phase 3: Enforce Delta ⊂ span(H)
**Canon ref:** §83 (Synthesis AB), §71 (Conservation Law)

The conservation law R + η_nullity = 27 requires all 54 Delta columns to lie
in span(H). Measure the projection residual: `||Delta - H @ lstsq(H, Delta)||`.

- [ ] Add Delta-containment residual as penalty
- [ ] Multi-start with fitness + containment penalty
- [ ] Compare with Phase 2 results

**Script:** `variety_search/phase3_delta_containment.py`
**Expected outcome:** This is the full variety condition. If satisfied with low fitness → we're on the variety.

---

## Phase 4: Enforce Gamma·Sigma = 3I₉ and Gamma·Nuisance = 0
**Canon ref:** §41 T2-T3

The exact solvability condition. Once α,β satisfy Phases 1-3,
γ is determined by the linear system.

- [ ] Given Phase 3 best (α,β), solve for γ exactly
- [ ] Measure fitness of the exact γ
- [ ] If fitness ≈ 0: we found an exact decomposition

**Script:** `variety_search/phase4_exact_gamma.py`
**Expected outcome:** Either fitness → 0 (exact decomposition!) or a proven obstruction.

---

## Progress Log

| Phase | Date | Best Fitness | rank(H) | rank(Nuis) | Delta Resid | Notes |
|-------|------|-------------|---------|------------|-------------|-------|
| — | — | 0.07342 | ? | ? | ? | slp_turbo_best baseline |
