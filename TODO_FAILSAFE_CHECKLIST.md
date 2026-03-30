# ADE3x3 Failsafe Checklist — Road to Rank ≤ 19

**Goal:** Exact rank-19 (or lower) decomposition of ⟨3,3,3⟩ matrix multiplication tensor.

**Current best:** R=23 exact (AlphaTensor). R=19 best inexact residual: 5.27e-4.

**Known sub-tensor results:** {A,B} rank-19 exact, [A,B] rank-20 exact (Step 75).

---

## Status Key

- [ ] Not started
- [~] In progress
- [x] Done
- [—] Ruled out / dead end

## Exhausted Routes (Do Not Revisit)

- [—] Warm-start continuation from R=23 (Phases 16, 23–30: identity-stratum obstruction)
- [—] Cold gradient restart at R≤22 (Phase 4: 592 runs, 0 exact hits)
- [—] Border rank singular-pair fusion (Step 77: 0/59 pairs worked)
- [—] Parity channel tightening (Phase 2: theorem is false)
- [—] Single/pair term removal from AlphaTensor (Phase 5: all 506 pairs give gain 0)
- [—] Factorized defect families (Phase 37: collapse Omega, can't sustain Delta-containment)

---

## 1. Hamilton Term-Sharing Analysis

**Priority: HIGH | Effort: Days | Uses existing data**

Step 75 found {A,B}=rank-19, [A,B]=rank-20, T = ½{A,B} + ½[A,B]. Naive union = 39 terms. Step 75 explicitly flags "sharing analysis deferred."

- [x] Extract the 19 anticommutator factor triples (alpha, beta, gamma) from Step 75 CSV
- [x] Extract the 20 commutator factor triples by reconstructing the Step 75 seed and exporting them
- [x] Check for scalar-multiple matches between any anticomm and comm term (direct cancellation)
- [x] Check if any small subset of terms across both sides is linearly dependent
- [x] Compute T - ½·(rank-19 anticomm reconstruction) and measure residual rank directly
- [x] If residual tensor has rank < 20, the combined rank < 39; find tightest upper bound
- [x] Document: best achievable combined rank from this split

**Route 1 verdict (Step 78):** no easy Hamilton sharing.

- Direct shared normalized rank-1 terms: **0**
- Span intersection dimension between the 19 anticommutator terms and 20 commutator terms: **0**
- Union span rank: **39**
- Best single-term projection into the opposite span: relative residual **~0.948**
- Principal-cosine overlap is modest only: top value **~0.405**
- Residual after subtracting ½·T_anti is exactly the commutator piece, with flattening ranks **[8,8,8]**
- Current tight upper bound from this split remains **39**, with no immediate algebraic compression visible

Exports created:
- `outputs/exports/step78_commutator_rank20_coefficients.csv`
- `outputs/exports/step78_hamilton_pairwise_overlap.csv`
- `outputs/exports/step78_hamilton_top_pairwise_overlap.csv`
- `outputs/exports/step78_hamilton_cross_span_membership.csv`
- `outputs/exports/step78_hamilton_principal_angles.csv`
- `outputs/exports/step78_hamilton_combined_term_table.csv`
- `outputs/exports/step78_summary.csv`

---

## 2. Basis-Rotated Search

**Priority: HIGH | Effort: Days | Reuses existing search code**

All searches so far use the standard computational basis. GL(9)³ acts on T without changing rank. A random basis change creates a different optimization landscape.

- [x] Implement preprocessing wrapper: sample random A,B,C ∈ GL(9), form T' = (A⊗B⊗C)·T
- [~] Run Phase 4 / Phase 16 search machinery on T' at R=19, 20, 21, 22
- [ ] Try 50–100 random basis rotations × existing restart count
- [ ] Try structured rotations: DFT basis, Hadamard, random orthogonal (condition-bounded)
- [ ] If any T' yields exact hit, recover original-basis factors via inverse transform
- [ ] Document: best residuals per rank under rotated bases

**Route 2 pilot verdict (Step 79):** negative at low budget; not yet worth escalation.

- Public rank-23 basis transport calibrates exactly on all tested rotations (residual ~1e-15), so the wrapper is correct
- Tested 4 rotated targets: 1 structured DCT basis and 3 random well-conditioned GL(9)^3 transforms
- Pilot scan budget: 4 cold restarts per rank for R = 19, 20, 21, 22
- Best pilot result: rotation `rot_gl_02`, rank 22, max-abs residual **7.15e-2**
- Best pilot rank-19 result: **1.15e-1**, far worse than current best **5.27e-4**
- No exact hits, and no rotated pilot beat the known rank-19 baseline

Exports created:
- `outputs/exports/step79_basis_rotated_results.csv`
- `outputs/exports/step79_basis_rotated_summary.csv`
- `outputs/exports/step79_basis_rotated_report.md`

Interpretation:
- The basis-change machinery is valid, but this first cold-start pilot does not show evidence that rotated coordinates open an easier basin.
- If Route 2 is revisited later, the next escalation should be a much larger restart budget and/or warm starts from transported rank-23 factors, not another tiny cold pilot.

---

## 3. Homotopy Continuation

**Priority: MEDIUM-HIGH | Effort: ~1 week | Mathematically rigorous global method**

Polynomial homotopy avoids gradient local minima by tracking all solution paths from a known start system.

- [x] Install Julia + HomotopyContinuation.jl (or Python PHCpack bindings)
- [ ] Formulate rank-R decomposition of T as polynomial system (R·27 unknowns, 729 equations)
- [ ] Construct start system: direct-sum tensor with known rank-19 structure
- [ ] Run parameter homotopy deforming start tensor → T, tracking all paths
- [ ] If path count is tractable (< 10^6), enumerate; otherwise sample monodromy loops
- [ ] Document: number of paths, convergence, any exact endpoints

**Route 3 status (Step 80 refresh):** backend ready; proceed to formulation.

- Julia 1.12.5 installed locally at `C:\Users\madsc\AppData\Local\Programs\Julia-1.12.5\bin\julia.exe`
- `HomotopyContinuation.jl` installed and loadable; Step 80 now reports `backend_ready = True`, `preferred_backend = julia`
- Minimal smoke test solved `x^2 - 1 = 0` with 2 tracked / 2 real solutions
- The mathematical caution remains unchanged: rank-19 affine CP coordinates still give 475 effective unknowns against 729 equations after gauge fixing, so the next step must use a square subsystem or parameter-homotopy design rather than a naive full-system solve

**Route 3 pilot verdict (Step 81):** GO for the homotopy bridge; the ADE3X3 export/track/reconstruct loop is now validated.

- Step 81 builds a support-restricted rank-23 square subsystem from the public exact witness in the ADE3X3 orientation
- After the obvious `2R` scaling gauge fix, the support chart still had 114 coordinates but witness Jacobian rank only 105, so 9 extra locally redundant coordinates were frozen at witness values before export
- HomotopyContinuation tracked 1 start path on a `105 x 105` subsystem and returned 1 solution
- Endpoint residual on the selected subsystem: **2.22e-16**
- Endpoint residual on the full 729-entry tensor: **2.22e-16**
- Endpoint matched the exact witness to machine precision with zero imaginary drift
- Interpretation: route 3 is no longer a tooling gamble; the next substantive task is to design an analogous square chart / start family for a nontrivial rank-19 or low-rank deformation problem

---

## 4. SAT / Finite-Field Exhaustive Search

**Priority: MEDIUM | Effort: ~1 week | Completely orthogonal approach**

Over GF(p), the search space is finite; SMT/SAT solvers can handle constrained combinatorial searches.

- [ ] Reduce T mod 2 and mod 3; verify tensor is well-defined over GF(2), GF(3)
- [ ] Encode rank-19 decomposition over GF(2) as SAT instance (or rank-22 first as easier target)
- [ ] Use CryptoMiniSat, Kissat, or Z3 to search; exploit S₃³ symmetry breaking
- [ ] If solution over GF(p) found → Hensel-lift to ℤ, then refine over ℝ
- [ ] If UNSAT over GF(2) at rank 19 → this is a rank lower bound certificate
- [ ] Document: SAT/UNSAT results per field per rank

---

## 5. Symmetry-Constrained Search

**Priority: MEDIUM | Effort: Days | Complements route 2**

Use S₃×S₃×S₃ (order 216) orbit structure to reduce search dimensionality.

- [ ] Enumerate orbit types for rank-1 terms under the group action
- [ ] Parameterize rank-19 decompositions as unions of complete orbits + fixed-point residuals
- [ ] Reduced parameter count: ~(# orbit reps) × (rep dimension) instead of 19×27
- [ ] Run gradient/ALS search in reduced parameter space
- [ ] Document: parameter reduction ratio, best residuals

---

## 6. Recursive / Block Decomposition

**Priority: MEDIUM-LOW | Effort: ~1 week | Classical techniques**

- [ ] Try ⟨3,3,3⟩ as sub-tensor of ⟨4,4,4⟩ (pad with zeros, restrict known decompositions)
- [ ] Try Kronecker-product decompositions: ⟨3,3,3⟩ via ⟨3,1,3⟩⊗⟨1,3,1⟩ etc.
- [ ] Survey Duan/Wu/Zhou (2023) and similar for finite-dimensional adaptations of laser method
- [ ] Document: any new upper bound constructions

---

## 7. Reinforcement Learning (Long-shot)

**Priority: LOW feasibility / HIGH payoff | Effort: Weeks+**

- [ ] Replicate AlphaTensor action-space for ⟨3,3,3⟩ at smaller scale
- [ ] Incorporate Omega/conservation-law structure as reward shaping
- [ ] Target R=22 first (smaller gap from known R=23)
- [ ] Document: training curves, best decompositions found

---

## Decision Gate

After completing **Route 1 (Hamilton sharing)** and **Route 2 (basis rotation)**:

- If combined Hamilton rank < 23 → pursue that construction aggressively
- If basis-rotated search finds exact R≤22 → celebrate and verify
- If both fail → commit to Route 3 (homotopy) or Route 4 (SAT) as the next major investment
- If Routes 3+4 also fail → the evidence strongly suggests R(⟨3,3,3⟩) = 23

---

*Last updated: 2026-03-29. Created from audit of 37 attack phases + 77 steps.*
