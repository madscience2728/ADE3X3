# Gate Progress — Algebraic Experiments

## Summary

Ten pure-algebra experiments (no optimization) evaluated structural properties
of the symmetric sector at generic random points across all 13 orbit configurations.

**Key result**: Gate 2 is generically free. Gate 3 (Sigma rank deficiency) is the
universal bottleneck.

---

## Experiment Results

### EXP 1 — Generic ranks of all blocks

| R | Orbits | eff_p | rk(Σ) | rk(H) | rk(Δ) | rk(N) | tgt_H | gap_H | gap_N | rk([Σ\|N]) | G3? |
|--:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|----------:|----:|
| 7 | [1,6] | 12 | 3 | 3 | 3 | 3 | −2 | 5 | 5 | 3 | no |
| 8 | [8] | 6 | 4 | 8 | 8 | 8 | −1 | 9 | 9 | 8 | no |
| 9 | [1,8] | 9 | 5 | 9 | 9 | 9 | 0 | 9 | 9 | 9 | no |
| 12 | [12] | 12 | 2 | 4 | 4 | 4 | 3 | 1 | 1 | 4 | no |
| 13 | [1,12] | 15 | 3 | 5 | 5 | 5 | 4 | 1 | 1 | 5 | no |
| 14 | [6,8] | 15 | 6 | 10 | 10 | 10 | 5 | 5 | 5 | 10 | no |
| 15 | [1,6,8] | 18 | 7 | 11 | 11 | 11 | 6 | 5 | 5 | 11 | no |
| 18 | [6,12] | 21 | 4 | 6 | 6 | 6 | 9 | −3 | −3 | 6 | no |
| 19 | [1,6,12] | 24 | 5 | 7 | 7 | 7 | 10 | −3 | −3 | 7 | no |
| 20 | [8,12] | 18 | 6 | 12 | 12 | 12 | 11 | 1 | 1 | 12 | no |
| 21 | [1,8,12] | 21 | 7 | 13 | 13 | 13 | 12 | 1 | 1 | 13 | no |
| 26 | [6,8,12] | 27 | 7 | 13 | 14 | 14 | 17 | −4 | −3 | 14 | no |
| 27 | [1,6,8,12] | 30 | 8 | 14 | 15 | 15 | 18 | −4 | −3 | 15 | no |

Gate 3 fails for **every** configuration generically.

### EXP 2 — Delta leak is an algebraic constant

rk(N) − rk(H) = 0 for all R ≤ 21. Exactly 1 for R=26,27 (configs
containing Edge+Interior+Face). This is a structural invariant.

### EXP 3 — Column-space containment

Δ lies entirely in col(H) for all R ≤ 21 (residual rank = 0).
Only R=26,27 have 1 dimension of Delta leaking outside span(H).

### EXP 4 — Eta1 vs Eta2 structure

Per-seed Eta ranks:

| Seed | rk(Eta1) | rk(Eta2) |
|-----:|---------:|---------:|
| Corner | 1 | 0 |
| Edge | 2 | 0 |
| Face | 4 | 2 |
| Interior | 8 (=7 standalone) | 4 |

Eta2 is generically much smaller than Eta1. Overlap (E1∩E2 dim) ranges 0–3.

### EXP 5 — Outer-product rank

rk([α_k ⊗ β_k]) = rk(Nuisance) for every R. The row spaces coincide exactly —
terms in the same orbit contribute identical row-space directions (massive redundancy).

### EXP 6 — Sigma is universally deficient

No configuration achieves rk(Σ) = 9 generically. Maximum is rk(Σ) = 8 at R=27.
Gate 3 is the universal bottleneck.

| R | rk(Σ) | deficit |
|--:|------:|--------:|
| 7 | 3 | 6 |
| 8 | 4 | 5 |
| 12 | 2 | 7 |
| 13 | 3 | 6 |
| 19 | 5 | 4 |
| 20 | 6 | 3 |
| 27 | 8 | 1 |

### EXP 7 — Conservation law R + η_null

| Group | R values | R + (18 − rk(H)) | deficit from 27 |
|------:|---------|------------------:|----------------:|
| Close | 12, 13, 20, 21 | 26 | 1 |
| Overshoot | 18, 19 | 30 | −3 |
| Far | 7, 14, 15 | 22 | 5 |
| Farthest | 8, 9 | 18 | 9 |
| Large | 26, 27 | 31 | −4 |

R=18,19 overshoot: generic rk(H) is too low, conservation says they have room.

### EXP 8 — Seed rank is perfectly additive

| Seed | H rank contribution |
|-----:|-------------------:|
| Corner (0,0,0) | 1 |
| Edge (0,0,1) | 2 |
| Face (0,1,1) | 4 |
| Interior (1,1,1) | 8 |

Combined rank = sum of individual contributions for every configuration tested.
No overlap between orbit contributions in H.

### EXP 9 — Per-channel Delta leak

All 6 channels Δ[s,t] (s≠t) have zero leak off H for R=13, 19, 20.
Individual channel ranks vary (4–9) but all lie within col(H).

### EXP 10 — Jacobian rank of the bilinear map

| R | params → output dim | Jacobian rank |
|--:|--------------------:|--------------:|
| 7 | 36 → 567 | 11 |
| 13 | 36 → 1053 | 14 |
| 19 | 54 → 1539 | 22 |
| 20 | 36 → 1620 | 17 |
| 27 | 72 → 2187 | 28 |

For R=19: 22 of 24 effective parameters are "visible" in the (Σ,H,Δ) image,
leaving a 2-dimensional kernel of the bilinear map.

---

## Implications

1. **Gate 2 is not the obstacle.** Generically rk(N) = rk(H), so Δ ⊂ span(H)
   holds at every generic point for all R ≤ 21.

2. **Gate 3 is the universal bottleneck.** rk(Σ) < 9 for every configuration.
   The missing Sigma dimensions must come from special parameter values where
   γ corrections compensate.

3. **R=19 specifics**: rk(H)=7 (need 10), rk(Σ)=5 (need 9), conservation
   overshoots by 3. The 3-unit gap in rk(H) and the 4-unit gap in rk(Σ) must
   both be closed simultaneously on a special subvariety.

4. **The gate2_algebra.py obstruction for R=13,20** (rk(N)=rk(H)+1 on the
   rank(H)=target subvariety) is a **non-generic** phenomenon: the leak appears
   only when you force rank(H) down to target. Generically, Gate 2 is free.

---

## Phase 4 Results (2026-04-04)

### Tool 1 — AlphaTensor 4-Term Deletion Search

- **Search space**: C(23,4) = 8,855 deletions (exhaustive)
- **Result**: ZERO deletions pass Gate 1. ALL pass Gate 2 and Gate 3.
- **Root cause**: AlphaTensor R=23 has rank(H)=14. Deleting 4 terms reduces rank
  by at most 2 (best observed: rank(H)=12, gate1_gap=2). Cannot reach rank(H)=10.
- **Conclusion**: R=19 is definitively NOT obtainable by 4-term deletion from AlphaTensor.
- **Output**: `CANON_DATABASE/alphatensor_delete_results.json` (top 100 by score)

### Tool 2 — Full-Packet Swap Optimizer (1-hour run, 20 workers)

- **Starting point**: random Gate-1-satisfying 19-term packets from TermDB
- **Best score reached**: `(0, 0, ~7.5e-15, 9, 2.8)` — tuple is `(gate1_gap, delta_leak, delta_resid, augmented_gap, recon_err)`
- **Gate 1** (gate1_gap=0): satisfied from early in the run.
- **Gate 2** (delta_leak=0, delta_resid~0): achieved at ~759s after starting at delta_leak=2.
  This is significant — Gate-2-neutral and Gate-1-satisfying held simultaneously,
  disproving the Phase 3 disjointness finding for individual terms (it does not
  extend to full R-packets).
- **Gate 3** (augmented_gap=9): NOT satisfied. rank([Sigma|H|Delta]) = 10, need 19.
  Gap of 9 is unchanged throughout 3.8M swaps tested in the Pareto neighborhood.
  Allowing Gate 2 violations (delta_leak > 0) buys zero improvement in augmented_gap.
  Exchange rate = 0.
- **Active bottleneck**: Gate 3 / augmented rank. The packet reconstruction error
  (~2.8) confirms gamma coefficients cannot fit the target tensor with the
  current H+Sigma structure.
- **Output**: `CANON_DATABASE/swap_improvements.jsonl`, `CANON_DATABASE/swap_checkpoint.json`
- **Tools**: `CANON_DATABASE/scripts/swap_*.py`, `CANON_DATABASE/scripts/alphatensor_delete.py`

---

## Phase 5 Results — Quotient Obstruction Test (2026-04-04)

### The Determinantal Criterion

For a rank-R packet, let K = col(H) ⊆ ℝᴿ (a TGT-plane) and U span K⊥. Then:

- Gate 3 ⟺ rank(U Σ) = 9

Script: `CANON_DATABASE/quotient_test.py`

### R=19 — 1,184 Gate-1 packets (296 files × ~4 draws)

Full (r_Δ, r_Σ) histogram after tolerance fix (absolute floor 1e-9 on SVD):

| (r_Δ, r_Σ) | count |  | (r_Δ, r_Σ) | count |
|------------|------:|--|------------|------:|
| (3,0) | 8 | | (8,0) | 109 |
| (3,1) | 98 | | (8,1) | 23 |
| (3,3) | 200 | | (8,2) | 14 |
| (4,0) | 28 | | (8,3) | 10 |
| (5,0) | 64 | | (8,4) | 19 |
| (5,1) | 7 | | (8,5) | 4 |
| (6,0) | 134 | | (9,0) | 41 |
| (6,1) | 15 | | (9,1) | 23 |
| (7,0) | 155 | | (9,2) | 53 |
| (7,1) | 16 | | (9,3) | 67 |
| (7,2) | 4 | | (9,4) | 54 |
| (7,3) | 2 | | (9,5) | 25 |
| | | | (9,6) | 8 |
| | | | (9,7) | 3 |

**Key findings (corrected after tolerance bug fix):**
- r_Δ=0 does NOT appear in the sweep, but this is a **selection artifact**: packets were generated under a Gate-1-only criterion, not Gate-2 optimized. The Phase-4 best (swap-optimized) DID achieve r_Δ=0.
- r_Σ=9 disappears entirely after the fix — those were numerical noise from relative tolerance applied to near-zero singular values.
- The Phase-4 best packet: rank(H)=10 ✓, r_Δ=0 ✓ (Gate 2), r_Σ=0 (Gate 3 blocked).
- **Gate 3 is the obstruction**, exactly as canon states.

**Tolerance bug:** The `rank()` function using `tol = 1e-9 * sv[0]` inflates rank when `sv[0]` is itself near machine epsilon (~1e-15), making the threshold ~1e-24 and counting noise as genuine singular values. Fix: `threshold = max(1e-9 * sv[0], 1e-9)` (absolute floor).

### Cross-Rank Confirmation (2026-04-04, corrected)

Extended sweep to R=13 (target rank(H)=4) and R=20 (target rank(H)=11). All have null_dim=9.

| R | packets tested | r_Δ=0 in sweep | r_Σ=9 in sweep | (0,9) hits | Phase-best (r_Δ, r_Σ) |
|---|---|---|---|---|---|
| 13 | 40 | 0 (selection artifact) | 0 | 0 | unknown |
| 19 | 1184 | 0 (selection artifact) | 0 | 0 | (0, 0) — Gate 3 blocked |
| 20 | 1024 | 0 (selection artifact) | 0 | 0 | unknown |

The sweep packets are Gate-1-only generated — none were Gate-2 optimized, so r_Δ=0 absence is expected, not a theorem. For R=19, the swap optimizer DID find r_Δ=0, confirming Gate 2 is achievable. Gate 3 (r_Σ=9) has not been achieved at any r_Δ value across all ranks tested.

### The True Obstruction

**Gate 3 is the bottleneck**, exactly as the canon states. The Phase-4 best sits at (r_Δ=0, r_Σ=0): Gates 1+2 simultaneously satisfied, Gate 3 blocked. The question is whether there exists any packet — integer or real — achieving (r_Δ=0, r_Σ=9).

### Open Question

Does there exist a 19-term set from TermDB (or its real relaxation) satisfying Gates 1+2 with r_Σ=9? If not, what is the maximum r_Σ achievable subject to r_Δ=0? A targeted search starting from the Phase-4 best and exploring its neighborhood under Gate-2-preserving moves would bound this.

---

## Phase 6 Results — Unconditional Gate-3 Sweep (2026-04-04)

### The aug_gap = 0 Universal Identity

Script: `CANON_DATABASE/gate3_free_sweep.py`

**The augmented rank criterion** (Gate 3 unconditional form):

$$\text{aug\_gap} = \text{rank}([\Sigma \mid \text{Nuisance}]) - \text{rank}(\text{Nuisance})$$

Gate 3 requires aug_gap = 9. Tested across 16,432 total draws:

| R | Random draws | Packet draws | aug_gap observed | aug_gap=9? |
|---|---|---|---|---|
| 13 | 5000 | 40 | **0 only** | Never |
| 19 | 5000 | 1184 | **0 only** | Never |
| 20 | 5000 | 1024 | **0 only** | Never |

**aug_gap = 0 universally.** For every (α,β) pair drawn from the integer TermDB,
col(Σ) ⊆ col([H | Δ]) exactly. No exceptions across any rank or draw strategy.

### Consequence

This is not a search landscape issue. It is an algebraic identity: for any pair of
3×3 integer matrices α,β over {−1,0,1}, the product σ = αβ (flattened to 9 entries)
lies in the span of {η₁, η₂, δ_{st}} — the H and Delta columns generated by the same
(α,β).

**If this identity holds over ℝ** (not just over {−1,0,1}), then Gate 3 is permanently
blocked by the Step-51 coordinate structure itself, and R=19 (and R=13, 20) are provably
impossible regardless of which terms are chosen.

**If the identity only holds over {−1,0,1}** (integer alphabet), then real-valued (α,β)
can escape it, and a continuous relaxation from the best integer packet could reach
aug_gap > 0.

### The Identity to Prove or Refute

For α,β ∈ Mat(3×3),  define per the Step-51 coordinates:

- $\sigma[r,u] = \sum_s \alpha[r,s]\beta[s,u]$ (diagonal products)
- $\eta_1[r,u] = \alpha[r,0]\beta[0,u] - \alpha[r,1]\beta[1,u]$
- $\eta_2[r,u] = \alpha[r,1]\beta[1,u] - \alpha[r,2]\beta[2,u]$  
- $\delta[r,s,t,u] = \alpha[r,s]\beta[t,u]$ for $s \neq t$ (off-diagonal Kronecker pieces)

**Claim:** For all α,β ∈ Mat(3×3,ℝ), σ ∈ span{η₁, η₂, δ} as vectors in ℝ⁹.

Proof sketch: Note σ = η₁ + η₂ + α[:,2]·β[2,:] and also
σ[r,u] = Σ_s α[r,s]β[s,u] = (diagonal sum). The δ terms include all
α[r,s]β[t,u] with s≠t. With η₁, η₂ providing the s=t differences, the full
αβ product is constructible from {η₁, η₂, δ}. **This is a universal identity
over ℝ.** No alphabet restriction needed.

### Critical Implication

If the proof sketch above is correct, **aug_gap = 0 is a theorem over ℝ**, not just
a numerical observation over {−1,0,1}. This means:

> **Gate 3 is permanently and provably blocked by the Step-51 coordinate definition.**
> The augmented rank can never reach rank(Nuisance) + 9 for any (α,β), over any field.

This would mean the solvability criterion from CANON_CONSTRAINTS.md (§T2–T3) cannot
be satisfied by the Step-51 coordinate map — and the R=19 problem requires either a
different coordinate system or genuine impossibility.

### Next Steps

1. **Verify the identity algebraically** — prove or refute col(Σ) ⊆ col([H|Δ]) over ℝ.
2. **If identity holds over ℝ:** the canon's solvability criterion is vacuous; revisit the
   Step-51 derivation for an error in how aug_gap relates to Γ-solvability.
3. **If identity only holds over ℤ:** continuous relaxation from Phase-4 best can escape
   it — go to step 4 of the plan.

---

## Phase 7 Results — Symmetry Breaking Test (2026-04-04)

### Correction to Phase 6 "Universal Identity" Claim

**The Phase 6 conclusion was partially misleading.** The aug_gap=0 observation for
random draws was **trivially zero** — for 199/200 random draws, rank(Nuisance) = R = 19
(full row rank). Since Sigma ∈ ℝ^19 and col(Nuisance) = ℝ^19, aug_gap=0 says only
"sigma is a vector in ℝ^19." This is vacuous.

The ONLY genuinely informative aug_gap measurement is for the Phase-4 best, where
rank(Nuisance) = 10 (proper subspace). There, aug_gap=0 means col(Σ) ⊆ col(H)
non-trivially — sigma lies in the same 10-dim subspace as η₁, η₂, and Δ.

The Phase 6 "proof sketch" was wrong: the claim σ ∈ span{η₁,η₂,δ} as 9-vectors
is trivially true via the algebraic identity σ = η₁ + η₂ + 2d₂ combined with
d₂ = α[:,2]⊗β[2,:] which cannot be written as a fixed linear combination of
{η₁, η₂, δ} unless d₂ ∈ col(Nuis), which requires col(Nuis) = ℝ^R (trivial).

### Key Experimental Findings

Script: `CANON_DATABASE/symmetry_break_test.py`

**Phase-4 best baseline (integer values):**
| rank(H) | rank(Nuis) | rank(UΣ) | aug_gap_H |
|--------:|-----------:|---------:|----------:|
| 10 | 10 | 0 | 0 |

**Perturbation sweep (real-valued noise, ε = 0.001 to 0.5):**

Any real perturbation, however small, **immediately destroys Gate-1+2**:
- rank(H) jumps from 10 → 18
- rank(Nuis) jumps from 10 → 19
- aug_gap_H = 1 (sigma escapes col(H) generically!)

| ε | rank(H) | rank(Nuis) | rank(UΣ) |
|----:|--------:|-----------:|----------:|
| 0.000 | 10 | 10 | 0 |
| 0.001 | 18 | 19 | 1 |
| 0.010 | 18 | 19 | 1 |
| 0.100 | 18 | 19 | 1 |
| 0.500 | 18 | 19 | 1 |

**Gate-1+2-preserving random search (14,000 trials across 6 ε values):**
- Found **zero** Gate-1+2 configurations among 14,000 random real perturbations
- Gate-1+2 is a **discrete phenomenon**: achievable only at isolated integer-lattice points

### Critical Conclusions

1. **Col(Σ) ⊆ col(H) is NOT a theorem over ℝ.** Generically (perturbed), sigma
   escapes col(H) with aug_gap_H = 1. The identity only holds at special discrete points.

2. **Gate-1+2 is a discrete/combinatorial condition.** The manifold of Gate-1+2
   configurations has measure zero in the real parameter space — it only manifests at
   isolated integer-alphabet solutions. Continuous symmetry breaking does not apply.

3. **The Phase-4 best is the unique known Gate-1+2 point.** Among all explored configurations
   (Phase-4 swap optimizer), only this one packet achieves rank(H) = rank(Nuis) = 10.
   For this specific packet, col(Σ) ⊆ col(H) holds as an **accident of the discrete structure**,
   not as a consequence of a real algebraic identity.

4. **Algebraic structure:** σ = η₁ + η₂ + 2d₂, where d₂ = α[:,r,2]⊗β[:,2,u]. For d₂ ∈ col(H),
   one sufficient condition is α[:,r,2] ∈ span{α[:,r,0], α[:,r,1]} as R-vectors (then d₂ is a
   fixed combination of delta columns in col(H) via Gate 2). Whether this or a symmetric
   condition on β is forced by Gate 1+2 over integers is an open combinatorial question.

### Mathematical Resolution (Phase 7 Derivation)

From $\eta_1 = d_0 - d_1$ and $\eta_2 = d_1 - d_2$ both in col$(H)$ by definition, left-multiplying by $U = \ker(H^T)$:

$$U(d_0 - d_1) = 0, \quad U(d_1 - d_2) = 0 \implies UD_0 = UD_1 = UD_2$$

$$\therefore \quad U\Sigma = U(D_0 + D_1 + D_2) = 3 \cdot UD_0$$

**Gate 3 reduces to rank$(UD_0) = 9$.** Gate 3 is not a constraint on σ at all — it is a constraint on the first-column diagonal rank-1 pieces $d_0^{(k)} = \alpha_{r0}^{(k)}\beta_{0u}^{(k)}$.

Gates 1+2 constrain the *off-diagonal* pieces $\delta_{st}$ (s≠t) to lie in col$(H)$, but say **nothing** about whether $D_0 \in$ col$(H)$. These are orthogonal constraints on different parts of the CP structure.

The Phase-4 best has $UD_0 = 0$ — an **additional accidental coincidence** beyond Gates 1+2. The search was not rewarded for $\|UD_0\| > 0$, so it settled here.

### New Objective

$$\text{Score} = \text{gate1\_gap} + \text{delta\_leak} - \lambda \cdot \|UD_0\|_F^2$$

This directly targets Gate 3 without conflicting with Gates 1+2. Script: `CANON_DATABASE/ud0_optimizer.py`

---

## Phase 8 Results — UD₀-Targeted Optimizer (2026-04-04)

Script: `CANON_DATABASE/ud0_optimizer.py`

### Phase 8 First Run (old objective: lex on gate1, delta_leak, ‖UD₀‖)

- Best global: (gate1=0, dleak=0, ‖UD₀‖²=0) — stuck at Phase-4 best
- **But**: worker 4 transiently reached gate1=0, rk(UD₀)=1, delta_leak=1 at t≈85min
- The checkpoint only saved the lexicographic global best (gate1=0, dleak=0), discarding the rk=1 discovery

**Root cause:** Lexicographic (gate1, delta_leak, -‖UD₀‖) treats any delta_leak>0 as worse
than delta_leak=0, even if rk(UD₀) improves. This prevents the optimizer from exploring
the (gate1=0, dleak>0, rk>0) region.

### Phase 8 Second Run (fixed objective, 6 workers, 60 min target)

New objective: `(gate1_gap, -rk_ud0, delta_leak, -‖UD₀‖², recon_err)` — rk prioritized over Gate 2.

All rk>0 events logged to `CANON_DATABASE/ud0_rk_log.jsonl`.

**Early output (t ≈ 8 min):**
| time | worker | gate1 | rk(UD₀) | dleak | ‖UD₀‖² | recon |
|-----:|-------:|------:|--------:|------:|-------:|------:|
| 5.3 | 4 | 0 | **1** | 3 | 6.0000 | 1.857 | ← first crack |
| 8.4 | 4 | 0 | **1** | 3 | 6.0000 | 1.728 | improving |

**Wall cracked: rk(UD₀)=1 achieved**, delta_leak=3. Full Gate 3 requires rk=9.

---

## Phase 9 Results — Impossibility Theorem (R=13, 19, 20) (2026-04-04)

Script: `CANON_DATABASE/d0_identity_test.py`

### The D₀ Containment Identity

From the Phase 8 run, all 370 logged events satisfy **rk(UD₀) ≤ delta_leak** exactly (min_dleak(rk=k) = k for all k observed). This is not coincidence — it is a structural theorem.

**Theorem:** For any R-packet over {-1,0,1}, $D_0 \in \text{col}([H; \Delta])$.

**Proof:**
1. $\eta_1 = D_0 - D_1 \in \text{col}(H)$ and $\eta_2 = D_1 - D_2 \in \text{col}(H)$ by construction
2. $\Sigma = D_0 + D_1 + D_2 = 3D_0 - 2\eta_1 - \eta_2$
3. aug\_gap = 0 (Phase 6) means $\Sigma \in \text{col}([H; \Delta])$
4. Since $\eta_1, \eta_2 \in \text{col}(H) \subset \text{col}([H;\Delta])$, we get $3D_0 \in \text{col}([H;\Delta])$
5. Over ℝ, $3 \neq 0$, so $D_0 \in \text{col}([H;\Delta])$ ∎

**Numerical verification:** 300/300 random packets at each of R=13, 19, 20 confirm rank([H;Δ;D₀]) = rank([H;Δ]).

### Corollary: Gates 2 and 3 Are Mutually Exclusive

Let $U = \ker(H)$ (right null space, shape $(R-\text{rank}(H)) \times R$). Then:

$$D_0 \in \text{col}([H;\Delta]) \implies UD_0 \in U \cdot \text{col}(\Delta)$$

$$\text{rank}(UD_0) \leq \dim(U \cdot \text{col}(\Delta)) \leq \text{delta\_leak}$$

| Gate | Condition | Implication |
|------|-----------|-------------|
| Gate 2 | delta\_leak = 0 | rank(UD₀) ≤ 0, so rank(UD₀) = 0 |
| Gate 3 | rank(UD₀) = 9 | delta\_leak ≥ 9 |
| Gates 2 + 3 | both | **Contradiction** — impossible |

**Confirmed empirically:** Over 370 rk>0 events in the Phase-8 optimizer run, min\_dleak at rk=k equals k for k = 1…6. No event ever had rk > 0 with dleak = 0.

### Scope of the Impossibility

This impossibility holds for **all three candidate ranks**:

| R | target rank(H) | null\_dim | Status |
|---|---|---|---|
| **13** | 4 | 9 | **IMPOSSIBLE** — Gates 2+3 mutually exclusive |
| **19** | 10 | 9 | **IMPOSSIBLE** — Gates 2+3 mutually exclusive |
| **20** | 11 | 9 | **IMPOSSIBLE** — Gates 2+3 mutually exclusive |

The theorem applies to any R where Gate 3 requires rank$(UD_0) = 9$ — i.e., all R with null\_dim = 9.

### Conclusion

**There is no rank-19 (or rank-13, or rank-20) CP decomposition of the 3×3 matrix multiplication tensor over the {-1, 0, 1} alphabet under the Step-51 coordinate system.** The obstruction is not a search limitation — it is a proven algebraic identity (D₀ ∈ col([H;Δ])) that makes Gates 2 and 3 structurally incompatible.

The path forward requires either:
1. A fundamentally different coordinate system (not Step-51), or
2. Allowing γ coefficients outside {−1,0,1} (real-valued leaf multiplications), or
3. Demonstrating that the aug\_gap=0 identity fails for real-valued α, β (it does — Phase 7 confirmed col(Σ) ⊄ col(H) generically over ℝ), motivating a purely real relaxation beyond the integer TermDB.

---

## Phase 10 Results — The Lever: Circular Argument Identification and GAP Algorithm (2026-04-04)

### Diagnosis: Phase 9 Is a Circular Argument

The Phase 9 "impossibility theorem" was identified as circular upon reading the canon index.

The proof chain is:
1. $\eta_1, \eta_2 \in \operatorname{col}(H)$ — **true by definition** (they are columns of H)
2. $\Sigma \in \operatorname{col}([H;\Delta])$, i.e., aug\_gap = 0 — **taken from Phase 6 observation**
3. Therefore $3D_0 = \Sigma + 2\eta_1 + \eta_2 \in \operatorname{col}([H;\Delta])$ ∎

Step 2 is the load-bearing claim. But Phase 7 proved that Phase 6's aug\_gap=0 was **vacuous for all random draws**: when rank(Nuisance) = R = 19, every vector in ℝ¹⁹ trivially lies in col([H;Δ]). The only non-vacuous measurement was at the **single Phase-4 best integer point** where rank(Nuisance) = 10. So Phase 9 "proves" D₀ ∈ col([H;Δ]) by assuming aug_gap=0 at the one point where the condition is non-trivial — the exact thing it is trying to explain.

**The Phase 9 theorem is not a universal algebraic identity. It is a tautological observation about one discrete lattice point.**

### Is D₀ ∈ col([H;Δ]) Algebraically Forced Over ℝ?

No. Here is the analysis.

$D_0[:,u] \in \operatorname{col}([H;\Delta])$ as ℝ¹⁹-vectors requires:

$$U \cdot D_0[:,u] = 0 \quad \forall u=0,\ldots,8$$

where $U = \ker([H;\Delta]^T)$ is the 9-dimensional left null (when Gate-1+2 is satisfied). Gate-2 forces $U \cdot \delta_{st}[:,u] = 0$ for all off-diagonal $s \neq t$, where $\delta_{st}[k,u] = \alpha_{k,r,s}\beta_{k,t,u}$. But $D_0[k,u] = \alpha_{k,r,0}\beta_{k,0,u}$ — the **diagonal** $s=t=0$ term. There is no polynomial identity relating $\alpha_{k,r,0}\beta_{k,0,u}$ (diagonal) to $\alpha_{k,r,s}\beta_{k,t,u}$ for $s \neq t$ for general real $\alpha, \beta$.

Gates 1+2 constrain the off-diagonal pieces $\delta_{st}$ to lie in $\operatorname{col}(H)$, but say **nothing** about whether $D_0 \in \operatorname{col}(H)$. These are algebraically independent constraints on different parts of the bilinear structure.

### Where The System Can Actually Break

The inference chain is:

$$\underbrace{D_0 \in \operatorname{col}([H;\Delta])}_{\text{not algebraically forced over } \mathbb{R}} \implies UD_0 = 0 \implies U\Sigma = 0 \implies aug\_gap = 0$$

The first implication is the breakpoint. It holds accidentally at all known integer Gate-1+2 solutions — a discrete lattice coincidence, not a continuous identity.

**The Gate-1+2 real manifold $\mathcal{M}$** is defined by:

$$\mathcal{M} = \{(\alpha,\beta) \in \mathbb{R}^{19\times9\times2} : \operatorname{col}([H(\alpha,\beta)\mid\Delta(\alpha,\beta)]) = V,\ V \in \operatorname{Gr}(10,19)\}$$

On $\mathcal{M}$, aug\_gap $= \operatorname{rank}(P_{V^\perp}\Sigma)$ is a continuous integer-valued function. At known integer points it equals 0. **There is no algebraic reason it must equal 0 everywhere on $\mathcal{M}$.** The generic value — and thus the Gate-3-satisfying value — is 9.

### Why All Previous Optimizers Failed

| Tool | rank(H) achieved | On variety? | Reason |
|------|-----------------|-------------|--------|
| slp_turbo (best 0.073) | 18 | No — conservation = 19 ≠ 27 | Never reached $\mathcal{M}$; descends in generic open set |
| real_cp_optimizer (best 3.0) | 18 | No | ALS lives on rank(H)=18 stratum by default |
| phase4 swaps | 10 ✓ | Partial — Gate-2 not hard | Stays near integer lattice degenerate points |
| ud0_optimizer | 10 ✓ | Partial — Gate-2 soft | Empirically locked to rk = dleak (sees the accident) |

None of these tools enforce Gate-2 as a **hard continuous constraint** over ℝ. Consequence: they either miss $\mathcal{M}$ entirely, or they can only visit the integer-degenerate corners where aug\_gap = 0 accidentally.

### The Correct Algorithm: GAP (Grassmannian Alternating Projection)

Fix $V \in \operatorname{Gr}(10, 19)$. The joint Gate-1+2 constraint with subspace $V$ is:

$$\forall k, u: \quad H[k,:] \in V, \quad \Delta[k,:] \in V$$

For **fixed $V$** and fixed $\alpha_k$, this is **linear in $\beta_k$** (projection onto the subspace defined by the constraints). For **fixed $(\alpha_k, \beta_k)$**, the optimal $V$ is the top-10 left singular subspace of $[H \mid \Delta]$.

**GAP Algorithm:**
1. Initialize random $V \in \operatorname{Gr}(10, 19)$; initialize $(\alpha_k, \beta_k)$ randomly
2. Fix $V$: for each $k$, solve $\beta_k$ minimizing $\|P_{V^\perp}[\eta_1[k,:],\eta_2[k,:],\delta_{st}[k,:]]\|^2$ (linear projection)
3. Fix $(\alpha_k, \beta_k)$: update $V =$ top-10 left singular subspace of $[H \mid \Delta]$
4. At convergence: compute aug\_gap = rank($P_{V^\perp}\Sigma$)
   - If aug\_gap = 9: solve $\Gamma = 3(P_{V^\perp}\Sigma)^{-1}P_{V^\perp}$ and verify exact reconstruction
   - If aug\_gap = 0: we are at a degenerate integer-accident point — perturb $V$ and restart
5. Repeat with diverse initializations

Once $(\alpha,\beta)$ satisfies Gates 1+2 (i.e., sits on $\mathcal{M}$), $\Gamma$ is **uniquely determined** by:

$$\Gamma = 3 \cdot (P_{V^\perp}\Sigma)^{-1} \cdot P_{V^\perp}$$

This automatically satisfies $\Gamma \cdot [H \mid \Delta] = 0$ (since $V = \ker(\Gamma)$) and $\Gamma \cdot \Sigma = 3I_9$. The exact multiplication verification reduces to a single linear check.

### Key Diagnostic: aug_gap Must Be Monitored On $\mathcal{M}$

The new invariant to track is:

$$aug\_gap\_on\_M = \operatorname{rank}(P_{V^\perp}\Sigma) \quad \text{where } V = \operatorname{col}([H\mid\Delta])$$

| Value | Interpretation |
|-------|---------------|
| 0 | Degenerate integer-accident point — $\Sigma \subset V$, Gate 3 blocked |
| 1–8 | Partial escape from the accident locus |
| 9 | **Gate 3 satisfied** — solve for Γ immediately |

Any optimizer that observes aug\_gap\_on\_M > 0 while enforcing Gates 1+2 as hard constraints will be the first to reach a genuine Gate-3 candidate over ℝ.

### Conclusion

The Phase 9 impossibility is not a proof of impossibility over ℝ — it is a proof that the integer lattice is a degenerate locus where $D_0$ accidentally falls into $\operatorname{col}([H;\Delta])$. The correct target space is the smooth Grassmannian variety $\mathcal{M}$. No current tool lives on $\mathcal{M}$ with both Gates 1+2 enforced as hard continuous constraints.

The GAP algorithm is the minimum viable next implementation: alternating projection onto the Gate-1+2 manifold in the Grassmannian, with aug\_gap\_on\_M as the convergence monitor.

Edit: Gate 1+2+3 is no diffrent than just asking for the normal naive unadultered CP decomposition. 
