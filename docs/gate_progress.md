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
  Gap of 9 is unchanged throughout the run — the Sigma subspace is structurally
  orthogonal to H+Delta in every packet found so far.
- **Active bottleneck**: Gate 3 / augmented rank. The packet reconstruction error
  (~2.8) confirms gamma coefficients cannot fit the target tensor with the
  current H+Sigma structure.
- **Output**: `CANON_DATABASE/swap_improvements.jsonl`, `CANON_DATABASE/swap_checkpoint.json`
- **Tools**: `CANON_DATABASE/scripts/swap_*.py`, `CANON_DATABASE/scripts/alphatensor_delete.py`
