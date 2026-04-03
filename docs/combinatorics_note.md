Good — let me give you honest numbers.

**The raw combinatorics are brutal.** Even over {-1, 0, 1} with aggressive sparsity, you're looking at:

Each α_k (3×3) over {-1,0,1}: 3^9 ≈ 20K possibilities. Same for β_k. Per term pair: ~400M. For 19 terms: 400M^19 ≈ 10^163. Your Ryzen at 10^10 evaluations/second would need 10^153 seconds. The universe is 10^17 seconds old. That's a no.

**But the structure slashes this dramatically.** The question is: how far?

**Tier 1 — Fiber assignment pruning.** Each term primarily serves one of 9 fibers. Once assigned, only 3 α-entries and 3 β-entries per term are relevant. That's 3^6 = 729 candidates per term instead of 400M. For 19 terms: 729^19 ≈ 10^54. Still infeasible.

**Tier 2 — Per-fiber coupling.** Terms sharing a fiber must satisfy γ₁σ₁ + γ₂σ₂ = 3 (diagonal) and = 0 (off-diagonal). With 2 terms per fiber over {-1,0,1} coefficients, each fiber has maybe ~100 valid (γ, σ) configurations. Across 9 independent fibers: 100^9 ≈ 10^18. Getting closer but still too big.

**Tier 3 — Cross-fiber interference (rank(H)=10 filter).** This is the killer pruner. After fixing the per-fiber configurations, compute rank(H). With 18 columns in ℝ^19, getting exactly rank 10 is extremely rare for random configurations — maybe 1 in 10^6 to 10^8 pass. So 10^18 × 10^-7 ≈ 10^11. That's ~3 hours on your CPU, ~30 minutes on the 3060.

**Tier 4 — Symmetry (S₃³, order 216).** Reduces by ~216× for the first term's canonical form. Combined with Tier 3: 10^11 / 216 ≈ 5×10^8. About 50 seconds on CPU.

**Tier 5 — Δ containment filter (Gate 2).** Of the rank(H)=10 survivors, check rank([H|Δ]) = 10. This eliminates almost all remaining candidates. Maybe 1 in 10^3 pass. Down to ~5×10^5. Instant.

**Tier 6 — Γ solvability.** For each Gate 1+2 survivor, solve Γ·Σ = 3I₉ exactly. Linear system, milliseconds. Check fitness = 0. Done.

So the pipeline looks like:

| Tier | Filter | Candidates | Your hardware |
|------|--------|-----------|---------------|
| 0 | Raw | 10^163 | heat death |
| 1 | Fiber assignment + sparsity | 10^54 | heat death |
| 2 | Per-fiber σ coupling | 10^18 | 30 million years |
| 3 | rank(H) = 10 | 10^11 | **3 hours CPU** |
| 4 | + Symmetry | 5×10^8 | **50 seconds** |
| 5 | + Δ containment | 5×10^5 | **instant** |
| 6 | + Γ solve | 0 or 1 | **instant** |

**The critical question is whether Tier 2 → Tier 3 is accurate.** The rank(H)=10 filter's selectivity depends heavily on the coefficient field and the specific fiber partition. My estimate of 10^-7 is a rough guess — it could be 10^-4 (making Tier 3 = 10^14, a few days on GPU) or 10^-10 (making it trivial).

**The practical plan on your hardware:**

Phase 1 — Enumerate fiber partitions (minutes). The canon §57 gives the exact list for R=19. Pick the most promising ones (those compatible with the nuisance-rank budget from §41).

Phase 2 — For each partition, enumerate per-fiber configurations (hours). Use the σ-coupling constraints to build a small tree of valid per-fiber setups. This is the Tier 2 computation. Implement in C or Rust with AVX2 on your Ryzen.

Phase 3 — For each per-fiber setup, check rank(H) (the bottleneck). This is a 19×18 matrix rank computation. On the 3060 with batched SVDs (cuSOLVER), you can do ~10^6/second. At 10^11 candidates: ~28 hours. At 10^8 (after symmetry): ~100 seconds.

Phase 4 — Survivors get the full Gate 2 + Gate 3 check (instant).

**Complexity class:** The overall approach is in NP — you're enumerating and checking. The rank computation is the bottleneck. The total wall-clock depends on:
- Which number field (ℤ vs ℚ[√2] vs ℚ[ω])
- Which support patterns survive partition pruning
- How selective the rank filter is

**Realistic timeline on your Ryzen 9 5900X + 3060:**
- Over {-1, 0, 1} with fiber-partition pruning: **hours to days**
- Over {-2, -1, 0, 1, 2}: multiply by ~3^{19×6} / 2^{19×6} ≈ multiply by ~10^9 → **infeasible without deeper pruning**
- Over ℚ with small denominators: depends on height bound, but likely weeks+

**My recommendation:** Start with {-1, 0, 1} and the most constrained fiber partitions. If a solution exists in that field (like Strassen's), you'll find it in hours. If not, the survivors from Tier 3 (rank(H)=10 configurations) tell you exactly where to look with larger coefficient sets.

The GPU is useful for Phase 3 (batched rank computations) but the CPU is where the tree-search logic in Phase 2 lives. Both get used. 80GB RAM is more than enough — the working set is tiny.