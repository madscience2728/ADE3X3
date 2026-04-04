# Packet Analysis Report — Phase 3
Generated: 2026-04-04 11:03:28
Completions per packet: 50

## R=13 (500 sampled configurations)

### Gate Pass Rates

| Gate | Tested | Passed | Rate |
|------|--------|--------|------|
| Gate 1 rank(H)=R-9 | 500 | 500 | 100.0% |
| Gate 2 Δ⊂span(H) | 500 | 0 | 0.0% |
| Gate 3 | — | — | — |

### Sigma Innovation Distribution

| σ_innov | Count | Fraction |
|---------|-------|----------|
| 0 | 500 | 100.0% |

**Maximum σ_innov observed: 0**

### Conservation Law

R + η_null = 27 satisfied: 500/500 (100.0%)

### Key Metric Distributions

| Metric | min | mean | median | max |
|--------|-----|------|--------|-----|
| augmented_gap | 0.0000 | 1.8040 | 2.0000 | 5.0000 |
| sigma_innovation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tensor_reconstruction_error | 0.8815 | 1.0102 | 1.0170 | 1.0571 |
| H_spectral_gap | 0.4631 | 1.2144 | 1.1147 | 2.7788 |
| delta_residual_frobenius | 5.0662 | 8.3531 | 8.3532 | 13.4458 |
| delta_leak_dims | 4.0000 | 7.1960 | 7.0000 | 9.0000 |
| gamma_lstsq_residual | 7.5782 | 8.3424 | 8.3568 | 8.6444 |

### Interpretation

**Gate 2 is the active bottleneck.** Zero of 500 Gate-1-passing configs satisfy Δ⊂span(H). When Gate 2 fails, Nuisance=[H|Δ] achieves full row rank R, making σ_innov=0 a trivial algebraic consequence — Σ is automatically contained in a full-row-rank matrix. This is NOT an independent sigma obstruction; it is a Gate 2 cascade. The meaningful sigma analysis requires Gate-2-passing configurations.

**Strategy:** Understand structurally what forces Δ⊂span(H). The Stage-1 hit pool enforces H-row containment by construction. The delta containment (Gate 2) requires (α[r,s]·β[t,u]) for s≠t to lie in the span of [Eta1|Eta2] — a far stronger constraint than H-row membership.

## R=19 (14800 sampled configurations)

### Gate Pass Rates

| Gate | Tested | Passed | Rate |
|------|--------|--------|------|
| Gate 1 rank(H)=R-9 | 14800 | 14800 | 100.0% |
| Gate 2 Δ⊂span(H) | 14800 | 0 | 0.0% |
| Gate 3 | — | — | — |

### Sigma Innovation Distribution

| σ_innov | Count | Fraction |
|---------|-------|----------|
| 0 | 14800 | 100.0% |

**Maximum σ_innov observed: 0**

### Conservation Law

R + η_null = 27 satisfied: 14800/14800 (100.0%)

### Key Metric Distributions

| Metric | min | mean | median | max |
|--------|-----|------|--------|-----|
| augmented_gap | 0.0000 | 1.6530 | 1.0000 | 8.0000 |
| sigma_innovation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tensor_reconstruction_error | 0.7813 | 0.9829 | 0.9852 | 1.1714 |
| H_spectral_gap | 0.1797 | 1.1198 | 1.1116 | 1.8988 |
| delta_residual_frobenius | 2.0000 | 8.0749 | 8.0623 | 14.3984 |
| delta_leak_dims | 1.0000 | 7.3470 | 8.0000 | 9.0000 |
| gamma_lstsq_residual | 6.1414 | 7.4957 | 7.6786 | 8.1332 |

### Interpretation

**Gate 2 is the active bottleneck.** Zero of 14800 Gate-1-passing configs satisfy Δ⊂span(H). When Gate 2 fails, Nuisance=[H|Δ] achieves full row rank R, making σ_innov=0 a trivial algebraic consequence — Σ is automatically contained in a full-row-rank matrix. This is NOT an independent sigma obstruction; it is a Gate 2 cascade. The meaningful sigma analysis requires Gate-2-passing configurations.

**Strategy:** Understand structurally what forces Δ⊂span(H). The Stage-1 hit pool enforces H-row containment by construction. The delta containment (Gate 2) requires (α[r,s]·β[t,u]) for s≠t to lie in the span of [Eta1|Eta2] — a far stronger constraint than H-row membership.

## R=20 (12800 sampled configurations)

### Gate Pass Rates

| Gate | Tested | Passed | Rate |
|------|--------|--------|------|
| Gate 1 rank(H)=R-9 | 12800 | 12800 | 100.0% |
| Gate 2 Δ⊂span(H) | 12800 | 0 | 0.0% |
| Gate 3 | — | — | — |

### Sigma Innovation Distribution

| σ_innov | Count | Fraction |
|---------|-------|----------|
| 0 | 12800 | 100.0% |

**Maximum σ_innov observed: 0**

### Conservation Law

R + η_null = 27 satisfied: 12800/12800 (100.0%)

### Key Metric Distributions

| Metric | min | mean | median | max |
|--------|-----|------|--------|-----|
| augmented_gap | 0.0000 | 2.3366 | 2.0000 | 8.0000 |
| sigma_innovation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tensor_reconstruction_error | 0.7176 | 0.9610 | 0.9620 | 1.1673 |
| H_spectral_gap | 0.5011 | 1.0981 | 1.0921 | 1.9760 |
| delta_residual_frobenius | 2.0000 | 7.6199 | 7.6158 | 12.1244 |
| delta_leak_dims | 1.0000 | 6.6634 | 7.0000 | 9.0000 |
| gamma_lstsq_residual | 6.8099 | 7.6560 | 7.6678 | 8.0565 |

### Interpretation

**Gate 2 is the active bottleneck.** Zero of 12800 Gate-1-passing configs satisfy Δ⊂span(H). When Gate 2 fails, Nuisance=[H|Δ] achieves full row rank R, making σ_innov=0 a trivial algebraic consequence — Σ is automatically contained in a full-row-rank matrix. This is NOT an independent sigma obstruction; it is a Gate 2 cascade. The meaningful sigma analysis requires Gate-2-passing configurations.

**Strategy:** Understand structurally what forces Δ⊂span(H). The Stage-1 hit pool enforces H-row containment by construction. The delta containment (Gate 2) requires (α[r,s]·β[t,u]) for s≠t to lie in the span of [Eta1|Eta2] — a far stronger constraint than H-row membership.

