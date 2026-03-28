# Step 54: Analytical Low-Nuisance Construction
Generated: 2026-03-28T12:43:31

[EXACT_DERIVED]

Step 54 turns the constructive question into two parts: an exact correction to the dead-free term template, and a reproducible nuisance-rank experiment on low-rank factor families.

## Dead-Free Term Analysis

- D1: A nonzero rank-1 term is dead-free (all delta coordinates vanish) iff there exists a unique summation index s* such that alpha[:,t]=0 for all t!=s* and beta[t,:]=0 for all t!=s*. Equivalently, the alpha column support and beta row support are contained in the same singleton {s*}.
- D2: For a nonzero dead-free term with active index s*, define v = a[:,s*] tensor b[s*,:] in the 9-dimensional fiber space. Then Sigma contributes v, while Eta1 and Eta2 are fixed signed embeddings of the same v; only Delta vanishes.
- D3: Therefore dead-free does not imply nuisance-free in the Step 51 basis. A generic nonzero dead-free term has dead rank 0 but nuisance rank 1, because its anisotropy coordinates are nonzero unless v=0.
- D4: The Strassen-style pure-live versus cancellation split does not transfer verbatim to Step 51. What survives is only the weaker statement that dead-free terms contribute no dead-X nuisance; they still contribute anisotropy nuisance.
- D5: A family of dead-free terms has nuisance rows inside the union of three fixed 9-dimensional embeddings E0(v)=(v,0), E1(v)=(-v,v), E2(v)=(0,-v) of the fiber space into the 18-dimensional anisotropy space.

| active_sum_index | sigma_profile | eta1_profile | eta2_profile | dead_profile | nuisance_zero | generic_nuisance_rank |
|------------------|---------------|--------------|--------------|--------------|---------------|-----------------------|
| 0 | Sigma = v | Eta1 = v | Eta2 = 0 | Delta = 0 | False | 1 |
| 1 | Sigma = v | Eta1 = -v | Eta2 = v | Delta = 0 | False | 1 |
| 2 | Sigma = v | Eta1 = 0 | Eta2 = -v | Delta = 0 | False | 1 |

## Random Low-Rank Factor Sweep

The experiment samples alpha=A*C and beta=B*D with prescribed matrix ranks p and q, then computes Sigma, Eta1, Eta2, Delta, nuisance rank, and the quotient-space gain rank([Sigma Nuisance]) - rank(Nuisance).

| R | p | q | trials | target nuisance <= R-9 | nuisance min | nuisance median | nuisance max | quotient gain max | criterion holds count | nuisance target hits | full target hits |
|---|---|---|--------|------------------------|--------------|----------------|--------------|-------------------|-----------------------|---------------------|------------------|
| 22 | 3 | 3 | 24 | 13 | 9 | 9 | 9 | 0 | 0 | 24 | 0 |
| 22 | 3 | 4 | 24 | 13 | 12 | 12 | 12 | 0 | 0 | 24 | 0 |
| 22 | 3 | 5 | 24 | 13 | 15 | 15 | 15 | 0 | 0 | 0 | 0 |
| 22 | 3 | 6 | 24 | 13 | 18 | 18 | 18 | 0 | 0 | 0 | 0 |
| 22 | 4 | 3 | 24 | 13 | 12 | 12 | 12 | 0 | 0 | 24 | 0 |
| 22 | 4 | 4 | 24 | 13 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| 22 | 4 | 5 | 24 | 13 | 20 | 20 | 20 | 0 | 0 | 0 | 0 |
| 22 | 4 | 6 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |
| 22 | 5 | 3 | 24 | 13 | 15 | 15 | 15 | 0 | 0 | 0 | 0 |
| 22 | 5 | 4 | 24 | 13 | 20 | 20 | 20 | 0 | 0 | 0 | 0 |
| 22 | 5 | 5 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |
| 22 | 5 | 6 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |
| 22 | 6 | 3 | 24 | 13 | 18 | 18 | 18 | 0 | 0 | 0 | 0 |
| 22 | 6 | 4 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |
| 22 | 6 | 5 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |
| 22 | 6 | 6 | 24 | 13 | 22 | 22 | 22 | 0 | 0 | 0 | 0 |

### Best R=22 Samples By Family

| p | q | nuisance_rank | quotient_gain | full_product_rank | criterion_holds | meets_nuisance_target | meets_full_target |
|---|---|---------------|---------------|-------------------|-----------------|-----------------------|------------------|
| 3 | 3 | 9 | 0 | 9 | False | True | False |
| 3 | 4 | 12 | 0 | 12 | False | True | False |
| 3 | 5 | 15 | 0 | 15 | False | False | False |
| 3 | 6 | 18 | 0 | 18 | False | False | False |
| 4 | 3 | 12 | 0 | 12 | False | True | False |
| 4 | 4 | 16 | 0 | 16 | False | False | False |
| 4 | 5 | 20 | 0 | 20 | False | False | False |
| 4 | 6 | 22 | 0 | 22 | False | False | False |
| 5 | 3 | 15 | 0 | 15 | False | False | False |
| 5 | 4 | 20 | 0 | 20 | False | False | False |
| 5 | 5 | 22 | 0 | 22 | False | False | False |
| 5 | 6 | 22 | 0 | 22 | False | False | False |
| 6 | 3 | 18 | 0 | 18 | False | False | False |
| 6 | 4 | 22 | 0 | 22 | False | False | False |
| 6 | 5 | 22 | 0 | 22 | False | False | False |
| 6 | 6 | 22 | 0 | 22 | False | False | False |

## Strassen Lift Baseline

| component | multiplication_count | method | note |
|-----------|----------------------|--------|------|
| top_left_2x2_block | 7 | Strassen_on_A11_B11 | Apply 2x2 Strassen to the top-left 2x2 corner block. |
| top_left_outer_fixup | 4 | a12_times_b21_outer_product | The a12*b21^T correction contributes 4 scalar products to C11. |
| top_right_block | 6 | A11_b12_plus_a12_b22 | A11*b12 costs 4 and a12*b22 costs 2. |
| bottom_left_block | 6 | a21T_B11_plus_a22_b21T | a21^T*B11 costs 4 and a22*b21^T costs 2. |
| bottom_right_scalar | 3 | a21T_b12_plus_a22_b22 | The scalar corner uses 2 + 1 standard products. |
| naive_corner_strassen_total | 26 | sum_of_above | A direct 2x2-corner Strassen lift already exceeds 23 before any cross-block optimization. |

## Smirnov Status

- Smirnov_23x3x3_2013: not_available_in_repo_or_current_tool_set (The repository contains only a not_mapped status note from Step 47 and no authoritative alpha,beta,gamma term list to profile directly.)

[INTERPRETATION]

The exact correction to the Strassen template is that dead-free terms are not nuisance-free in the Step 51 basis: False. In the random-factor experiment, low nuisance at R=22 is easy to obtain numerically in some low-product-dimension families, but quotient-space independence never appeared in the sampled families: r22_any_meets_full_target = False. The recurring empirical pattern is that the nuisance rank tracks the full Hadamard product-space rank, while Sigma adds no extra quotient gain. That makes generic low-rank factor models poor constructive candidates even when their nuisance rank is below the Step 52 threshold. The direct 2x2-corner Strassen lift baseline is 26 scalar multiplications, so it is not competitive without additional cross-block structure.