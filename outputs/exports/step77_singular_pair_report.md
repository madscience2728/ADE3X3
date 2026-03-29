# Step 77: Border Rank Singular-Pair Witness Construction
Generated: 2026-03-29T00:44:05

[MEASURED_FROM_CODE]

Pairs analysed: 59 (all constant-multiple pairs from Step 70)
Gamma disjoint pairs:       23
Gamma proportional pairs:   1
Pair tensor rank-1 cases:   0
Search target rank:         22 (trying T_full - T_pair at rank 21)
Restarts per pair:          50
Exact hits (rank-22 found): 0

## Gamma-condition breakdown

| pair | gamma_i | gamma_j | intersect | disjoint | proportional | ratio | T_pair_rank_lb | exact_hit |
|------|---------|---------|-----------|----------|--------------|-------|----------------|-----------|
| t01,t02 | 2 | 3 | 2 | False | False |  | 2 | False |
| t01,t03 | 2 | 1 | 0 | True | False |  | 2 | False |
| t01,t06 | 2 | 1 | 0 | True | False |  | 2 | False |
| t01,t08 | 2 | 3 | 1 | False | False |  | 2 | False |
| t01,t09 | 2 | 1 | 0 | True | False |  | 2 | False |
| t02,t03 | 3 | 1 | 1 | False | False |  | 2 | False |
| t02,t04 | 3 | 3 | 2 | False | False |  | 2 | False |
| t02,t05 | 3 | 2 | 1 | False | False |  | 2 | False |
| t02,t06 | 3 | 1 | 1 | False | False |  | 2 | False |
| t02,t07 | 3 | 2 | 0 | True | False |  | 2 | False |
| t02,t08 | 3 | 3 | 2 | False | False |  | 2 | False |
| t02,t09 | 3 | 1 | 0 | True | False |  | 2 | False |
| t02,t11 | 3 | 1 | 0 | True | False |  | 2 | False |
| t02,t15 | 3 | 1 | 0 | True | False |  | 2 | False |
| t02,t20 | 3 | 5 | 3 | False | False |  | 2 | False |
| t03,t04 | 1 | 3 | 1 | False | False |  | 2 | False |
| t03,t05 | 1 | 2 | 0 | True | False |  | 2 | False |
| t03,t10 | 1 | 1 | 0 | True | False |  | 2 | False |
| t04,t05 | 3 | 2 | 1 | False | False |  | 2 | False |
| t04,t10 | 3 | 1 | 0 | True | False |  | 2 | False |
| t05,t06 | 2 | 1 | 0 | True | False |  | 2 | False |
| t05,t07 | 2 | 2 | 1 | False | False |  | 2 | False |
| t05,t08 | 2 | 3 | 0 | True | False |  | 2 | False |
| t05,t10 | 2 | 1 | 1 | False | False |  | 2 | False |
| t05,t11 | 2 | 1 | 1 | False | False |  | 2 | False |
| t05,t12 | 2 | 1 | 0 | True | False |  | 2 | False |
| t06,t11 | 1 | 1 | 0 | True | False |  | 2 | False |
| t07,t08 | 2 | 3 | 1 | False | False |  | 2 | False |
| t07,t09 | 2 | 1 | 1 | False | False |  | 2 | False |
| t07,t10 | 2 | 1 | 0 | True | False |  | 2 | False |
| t07,t11 | 2 | 1 | 1 | False | False |  | 2 | False |
| t07,t12 | 2 | 1 | 1 | False | False |  | 2 | False |
| t07,t13 | 2 | 4 | 2 | False | False |  | 2 | False |
| t07,t14 | 2 | 3 | 1 | False | False |  | 2 | False |
| t07,t15 | 2 | 1 | 0 | True | False |  | 2 | False |
| t07,t23 | 2 | 3 | 1 | False | False |  | 2 | False |
| t08,t09 | 3 | 1 | 1 | False | False |  | 2 | False |
| t09,t13 | 1 | 4 | 1 | False | False |  | 2 | False |
| t10,t11 | 1 | 1 | 0 | True | False |  | 2 | False |
| t10,t12 | 1 | 1 | 0 | True | False |  | 2 | False |
| t10,t16 | 1 | 2 | 1 | False | False |  | 2 | False |
| t11,t12 | 1 | 1 | 0 | True | False |  | 2 | False |
| t11,t14 | 1 | 3 | 1 | False | False |  | 2 | False |
| t11,t23 | 1 | 3 | 1 | False | False |  | 2 | False |
| t12,t19 | 1 | 2 | 1 | False | False |  | 2 | False |
| t13,t14 | 4 | 3 | 2 | False | False |  | 2 | False |
| t13,t15 | 4 | 1 | 0 | True | False |  | 2 | False |
| t13,t19 | 4 | 2 | 2 | False | False |  | 2 | False |
| t13,t23 | 4 | 3 | 2 | False | False |  | 2 | False |
| t14,t15 | 3 | 1 | 1 | False | False |  | 2 | False |
| t14,t19 | 3 | 2 | 0 | True | False |  | 2 | False |
| t14,t21 | 3 | 2 | 2 | False | False |  | 2 | False |
| t14,t23 | 3 | 3 | 2 | False | False |  | 2 | False |
| t15,t17 | 1 | 1 | 1 | False | True | 1 | 2 | False |
| t15,t19 | 1 | 2 | 0 | True | False |  | 2 | False |
| t15,t20 | 1 | 5 | 1 | False | False |  | 2 | False |
| t15,t21 | 1 | 2 | 0 | True | False |  | 2 | False |
| t18,t21 | 3 | 2 | 2 | False | False |  | 2 | False |
| t21,t23 | 2 | 3 | 1 | False | False |  | 2 | False |

## Interpretation

NEGATIVE: No pair produced an exact rank-22 replacement at 50 restarts per pair.
Best residual across all pairs: 6.3614e-01 (pair t15,t17).

The gamma-support conditions provide the algebraic filter:
  23 / 59 pairs have disjoint gamma support (cleanest fusion condition).
  1 / 59 pairs have proportional gamma vectors.
  0 / 59 pair sub-tensors are already rank-1 (these are trivially fused without deformation, but the residual T_full - T_pair still needs rank <= 21 for a net gain).

A true border-rank argument would need to exhibit an explicit one-parameter family of rank-22 tensors converging to T_full, which requires the residual (T_full - T_pair) to itself be expressible at rank <= 21.  The bounded LP-BFGS search here is a necessary (not sufficient) test at the chosen restart budget.
