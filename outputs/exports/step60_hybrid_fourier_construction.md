# Step 60: Hybrid Fourier Construction
Generated: 2026-03-28T15:48:11

[EXACT_DERIVED]

Step 60 formalizes the hybrid split suggested by Step 59: same-fiber Fourier triples are exact modular blocks, while the remaining spreader part is a reduced output-fiber tensor problem. The main computational filter is the symmetry-reduced flattening-rank scan of residual fiber sets.

Fourier subsets checked: 512
Fourier subsets passing exactly: 512
Six-fiber orbit count: 6
Six-fiber max-flattening lower bound range: 9..9
Six-fiber flattening candidates at R=19..23: 6, 6, 6, 6, 6

## Task 1: Fourier Block Modularity

| subset size | subset count | passing count | max abs residual |
|-------------|--------------|---------------|------------------|
| 0 | 1 | 1 | 0.0000000000 |
| 1 | 9 | 9 | 0.0000000000 |
| 2 | 36 | 36 | 0.0000000000 |
| 3 | 84 | 84 | 0.0000000000 |
| 4 | 126 | 126 | 0.0000000000 |
| 5 | 126 | 126 | 0.0000000000 |
| 6 | 84 | 84 | 0.0000000000 |
| 7 | 36 | 36 | 0.0000000000 |
| 8 | 9 | 9 | 0.0000000000 |
| 9 | 1 | 1 | 0.0000000000 |

Conclusion: the same-fiber Fourier block is perfectly modular. Covering some fibers with Fourier triples leaves the tensor on the complementary fibers unchanged.

## Task 2d: Reduced Residual Systems

| remaining fibers | Fourier fibers | reduced outputs | fiber-sum eqs | anisotropy eqs | dead-X eqs | total reduced eqs | R=19 spreaders | R=20 spreaders | R=21 spreaders | R=22 spreaders | R=23 spreaders |
|------------------|----------------|-----------------|---------------|----------------|------------|-------------------|---------------|---------------|---------------|---------------|---------------|
| 1 | 8 | 1 | 9 | 18 | 54 | 81 | -5 | -4 | -3 | -2 | -1 |
| 2 | 7 | 2 | 18 | 36 | 108 | 162 | -2 | -1 | 0 | 1 | 2 |
| 3 | 6 | 3 | 27 | 54 | 162 | 243 | 1 | 2 | 3 | 4 | 5 |
| 4 | 5 | 4 | 36 | 72 | 216 | 324 | 4 | 5 | 6 | 7 | 8 |
| 5 | 4 | 5 | 45 | 90 | 270 | 405 | 7 | 8 | 9 | 10 | 11 |
| 6 | 3 | 6 | 54 | 108 | 324 | 486 | 10 | 11 | 12 | 13 | 14 |
| 7 | 2 | 7 | 63 | 126 | 378 | 567 | 13 | 14 | 15 | 16 | 17 |
| 8 | 1 | 8 | 72 | 144 | 432 | 648 | 16 | 17 | 18 | 19 | 20 |
| 9 | 0 | 9 | 81 | 162 | 486 | 729 | 19 | 20 | 21 | 22 | 23 |

## Task 4c: Six-Fiber Sub-Tensor Flattening Orbits

| representative residual fibers | orbit size | row profile | col profile | rectangle | rank A|(BC) | rank B|(AC) | rank C|(AB) | max lower bound | known exact rank if any |
|-------------------------------|------------|-------------|-------------|-----------|------------|------------|------------|-----------------|-------------------------|
| (0,0); (0,1); (0,2); (1,0); (1,1); (1,2) | 3 | 3,3,0 | 2,2,2 | True | 6 | 9 | 6 | 9 | 15 |
| (0,0); (0,1); (0,2); (1,0); (1,1); (2,0) | 36 | 3,2,1 | 3,2,1 | False | 9 | 9 | 6 | 9 |  |
| (0,0); (0,1); (0,2); (1,0); (1,1); (2,2) | 18 | 3,2,1 | 2,2,2 | False | 9 | 9 | 6 | 9 |  |
| (0,0); (0,1); (1,0); (1,1); (2,0); (2,1) | 3 | 2,2,2 | 3,3,0 | True | 9 | 6 | 6 | 9 | 15 |
| (0,0); (0,1); (1,0); (1,1); (2,0); (2,2) | 18 | 2,2,2 | 3,2,1 | False | 9 | 9 | 6 | 9 |  |
| (0,0); (0,1); (1,0); (1,2); (2,1); (2,2) | 6 | 2,2,2 | 2,2,2 | False | 9 | 9 | 6 | 9 |  |

## Task 3f: Hybrid Feasibility Table

| Fourier fibers f | Fourier terms | remaining fibers | R=19 spreaders | R=20 spreaders | R=21 spreaders | R=22 spreaders | R=23 spreaders | flattening lower-bound range over residual orbits | R19 cand. | R20 cand. | R21 cand. | R22 cand. | R23 cand. |
|------------------|---------------|------------------|---------------|---------------|---------------|---------------|---------------|-------------------------------------------|-----------|-----------|-----------|-----------|-----------|
| 8 | 24 | 1 | -5 | -4 | -3 | -2 | -1 | 3..3 | 0 | 0 | 0 | 0 | 0 |
| 7 | 21 | 2 | -2 | -1 | 0 | 1 | 2 | 6..6 | 0 | 0 | 0 | 0 | 0 |
| 6 | 18 | 3 | 1 | 2 | 3 | 4 | 5 | 6..9 | 0 | 0 | 0 | 0 | 0 |
| 5 | 15 | 4 | 4 | 5 | 6 | 7 | 8 | 6..9 | 0 | 0 | 1 | 1 | 1 |
| 4 | 12 | 5 | 7 | 8 | 9 | 10 | 11 | 9..9 | 0 | 0 | 7 | 7 | 7 |
| 3 | 9 | 6 | 10 | 11 | 12 | 13 | 14 | 9..9 | 6 | 6 | 6 | 6 | 6 |
| 2 | 6 | 7 | 13 | 14 | 15 | 16 | 17 | 9..9 | 3 | 3 | 3 | 3 | 3 |
| 1 | 3 | 8 | 16 | 17 | 18 | 19 | 20 | 9..9 | 1 | 1 | 1 | 1 | 1 |
| 0 | 0 | 9 | 19 | 20 | 21 | 22 | 23 | 9..9 | 1 | 1 | 1 | 1 | 1 |

## Explicit Cases

| case | R total | remaining fibers | available spreaders | max flattening lower bound | known exact rank | verdict | justification |
|------|---------|------------------|---------------------|----------------------------|------------------|---------|---------------|
| R23_two_fiber_row_case | 23 | (2,1); (2,2) | 2 | 6 | 6 | ruled_out_by_known_exact_rank | known exact rank 6 versus available spreaders 2 |
| R22_three_fiber_row_case | 22 | (2,0); (2,1); (2,2) | 4 | 9 | 9 | ruled_out_by_known_exact_rank | known exact rank 9 versus available spreaders 4 |
| R22_three_fiber_column_case | 22 | (0,2); (1,2); (2,2) | 4 | 9 | 9 | ruled_out_by_known_exact_rank | known exact rank 9 versus available spreaders 4 |
| R22_three_fiber_diagonal_case | 22 | (0,0); (1,1); (2,2) | 4 | 9 |  | ruled_out_by_flattening | flattening lower bound 9 versus available spreaders 4 |
| R22_six_fiber_rectangle_case | 22 | (0,0); (0,1); (0,2); (1,0); (1,1); (1,2) | 13 | 9 | 15 | ruled_out_by_known_exact_rank | known exact rank 15 versus available spreaders 13 |

[INTERPRETATION]

The clean result is modularity: the Fourier block really does decouple. That means the hybrid question is exactly the rank question for the residual output-fiber tensor, not a coupled correction problem. Extending the same flattening filter down to R=19, R=20, and R=21 does not shrink the six-fiber survivor set at all: every six-fiber orbit still has lower bound 9, so all six remain compatible with spreader budgets 10, 11, 12, 13, and 14. But the rectangular 2x3 case is already known to need rank 15, so flattening is too weak there. At the small end, the explicit two-fiber and three-fiber row/column/diagonal residuals are ruled out immediately, and four-fiber residuals only begin to survive at the flattening level once the budget reaches 7. The remaining unresolved hybrid territory therefore still sits in non-rectangular 4-, 5-, 6-, 7-, and 8-fiber residual patterns, where flattening lower bounds alone do not decide feasibility.