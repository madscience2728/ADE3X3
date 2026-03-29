# Step 72: Depth-2 Circuit Dimension Census
Generated: 2026-03-28T20:59:39

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

## Task 1 / 2: Exact Formulas

- Parameter formula: P = 27R1 + R1R2 + 18R2 + n_QQ(R1 - 9)
- Raw upper-bound constraint formula: C_upper = 729 + 3645*1_AA + 3645*1_BB + 18225*1_QQ
- QQ terms add parameters exactly when R1 > 9, break even at R1 = 9, and reduce the raw parameter count relative to AA/BB at R1 < 9.
- Exact census rows written: 12650
- Raw upper-bound positive deficits: 0
- Best raw upper-bound deficit: -135 at R_total=22, R1=22, R2=0, AA=0, BB=0, QQ=0

## Task 4: AA-Only Case Study

| R_total | R1 | R2 | parameters | raw D vs 4374/729 rule | D vs 729 only |
|---------|----|----|------------|------------------------|---------------|
| 22 | 11 | 11 | 616 | -3758 | -113 |
| 22 | 12 | 10 | 624 | -3750 | -105 |
| 22 | 13 | 9 | 630 | -3744 | -99 |
| 22 | 14 | 8 | 634 | -3740 | -95 |
| 22 | 15 | 7 | 636 | -3738 | -93 |
| 22 | 16 | 6 | 636 | -3738 | -93 |
| 22 | 17 | 5 | 634 | -3740 | -95 |
| 22 | 18 | 4 | 630 | -3744 | -99 |
| 22 | 19 | 3 | 624 | -3750 | -105 |
| 22 | 20 | 2 | 616 | -3758 | -113 |
| 22 | 21 | 1 | 606 | -3768 | -123 |
| 22 | 22 | 0 | 594 | -135 | -135 |

## Task 5: Measured Jacobian Sweep

- Jacobian method: analytic
- Measured configurations: 506 total = 271 AA/BB + 235 AA/QQ
- Positive measured local fiber dimension count: 506
- Full-column-rank count: 0
- Maximum measured local fiber dimension: 43

| family | R_total | R1 | R2 | AA | BB | QQ | parameters | rows | rank(J) | local fiber | raw upper D | status |
|--------|---------|----|----|----|----|----|------------|------|---------|-------------|-------------|--------|
| AA_BB_only | 22 | 20 | 2 | 2 | 0 | 0 | 616 | 4374 | 573 | 43 | -3758 | positive_local_fiber |
| AA_BB_only | 21 | 20 | 1 | 0 | 1 | 0 | 578 | 4374 | 536 | 42 | -3796 | positive_local_fiber |
| AA_BB_only | 22 | 21 | 1 | 0 | 1 | 0 | 606 | 4374 | 564 | 42 | -3768 | positive_local_fiber |
| AA_BB_only | 22 | 20 | 2 | 1 | 1 | 0 | 616 | 8019 | 574 | 42 | -7403 | positive_local_fiber |
| AA_BB_only | 22 | 18 | 4 | 2 | 2 | 0 | 630 | 8019 | 588 | 42 | -7389 | positive_local_fiber |
| AA_BB_only | 21 | 19 | 2 | 2 | 0 | 0 | 587 | 4374 | 546 | 41 | -3787 | positive_local_fiber |
| AA_BB_only | 22 | 19 | 3 | 1 | 2 | 0 | 624 | 8019 | 583 | 41 | -7395 | positive_local_fiber |
| AA_BB_only | 20 | 19 | 1 | 1 | 0 | 0 | 550 | 4374 | 510 | 40 | -3824 | positive_local_fiber |
| AA_BB_only | 21 | 20 | 1 | 1 | 0 | 0 | 578 | 4374 | 538 | 40 | -3796 | positive_local_fiber |
| AA_BB_only | 21 | 19 | 2 | 1 | 1 | 0 | 587 | 8019 | 547 | 40 | -7432 | positive_local_fiber |
| AA_BB_only | 21 | 18 | 3 | 1 | 2 | 0 | 594 | 8019 | 554 | 40 | -7425 | positive_local_fiber |
| AA_BB_only | 21 | 17 | 4 | 0 | 4 | 0 | 599 | 4374 | 559 | 40 | -3775 | positive_local_fiber |

## Task 6: Landscape Status

- Search status: not_run_dimension_count_only
- The deliverable table is dimension-only in this step. The Jacobian landscape is exported, but no nonlinear solve was run here.
- Any positive local fiber dimension should be interpreted as a parameterization-level signal only. It does not certify an exact solution at R_total < 23.

[INTERPRETATION]

The raw upper-bound count remains brutally negative across the entire exact census: even the best parameter-rich family stays far below the naive coefficient-slot upper bound once the degree-3 and degree-4 cancellation equations are counted at face value. That confirms the prompt's warning that raw counts alone are too pessimistic to decide the question.
The measured Jacobian sweep gives the more relevant local picture. Instead of comparing parameter count to the full coefficient-slot total, Step 72 measures how many constraint directions are actually activated at a generic point for the requested depth-2 families. This is the right local dimension object for the census, but it still answers only a dimension question, not the existence question.
The other important Step 72 correction is methodological: the prompt proposed finite differences, but the actual sweep uses the exact analytical Jacobian. That improves numerical stability and lets the requested AA/BB and AA/QQ families run as a single measured landscape without changing the rank quantity being measured.