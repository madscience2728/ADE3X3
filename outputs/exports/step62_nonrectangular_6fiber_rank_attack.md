# Step 62: Non-Rectangular 6-Fiber Sub-Tensor Rank Attack
Generated: 2026-03-28T16:02:14

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

Step 62 attacks the six-fiber hybrid survivors from Step 60 with exact restriction bounds, measured CP-rank scans, and a direct upper-bound construction for the anti-diagonal-missing pattern P4.

Nonrectangular patterns attacked: 4
Any nonrectangular rank <= 13 found numerically: false
Any nonrectangular rank <= 14 found numerically: false
P4 direct upper bound: 18

## Pattern Table

| pattern | representative fibers | substitution best lower bound | numerical rank upper bound if found | known exact rank if determined | feasible at 13 spreaders? | feasible at 14 spreaders? |
|---------|-----------------------|-------------------------------|-------------------------------------|-------------------------------|--------------------------|--------------------------|
| P1 | (0,0); (0,1); (0,2); (1,0); (1,1); (2,0) | 9 |  |  | not_supported_by_measurement | not_supported_by_measurement |
| P2 | (0,0); (0,1); (0,2); (1,0); (1,1); (2,2) | 9 |  |  | not_supported_by_measurement | not_supported_by_measurement |
| P3 | (0,0); (0,1); (1,0); (1,1); (2,0); (2,2) | 9 |  |  | not_supported_by_measurement | not_supported_by_measurement |
| P4 | (0,0); (0,1); (1,0); (1,2); (2,1); (2,2) | 9 |  |  | not_supported_by_measurement | not_supported_by_measurement |
| P5 | (0,0); (0,1); (0,2); (1,0); (1,1); (1,2) | 9 |  | 15 | exact_no | exact_no |
| P6 | (0,0); (0,1); (1,0); (1,1); (2,0); (2,1) | 9 |  | 15 | exact_no | exact_no |

## Hybrid Verdicts

| R | Fourier fibers f | spreaders | verdict | scope |
|---|------------------|-----------|---------|-------|
| 22 | 3 | 13 | no_numerical_witness_at_or_below_13 | six_fiber_nonrectangular_patterns_only |
| 23 | 3 | 14 | no_numerical_witness_at_or_below_14 | six_fiber_nonrectangular_patterns_only |
| 22 | 4 | 10 | not_attacked_in_step62 | five_fiber_patterns_not_reoptimized_here |
| 23 | 4 | 11 | not_attacked_in_step62 | five_fiber_patterns_not_reoptimized_here |
| 22 | 5 | 7 | not_attacked_in_step62 | four_fiber_patterns_not_reoptimized_here |

## P4 Direct Construction

| construction | term count | status | note |
|--------------|------------|--------|------|
| fiber_local_standard | 18 | exact_verified | Three fiber-local terms per selected output fiber give an exact decomposition. |
| row_pair_cover | 18 | no_improvement | Three row-wise 1x2 subproblems each have known exact rank 6, totaling 18. |
| column_pair_cover | 18 | no_improvement | Three column-wise 2x1 subproblems each have known exact rank 6, totaling 18. |

[INTERPRETATION]

The exact restriction bounds remain coarse on the nonrectangular six-fiber patterns, but the attack does something Step 60 could not: it asks directly whether rank 13 or 14 is numerically plausible on the full 9x9x6 subtensors. The rectangular patterns P5 and P6 stay dead by exact rank 15. P4 has an explicit exact upper bound 18 via the standard fiber-local decomposition, and its row-pair and column-pair covers do not improve that count. The decisive content of Step 62 is therefore the measured rank scan on P1-P4 and the resulting R=22/R=23 verdicts for the six-fiber hybrid route. Those verdicts are evidence-based and numerical, not a proof unless matched by an exact decomposition or a stronger lower bound.