# Step 47: Mixed Pair Resolution + Algorithm Constraint Extraction
Generated: 2026-03-28T12:43:57

[EXACT_DERIVED]

## Task 1: Mixed-Pair Resolution

Mixed orbit pairs: 164
Mixed-pair witnesses: 41,688
- Fully resolved by (s,t): 0 / 164
- Fully resolved by (r,s,t,u): 0 / 164
- Output-set invariant across all occupied (s,t) strata: 164 / 164
- Output-set invariant across all occupied (r,s,t,u) strata: 164 / 164

| orbit_A | orbit_B | output_orbits | n_strata_by_st | fully_resolved_by_st |
|---------|---------|---------------|----------------|----------------------|
| 30 | 10 | [20, 40] | 3 | False |
| 30 | 20 | [10, 40] | 3 | False |
| 30 | 30 | [0, 30] | 3 | False |
| 30 | 34 | [4, 34] | 3 | False |
| 30 | 40 | [10, 20] | 3 | False |
| 30 | 95 | [50, 95] | 3 | False |
| 34 | 143 | [20, 40] | 3 | False |
| 34 | 161 | [10, 40] | 3 | False |
| 34 | 179 | [0, 30] | 3 | False |
| 34 | 185 | [4, 34] | 3 | False |
| 34 | 191 | [4, 34] | 3 | False |
| 34 | 197 | [10, 20] | 3 | False |
| 34 | 296 | [50, 95] | 3 | False |
| 95 | 995 | [20, 40] | 3 | False |
| 95 | 1010 | [10, 40] | 3 | False |
| 95 | 1025 | [0, 30] | 3 | False |
| 95 | 1031 | [4, 34] | 3 | False |
| 95 | 1040 | [10, 20] | 3 | False |
| 95 | 1100 | [50, 95] | 3 | False |
| 95 | 1175 | [50, 95] | 3 | False |
| ... | ... | ... | ... | ... |

## Task 2: Multiplication Tensor in Orbit Coordinates

| orbit_id | pair_count | fraction | stab_type |
|----------|------------|----------|-----------|
| 0 | 27 | 0.500000 | (Z2)^3 |
| 30 | 27 | 0.500000 | Z2xZ2 |

## Task 3: Algorithm Constraints

- mixed_pair_count: 164
- resolved_by_st: 0/164
- resolved_by_rstu: 0/164
- constant_outputset_by_st: 164/164
- constant_outputset_by_rstu: 164/164
- tensor_same_fiber_orbits: [0, 30]
- tensor_doubly_live_outside_0_30: 0
- rank1_exact_orbit_model_status: underdetermined_without_coefficient_or_support_model

[INTERPRETATION]

The mixed pairs are the exact forks that an algorithm must resolve by committing to
finer interface data than orbit identity alone. The multiplication tensor itself lives
on a very small same-fiber slice of the 64-orbit algebra, so any algorithm inside this
sub-algebra must reproduce that orbit profile while choosing consistent coordinate-level
branches across the 164 mixed pairs.