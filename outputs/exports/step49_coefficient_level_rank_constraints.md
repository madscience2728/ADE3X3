# Step 49: Coefficient-Level Rank Constraints in the Same-Fiber Algebra
Generated: 2026-03-28T12:43:11

[EXACT_DERIVED]

## Task 3c: The 8 Equation Types

| equation | structure | orbit_size | RHS | representative | transparent rewrite |
|----------|-----------|------------|-----|----------------|---------------------|
| E0 | s=t, r=r', u=u' | 27 | 1 | sum_k alpha_k[0,0] * beta_k[0,0] * gamma_k[0,0] = 1 | sum_k p_k * gamma_k[0,0] = 1 |
| E1 | s=t, r=r', u!=u' | 54 | 0 | sum_k alpha_k[0,0] * beta_k[0,0] * gamma_k[0,1] = 0 | sum_k p_k * gamma_k[0,1] = 0 |
| E2 | s=t, r!=r', u=u' | 54 | 0 | sum_k alpha_k[0,0] * beta_k[0,0] * gamma_k[1,0] = 0 | sum_k p_k * gamma_k[1,0] = 0 |
| E3 | s=t, r!=r', u!=u' | 108 | 0 | sum_k alpha_k[0,0] * beta_k[0,0] * gamma_k[1,1] = 0 | sum_k p_k * gamma_k[1,1] = 0 |
| E4 | s!=t, r=r', u=u' | 54 | 0 | sum_k alpha_k[0,0] * beta_k[1,0] * gamma_k[0,0] = 0 | sum_k q_k * gamma_k[0,0] = 0 |
| E5 | s!=t, r=r', u!=u' | 108 | 0 | sum_k alpha_k[0,0] * beta_k[1,0] * gamma_k[0,1] = 0 | sum_k q_k * gamma_k[0,1] = 0 |
| E6 | s!=t, r!=r', u=u' | 108 | 0 | sum_k alpha_k[0,0] * beta_k[1,0] * gamma_k[1,0] = 0 | sum_k q_k * gamma_k[1,0] = 0 |
| E7 | s!=t, r!=r', u!=u' | 216 | 0 | sum_k alpha_k[0,0] * beta_k[1,0] * gamma_k[1,1] = 0 | sum_k q_k * gamma_k[1,1] = 0 |

These 8 structural types partition all 729 tensor equations.

## Task 1a-1b: Orbit Verification and Standard 27-Term Check

| equation | orbit_size_match | step48_match | observed_standard_sum | expected_rhs |
|----------|------------------|--------------|-----------------------|--------------|
| E0 | True | True | 1 | 1 |
| E1 | True | True | 0 | 0 |
| E2 | True | True | 0 | 0 |
| E3 | True | True | 0 | 0 |
| E4 | True | True | 0 | 0 |
| E5 | True | True | 0 | 0 |
| E6 | True | True | 0 | 0 |
| E7 | True | True | 0 | 0 |

Full 729-equation standard-basis mismatches: 0

## Task 2: Transparent Rewriting and Matrix Form

Define p_k = alpha_k[0,0] * beta_k[0,0] and q_k = alpha_k[0,0] * beta_k[1,0].
Then E0-E3 are the p_k equations against gamma_k[0,0], gamma_k[0,1], gamma_k[1,0], gamma_k[1,1].
E4-E7 are the q_k equations against those same four gamma coordinates.

T[:,:,C[0,0]] nonzero positions: (A[0,0],B[0,0]), (A[0,1],B[1,0]), (A[0,2],B[2,0])

## Task 4: Strassen 2x2 Worked Example

Strassen full 64-equation mismatches: 0
2x2 equation orbit classes under S2 x S2 x S2: 8
Coefficient orbits used by the 7 Strassen terms: 4

| term | coefficient_orbit_id | orbit_size | live_X_nonzero | dead_X_nonzero | active_equation_types |
|------|----------------------|------------|----------------|----------------|-----------------------|
| m1 | 0 | 4 | 8 | 8 | [0, 3, 5, 6] |
| m2 | 1 | 8 | 4 | 4 | [0, 1, 4, 5] |
| m3 | 2 | 8 | 4 | 4 | [0, 2, 4, 6] |
| m4 | 2 | 8 | 4 | 4 | [0, 2, 4, 6] |
| m5 | 1 | 8 | 4 | 4 | [0, 1, 4, 5] |
| m6 | 3 | 8 | 16 | 0 | [0, 1, 2, 3] |
| m7 | 3 | 8 | 16 | 0 | [0, 1, 2, 3] |

| coefficient_orbit_id | coefficient_orbit_size | strassen_terms |
|----------------------|------------------------|----------------|
| 0 | 4 | ['m1'] |
| 1 | 8 | ['m2', 'm5'] |
| 2 | 8 | ['m3', 'm4'] |
| 3 | 8 | ['m6', 'm7'] |

| equation_type | structure | orbit_size | RHS |
|---------------|-----------|------------|-----|
| Type 0 | s=t, r=r', u=u' | 8 | 1 |
| Type 1 | s=t, r=r', u!=u' | 8 | 0 |
| Type 2 | s=t, r!=r', u=u' | 8 | 0 |
| Type 3 | s=t, r!=r', u!=u' | 8 | 0 |
| Type 4 | s!=t, r=r', u=u' | 8 | 0 |
| Type 5 | s!=t, r=r', u!=u' | 8 | 0 |
| Type 6 | s!=t, r!=r', u=u' | 8 | 0 |
| Type 7 | s!=t, r!=r', u!=u' | 8 | 0 |

## Task 5: 3x3 Search Space Dimensions

| R | K | dense_variables_27R | sparse_upper_bound_3KR | constraints | dense_balance | sparse_balance |
|---|---|---------------------|-----------------------|-------------|---------------|----------------|
| 23 | 9 | 621 | 621 | 729 | over | over |
| 23 | 3 | 621 | 207 | 729 | over | over |
| 23 | 2 | 621 | 138 | 729 | over | over |
| 22 | 9 | 594 | 594 | 729 | over | over |
| 21 | 9 | 567 | 567 | 729 | over | over |
| 20 | 9 | 540 | 540 | 729 | over | over |
| 19 | 9 | 513 | 513 | 729 | over | over |

[INTERPRETATION]

The 8 equation types are the core structural reduction, but they only replace all 729 equations when the full term multiset is closed under the S3 x S3 x S3 action.
This is not a lower bound on general algorithms: non-group-closed decompositions, including known fast algorithms, still have to satisfy all 729 instantiated equations directly.
Strassen 2x2 confirms the cancellation pattern explicitly: dead-X activations do occur, and the weighted gamma outputs must cancel them.
The search-space table shows that even sparse ansatze remain heavily overdetermined in raw equation count, but because the system is trilinear that count alone does not settle feasibility or emptiness of the solution variety.