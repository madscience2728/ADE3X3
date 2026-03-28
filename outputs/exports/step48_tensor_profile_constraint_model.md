# Step 48: Tensor Profile Constraint Model
Generated: 2026-03-28T12:43:11

[EXACT_DERIVED]

## Task 3a: XC-Orbit Decomposition of the 729 Tensor Equations

XC orbit classes: 8
Positive XC orbit ids: [0]
Positive tensor equations: 27 of 729

| xc_orbit_id | orbit_size | x_live | c_relation_to_target | rhs_constant | rhs_sum |
|-------------|------------|--------|----------------------|--------------|---------|
| 0 | 27 | True | exact_target | 1 | 27 |
| 1 | 54 | True | same_row_diff_col | 0 | 0 |
| 2 | 54 | True | diff_row_same_col | 0 | 0 |
| 3 | 108 | True | diff_row_diff_col | 0 | 0 |
| 4 | 54 | False | exact_target | 0 | 0 |
| 5 | 108 | False | same_row_diff_col | 0 | 0 |
| 6 | 108 | False | diff_row_same_col | 0 | 0 |
| 7 | 216 | False | diff_row_diff_col | 0 | 0 |

Only XC orbit 0 carries tensor RHS 1. The other 7 XC orbits are exact zero classes.

## Task 2b: Exact Coordinate Tensor System

For each rank-R decomposition, the exact equations are:
sum_k a_k[r,s] * b_k[t,u] * c_k[r',u'] = delta_(s=t) * delta_(r=r') * delta_(u=u')

This is a 729-equation trilinear system on 27R scalar unknowns.

## Task 1 and 2a: Rank-1 Fiber Activation Profiles

A generic rank-1 term activates exactly 3 live X atoms in every output fiber C[r,u].

| fiber_id | target_c_name | activated_live_atoms_generic | coefficient_monomials |
|----------|---------------|------------------------------|-----------------------|
| 0 | C[0,0] | 3 | ['a[0,0]*b[0,0]', 'a[0,1]*b[1,0]', 'a[0,2]*b[2,0]'] |
| 1 | C[0,1] | 3 | ['a[0,0]*b[0,1]', 'a[0,1]*b[1,1]', 'a[0,2]*b[2,1]'] |
| 2 | C[0,2] | 3 | ['a[0,0]*b[0,2]', 'a[0,1]*b[1,2]', 'a[0,2]*b[2,2]'] |
| 3 | C[1,0] | 3 | ['a[1,0]*b[0,0]', 'a[1,1]*b[1,0]', 'a[1,2]*b[2,0]'] |
| 4 | C[1,1] | 3 | ['a[1,0]*b[0,1]', 'a[1,1]*b[1,1]', 'a[1,2]*b[2,1]'] |
| 5 | C[1,2] | 3 | ['a[1,0]*b[0,2]', 'a[1,1]*b[1,2]', 'a[1,2]*b[2,2]'] |
| 6 | C[2,0] | 3 | ['a[2,0]*b[0,0]', 'a[2,1]*b[1,0]', 'a[2,2]*b[2,0]'] |
| 7 | C[2,1] | 3 | ['a[2,0]*b[0,1]', 'a[2,1]*b[1,1]', 'a[2,2]*b[2,1]'] |
| 8 | C[2,2] | 3 | ['a[2,0]*b[0,2]', 'a[2,1]*b[1,2]', 'a[2,2]*b[2,2]'] |

Support-profile orbit types: 531
Distinct support-level (CXXC orbit 0, CXXC orbit 30) contribution pairs: 111
Generic support profile unique: True
Generic support profile: (3, 3, 3, 3, 3, 3, 3, 3, 3) -> (orbit 0 support, orbit 30 support) = (27, 27)

| profile_orbit_id | canonical_profile | cxxc_orbit0_support | cxxc_orbit30_support | support_pair_count |
|------------------|-------------------|---------------------|----------------------|--------------------|
| 1 | (0, 0, 0, 0, 0, 0, 0, 0, 0) | 0 | 0 | 3375 |
| 2 | (0, 0, 0, 0, 0, 0, 0, 0, 1) | 1 | 0 | 6075 |
| 3 | (0, 0, 0, 0, 0, 0, 0, 0, 2) | 2 | 1 | 405 |
| 4 | (0, 0, 0, 0, 0, 0, 0, 0, 3) | 3 | 3 | 9 |
| 5 | (0, 0, 0, 0, 0, 0, 0, 1, 1) | 2 | 0 | 6885 |
| 6 | (0, 0, 0, 0, 0, 0, 0, 1, 2) | 3 | 1 | 1674 |
| 7 | (0, 0, 0, 0, 0, 0, 0, 1, 3) | 4 | 3 | 54 |
| 8 | (0, 0, 0, 0, 0, 0, 0, 2, 2) | 4 | 2 | 459 |
| 9 | (0, 0, 0, 0, 0, 0, 0, 2, 3) | 5 | 4 | 54 |
| 10 | (0, 0, 0, 0, 0, 0, 0, 3, 3) | 6 | 6 | 9 |
| 11 | (0, 0, 0, 0, 0, 0, 1, 1, 1) | 3 | 0 | 2853 |
| 12 | (0, 0, 0, 0, 0, 0, 1, 1, 2) | 4 | 1 | 1755 |

At the weighted same-fiber level, with lambda_[r,u,s] = a[r,s]*b[s,u], the exact one-term polynomials are:
- weighted orbit-0 = sum_(r,u,s) lambda_[r,u,s]^2
- weighted orbit-30 = sum_(r,u) sum_(s1<s2) lambda_[r,u,s1] * lambda_[r,u,s2]

## Task 3b-3c: XC Orbit-Sum Constraint Model

For one rank-1 term k define:
- L_k[r,u] = sum_s a_k[r,s]*b_k[s,u]
- D_k[r,u] = alpha_k[r]*beta_k[u] - L_k[r,u]
- alpha_k[r] = sum_s a_k[r,s], beta_k[u] = sum_t b_k[t,u]
- R_k[r] = sum_v c_k[r,v], U_k[u] = sum_w c_k[w,u], S_k = sum_(r,u) c_k[r,u]

| xc_orbit_id | orbit_sum_formula | target_rhs_sum |
|-------------|-------------------|----------------|
| 0 | q_k,0 = sum_(r,u) L_k[r,u] * c_k[r,u] | 27 |
| 1 | q_k,1 = sum_(r,u) L_k[r,u] * (R_k[r] - c_k[r,u]) | 0 |
| 2 | q_k,2 = sum_(r,u) L_k[r,u] * (U_k[u] - c_k[r,u]) | 0 |
| 3 | q_k,3 = sum_(r,u) L_k[r,u] * (S_k - R_k[r] - U_k[u] + c_k[r,u]) | 0 |
| 4 | q_k,4 = sum_(r,u) D_k[r,u] * c_k[r,u] | 0 |
| 5 | q_k,5 = sum_(r,u) D_k[r,u] * (R_k[r] - c_k[r,u]) | 0 |
| 6 | q_k,6 = sum_(r,u) D_k[r,u] * (U_k[u] - c_k[r,u]) | 0 |
| 7 | q_k,7 = sum_(r,u) D_k[r,u] * (S_k - R_k[r] - U_k[u] + c_k[r,u]) | 0 |

The exact aggregated necessary condition is:
- sum_k q_(k,0) = 27
- sum_k q_(k,1) = 0
- sum_k q_(k,2) = 0
- sum_k q_(k,3) = 0
- sum_k q_(k,4) = 0
- sum_k q_(k,5) = 0
- sum_k q_(k,6) = 0
- sum_k q_(k,7) = 0

Orbit-sum counterexample vector: [27, 0, 0, 0, 0, 0, 0, 0]
Full tensor equation mismatches for that one-term witness: 27

[INTERPRETATION]

The orbit-sum model is exact as a necessary condition, but it is far too coarse to force a rank lower bound by itself.
A single sparse rank-1 term already matches the 8 XC-orbit sums ((27,0,0,0,0,0,0,0)) while failing the full 729 tensor equations.
So the Step 48 payoff is a clean obstruction statement: any lower-bound attack has to use structure finer than the XC orbit-sum linearization, together with real coefficient constraints rather than support counts alone.