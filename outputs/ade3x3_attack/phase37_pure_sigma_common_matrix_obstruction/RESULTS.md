# Phase 37 Results: Pure-Sigma Common-Matrix Obstruction
Generated: 2026-03-29T19:17:12

## 37a. Exact Pure-Sigma Obstruction Operator

[EXACT_DERIVED]

Let Sigma, H, Delta be the Step 51/Phase 17 fiber-mode blocks, with H = [eta1 | eta2].
Define the silent sigma-sector

  Z = ker(H^T) intersect ker(Delta^T).

If dim(Z) = 9 and Sigma_Z := Sigma^T|_Z is invertible, define the 9x9 operator

  Omega = (Gamma|_Z) * Sigma_Z^(-1).

Then every z in Z satisfies

  Gamma z = Omega (Sigma^T z).

Now suppose Delta subset span(H). Any Phase 36 witness w already obeys H^T w = 0, so it also obeys
Delta^T w = 0 and therefore lies in Z. Because Sigma^T w = 3 vec(C(w)), every nonzero witness matrix must satisfy

  Omega vec(C(w)) = 0.

So invertibility of Omega is an exact common-matrix obstruction inside the Delta-contained regime.

## 37b. Known Exact Decompositions

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

| case | rank(H) | rank(Delta) | rank([H|Delta]) | dim Z | rank Sigma_Z | det Sigma_Z | det Omega | min eig(Omega) | max eig(Omega) | symmetric? | positive definite? |
|------|---------|-------------|------------------|-------|--------------|------------|-----------|----------------|----------------|------------|--------------------|
| alphatensor_rank23 | 14 | 10 | 14 | 9 | 9 | 19683 | 77875/19683 | 0.189392234833 | 4.447777472960 | True | True |
| standard_rank27 | 18 | 0 | 18 | 9 | 9 | 19683 | 1 | 1.000000000000 | 1.000000000000 | True | True |

Two exact facts stand out.

1. The silent sigma-sector has the clean expected size dim(Z) = 9 for both exact decompositions.
2. Omega is symmetric positive definite in both cases, so the common-matrix equation Omega vec(C) = 0 forces C = 0.

In particular, the standard algorithm collapses to the strongest possible certificate:

  Omega_standard = I_9.

For AlphaTensor the operator is no longer diagonal, but it remains exact, symmetric, and invertible. Its leading principal minors are all positive, so positivity survives the entangled case too.

AlphaTensor exact characteristic polynomial:

  (19683*x**9 - 314928*x**8 + 2053593*x**7 - 7109208*x**6 + 14304924*x**5 - 17208855*x**4 + 12251790*x**3 - 4925403*x**2 + 1004694*x - 77875)/19683

## 37c. Structured Factorized Defect Branches

[MEASURED_FROM_CODE]

These are the Phase 19-style factorized defect projections with gamma re-solved by least squares.

| base | family | dim Z | rank Sigma_Z | operator defined? | min eig(Omega) | tensor max residual |
|------|--------|-------|--------------|-------------------|----------------|---------------------|
| alphatensor_rank23 | pair_equal_01 | 5 | 0 | False | n/a | 0.500000000000 |
| alphatensor_rank23 | pair_equal_12 | 4 | 0 | False | n/a | 0.500000000000 |
| alphatensor_rank23 | all_equal | 14 | 0 | False | n/a | 0.666666666667 |
| alphatensor_rank23 | collinear | 3 | 3 | False | n/a | 0.758861599108 |
| standard_rank27 | pair_equal_01 | 9 | 0 | False | n/a | 0.500000000000 |
| standard_rank27 | pair_equal_12 | 9 | 0 | False | n/a | 0.500000000000 |
| standard_rank27 | all_equal | 18 | 0 | False | n/a | 0.666666666667 |
| standard_rank27 | collinear | 9 | 9 | True | 1.000000000000 | 0.000000000000 |

The clean obstruction operator disappears exactly on the rigid defect loci that already showed hard tensor residual floors in Phase 19.

- Pair-equality and all-equal projections force rank(Sigma_Z) = 0, so the common sigma image collapses completely.
- The standard collinear branch keeps the exact operator intact and stays exact, matching the older observation that collinearity is not itself an obstruction.
- The AlphaTensor collinear branch shrinks the silent sector itself to dim(Z) = 3, so the 9x9 common-matrix operator no longer exists in that family.

## 37d. Wildcard Random Channel Rescalings

[WILDCARD]

Each wildcard trial rescales every term's three live channels independently, then re-solves gamma by least squares.

| base | trials | operator-defined trials | dim Z range | rank Sigma_Z range | tensor residual range |
|------|--------|------------------------|-------------|--------------------|-----------------------|
| alphatensor_rank23 | 12 | 0 | 1..1 | 1..1 | 0.479073506549..0.779925965050 |
| standard_rank27 | 12 | 12 | 9..9 | 9..9 | 0.000000000000..0.000000000000 |

The wildcard branch splits sharply by base family. Standard-derived channel rescalings preserve the clean 9-dimensional sigma-only sector and Omega = I in every sampled trial, while AlphaTensor-derived rescalings collapse immediately to a 1-dimensional silent sector and a visible tensor residual.

## 37e. Interpretation

Phase 37 converts the witness-matrix question into a smaller exact object. Instead of asking directly whether a nonzero w in ker(Gamma) can satisfy channel collapse, we isolate the sigma-only silent sector Z and the induced 9x9 output operator Omega.

For the known exact decompositions, Omega is already enough to rule out common matrices once Delta containment is granted: standard gives Omega = I, while AlphaTensor gives an exact symmetric positive-definite rational matrix. So the Phase 36 witness problem has now been compressed to a finite 9x9 certificate on the sigma-only sector, rather than the full R-dimensional term space.

This is not yet a universal theorem, because Phase 37 still uses the Delta-contained regime rather than proving it universally. But it sharpens the obstruction target substantially: the next theorem-level step is to show that every valid minimum-rank decomposition has the same clean 9-dimensional sigma-only sector and an invertible induced Omega.
