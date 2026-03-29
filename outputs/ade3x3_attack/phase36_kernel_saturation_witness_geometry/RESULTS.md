# Phase 36 Results: Kernel-Saturation Witness Geometry
Generated: 2026-03-29T17:55:53

## 36a. Saturation Witness Theorem

[EXACT_DERIVED]

Fix a 3x3 right-inverse family P_0, P_1, P_2 with Gamma P_s = I_9 and
H = [P_0-P_1 | P_1-P_2]. Then

  ker(Gamma) ∩ ker(H^T) = { w : Gamma w = 0 and P_0^T w = P_1^T w = P_2^T w }.

So nonsaturation is equivalent to the existence of a nonzero witness w whose three
channel images collapse to one common 3x3 matrix C(w). Phase 36 attacks the universal
theorem by measuring the geometry of that witness space directly.

## 36b. Known Exact Decompositions

[MEASURED_FROM_CODE]

| case | dim ker(Gamma) | rank(H) | saturation defect | witness dim | common-image dim | common-matrix rank set |
|------|----------------|---------|-------------------|-------------|------------------|------------------------|
| strassen_2x2 | 3 | 3 | 0 | 0 | 0 | none |
| standard_rank27 | 18 | 18 | 0 | 0 | 0 | none |
| alphatensor_rank23 | 14 | 14 | 0 | 0 | 0 | none |

As expected, the known exact decompositions have witness dimension 0. So the Phase 36
question is not whether a witness can exist abstractly, but what structure a witness would
have to force inside a valid low-rank decomposition.

## 36c. Structured Defect Families Over Fixed Gamma

[MEASURED_FROM_CODE]

These families keep Gamma P_s = I_9 exactly but replace the actual channel faces by synthetic
kernel lifts P_s = X_0 + K_s with K_s in ker(Gamma).

| case | dim ker(Gamma) | rank(H) | saturation defect | witness dim | common-image dim | common-matrix rank set |
|------|----------------|---------|-------------------|-------------|------------------|------------------------|
| standard_rank27_all_equal | 18 | 0 | 18 | 18 | 9 | 3 |
| standard_rank27_k0_equals_k1 | 18 | 9 | 9 | 9 | 9 | 3 |
| standard_rank27_all_collinear | 18 | 1 | 17 | 17 | 9 | 0,1,2 |
| standard_rank27_two_mode_split | 18 | 6 | 12 | 12 | 9 | 3 |
| alphatensor_rank23_all_equal | 14 | 0 | 14 | 14 | 9 | 3 |
| alphatensor_rank23_k0_equals_k1 | 14 | 9 | 5 | 5 | 5 | 3 |
| alphatensor_rank23_all_collinear | 14 | 1 | 13 | 13 | 9 | 1,2,3 |
| alphatensor_rank23_two_mode_split | 14 | 6 | 8 | 8 | 8 | 3 |

The exact defect dimension always matches the witness-space dimension, as it must. The useful
extra measurement is the common-image dimension: it tracks how many independent shared 3x3
matrices survive once the three channel lifts are forced together.

## 36d. Witness Basis Samples

[MEASURED_FROM_CODE]

| case | basis vector | common rank | common matrix |
|------|--------------|-------------|---------------|
| standard_rank27_all_equal | b0 | 3 | [[3, -2, -1], [1, -2, -2], [-1, 0, 2]] |
| standard_rank27_all_equal | b1 | 3 | [[8, -7, -4], [4, -8, -8], [-4, 0, 8]] |
| standard_rank27_all_equal | b2 | 3 | [[-8, 8, 5], [-4, 8, 8], [4, 0, -8]] |
| standard_rank27_all_equal | b3 | 3 | [[-1, 2, 1], [-1, 2, 2], [1, 0, -2]] |
| standard_rank27_all_equal | b4 | 3 | [[10, -9, -5], [5, -10, -10], [-5, 0, 10]] |
| standard_rank27_all_equal | b5 | 3 | [[-4, 4, 3], [-2, 4, 4], [2, 0, -4]] |
| standard_rank27_all_equal | b6 | 3 | [[-2, 2, 1], [0, 2, 2], [1, 0, -2]] |
| standard_rank27_all_equal | b7 | 3 | [[-6, 6, 3], [-3, 7, 6], [3, 0, -6]] |
| standard_rank27_all_equal | b8 | 3 | [[4, -4, -2], [2, -4, -3], [-2, 0, 4]] |
| standard_rank27_all_equal | b9 | 3 | [[2, -2, -1], [2, -2, -2], [-1, 0, 2]] |
| standard_rank27_all_equal | b10 | 3 | [[-6, 6, 3], [-3, 7, 6], [3, 0, -6]] |
| standard_rank27_all_equal | b11 | 3 | [[-4, 4, 2], [-2, 4, 5], [2, 0, -4]] |
| standard_rank27_all_equal | b12 | 3 | [[2, -2, -1], [1, -2, -2], [0, 0, 2]] |
| standard_rank27_all_equal | b13 | 3 | [[12, -12, -6], [6, -12, -12], [-6, 1, 12]] |
| standard_rank27_all_equal | b14 | 3 | [[-2, 2, 1], [-1, 2, 2], [1, 0, -1]] |
| standard_rank27_all_equal | b15 | 3 | [[-2, 2, 1], [-1, 2, 2], [2, 0, -2]] |
| standard_rank27_all_equal | b16 | 3 | [[12, -12, -6], [6, -12, -12], [-6, 1, 12]] |
| standard_rank27_all_equal | b17 | 3 | [[2, -2, -1], [1, -2, -2], [-1, 0, 3]] |

In the synthetic defect families the common matrices are typically low-rank or sparse-patterned.
That is not yet a theorem, but it suggests a more concrete universal target: show that any common
matrix C(w) forced by a nonzero witness is incompatible with exact multiplication identities.

## 36e. Wildcard Random Right-Inverse Scan

[WILDCARD]

| gamma family | trials | saturated trials | deficient trials | max witness dim seen |
|--------------|--------|-----------------|-----------------|----------------------|
| standard_rank27 | 18 | 18 | 0 | 0 |
| alphatensor_rank23 | 18 | 18 | 0 | 0 |

This wildcard branch is only a stress test. It asks how often random kernel lifts accidentally
create shared-channel witnesses. Generic lifts should saturate; deficient families should appear
as special coincidence loci.

## 36f. Interpretation

Phase 36 starts the direct saturation program rather than another spectral pass. The key shift is
to treat nonsaturation as an exact witness geometry problem: find or rule out nonzero w in ker(Gamma)
that produce one common channel matrix across all three faces. The next theorem target is therefore
not an eigenvalue bound but a structural impossibility statement for such common matrices inside valid
minimum-rank decompositions.
