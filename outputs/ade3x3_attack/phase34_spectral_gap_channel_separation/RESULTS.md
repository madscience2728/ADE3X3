# Phase 34: Spectral Gap of the Channel-Separation Quadratic Form
Generated: 2026-03-29T17:15:48

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

The Phase 34 target is the exact defect condition from Phase 33b:

  W_0 = W_1 = ... = W_{n-1} for some nonzero w in ker(Gamma).

Equivalently, the total channel-separation quadratic form

  Q_chan(w) = sum_{s<t} ||W_s - W_t||_F^2

must vanish on a nonzero kernel vector. This phase computes Q_chan exactly on ker(Gamma),
together with the individual pair forms Q_st, so the spectral-gap statement is now explicit.

## Track A: Exact Pair and Total Spectra

### alphatensor_rank23

- ker(Gamma) dimension: 14
- total Q_chan exact spectrum: roots of lambda**14 - 124*lambda**13 + 6582*lambda**12 - 197458*lambda**11 + 3731176*lambda**10 - 46947434*lambda**9 + 405911585*lambda**8 - 2453232462*lambda**7 + 10430747671*lambda**6 - 31089865444*lambda**5 + 63998400093*lambda**4 - 88241687082*lambda**3 + 76999874076*lambda**2 - 38032056360*lambda + 8023617000
- total Q_chan numerical spectrum: 0.834405031117 x 1; 1.027471237850 x 1; 1.710641909229 x 1; 2.352345040120 x 1; 2.648231163759 x 1; 3.350388740865 x 1; 4.125083289338 x 1; 5.444376772489 x 1; 6.931756677004 x 1; 8.978034776123 x 1; 13.230232274745 x 1; 20.964658410373 x 1; 22.985777465540 x 1; 29.416597211448 x 1
- total spectral gap: root of lambda**14 - 124*lambda**13 + 6582*lambda**12 - 197458*lambda**11 + 3731176*lambda**10 - 46947434*lambda**9 + 405911585*lambda**8 - 2453232462*lambda**7 + 10430747671*lambda**6 - 31089865444*lambda**5 + 63998400093*lambda**4 - 88241687082*lambda**3 + 76999874076*lambda**2 - 38032056360*lambda + 8023617000 ≈ 0.834405031117
- total nullity on ker(Gamma): 0

| pair | rank on ker(Gamma) | nullity on ker(Gamma) | exact spectrum | numerical spectrum | min positive eigenvalue |
|------|--------------------|-----------------------|----------------|--------------------|-------------------------|
| 01 | 6 | 8 | roots of lambda**14 - 48*lambda**13 + 824*lambda**12 - 6213*lambda**11 + 20920*lambda**10 - 29664*lambda**9 + 13440*lambda**8 | 0.000000000000 x 8; 0.838536614790 x 1; 2.092959729337 x 1; 3.426263226875 x 1; 8.121039863178 x 1; 14.374142230596 x 1; 19.147058335224 x 1 | root of lambda**14 - 48*lambda**13 + 824*lambda**12 - 6213*lambda**11 + 20920*lambda**10 - 29664*lambda**9 + 13440*lambda**8 ≈ 0.838536614790 |
| 02 | 9 | 5 | roots of lambda**14 - 39*lambda**13 + 584*lambda**12 - 4433*lambda**11 + 18860*lambda**10 - 46989*lambda**9 + 69312*lambda**8 - 58906*lambda**7 + 26319*lambda**6 - 4727*lambda**5 | 0.000000000000 x 5; 0.556794545079 x 1; 0.820769896791 x 1; 1.435235759135 x 1; 1.480402758325 x 1; 1.819870272552 x 1; 3.612713239405 x 1; 6.085221711569 x 1; 8.023319364763 x 1; 15.165672452381 x 1 | root of lambda**14 - 39*lambda**13 + 584*lambda**12 - 4433*lambda**11 + 18860*lambda**10 - 46989*lambda**9 + 69312*lambda**8 - 58906*lambda**7 + 26319*lambda**6 - 4727*lambda**5 ≈ 0.556794545079 |
| 12 | 8 | 6 | roots of lambda**14 - 37*lambda**13 + 508*lambda**12 - 3441*lambda**11 + 12749*lambda**10 - 26787*lambda**9 + 31389*lambda**8 - 18732*lambda**7 + 4292*lambda**6 | 0.000000000000 x 6; 0.576313144292 x 1; 1.288811053824 x 1; 1.599972577017 x 1; 2.108913126238 x 1; 2.738054588268 x 1; 5.046166261315 x 1; 7.847722223594 x 1; 15.794047025451 x 1 | root of lambda**14 - 37*lambda**13 + 508*lambda**12 - 3441*lambda**11 + 12749*lambda**10 - 26787*lambda**9 + 31389*lambda**8 - 18732*lambda**7 + 4292*lambda**6 ≈ 0.576313144292 |

### standard_rank27

- ker(Gamma) dimension: 18
- total Q_chan exact spectrum: 3 x 18
- total Q_chan numerical spectrum: 3.000000000000 x 18
- total spectral gap: 3 ≈ 3.000000000000
- total nullity on ker(Gamma): 0

| pair | rank on ker(Gamma) | nullity on ker(Gamma) | exact spectrum | numerical spectrum | min positive eigenvalue |
|------|--------------------|-----------------------|----------------|--------------------|-------------------------|
| 01 | 9 | 9 | 0 x 9; 2 x 9 | 0.000000000000 x 9; 2.000000000000 x 9 | 2 ≈ 2.000000000000 |
| 02 | 9 | 9 | 0 x 9; 2 x 9 | 0.000000000000 x 9; 2.000000000000 x 9 | 2 ≈ 2.000000000000 |
| 12 | 9 | 9 | 0 x 9; 2 x 9 | 0.000000000000 x 9; 2.000000000000 x 9 | 2 ≈ 2.000000000000 |

### strassen_2x2

- ker(Gamma) dimension: 3
- total Q_chan exact spectrum: 2 x 2; 10 x 1
- total Q_chan numerical spectrum: 2.000000000000 x 2; 10.000000000000 x 1
- total spectral gap: 2 ≈ 2.000000000000
- total nullity on ker(Gamma): 0

| pair | rank on ker(Gamma) | nullity on ker(Gamma) | exact spectrum | numerical spectrum | min positive eigenvalue |
|------|--------------------|-----------------------|----------------|--------------------|-------------------------|
| 01 | 3 | 0 | 2 x 2; 10 x 1 | 2.000000000000 x 2; 10.000000000000 x 1 | 2 ≈ 2.000000000000 |

## Track B: Standard-Family Pattern

For the standard n x n algorithm, ker(Gamma) splits into n^2 independent output fibers, each
isomorphic to the zero-sum hyperplane {x in R^n : sum_i x_i = 0}. On one such fiber,
Q_st(x) = (x_s - x_t)^2 has spectrum {2, 0, ..., 0} and the total form
Q_chan(x) = sum_{s<t} (x_s - x_t)^2 = n ||x||^2 on that hyperplane.

| family | R | ker(Gamma) dim | pair spectrum | pair gap | total spectrum | total gap |
|--------|---|----------------|---------------|----------|----------------|-----------|
| standard_2x2 | 8 | 4 | 2 x 4 | 2 | 2 x 4 | 2 |
| standard_3x3 | 27 | 18 | 0 x 9; 2 x 9 | 2 | 3 x 18 | 3 |
| standard_4x4 | 64 | 48 | 0 x 32; 2 x 16 | 2 | 4 x 48 | 4 |

## Track C: Lower-Bound Status

The exact-defect condition is absent whenever Q_chan is positive definite on ker(Gamma),
equivalently when the stacked pair-difference map has full rank on ker(Gamma).

| decomposition | total rank on ker(Gamma) | total nullity on ker(Gamma) | exact total spectral gap | conclusion |
|---------------|--------------------------|-----------------------------|--------------------------|------------|
| alphatensor_rank23 | 14 | 0 | root of lambda**14 - 124*lambda**13 + 6582*lambda**12 - 197458*lambda**11 + 3731176*lambda**10 - 46947434*lambda**9 + 405911585*lambda**8 - 2453232462*lambda**7 + 10430747671*lambda**6 - 31089865444*lambda**5 + 63998400093*lambda**4 - 88241687082*lambda**3 + 76999874076*lambda**2 - 38032056360*lambda + 8023617000 ≈ 0.834405031117 | exact defect absent |
| standard_rank27 | 18 | 0 | 3 ≈ 3.000000000000 | exact defect absent |
| strassen_2x2 | 3 | 0 | 2 ≈ 2.000000000000 | exact defect absent |

AlphaTensor's exact spectrum is algebraic rather than rational in this basis-independent
generalized-eigenvalue sense; the standard family and Strassen give exact rational spectra.

## Track D: Nullspace / Intersection Structure

### alphatensor_rank23

| pair subset | common nullspace dimension inside ker(Gamma) |
|-------------|----------------------------------------------|
| 01 | 8 |
| 02 | 5 |
| 12 | 6 |
| 01,02 | 0 |
| 01,12 | 0 |
| 02,12 | 0 |
| 01,02,12 | 0 |

### standard_rank27

| pair subset | common nullspace dimension inside ker(Gamma) |
|-------------|----------------------------------------------|
| 01 | 9 |
| 02 | 9 |
| 12 | 9 |
| 01,02 | 0 |
| 01,12 | 0 |
| 02,12 | 0 |
| 01,02,12 | 0 |

### strassen_2x2

| pair subset | common nullspace dimension inside ker(Gamma) |
|-------------|----------------------------------------------|
| 01 | 0 |

## Track E: Strassen 2x2 Exact Positivity Certificate

On Strassen there is only one channel pair, so Q_chan = Q_01. In the exact kernel basis
returned by SymPy, the basis Gram of Q_chan is:

- basis Gram: [[6, -2, 2], [-2, 6, -2], [2, -2, 26]]
- leading principal minors: 6, 32, 800
- generalized spectrum on ker(Gamma): 2 x 2; 10 x 1
- exact gap: 2 ≈ 2.000000000000

All leading principal minors are strictly positive, so the basis Gram is positive definite.
Therefore Q_chan(w) > 0 for every nonzero w in ker(Gamma), and Strassen admits no exact
channel-equality defect.

## Track F: Integer Gram Structure

Clearing denominators in the kernel basis gives an integral basis for ker(Gamma).
In that basis, the total channel-separation Gram matrix is integral for all three known
exact decompositions, so the obstruction can be studied as an honest integer quadratic form.

| decomposition | integer Gram determinant | integer kernel basis available? |
|---------------|--------------------------|----------------------------------|
| alphatensor_rank23 | 624839173875000 | yes |
| standard_rank27 | 7625597484987 | yes |
| strassen_2x2 | 800 | yes |

## Conclusions

Phase 34 makes the spectral target exact. The total channel-separation form Q_chan is
strictly positive on ker(Gamma) for AlphaTensor, the standard 3x3 algorithm, and Strassen.
So the exact defect condition W_0 = W_1 = ... = W_{n-1} does not occur on any known exact
minimum-rank decomposition in the repository.

For the standard family the pattern is completely clean: pair gaps stay fixed at 2, while
the total gap is exactly n. That makes the right next theorem target sharper: prove that the
minimum total channel-separation eigenvalue stays uniformly positive over the admissible
rank-R multiplication variety, rather than chasing a termwise identity.
