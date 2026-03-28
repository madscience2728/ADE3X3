# Step 64: Small Integer Coefficient Enumeration
Generated: 2026-03-28T16:43:21

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

Step 64 treats alpha,beta in {-1,0,1}^{n x n} as a finite exact pool. The main exact reduction is that a bilinear profile is an outer product of two nonzero ternary vectors, so duplicate elimination and sign quotienting are projective-line counts rather than a 387M-object hash table.

## Pool Statistics

- 3x3 raw pairs = 387420489
- 3x3 nonzero pairs = 387381124
- 3x3 distinct nonzero profiles modulo duplicates and negation = 96845281
- 3x3 symmetry-reduced profile orbits = 570521
- 3x3 dead-free nonzero profiles = 0

## Top Usefulness Profiles

- rank 1: score=3.674235, projection=27, nuisance_sq=54
- rank 2: score=3.674235, projection=27, nuisance_sq=54
- rank 3: score=3.674235, projection=27, nuisance_sq=54
- rank 4: score=3.674235, projection=27, nuisance_sq=54
- rank 5: score=3.464102, projection=18, nuisance_sq=27
- rank 6: score=3.464102, projection=18, nuisance_sq=27
- rank 7: score=3.464102, projection=18, nuisance_sq=27
- rank 8: score=3.464102, projection=18, nuisance_sq=27
- rank 9: score=3.360672, projection=24, nuisance_sq=51
- rank 10: score=3.360672, projection=24, nuisance_sq=51

## 3x3 Collapsed 81D Greedy

- first exact zero residual at rank 3

## 2x2 Verification

- corrected full-tensor greedy does not reach exact zero through rank 7
- 2x2 distinct profiles modulo sign = 1600
- 2x2 symmetry-reduced profiles = 320

[INTERPRETATION]

The finite pool itself is exact and much larger than the initial rough expectation: the 3x3 duplicate/negation quotient already leaves 96,845,281 distinct nonzero profiles before symmetry reduction. The collapsed 81D greedy is therefore a search over a genuine large finite dictionary, but it is still too weak to certify a full matrix-multiplication algorithm because it ignores the 9-component gamma output layer. That is why the 2x2 validation in this step uses the corrected full tensor with optimal gamma vectors per selected profile.