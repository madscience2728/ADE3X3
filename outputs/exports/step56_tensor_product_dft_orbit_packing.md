# Step 56: Tensor-Product DFT Construction + Orbit Packing
Generated: 2026-03-28T13:42:05

[EXACT_DERIVED]

Step 56 moves to the p=q=4 regime, where the Hadamard space has dimension 16 and therefore leaves room for at most 7 nuisance directions if the quotient gain is to reach 9. The top-signal computation is the complete tensor-product DFT sweep over all 126 x 126 mode pairs.

## Track A: p=q=4 Structured Families

**Hadamard nuisance cap for p=q=4:** 7
**Best DFT quotient gain:** 1
**Best DFT nuisance rank:** 15
**Best DFT mode pair:** 00|01|02|10 :: 00|01|10|20

### Complete DFT Sweep: Top 10 Mode Pairs

| rank | C modes | D modes | C has DC | D has DC | sigma_rank | nuisance_rank | quotient_gain | meets nuisance<=7 |
|------|---------|---------|----------|----------|------------|---------------|---------------|-------------------|
| 1 | 00|01|02|10 | 00|01|10|20 | True | True | 4 | 15 | 1 | False |
| 2 | 00|01|02|10 | 00|01|11|21 | True | True | 4 | 15 | 1 | False |
| 3 | 00|01|02|10 | 00|02|10|20 | True | True | 4 | 15 | 1 | False |
| 4 | 00|01|02|10 | 00|02|12|22 | True | True | 4 | 15 | 1 | False |
| 5 | 00|01|02|10 | 01|02|11|21 | True | False | 4 | 15 | 1 | False |
| 6 | 00|01|02|10 | 01|02|12|22 | True | False | 4 | 15 | 1 | False |
| 7 | 00|01|02|11 | 00|10|20|21 | True | True | 4 | 15 | 1 | False |
| 8 | 00|01|02|11 | 00|10|20|22 | True | True | 4 | 15 | 1 | False |
| 9 | 00|01|02|11 | 01|11|20|21 | True | False | 4 | 15 | 1 | False |
| 10 | 00|01|02|11 | 01|11|21|22 | True | False | 4 | 15 | 1 | False |

### DC Usage in the Top 10

| DC pattern | top-10 count |
|------------|--------------|
| C_dc=True,D_dc=False | 4 |
| C_dc=True,D_dc=True | 6 |

### Coordinate-vs-Term Verification for the Top DFT Pairs

| rank | C modes | D modes | coordinate nuisance | actual nuisance | coordinate gain | actual gain | matches |
|------|---------|---------|--------------------|-----------------|-----------------|-------------|---------|
| 1 | 00|01|02|10 | 00|01|10|20 | 15 | 15 | 1 | 1 | True |
| 2 | 00|01|02|10 | 00|01|11|21 | 15 | 15 | 1 | 1 | True |
| 3 | 00|01|02|10 | 00|02|10|20 | 15 | 15 | 1 | 1 | True |
| 4 | 00|01|02|10 | 00|02|12|22 | 15 | 15 | 1 | 1 | True |
| 5 | 00|01|02|10 | 01|02|11|21 | 15 | 15 | 1 | 1 | True |
| 6 | 00|01|02|10 | 01|02|12|22 | 15 | 15 | 1 | 1 | True |
| 7 | 00|01|02|11 | 00|10|20|21 | 15 | 15 | 1 | 1 | True |
| 8 | 00|01|02|11 | 00|10|20|22 | 15 | 15 | 1 | 1 | True |
| 9 | 00|01|02|11 | 01|11|20|21 | 15 | 15 | 1 | 1 | True |
| 10 | 00|01|02|11 | 01|11|21|22 | 15 | 15 | 1 | 1 | True |

### Other Structured Families

| family | best quotient_gain | best nuisance_rank |
|--------|--------------------|--------------------|
| ternary random sample (10000) | 0 | 16 |
| algebraic random sample (10000) | 0 | 16 |
| interpreted correction-block sweep | 0 | 16 |
| numpy local search surrogate | 0 | 16 |

[INTERPRETATION]

The full DFT sweep is definitive for this construction family. If the best quotient gain stays low here, the tensor-product DFT basis is not by itself the missing structured ansatz. The p=q=4 Hadamard cap of 7 is looser than the p=3,q=4 cap of 3, but still severe: the DFT family must both compress 72 nuisance columns into at most 7 dimensions and free 9 quotient dimensions. The coordinate-vs-term verification confirms that the Hadamard-coordinate computation is faithfully tracking the actual 22-term matrices for the best DFT pairs.

## Track B: Discrete Orbit Packing

| branch | count | fraction |
|--------|-------|----------|
| 30 o 30 -> 0 | 54 | 0.500000 |
| 30 o 30 -> 30 | 54 | 0.500000 |

| removed term | removed fiber | orbit0 after removal | orbit30 after removal | delta orbit0 | delta orbit30 |
|--------------|--------------|----------------------|----------------------|--------------|--------------|
| m[0,0,0] | (0, 0) | 26 | 25 | -1 | -2 |
| m[0,0,1] | (0, 1) | 26 | 25 | -1 | -2 |
| m[0,0,2] | (0, 2) | 26 | 25 | -1 | -2 |
| m[0,1,0] | (0, 0) | 26 | 25 | -1 | -2 |
| m[0,1,1] | (0, 1) | 26 | 25 | -1 | -2 |
| m[0,1,2] | (0, 2) | 26 | 25 | -1 | -2 |
| m[0,2,0] | (0, 0) | 26 | 25 | -1 | -2 |
| m[0,2,1] | (0, 1) | 26 | 25 | -1 | -2 |
| m[0,2,2] | (0, 2) | 26 | 25 | -1 | -2 |

| algorithm | fibers hit | total aligned occurrences | unique live X atoms | unique distinct same-fiber pairs | occurrence distinct same-fiber pairs |
|-----------|-----------|--------------------------|---------------------|----------------------------------|--------------------------------------|
| standard_2x2_basis | 4 | 8 | 8 | 4 | 4 |
| strassen_2x2 | 4 | 8 | 8 | 4 | 4 |

[INTERPRETATION]

The same-fiber branching point remains exactly balanced: 30 o 30 splits 54 / 54. So there is no asymmetry lever hiding in the raw same-fiber table. For the standard 27-term algorithm, removing any one basis term reduces the same-fiber orbit profile by -1 on orbit 0 and -2 on orbit 30, matching the fact that each term belongs to one 3-element fiber. In the coarse 2x2 same-fiber live-support analog, Strassen and the standard 8-term basis algorithm have the same counts, so that analog is too coarse to expose Strassen's cancellation structure.

[INTERPRETATION]

The direct minimization track used a numpy local-search surrogate because PyTorch is not installed in the current environment. The reported values are exact measurements of the best observed restarts under that search, not a claim of global optimality.