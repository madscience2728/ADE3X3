# Step 55: Algebraic Nuisance Dependencies + Wildcard Exploration
Generated: 2026-03-28T12:26:38

[EXACT_DERIVED]

Step 55 follows the Step 54 obstruction into the Hadamard coordinate system. The main exact point is that once alpha=A*C and beta=B*D are restricted to a p*q-dimensional Hadamard space, full 9-dimensional quotient recovery forces a much tighter nuisance-rank target than the raw R-constraint alone.

## Track A: Hadamard-Space Dependency Arithmetic

| regime | Hadamard dim upper bound | quotient target | max nuisance from geometry | max nuisance from R=22 | combined target |
|--------|---------------------------|-----------------|----------------------------|------------------------|----------------|
| p=3,q=3 | 9 | 9 | 0 | 13 | 0 |
| p=3,q=4 | 12 | 9 | 3 | 13 | 3 |
| p=4,q=3 | 12 | 9 | 3 | 13 | 3 |

The p=q=3 regime has Hadamard dimension 9, so full quotient recovery would force nuisance rank 0. The p=3,q=4 regime raises the ambient Hadamard dimension to 12, but still forces nuisance rank <= 3 if Sigma is to contribute 9 independent quotient directions.

### Best Structured 3x4 Families

| family | label | c_rank | d_rank | hadamard_dim | sigma_rank | nuisance_rank | quotient_gain | meets nuisance<=3 | full quotient target |
|--------|-------|--------|--------|--------------|------------|---------------|---------------|-------------------|----------------------|
| dft_modes | dft_modes_3 | 3 | 4 | 12 | 3 | 11 | 1 | False | False |
| toeplitz_genericD | toeplitz_genericD_2 | 3 | 4 | 12 | 3 | 12 | 0 | False | False |

### Structured Family Sweep

| family | label | ranks(C,D) | nuisance_rank | quotient_gain | notes |
|--------|-------|------------|---------------|---------------|-------|
| toeplitz_circulant | toeplitz_circulant_1 | (2,3) | 6 | 0 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_2 | (2,3) | 6 | 0 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_3 | (2,3) | 6 | 0 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_4 | (2,3) | 6 | 0 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_5 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_6 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_7 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_8 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_9 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_10 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_11 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_12 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_13 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_14 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_15 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_circulant | toeplitz_circulant_16 | (3,3) | 8 | 1 | C uses summation-mode Toeplitz structure; D uses summation-mode circulant structure. |
| toeplitz_genericD | toeplitz_genericD_1 | (2,4) | 8 | 0 | Toeplitz C against a fixed full-rank 4x9 D. |
| toeplitz_genericD | toeplitz_genericD_2 | (3,4) | 12 | 0 | Toeplitz C against a fixed full-rank 4x9 D. |
| toeplitz_genericD | toeplitz_genericD_3 | (3,4) | 12 | 0 | Toeplitz C against a fixed full-rank 4x9 D. |
| toeplitz_genericD | toeplitz_genericD_4 | (3,4) | 12 | 0 | Toeplitz C against a fixed full-rank 4x9 D. |
| genericC_circulant | genericC_circulant_1 | (3,3) | 9 | 0 | Fixed full-rank 3x9 C against circulant D. |
| genericC_circulant | genericC_circulant_2 | (3,3) | 9 | 0 | Fixed full-rank 3x9 C against circulant D. |
| genericC_circulant | genericC_circulant_3 | (3,3) | 9 | 0 | Fixed full-rank 3x9 C against circulant D. |
| genericC_circulant | genericC_circulant_4 | (3,3) | 9 | 0 | Fixed full-rank 3x9 C against circulant D. |
| shared_latent_shadow | shared_latent_shadow | (3,3) | 9 | 0 | C and D both factor through a common 3-dimensional latent matrix; D rank cannot exceed 3 in this construction. |
| dft_modes | dft_modes_1 | (3,4) | 11 | 1 | C uses exact 3-point Fourier modes; D mixes Fourier rows with one extra row to allow q=4. |
| dft_modes | dft_modes_2 | (3,4) | 11 | 1 | C uses exact 3-point Fourier modes; D mixes Fourier rows with one extra row to allow q=4. |
| dft_modes | dft_modes_3 | (3,4) | 11 | 1 | C uses exact 3-point Fourier modes; D mixes Fourier rows with one extra row to allow q=4. |
| dft_modes | dft_modes_4 | (3,4) | 11 | 1 | C uses exact 3-point Fourier modes; D mixes Fourier rows with one extra row to allow q=4. |

### Symbolic Witness

- 4-parameter family `toeplitz_genericD_symbolic`: selected 4x4 minor rows (0, 1, 2, 3) and columns (0, 1, 2, 9) has determinant `-21*a**3*b + 8*a**3 + 25*a**2*b + 4*a**2 + 3*a*b - 8*a - 7*b`.
- Rank-drop locus witness: -21*a**3*b + 8*a**3 + 25*a**2*b + 4*a**2 + 3*a*b - 8*a - 7*b = 0

[INTERPRETATION]

The exact geometry tightens the sweet-spot target much more than Step 54 alone suggested: for p=3,q=4 the nuisance span must compress to dimension at most 3, not merely <= 13. In the tested structured families, the best actual 3x4 nuisance rank was 11 and the best quotient gain stayed 1. The shared-latent construction collapses q down to 3 automatically, so it cannot inhabit the intended (3,4) regime. The DFT-aligned and Toeplitz/circulant families therefore still look trapped inside the same Hadamard-space obstruction, even after imposing visible algebraic symmetry.

## Track B: Wildcards

[WILDCARD]

### Characteristic-2 Shadow

| flattening | shape | nonzero entries | GF(2) rank | real rank |
|------------|-------|-----------------|------------|-----------|
| A_vs_BC | 9x81 | 27 | 9 | 9 |
| B_vs_AC | 9x81 | 27 | 9 | 9 |
| C_vs_AB | 9x81 | 27 | 9 | 9 |

### Tropical Flattening Ranks

| flattening | shape | tropical rank | upper bound | implies R>=23? |
|------------|-------|---------------|-------------|----------------|
| A_vs_BC | 9x81 | 9 | 9 | False |
| B_vs_AC | 9x81 | 9 | 9 | False |
| C_vs_AB | 9x81 | 9 | 9 | False |

### Fiber Commutators

| algorithm | term_count | ordered_pairs | zero_commutators | nonzero_commutators | max_commutator_rank |
|-----------|------------|---------------|------------------|---------------------|---------------------|
| standard_3x3 | 27 | 729 | 81 | 648 | 2 |
| strassen_2x2 | 7 | 49 | 7 | 42 | 2 |

[INTERPRETATION]

Over GF(2), all three flattenings still have rank 9, so the characteristic-2 shadow only returns the obvious lower bound 9. The tropical flattening ranks are also 9, and this route cannot possibly certify R >= 23 because a 9x81 flattening has tropical rank at most 9. The commutator wildcard is more informative structurally: the standard 27-term decomposition has only 81 zero ordered-pair commutators out of 729, so 'different target entry' does not force vanishing. Strassen likewise has a nontrivial overlap pattern, with only 7 zero ordered pairs out of 49.