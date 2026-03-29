# Phase 5 Diagnostics Results

## Completed In This Pass

### 5b. Nuisance Dimension Trajectory

- In the original AlphaTensor term order, rank(H) first reaches 14 at k = 18.
- The full Step 52 quotient gain of 9 is reached only at k = 23 in every tested ordering.
- Across 20 random orderings, the first k with rank(H) = 14 ranged from 14 to 21.
- The final profile is order-invariant in this sample: rank(H) = 14, rank(Delta) = 10, rank(Nuisance) = 14, quotient gain = 9.
- Exactly 9 terms contribute zero additional nuisance rank in every tested ordering summary.

### 5e. AlphaTensor Anisotropy Identities

- The anisotropy matrix H has exact rank 14 and coordinate nullity 4.
- A primitive integer basis for the four vanishing anisotropy identities is:
  - eta1[0,1] + eta1[0,2] - eta1[1,1] = 0
  - eta1[0,0] - eta1[2,1] = 0
  - eta1[0,0] - eta1[1,0] + eta1[2,2] = 0
  - eta2[0,2] + eta2[2,1] = 0

### 5a. Projection Matrix M

- The exact AlphaTensor projection matrix M has shape 18 x 54, exact rank 10, and nullity 44.
- Its nonzero coefficient alphabet is exactly {+1, -1}.
- Delta-column support sizes range from 0 to 4, and eta-row incidence counts range from 0 to 9.
- Simple reshape tests do not show a trivial one-factor Kronecker collapse: the coarse block unfoldings have ranks 2 and 9 rather than 1.

### 5c. Pairwise Interaction Matrix

- All 506 ordered term pairs have pairwise anisotropy compression 0 and pairwise nuisance compression 0.
- In this exact pairwise sense, the AlphaTensor structure is not built from especially entangled two-term blocks; the compression appears only at larger collective scale.

### 5d. Standard 27-Term Baseline

- The standard algorithm verifies exactly with rank(H) = 18, rank(Delta) = 0, rank(Nuisance) = 18, rank(Sigma) = 9, quotient gain = 9.
- This gives the expected calibration R = 27 = 9 + 18, with no anisotropy nullspace and no dead-X correction layer.

## Interpretation

The AlphaTensor rank deficiency is not a fragile numerical accident: it is enforced by four exact linear identities among the 18 anisotropy coordinates. The trajectory data then shows that these dependencies accumulate gradually and that the final quotient gain of 9 only closes when the full 23-term set is present. In this sample, term order changes when the anisotropy rank saturates, but not the final nuisance budget. The projection analysis and zero pairwise-compression scan then sharpen the structural picture: the remarkable compression is genuinely collective, not explained by any simple 2-term building block or trivial factorization of M. The standard 27-term baseline remains the clean contrast class with full anisotropy rank 18 and no dead layer at all.