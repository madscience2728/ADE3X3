# Step 73: Depth-2 Bilinear Theorem
Generated: 2026-03-28T21:28:53

[EXACT_DERIVED] + [INTERPRETATION] + [EXTERNAL_SOURCE]

## Task 1a / 1b: AA and QQ Sector Audit

The Step 72 parameterization already separates the coefficient sectors into bilinear, A^2B, AB^2, and A^2B^2 slots. Step 73 uses that same parameterization and checks the two families that mattered for the proposed depth-2 follow-up: pure AA second layers and pure QQ second layers.

| family | config | expected nonzero sectors | bilinear norm | A^2B norm | AB^2 norm | A^2B^2 norm | second-layer bilinear norm | reconstruction residual |
|--------|--------|--------------------------|---------------|------------|------------|--------------|----------------------------|-------------------------|
| AA_only | R22_r120_r22_aa2_bb0_qq0 | bilinear + A^2B | 115.043758813732 | 463.647715543417 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 1.375042478483e-13 |
| QQ_only | R22_r120_r22_aa0_bb0_qq2 | bilinear + A^2B^2 | 120.167035677964 | 0.000000000000 | 0.000000000000 | 7667.131437586283 | 0.000000000000 | 1.071255111195e-12 |

AA-only therefore contributes only bilinear + A^2B, and QQ-only contributes only bilinear + A^2B^2. In both cases the useful degree-2 part comes entirely from Layer 1.

## Exact-Derived Theorem Status

- aa_layer2_zero_bilinear_projection: In the Step 72 AA-only family, the second layer contributes only A^2B coefficients; its bilinear projection is identically zero. Evidence: Sample R22_r120_r22_aa2_bb0_qq0 has second-layer bilinear norm 0.000000000000 and unexpected-sector max abs 0.000000000000.
- qq_layer2_zero_bilinear_projection: In the Step 72 QQ-only family, the second layer contributes only A^2B^2 coefficients; its bilinear projection is identically zero. Evidence: Sample R22_r120_r22_aa0_bb0_qq2 has second-layer bilinear norm 0.000000000000 and unexpected-sector max abs 0.000000000000.
- division_free_depth_does_not_improve_bilinear_complexity: For a bilinear target and a division-free exact circuit, taking homogeneous degree-2 parts yields a bilinear circuit with no more multiplication gates. Exact division-free depth therefore does not lower bilinear complexity. Evidence: Higher-degree AA/QQ branches vanish in the bilinear projection, so any exact success would already be witnessed by the degree-2 truncation. Step 73 records the standard homogeneous-components argument explicitly and cites Bürgisser-Clausen-Shokrollahi as the external reference point.
- step72_positive_fibers_are_parameter_redundancy: Step 72 positive local fiber dimensions should be read as gauge or parameterization redundancy in the chosen depth-2 ansatz, not as evidence that AA/QQ branches create new exact bilinear directions. Evidence: Step 72 measured max local fiber 43 at R22_r120_r22_aa2_bb0_qq0, but Step 73 shows the AA and QQ second-layer branches have zero useful bilinear projection.

The exact consequence is immediate: if an AA/QQ-restricted depth-2 circuit computed 3x3 matrix multiplication exactly with R1 < 23, then its degree-2 truncation would already give a rank-R1 exact bilinear decomposition of the matrix-multiplication tensor. Step 73 therefore does not support a separate nonlinear AA/QQ solve below the exact rank wall; it collapses the question back to the ordinary bilinear-rank problem.

## Step 72 Reinterpretation

Step 72 measured a maximum local fiber dimension of 43 at R22_r120_r22_aa2_bb0_qq0. Step 73 reinterprets that number correctly: it is a gauge or parameterization redundancy signal inside the chosen depth-2 ansatz, not evidence that the AA or QQ branches open new useful bilinear directions.

## Border-Rank Literature Pass

| topic | source | claim |
|-------|--------|-------|
| exact_depth_theorem_reference | [Bürgisser, Clausen, Shokrollahi: Algebraic Complexity Theory](https://link.springer.com/book/10.1007/978-3-662-03338-8) | Standard reference for homogeneous-components and bilinear-complexity arguments used to justify degree-2 truncation for exact division-free circuits. |
| border_rank_lower_bound | [Landsberg-Ottaviani (2015), New Lower Bounds for the Border Rank of Matrix Multiplication](https://theoryofcomputing.org/articles/v011a011/) | For n x n matrix multiplication, border rank is at least 2n^2 - n; for n = 3 this gives underline(R)(<3,3,3>) >= 15. |
| exact_rank_status_online | [MathOverflow discussion: best known lower and upper bounds for matrix multiplication tensor rank of 3x3 matrices](https://mathoverflow.net/questions/383956/what-are-the-best-known-lower-and-upper-bounds-for-the-rank-of-the-matrix-multip) | The online summary still reports exact-rank bounds 19 <= R(<3,3,3>) <= 23 and points to Schonhage-style approximate constructions. |
| border_rank_escape_route | Step 73 literature synthesis | The only external escape route surfaced in this pass is border rank plus explicit correction terms. No explicit 3x3 border-rank witness with coefficients was recovered online here, so no correction computation was attempted. |

The clean sourced lower bound recovered in this pass is underlineR(<3,3,3>) >= 15. The online exact-rank status still reads 19 <= R(<3,3,3>) <= 23. No explicit 3x3 border-rank witness with coefficients was harvested in this pass, so the border-rank-plus-correction route remains only a literature pointer here, not a computable next artifact.