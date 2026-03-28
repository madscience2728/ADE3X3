# Step 65: Polyomino Sub-Tensor Ranks + AlphaTensor Tiling Analysis
Generated: 2026-03-28T17:38:03

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

- L-tromino symmetry classes under S3xS3: 1
- L-tromino-only tilings of the 3x3 grid: 0
- L-tromino-only tiling classes under S3xS3: 0
- AlphaTensor support clusters: 20
- Minimum tiling upper-bound cost found: 26

## L-Tromino Rank Table

- L_tromino_class_01: entries=(0,0); (0,1); (1,0), LB=6, numerical=none, AlphaTensor UB=14, exact=undetermined

## L-Tromino Tilings

- No tilings by three L-trominoes exist.

## AlphaTensor Implicit Tiling

- cluster_01: 2 terms on (0,0) (monomino)
- cluster_02: 1 terms on (0,0); (0,1); (1,1) (L_tromino)
- cluster_03: 1 terms on (0,0); (2,0) (disconnected_size_2)
- cluster_04: 1 terms on (0,0); (2,0); (2,1); (2,2) (disconnected_size_4)
- cluster_05: 1 terms on (0,0); (2,1) (disconnected_size_2)
- cluster_06: 1 terms on (0,1); (0,2) (domino_row)
- cluster_07: 1 terms on (0,1); (0,2); (1,0); (1,1); (1,2) (connected_size_5)
- cluster_08: 1 terms on (0,1); (0,2); (1,1) (L_tromino)
- cluster_09: 1 terms on (0,1); (1,1); (2,1) (tromino_col)
- cluster_10: 1 terms on (0,2) (monomino)

## Best Tiling Upper Bound

- signature=1x domino_col + 3x monomino + 1x square_tetromino
- pieces=(0,0) | (0,1) | (0,2) | (1,0); (2,0) | (1,1); (1,2); (2,1); (2,2)
- total upper-bound cost=26