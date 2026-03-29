# Phase 6 Results

## 6a. Exact Bilinear Expansion

- Four exact null-space identities were expanded into 9x9 bilinear matrices `Q_1,...,Q_4` and exported in `Q_matrices.csv`.
- All 23 public AlphaTensor terms satisfy all four bilinear equations exactly.
- Each identity is non-tautological: a small integer witness `(a,b)` was found for every `Q_j` with `a^T Q_j b != 0`.

## 6b-6c. Variety Defined by the Four Identities

- Random exact sampling on `V_4 = {(a,b) : a^T Q_j b = 0, j=1..4}` gave constraint rank 4 for generic `a` and Jacobian rank 4 at all sampled points.
- The observed dimension is therefore `18 - 4 = 14`.
- Sampled `sigma` vectors from `V_4` still span dimension 9, so the four identities do not collapse the 9-dimensional signal block.

## 6d. Search for a Fifth Compatible Identity

- Direct null-space extension on every 22-term single-removal subset failed: the minimum observed `rank(H)` is 14, so no subset reaches the required `rank(H) <= 13` for an immediate `R=22` opening.
- Symmetry-guided transport also does not produce a fifth compatible identity for the fixed decomposition.
- The original 4-dimensional identity space is preserved only by the identity action, a single transformed copy can enlarge the span to dimension 8, and the full orbit spans dimension 18 in eta-space.

## 6e. Single-Term Removal Verdict

- All 23 single-removal subsets preserve `rank(H)=14`, `eta`-nullity 4, `rank([H|Delta])=14`, and quotient gain 8.
- Verdict: no single removed AlphaTensor term reveals a fifth exact identity.
