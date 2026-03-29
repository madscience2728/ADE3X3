# Phase 19 Results

Phase 19 carried out both of the next compute tasks:

- a factorization-aware diagnostic on the actual valid decompositions, using the centered right inverse `X_0 = (P_0 + P_1 + P_2) / 3` and the induced kernel lifts `K_s = P_s - X_0`;
- a constrained search over explicit defect families in factorized `(alpha, beta)` space, with `gamma` solved by least squares against the exact multiplication tensor.

## 19a. Factorization-Aware Diagnostics On Known Decompositions

For both the AlphaTensor rank-23 decomposition and the standard rank-27 decomposition, the centered right-inverse picture behaves exactly as the theory predicts.

- AlphaTensor rank-23:
  - `rank(Gamma) = 9`
  - `dim(ker Gamma) = 14`
  - `rank(H) = 14`
  - defect dimension `dim(ker Gamma) - rank(H) = 0`
  - `Gamma * X_0 = I_9` exactly
  - each centered lift `K_s` lies in `ker(Gamma)` exactly
  - `K_0 + K_1 + K_2 = 0` exactly
  - the three centered lifts span a `2`-dimensional affine channel sector, which is the expected maximum once their sum is forced to vanish
- Standard rank-27:
  - `rank(Gamma) = 9`
  - `dim(ker Gamma) = 18`
  - `rank(H) = 18`
  - defect dimension `0`
  - the same exact centered-lift identities hold
  - again the lift-span rank is `2`

Interpretation:

The actual valid decompositions already sit in the most economical centered-right-inverse picture: the three lifts are constrained only by `K_0 + K_1 + K_2 = 0`, not by any extra linear defect in `ker(Gamma)`.

## 19b. Constrained Defect-Family Search

We then took the exact AlphaTensor and standard factors `(alpha, beta)`, projected them onto explicit defect loci, solved the best least-squares `gamma`, and measured the resulting multiplication residual.

The strongest families were:

- `pair_equal_01`: enforce `P_0 = P_1`
- `pair_equal_12`: enforce `P_1 = P_2`
- `all_equal`: enforce `P_0 = P_1 = P_2`
- `collinear`: enforce each term's three live products to remain channel-collinear

Best observed outcomes:

- AlphaTensor base:
  - `pair_equal_01`: best max-abs tensor residual `0.5`, best `rank(H) = 9`
  - `pair_equal_12`: best max-abs tensor residual `0.5`, best `rank(H) = 9`
  - `all_equal`: best max-abs tensor residual `2/3`, best `rank(H) = 0`
  - `collinear`: best max-abs tensor residual `0.7420931243330606`, best `rank(H) = 18`
- Standard base:
  - `pair_equal_01`: best max-abs tensor residual `0.5`, best `rank(H) = 9`
  - `pair_equal_12`: best max-abs tensor residual `0.5`, best `rank(H) = 9`
  - `all_equal`: best max-abs tensor residual `2/3`, best `rank(H) = 0`
  - `collinear`: exact residual `0.0`, exact loss `0.0`, best `rank(H) = 18`

## Interpretation

The compute split is informative.

1. Pair-equality defects are strongly incompatible with exact multiplication in both the AlphaTensor-derived and standard-derived searches. Forcing `P_0 = P_1` or `P_1 = P_2` immediately collapses `rank(H)` to `9` and leaves a hard residual floor of `0.5`.
2. The fully equal case is even more rigidly impossible: `rank(H) = 0` and the best residual floor is `2/3`.
3. But the broader “channel-collinear” condition is not an obstruction. The standard algorithm remains exact inside that family.

So the next obstruction search must be sharpened. It is not enough to prove that the three live-channel profiles are linearly related in some broad sense. The evidence now points to a narrower target: rule out pair-equality-type or similar low-codimension coincidence loci for the factorized lifts `(K_0, K_1, K_2)`, while avoiding overbroad conditions like per-term channel collinearity that already contain valid multiplication algorithms.