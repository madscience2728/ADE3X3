# Phase 20 Results

Phase 20 tested a specific codimension-one obstruction candidate suggested by the centered-lift derivation.

Let

- `X_0 = (P_0 + P_1 + P_2) / 3`
- `K_s = P_s - X_0`

so that always `K_0 + K_1 + K_2 = 0`.

If there were any second independent scalar relation among the three centered lifts, then after eliminating the forced sum-zero relation all three `K_s` would have to lie on a single matrix line. Equivalently, the flattened lift span would drop from rank `2` to rank `1`.

That gives a concrete candidate obstruction:

> valid multiplication decompositions should not lie in the affine-line lift locus `span{K_0, K_1, K_2}` of matrix rank `1`.

Phase 20 tested this in two ways.

## 20a. Affine-Line Lift Analysis On Exact Decompositions

For both known exact `3x3` multiplication decompositions, the centered lifts have span rank `2`, not `1`.

- AlphaTensor rank-23:
  - lift-span rank: `2`
  - best scalar-fit residuals between pairs of centered lifts:
    - `K_1 ≈ lambda K_0`: residual `1.0`
    - `K_2 ≈ lambda K_0`: residual `1.0`
    - `K_2 ≈ lambda K_1`: residual `0.6259541984732825`
- Standard rank-27:
  - lift-span rank: `2`
  - best scalar-fit residuals between any pair of centered lifts: `0.5`

Interpretation:

Neither exact decomposition is even close to the codimension-one affine-line defect. The forced relation `K_0 + K_1 + K_2 = 0` is present, but there is no second scalar relation collapsing the three lifts onto a single matrix direction.

## 20b. Factorized Affine-Line Family Search

We then searched directly inside a factorized family that forces the affine-line condition by construction.

The family was:

- `alpha_k[:,s] = a_s * u_k`
- `beta_k[s,:] = b_s * v_k^T`

with global channel scalars `a_s`, `b_s` and termwise vectors `u_k`, `v_k`.

This makes the live-channel matrices satisfy

- `P_s = (a_s b_s) * M`

for a common stacked matrix `M`, so the centered lift span is exactly `1`.

Results over random trials with `gamma` solved by least squares against exact multiplication:

- `R = 23`:
  - trials: `120`
  - best max-abs tensor residual: `0.7361786988441493`
  - best loss: `19.08981833812571`
  - best `rank(H) = 9`
  - best `rank([H | Delta]) = 9`
  - best lift-span rank: `1`
- `R = 27`:
  - trials: `120`
  - best max-abs tensor residual: `0.8063413133156774`
  - best loss: `19.661087559288475`
  - best `rank(H) = 9`
  - best `rank([H | Delta]) = 9`
  - best lift-span rank: `1`

Interpretation:

The affine-line family does exactly what the derivation predicts structurally: it forces a second scalar relation among the centered lifts and collapses `H` to rank at most `9`. But once that happens, the multiplication tensor fit becomes very poor. Even with least-squares-optimal `gamma`, the best residuals stay around `0.74` to `0.81`, far from an exact decomposition.

## Overall Interpretation

This phase gives a derived-and-tested obstruction candidate that survives direct compute.

1. The known exact decompositions do not sit on the affine-line lift defect.
2. A factorized family engineered to force that defect does not come close to exact multiplication.

So a second scalar relation among the centered lifts now looks genuinely incompatible with exact `3x3` multiplication. That does not yet prove the full conservation law, but it rules out a clean codimension-one failure mechanism and narrows the remaining bad locus further.