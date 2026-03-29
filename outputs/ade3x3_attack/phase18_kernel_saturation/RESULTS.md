# Phase 18 Results

Phase 18 turned the remaining conservation-law gap into a directly testable linear-algebra experiment around the per-channel right-inverse equations

- `Gamma * P_0 = I_9`
- `Gamma * P_1 = I_9`
- `Gamma * P_2 = I_9`

and the difference-sector identity

- `H = [P_0 - P_1 | P_1 - P_2]`.

The point of the phase was to test the reformulation

> `rank(H) = dim(ker Gamma)`

on exact known decompositions, on synthetic random triples inside the affine right-inverse model, and on explicit defective constructions that still satisfy the channel identities but are forced onto lower-dimensional loci.

## 18a. Known Exact Decompositions

Both known valid matrix-multiplication decompositions satisfy the per-channel identities exactly and saturate the kernel exactly.

- AlphaTensor public rank-23 decomposition:
  - `rank(Gamma) = 9`
  - `dim(ker Gamma) = 14`
  - `rank(H) = 14`
  - `rank(Q^T H) = 14` for a numeric kernel basis `Q`
  - `Gamma * P_s = I_9` exactly for `s = 0,1,2`
  - `Gamma * H = 0` exactly
- Standard rank-27 decomposition:
  - `rank(Gamma) = 9`
  - `dim(ker Gamma) = 18`
  - `rank(H) = 18`
  - `rank(Q^T H) = 18`
  - `Gamma * P_s = I_9` exactly for `s = 0,1,2`
  - `Gamma * H = 0` exactly

So the route-(b) target is not only consistent with the known exact decompositions; it is exactly realized by them in both numeric and exact arithmetic.

## 18b. Synthetic Right-Inverse Genericity Test

For each known `Gamma`, we fixed one right inverse `X_0` and sampled random kernel-valued lifts

- `P_0 = X_0 + K_0`
- `P_1 = X_0 + K_1`
- `P_2 = X_0 + K_2`

with each `K_s` having columns in `ker(Gamma)`.

Results:

- AlphaTensor `Gamma`:
  - trials: `250`
  - `dim(ker Gamma) = 14`
  - saturation count: `250 / 250`
  - observed `rank(H)` range: `14..14`
- Standard `Gamma`:
  - trials: `250`
  - `dim(ker Gamma) = 18`
  - saturation count: `250 / 250`
  - observed `rank(H)` range: `18..18`

Interpretation:

Inside the affine right-inverse model, full kernel saturation is overwhelmingly stable. In this sample it was not merely typical; it occurred in every trial. That does not prove the matrix-multiplication theorem, because random right inverses need not come from factorized rank-1 terms, but it gives direct computational support for the claim that failure of `rank(H) = dim(ker Gamma)` should lie on a special lower-dimensional locus.

## 18c. Explicit Defective Families

To show that the right-inverse model still admits failures on special loci, we built synthetic triples with forced relations among the kernel lifts.

For AlphaTensor `Gamma`:

- `K_0 = K_1 = K_2` gives `rank(H) = 0`, deficiency `14`
- `K_0 = K_1` gives `rank(H) = 9`, deficiency `5`
- `K_0, K_1, K_2` collinear in the same kernel-generated 9-column family gives `rank(H) = 9`, deficiency `5`

For standard `Gamma`:

- `K_0 = K_1 = K_2` gives `rank(H) = 0`, deficiency `18`
- `K_0 = K_1` gives `rank(H) = 9`, deficiency `9`
- collinear kernel-lift families again give `rank(H) = 9`, deficiency `9`

All of these defective examples still satisfy the channel identities `Gamma * P_s = I_9` up to floating-point roundoff and still satisfy `Gamma * H = 0`. So the failure mechanism is not violation of the affine right-inverse equations themselves. The failure comes from special algebraic coincidences among the kernel lifts.

## Interpretation

This is the first compute phase aimed directly at the remaining theorem target rather than at the older `Delta subset span(H)` formulation.

- The exact known decompositions saturate `ker(Gamma)` through `H`.
- Random triples inside the affine right-inverse model saturate `ker(Gamma)` in every trial.
- Failures can still be manufactured, but only by imposing special relations such as `K_0 = K_1` or full collinearity among the kernel lifts.

That is exactly the shape the theory predicted. The remaining proof problem is now sharper:

> show that a valid minimum-rank matrix-multiplication decomposition cannot land on one of those defective kernel-lift loci.

So the next compute target should not be more generic sampling. It should be factorization-aware diagnostics that ask which polynomial relations among `(K_0, K_1, K_2)` are compatible with coming from actual rank-1 terms `(alpha_k, beta_k, gamma_k)`.