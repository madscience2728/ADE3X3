# Phase 9 Results

## 9a. Generic Variety Dimension

- Random bilinear systems were sampled for `k=1,...,9`.
- Observed median dimensions: `k=4 -> 14.0`, `k=5 -> 13.0`, `k=6 -> 12.0`, `k=9 -> None`.
- Sampled sigma-image ranks stayed at `k=4 -> 9.0`, `k=5 -> 9.0`, `k=6 -> 9.0`, `k=9 -> None`.

## 9b. Sigma-Span on Random V5

- Trial count: 100.
- `rank(Sigma)=9` success rate from 22 sampled points: 1.000.
- Rank histogram: {'9': 100}.

## 9c. Tensor Feasibility on Random V5

- Trial count: 100.
- Exact/near-exact tensor-feasible random V5 count: 0.
- Best random V5 projection loss: 18.333884424173043.
- Best random V5 max-abs residual: 0.898833335862747.

## 9d. Outer Random Search Proxy

- A random search over 5-identity subspaces was run for 64 outer trials.
- Best observed bilevel proxy loss: 18.26324631761861.
- Best observed proxy max-abs residual: 0.9075379843352899.

## Phase 9 Verdict

- Generic 5-identity varieties remain geometrically large enough to support full 9-dimensional sigma span.
- Random V5 choices almost never satisfy the tensor constraint exactly, so the obstruction is not raw variety dimension but the special placement of the identity subspace relative to the tensor.
- The outer search problem is therefore highly structured: finding a useful 5-identity space is a rare-position problem, not a dimension-count problem.
