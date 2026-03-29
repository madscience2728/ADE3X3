# Phase 2: Anisotropy Fourier/Parity Check

## Verdicts

- The linear Fourier transform identities between `(eta1, eta2)` and `(xi+, xi-)`: YES.
- For real factors, `xi- = conjugate(xi+)`: YES.
- Proposed universal theorem `rank(H) = 2 * rank_C(Xi+)` and hence `rank(H)` always even: NO.
- AlphaTensor public rank-23 is consistent with the transform identities: YES.
- AlphaTensor public rank-23 supports the proposed parity theorem: NO.

## Exact Fiberwise Transform

For each output fiber `(r,u)`, with

- `lambda0 = a[r,0] b[0,u]`
- `lambda1 = a[r,1] b[1,u]`
- `lambda2 = a[r,2] b[2,u]`

and `omega = -1/2 + (sqrt(3)/2)i`, the exact relations are

- `xi+ = lambda0 + omega lambda1 + omega^2 lambda2 = eta1 - omega^2 eta2 = eta1 + eta2/2 + (sqrt(3)/2)i eta2`
- `xi- = lambda0 + omega^2 lambda1 + omega lambda2 = eta1 - omega eta2 = eta1 + eta2/2 - (sqrt(3)/2)i eta2`
- `eta2 = (xi+ - xi-) / (i sqrt(3))`
- `eta1 = (xi+ + xi-) / 2 + (xi+ - xi-) / (2 i sqrt(3))`

The script verifies these identities symbolically and on every AlphaTensor term/fiber pair.

## Real-Factor Conjugacy

If the factors are real, then `lambda0`, `lambda1`, `lambda2` are real and `conjugate(omega) = omega^2`, so

`conjugate(xi+) = lambda0 + omega^2 lambda1 + omega lambda2 = xi-`.

That part of the proposed theorem is correct.

## Why The Even-Rank Theorem Fails

The incorrect step is the jump from conjugacy to

`rank_R(H) = 2 * rank_C(Xi+)`.

What is always true is that `H` and the realification of `Xi+` carry the same real-linear information. But the realification of a complex column space need not have even dimension.

The simplest counterexample is a single dead-free term with active summation index `s*=0`:

- `eta1 = 1`
- `eta2 = 0`
- `xi+ = 1`
- `xi- = 1`

So `rank(H) = 1`, which is odd.

## AlphaTensor Check

For the public rank-23 decomposition:

- `rank(H) = 14`
- `rank_C(Xi+) = 9`
- `rank_R(realification(Xi+)) = 14`

So the AlphaTensor data itself disproves the proposed equality:

`14 != 2 * 9`.

The observed even value `14` is therefore incidental for this decomposition, not a universal parity law.

## Consequence For The Nuisance Budget

There is no valid parity tightening of the Step 52 nuisance budget from this argument. The per-algorithm Step 52 caps remain unchanged.

| R | Step 52 max nuisance rank | claimed even cap if theorem held | usable as theorem |
|---|---------------------------|----------------------------------|-------------------|
| 23 | 14 | 14 | False |
| 22 | 13 | 12 | False |
| 21 | 12 | 12 | False |
| 20 | 11 | 10 | False |
| 19 | 10 | 10 | False |

## Combined Phase 1 + Phase 2 Status

Phase 1 gives a strong positive result for the public AlphaTensor decomposition:

- `Delta` is contained in `span(H)` at the collection level.

But Phase 2 blocks the proposed universal follow-up theorem:

- there is no general even-rank law for `H`,
- so there is no justified parity tightening from `R=22 -> nuisance <= 12` on this route alone.

## Generated Files

- `parity_proof.py`
- `fourier_transform_formulas.csv`
- `alpha_tensor_fourier_verification.csv`
- `budget_table.csv`
- `parity_summary.json`