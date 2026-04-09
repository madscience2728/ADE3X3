# Induced Chart Obstruction

This note resolves a key ambiguity about the proposed "induced-equivariant" chart.

## Claim Being Tested

One might try to replace the stabilizer-fixed seed chart by

`phi(A,B,C) = sum_{g in G / H} g . (A,B,C)`

with a completely free seed `(A,B,C) in Mat_3^3`, where `H = Stab(seed_support)`.

The hope is that this gives a larger orbit chart than the `H`-fixed seed space.

## Why This Fails For One-Seed-Per-Orbit Transport

This formula is not well-defined unless the seed is `H`-fixed.

Reason:

- `G/H` is a set of left cosets, so to compute the sum one must choose one representative `t` from each coset.
- If `h in H`, then `t` and `t h` represent the same coset.
- Replacing the representative `t` by `t h` changes the contribution from `t.(A,B,C)` to `t.(h.(A,B,C))`.
- Therefore the orbit sum is independent of representative choice if and only if

  `h.(A,B,C) = (A,B,C)` for all `h in H`.

That is exactly the stabilizer-fixed condition.

## Equivalent Reynolds-Operator Statement

The only canonical group-theoretic construction from a free seed is the full group average

`Phi(A,B,C) = sum_{g in G} g.(A,B,C)`.

Writing `G = ⨆_{t in T} tH` for a set of coset representatives `T`,

`Phi(A,B,C) = sum_{t in T} t.( sum_{h in H} h.(A,B,C) )`.

So `Phi` factors through the Reynolds projection onto the `H`-fixed subspace:

`P_H(A,B,C) = (1 / |H|) sum_{h in H} h.(A,B,C)`.

Therefore, for the one-seed-per-orbit construction, only the `H`-fixed part of the seed contributes canonically.

## Numerical Check

Using the repo's exact factor action from `scripts/poly_rootfind.py`, an edge seed `(0,0,1)` was tested with a deliberately non-`H`-fixed triple `(A,B,C)`.

Observed:

- changing just one chosen coset representative by right multiplication with a stabilizer element changed the resulting orbit-sum tensor by Frobenius norm about `1477.96`;
- replacing the seed by its `H`-average produced a different tensor, and that averaged seed was exactly `H`-fixed.

So the free-seed coset-sum is not a well-defined chart on factor space.

## Consequence

The stabilizer-fixed seed spaces in `FULL_STABILIZER_SEED_SPACES.md` are not an accidental overconstraint. They are the canonical seed coordinates for the current one-seed-per-orbit transport model.

For that model, the seed-space dimensions are therefore exactly:

- orbit size `1`: `3`
- orbit size `6`: `9`
- orbit size `12`: `12`
- orbit size `8`: `6`

## Real Next Step If A Larger Family Exists

If the true symmetric sector is larger, the enlargement cannot come from a single free seed modulo `H`.

Instead it must come from a genuinely larger equivariant construction, for example:

- multiple seed copies per orbit type;
- seed values transforming in a nontrivial `H`-representation rather than the trivial one;
- an induced representation on labeled orbit fibers, followed by an explicit quotient back to unlabeled CP terms.

That is the real missing derivation.

So the next mathematical task is not:

- "compute `ker(L)` for a linear map `L : R^27 -> R^729` on free seed factors".

It is:

- "construct the correct larger equivariant factor bundle, identify its trivial-isotypic contribution to the unlabeled orbit-sum tensor, and then derive coordinates for that bundle explicitly".