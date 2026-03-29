# Phase 17 Results

Phase 17 split the conservation-law story into two distinct pieces:

- Part 1 is now proved: for any minimum-rank decomposition, the stacked fiber-mode block `[Sigma | H | Delta]` has rank `R`.
- Part 2 remains open: to deduce the exact `3x3` conservation law `R + eta_nullity = 27`, one still needs the universal containment statement `Delta subset span(H)`.

The three Phase 17 scripts were run successfully and their outputs agree on that split.

## 17a. Fiber-Mode Faithfulness

- Valid decompositions tested: 2.
- Both valid decompositions were faithful numerically and exactly.
- Both valid decompositions satisfied the corollary `dim(ker Gamma) = rank([H | Delta])` numerically and exactly.

Measured valid cases:

- AlphaTensor public rank-23 decomposition:
  - `rank(Sigma) = 9`
  - `rank(H) = 14`
  - `rank([H | Delta]) = 14`
  - `rank([Sigma | H | Delta]) = 23`
  - `eta_nullity = 4`
  - `R + eta_nullity = 27`
- Standard rank-27 decomposition:
  - `rank(Sigma) = 9`
  - `rank(H) = 18`
  - `rank([H | Delta]) = 18`
  - `rank([Sigma | H | Delta]) = 27`
  - `eta_nullity = 0`
  - `R + eta_nullity = 27`

Additional robustness checks:

- Perturbed AlphaTensor trials: `20 / 20` faithful.
- Random generic rank-1 collections: `25 / 25` faithful across `R in {9, 14, 18, 23, 27}`.

Interpretation:

The faithfulness statement is not just an artifact of the two known exact decompositions. It survived all perturbation and generic-sample checks performed here. This supports the proof text used in the script: the Step 51 fiber-mode map is an invertible linear change of coordinates on the `81` bilinear monomials, so a dependence in `[Sigma | H | Delta]` would induce a dependence among the bilinear rank-1 profiles and contradict minimum rank.

## 17b. Symbolic Single-Term Delta/Eta Analysis

- Live-product recovery from `(sigma, eta1, eta2)` was verified exactly.
- The recovered formulas are:
  - `p0 = (sigma + 2*eta1 + eta2) / 3`
  - `p1 = (sigma - eta1 + eta2) / 3`
  - `p2 = (sigma - eta1 - 2*eta2) / 3`
- Cross-ratio identity counts:
  - left identities: `324` total, `162` independent
  - right identities: `324` total, `162` independent
  - combined independent identities: `324`

Interpretation:

At the single-term level, the dead slices are not free variables. They are constrained by a large exact family of multiplicative compatibility identities tying them to the live slices. But the gamma-propagation analysis also showed that this is not a direct linear implication: `Gamma * Delta = 0` does not follow linearly from `Gamma * Eta = 0`. The obstruction is structural. Dead slices use cross-channel outer products `a[:,s] * b[t,:]^T` with `s != t`, so the control is rational/cross-ratio in nature rather than linear.

## 17c. Collective Delta Containment

### AlphaTensor inheritance

- The exact projection `Delta = H * M` from Phase 1 was re-verified on the public AlphaTensor rank-23 decomposition.
- Tested delta columns: `54`.
- Tested output rows: `9`.
- Weighted projection max residual: `0.0`.
- Nonzero gamma-weighted columns after projection: `0`.

### Single deletions of AlphaTensor

- Terms removed one at a time: `23` cases.
- Exact containment count: `23 / 23`.
- Numeric containment count: `23 / 23`.
- Failing terms: none.

### Off-manifold negative evidence

- Exported Step 75 anticommutator rank-19 fibermode:
  - `rank(H) = 17`
  - `rank(Delta) = 19`
  - `rank([H | Delta]) = 19`
  - `Delta subset span(H)`: false
- Random rank-23 generic collections:
  - trials: `1000`
  - containment count: `0`
  - containment rate: `0.0`
  - `rank(H) = 18` throughout
  - `rank([H | Delta]) = 23` throughout

Interpretation:

`Delta subset span(H)` is therefore very far from generic. It holds exactly on AlphaTensor and stays exact across all `23` single-term deletions, but it fails on generic random rank-23 collections and also fails on the exported Step 75 anticommutator fibermode data. That is the cleanest current evidence that Part 2 is a special matrix-multiplication phenomenon, not a generic tensor fact.

## Overall Interpretation

Phase 17 establishes the logical shape of the conservation law.

- Proved now: the fiber-mode decomposition is faithful, so every minimum-rank decomposition satisfies `rank([Sigma | H | Delta]) = R`.
- Verified on the known valid `3x3` algorithms: `dim(ker Gamma) = rank([H | Delta])` and `R + eta_nullity = 27`.
- Still open: a universal proof that `Delta subset span(H)` for all minimum-rank `3x3` matrix-multiplication decompositions.

So the conservation law is no longer a single unstructured conjecture. Its linear-algebra backbone is now proved; only the genuinely matrix-multiplication-specific Delta-containment step remains open.