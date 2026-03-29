# Phase 21 Results

Phase 21 tested the next direct bad-locus candidate on `col(H)` itself.

The correct codimension-one defect is not merely the existence of some nonzero functional `lambda` with

- `lambda^T H = 0`.

That condition is too weak, because every proper subspace has many annihilators and such a `lambda` need not interact with `ker(Gamma)` at all. The defect relevant to kernel saturation is:

> there exists a nonzero `lambda in ker(Gamma)` such that `lambda^T H = 0`.

Equivalently, after restricting `H` to a basis of `ker(Gamma)`, the restricted matrix must lose rank.

## 21a. Diagnostics On Known Exact Decompositions

For each known exact decomposition, we formed a numeric basis `Q` of `ker(Gamma)` and examined the restricted matrix

- `Q^T H`.

Results:

- AlphaTensor rank-23:
  - `dim(ker Gamma) = 14`
  - `rank(H) = 14`
  - `rank(Q^T H) = 14`
  - smallest singular value of `Q^T H`: `0.6245923009701736`
  - functional defect dimension: `0`
- Standard rank-27:
  - `dim(ker Gamma) = 18`
  - `rank(H) = 18`
  - `rank(Q^T H) = 18`
  - smallest singular value of `Q^T H`: `1.0`
  - functional defect dimension: `0`

Interpretation:

Neither known exact decomposition has even a near-annihilator inside `ker(Gamma)`. The restriction of `H` to `ker(Gamma)` is full rank in both cases and is quantitatively well-conditioned relative to the numeric scale of the data.

## 21b. Weighted Annihilator Family Search

We then built a factorized family that enforces an exact shared annihilator of `H` at the level of the live channels.

Starting from fixed factor matrices `(alpha, beta)`, we chose a random term-space weight vector `lambda`, then solved for channel-wise term rescalings so that

- `lambda^T P_0 = lambda^T P_1 = lambda^T P_2`

exactly up to numerical projection tolerance.

After that we solved the best least-squares `gamma` against exact multiplication and measured whether the resulting `lambda` was actually close to `ker(Gamma)`.

Best observed outcomes:

- AlphaTensor-derived family:
  - best max-abs tensor residual: `0.6680211384226139`
  - best loss: `3.878045707158768`
  - best `rank(H) = 14`
  - best functional defect dimension in `ker(Gamma)`: `0`
  - best distance from enforced `lambda` to `ker(Gamma)`: `1.1927905953557112`
- Standard-derived family:
  - best max-abs tensor residual: `2.220446049250313e-16`
  - best loss: `9.860761315262648e-32`
  - best `rank(H) = 18`
  - best functional defect dimension in `ker(Gamma)`: `0`
  - best distance from enforced `lambda` to `ker(Gamma)`: `1.4142135623730898`

Interpretation:

This is the crucial distinction.

The weighted-annihilator construction can enforce `lambda^T H = 0` exactly, but the resulting `lambda` does not approach `ker(Gamma)`. In the standard-derived family, exact multiplication can still survive because the enforced annihilator lives in the wrong place in term space. It is not a kernel annihilator.

So the naïve “shared annihilator of `H`” condition is not the right obstruction. The real obstruction candidate must enforce a functional annihilator that is itself tied to `ker(Gamma)`.

## Overall Interpretation

Phase 21 rules out a tempting but incorrect simplification.

1. A generic annihilator of `H` is too weak to matter.
2. The relevant defect is specifically a nonzero element of `ker(Gamma)` annihilating `H`.
3. The known exact decompositions show no such defect.
4. A factorized family can force `lambda^T H = 0` without producing any kernel defect at all.

So the next compute step has to couple the annihilator directly to `ker(Gamma)` rather than treating the two conditions separately.