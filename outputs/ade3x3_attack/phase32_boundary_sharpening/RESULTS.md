# Phase 32 Results

Phase 32 sharpens the Phase 31 feasibility boundary with a denser rotation-cap grid and a slightly larger budget range.

Scanned rotation caps:

- `2.0, 2.5, 3.0, ..., 14.0`

Scanned budgets:

- `0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0`

The observable is unchanged:

- smallest per-step mode-rotation cap for which any regular descent exists.

## 32a. Sharpened Frontier

The boundary remains completely flat across the tested budget range, but the denser cap grid moves the AlphaTensor threshold lower.

For every tested budget:

- AlphaTensor: minimum feasible rotation cap = `2.0` degrees
- Standard algorithm: minimum feasible rotation cap = `12.0` degrees

So the sharpened gap is:

- `standard minus alpha = 10.0` degrees

This is stronger than Phase 31, which only established a gap of `8` degrees on the coarser grid.

## 32b. Budget Dependence

Although the threshold cap stays flat, the amount of motion at that threshold grows with budget.

At the frontier cap:

- AlphaTensor total gain rises from about `0.00299` at budget `0.02` to about `0.32746` at budget `3.0`.
- Standard total gain rises from about `0.01124` at budget `0.02` to about `0.76662` at budget `3.0`.

So the two families differ first in **feasibility threshold**, not in whether large enough budget eventually yields significant motion once the threshold is crossed.

## 32c. Interpretation

Phase 32 strengthens the continuation-side structural picture.

1. The feasibility-boundary separator is not a coarse-grid artifact.
2. The standard family still cannot descend regularly below a `12` degree cap anywhere on the scanned budget range.
3. AlphaTensor remains feasible already at the smallest scanned cap, `2` degrees, across the whole same range.

This makes the current empirical thesis sharper:

- the two families occupy different smooth-feasibility phases,
- and the relevant invariant now looks like a **critical rotation threshold** rather than a raw gain score.

So the next natural refinement is to probe whether AlphaTensor's true threshold lies below `2` degrees, and whether the standard threshold truly sits at `12` or merely in the interval `[11.5, 12.0]`.