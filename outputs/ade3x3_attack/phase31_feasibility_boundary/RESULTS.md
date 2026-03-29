# Phase 31 Results

Phase 31 replaces scalar continuation scores by a direct feasibility map.

The question is:

- for a given continuation budget and per-step mode-rotation cap, does any regular descent remain feasible?

The scan keeps the wildcard branch enabled and measures, for each budget level, the **smallest rotation cap** at which regular descent exists.

## 31a. Main Result

Across the full scanned budget range

- `0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0`

the frontier is stable:

- AlphaTensor: minimum feasible rotation cap = `4` degrees
- Standard algorithm: minimum feasible rotation cap = `12` degrees

So the rotation-cap gap is

- `standard minus alpha = 8` degrees

for every tested budget.

## 31b. Interpretation

This is stronger than the isolated Phase 30 observation.

Phase 30 showed that under one strict policy (`8` degree cap), the standard algorithm stalled while AlphaTensor still moved. Phase 31 shows that this is not an accidental single-point effect. It persists across a whole slice of the budget axis.

In the scanned region, the two families have different feasibility thresholds:

- AlphaTensor remains compatible with very tight smoothness,
- the standard algorithm requires a much looser rotation allowance before any regular descent is possible.

## 31c. Consequence For The Program

This means the promising invariant is no longer a single efficiency number.

The useful object is a **feasibility-boundary profile**:

- smallest admissible rotation cap as a function of budget,
- or equivalently the regular-descent phase diagram in `(rotation cap, budget)` space.

At least on the current grid, that profile separates AlphaTensor from the standard algorithm cleanly.

So the next natural refinement is not another continuation heuristic. It is to sharpen this boundary:

- densify the rotation-cap grid near `4` and `12`,
- check whether the boundary remains flat under larger step budgets or finer cap resolution,
- and test whether the gap can be connected back to an exact structural statement about `H`, `Delta`, or the coupled kernel geometry.