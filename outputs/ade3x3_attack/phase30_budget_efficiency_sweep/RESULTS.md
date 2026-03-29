# Phase 30 Results

Phase 30 follows the two empirical directions suggested after Phase 29:

1. export explicit **gain-per-budget** and **gain-per-rotation** quantities,
2. run a small **policy sweep** over rotation caps and budget weights.

The goal is to test whether the Phase 29 budget-aware story is robust, or whether it was just an artifact of one specific policy choice.

## 30a. Policy Grid

The sweep used five policies:

- `rot8, budget_weight 0.0, max_steps 16`
- `rot12, budget_weight 0.0, max_steps 16`
- `rot12, budget_weight 0.3, max_steps 16`
- `rot12, budget_weight 0.6, max_steps 24`
- `rot16, budget_weight 0.6, max_steps 24`

For each policy and each family, the script recorded:

- total gain `initial_kappa_ker - final_kappa_ker`,
- total gain per unit continuation budget,
- total gain per degree of cumulative mode rotation,
- wildcard-win counts,
- and the accepted trajectory.

## 30b. The First Robust Asymmetry: A Hard Rotation Wall

The strictest rotation-cap policy produced the first genuinely new asymmetric event in this continuation branch.

Under `rotation_cap = 8` degrees and no budget penalty:

- AlphaTensor accepts `7` steps and moves from `0.6245923009701738` to `0.44797689207527414`.
- The standard algorithm accepts `0` steps and remains at `1.0`.

So the standard family hits a wall immediately under that stricter smoothness requirement, while AlphaTensor still admits exact, no-collapse descent.

This is the strongest separator seen in the continuation program since the move to the honest coupled quantity.

## 30c. Naive Efficiency Ratios Are Not Separators

The original budget-aware hope does **not** survive the sweep in the naive form “gain per budget.”

For the looser policies where both families move, the standard algorithm is actually more efficient by that metric.

Examples:

- `rot12, budget_weight 0.0`:
  - AlphaTensor gain per budget `0.1510256561436216`
  - Standard gain per budget `0.38722979689553566`
- `rot12, budget_weight 0.3`:
  - AlphaTensor gain per budget `0.18913387584597885`
  - Standard gain per budget `0.4877662518759931`
- `rot12, budget_weight 0.6`:
  - AlphaTensor gain per budget `0.1938637680810227`
  - Standard gain per budget `0.6372202781608317`

The same basic reversal holds for gain per cumulative mode rotation.

So a simple efficiency quotient is not the right invariant by itself.

## 30d. Interpretation

Phase 30 resolves one uncertainty and creates another.

1. The Phase 29 budget narrative was not robust in its naive quantitative form. If the proposed invariant is just “defect reduction per unit budget,” the standard family looks **better**, not worse.
2. But the sweep also revealed a sharper qualitative effect: the standard continuation is much more sensitive to a tight rotation cap. At `8` degrees it stalls completely, while AlphaTensor still moves.
3. Therefore the next plausible separator is not a scalar budget-efficiency ratio alone. It is a **feasibility boundary** in the joint space of
   - allowed mode rotation,
   - allowed continuation cost,
   - and achievable defect decrease.

So the next compute target should be a phase diagram, not a single score.

The concrete question is now:

- for each family, what is the smallest rotation cap under which any regular descent is still feasible?

That is more promising than asking which family has the largest raw gain-per-budget ratio.