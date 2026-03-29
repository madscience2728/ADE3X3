# Phase 26 Results

Phase 26 applies the first natural regularity filter to the Phase 25 paths.

The goal is simple:

- keep only path states that remain numerically exact,
- keep only path states whose kernel-restricted rank does **not** collapse,
- then ask how low `kappa_ker` can go under that no-collapse filter.

In this phase, the regularity condition was:

- tensor max-abs residual `<= 1e-10`,
- `restricted_rank` equal to its base value for that path.

So this explicitly removes the standard softest-path endpoint collapse seen in Phase 25.

## 26a. Softest Paths Under The No-Collapse Filter

Results:

- AlphaTensor, softest path:
  - base `restricted_rank = 14`
  - all 11 sampled states remain regular
  - minimum regular `kappa_ker = 0.004687401734031592` at `t = 1.0`
- Standard, softest path:
  - base `restricted_rank = 18`
  - only 10 sampled states remain regular, because the endpoint `t = 1.0` collapses
  - minimum regular `kappa_ker = 0.09999999999999981` at `t = 0.9`

Interpretation:

This removes the most obvious standard loophole from Phase 25.

Once the singular endpoint collapse is excluded, the standard softest path no longer reaches an honest defect. It bottoms out at `0.1`, while the AlphaTensor softest path continues smoothly down to about `4.69e-3`.

So on the softest path, the regularity filter produces a real separation.

## 26b. Remaining Standard Escape Routes

The same filter does **not** produce a global separator across all sampled directions.

Filtered minima:

- AlphaTensor random directions:
  - `0.045015644556162006`
  - `0.06486190587044158`
- Standard random directions:
  - `0.08198406982723286`
  - `0.005706537391332286`

Interpretation:

One standard random corrected path still reaches about `5.71e-3` while staying regular under the present filter. So the no-collapse condition is necessary, but not sufficient, to isolate a universal obstruction.

## Overall Interpretation

Phase 26 sharpens the target in a useful way.

1. Excluding singular endpoint collapse materially changes the standard softest-path story.
2. Under that filter, the AlphaTensor softest path approaches the honest kernel-linked defect much more aggressively than the standard softest path.
3. But the filter is still too weak globally, because some highly symmetric standard corrected directions survive it.

So the next regularity condition has to be stronger than “no rank collapse” alone. It likely needs to control one of the remaining high-symmetry escape channels, not just catastrophic degeneration.