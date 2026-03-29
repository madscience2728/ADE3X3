# Phase 29 Results

Phase 29 executes both requested refinements on top of Phase 28.

It compares two wildcard-aware continuation policies:

1. `smooth_greedy`: keep the wildcard search, but require each accepted probe to satisfy the old exactness / no-collapse filter **and** a mode-rotation cap of `12` degrees.
2. `smooth_budgeted_long`: keep the same mode-rotation cap, but switch from greedy one-step descent to a longer-horizon budgeted policy that penalizes step cost and runs for up to `24` accepted steps with total budget `1.2`.

So Phase 29 asks two separate questions:

- does a smoothness cap kill the wildcard escape routes?
- if not, does a budgeted long-horizon policy reveal a more stable separation?

## 29a. Smooth Greedy Policy

Results:

- AlphaTensor:
  - base `kappa_ker = 0.6245923009701738`
  - final / minimum `kappa_ker = 0.37602350369115983`
  - `8` accepted steps
  - all `8` accepted steps are wildcard steps
  - cumulative budget used `1.956624093599254`
- Standard algorithm:
  - base `kappa_ker = 1.0`
  - final / minimum `kappa_ker = 0.607502204723264`
  - `8` accepted steps
  - all `8` accepted steps are wildcard steps
  - cumulative budget used `0.9134797844495537`

Interpretation:

The mode-rotation cap by itself is not enough to eliminate wildcard-assisted descent. Both families still admit smooth, exact, no-collapse wildcard continuation, and the accepted AlphaTensor steps are still quite aggressive. So the extra smoothness constraint strengthens the test, but it does not yet isolate a universal obstruction.

## 29b. Smooth Budgeted Long-Horizon Policy

Results:

- AlphaTensor:
  - base `kappa_ker = 0.6245923009701738`
  - final / minimum `kappa_ker = 0.6099009207075097`
  - `24` accepted steps
  - only `2` wildcard wins
  - total budget used `0.07624610604030692`
- Standard algorithm:
  - base `kappa_ker = 1.0`
  - final / minimum `kappa_ker = 0.7966188942955966`
  - `24` accepted steps
  - `0` wildcard wins
  - total budget used `0.31916923028785094`

The accepted steps are qualitatively different from Phase 28.

- The policy almost always chooses `eps = 1e-3` for AlphaTensor.
- For the standard algorithm it takes four initial `eps = 5e-2` structured steps, then switches to tiny `eps = 1e-3` structured refinements.
- The standard long-horizon policy exhibits essentially zero mode rotation on all accepted steps after the initial drop.

Interpretation:

Once the objective includes explicit cost control, the continuation no longer prefers aggressive wildcard moves. The search collapses mostly onto tiny structured steps, especially for the standard algorithm.

## 29c. Overall Interpretation

Phase 29 refines the Phase 28 picture in a useful way.

1. Smoothness alone is too weak. A `12` degree mode-rotation cap still permits substantial wildcard-assisted descent in both families.
2. Budget changes the geometry. The long-horizon cost-penalized policy sharply suppresses wildcards and produces much slower but more stable continuation.
3. Under that budgeted policy, AlphaTensor still makes progress, but only slightly: from about `0.6246` to `0.6099`.
4. The standard algorithm also makes progress, but more expensively in total budget and without any wildcard wins.

So the next useful separator is likely not just “smooth” versus “nonsmooth.” It likely needs to measure a stronger persistence tradeoff, such as:

- defect reduction per unit continuation budget,
- cumulative mode rotation versus gain,
- or a curvature-style invariant along the accepted path rather than a per-step cap alone.