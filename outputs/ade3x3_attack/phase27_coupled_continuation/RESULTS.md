# Phase 27 Results

Phase 27 implements the first prototype of the actual coupled continuation problem derived in Phase 25.

Instead of freezing `lambda`, it solves a linearized constrained problem in both

- the decomposition variables `dot(x)` (represented here by the channel-scaling variables), and
- the term-space vector `dot(lambda)`.

The prototype enforces to first order:

1. exact tensor preservation,
2. moving-kernel consistency,
3. normalization of `lambda`.

Then it minimizes the linearized honest defect cost.

## 27a. First-Order Solve

At the base exact states, the prototype predicts an almost perfect first-order kill of the honest defect in both families.

Predicted reduction factors:

- AlphaTensor: about `5.39e-15`
- Standard: about `2.02e-15`

So at the purely infinitesimal linearized level, both geometries look flexible enough to cancel the current honest defect.

This is consistent with the broader picture already emerging from Phase 24:

- a purely first-order obstruction is too weak.

## 27b. Finite Probes

The useful test is therefore not the linearized prediction itself, but what happens when the coupled step is probed at small finite scale.

### AlphaTensor

Using the soft kernel mode at the base state:

- base honest defect is about `0.624592`
- at probe `eps = 1e-3`: honest defect `0.623968`
- at probe `eps = 1e-2`: honest defect `0.618332`
- at probe `eps = 5e-2`: honest defect `0.593058`

Tensor residual stays essentially exact throughout these probes.

### Standard

Using the soft kernel mode at the base state:

- base honest defect is `1.0`
- at probe `eps = 1e-3`: honest defect `0.9989999`
- at probe `eps = 1e-2`: honest defect `0.9899939`
- at probe `eps = 5e-2`: honest defect `0.9498499`

Tensor residual also stays exact throughout these probes.

## 27c. Interpretation

The Phase 27 prototype says something precise.

The coupled linearized problem can be solved extremely well, but that success does **not** translate into a dramatic finite-step reduction of the honest defect. At small finite scales, both AlphaTensor and the standard algorithm only reduce the defect modestly.

So the next barrier is not the existence of a coupled first-order direction. It is the **nonlinear persistence** of the honest defect beyond first order.

This is stronger than the Phase 24 conclusion:

- Phase 24 showed that freezing `lambda` misses the moving-kernel geometry.
- Phase 27 shows that even after restoring the moving-kernel variable `lambda`, the linearized solve is still too optimistic.

## Additional Observation

In this prototype, the result is the same for all three candidate labels inside each base decomposition. That is not a bug in interpretation: the solver immediately reanchors to the current soft kernel mode of `Q_x^T H_x`, so the initial candidate label becomes irrelevant at the base point.

## Overall Interpretation

Phase 27 narrows the next target again.

1. The coupled `(dot(x), dot(lambda))` system is the right local model.
2. But local solvability does not imply meaningful finite reduction of the honest defect.
3. Therefore the real obstruction problem now lives in the nonlinear regularity of the continuation, not in the mere existence of a first-order coupled descent direction.

The next compute refinement should therefore combine the Phase 26 regularity filters with the Phase 27 coupled step, and ask which smooth coupled continuations actually produce sustained decrease of `kappa_ker` rather than only infinitesimal cancellation.