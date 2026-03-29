# Phase 24 Results

Phase 24 tested whether the Phase 22 instability is already a first-order obstruction or whether it only appears after quotienting out exact tangent freedom incorrectly.

The setup was:

1. keep the same kernel-linked annihilator forcing model as Phase 22,
2. linearize the exact-multiplication condition after optimally re-solving `gamma`,
3. compare two forcing directions:
   - the canonical minimum-norm annihilator-forcing direction used implicitly by Phase 22,
   - the best corrected direction inside the same affine annihilator constraint that minimizes first-order tensor error,
4. then follow the corrected path at finite scales and track three quantities together:
   - tensor residual,
   - progress toward `lambda^T H = 0`,
   - angle from `span(lambda)` to the new `ker(Gamma_t)`.

## 24a. The Canonical Phase 22 Direction Is Already Transverse On AlphaTensor

For the minimum-norm annihilator-forcing direction, the first-order tensor residual is already large on AlphaTensor:

- softest kernel direction:
  - first-order residual `L2 = 2.9108098588350466`
  - max-abs entry `0.7440549635978719`
- random kernel directions:
  - first-order residual `L2 = 3.217597878814448`, max-abs `0.7027892450594478`
  - first-order residual `L2 = 2.703082223963728`, max-abs `0.6752382860672448`

By contrast, for the standard rank-27 algorithm the same minimum-norm direction is already tangent to the exact variety:

- all three tested directions have first-order residual exactly `0` to numerical precision.

This explains the Phase 22 asymmetry more precisely:

- on AlphaTensor, the most natural annihilator-forcing path immediately pushes out of exactness,
- on the standard algorithm, the same gauge choice sits inside exact tangent freedom.

## 24b. First-Order Obstruction Alone Still Fails

Phase 24 then allowed corrections inside the affine annihilator constraint and minimized the first-order tensor error.

Result:

- AlphaTensor also admits corrected directions with first-order residual essentially zero:
  - optimized first-order residuals are about `1e-14`
- Standard again admits exact tangent directions, with optimized first-order residual exactly zero.

So the universal obstruction is not:

> no first-order exact tangent direction exists.

That statement is false even for AlphaTensor.

What *does* distinguish the two cases is how much correction is needed to repair the canonical forcing direction.

For AlphaTensor, the tangent correction is substantial:

- correction ratios relative to the minimum-norm direction are about `0.56` to `0.69`.

For the standard algorithm, the tangent correction is exactly zero in all three tested cases.

Interpretation:

The standard algorithm has enough symmetry/redundancy that the canonical forcing direction is already tangent. AlphaTensor only regains tangency after a substantial sideways correction.

## 24c. The Corrected Exact Path Escapes By Losing Kernel Linkage

The corrected AlphaTensor paths remain exact to floating-point precision even at the fully enforced endpoint while reducing `lambda^T H` all the way to numerical zero.

For the softest-kernel-direction corrected path:

- at `t = 0.1`:
  - tensor residual `3.33e-15`
  - `lambda^T H` ratio `0.9`
  - angle to `ker(Gamma_t)` `1.1508368370922002` degrees
- at `t = 0.2`:
  - tensor residual `3.62e-15`
  - `lambda^T H` ratio `0.8`
  - angle `2.57814501329677` degrees
- at `t = 0.5`:
  - tensor residual `2.99e-15`
  - `lambda^T H` ratio `0.5`
  - angle `9.832781539614492` degrees
- at `t = 1.0`:
  - tensor residual `1.64e-13`
  - `lambda^T H` ratio about `1.38e-15`
  - angle `89.99999999999642` degrees

So the corrected path does not refute the kernel-linked obstruction picture. It shows the escape mechanism explicitly:

- exactness can be preserved,
- the annihilator can be enforced,
- but only by rotating `lambda` out of the moving kernel.

The same corrected-path phenomenon appears in the standard algorithm as well: exactness can survive while the angle to `ker(Gamma_t)` grows substantially for some directions.

## Overall Interpretation

Phase 24 sharpens the missing gap.

1. The raw Phase 22 minimum-norm forcing direction is already informative: it is transverse on AlphaTensor but tangent on the standard algorithm.
2. However, there is no universal first-order obstruction, because AlphaTensor still has corrected tangent-compatible directions.
3. Those corrected exact paths escape only by breaking the kernel-link condition: `lambda^T H` can go to zero while `lambda` rotates far away from `ker(Gamma_t)`.

So the next obstruction cannot be purely local and cannot talk only about annihilator progress or only about exact tangent freedom. It has to control the **simultaneous nonlinear coupling**:

- keep exactness,
- drive `lambda^T H` down,
- and keep `lambda` genuinely linked to `ker(Gamma_t)`.

That is now the sharp remaining compute target.