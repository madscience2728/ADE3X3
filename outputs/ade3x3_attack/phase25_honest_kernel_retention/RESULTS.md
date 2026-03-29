# Phase 25 Derivation Session

Phase 24 identified the exact escape route that defeats the naive kernel-linked obstruction:

- exactness can be preserved,
- `lambda^T H` can be driven to `0`,
- but this can happen by rotating `lambda` out of the moving `ker(Gamma_t)`.

So the next phase should not search for another defect family. It should formalize the **honest kernel-linked continuation problem**.

## 25a. The Correct Coupled Defect Operator

For a decomposition state `x`, let

- `Gamma = Gamma(x)` be the `9 x R` output matrix,
- `H = H(x)` be the `R x 18` eta block.

A genuine kernel-linked annihilator is a nonzero term-space vector `lambda in R^R` such that

- `Gamma lambda = 0`,
- `lambda^T H = 0`.

Equivalently, `lambda` lies in the kernel of the stacked linear operator

`C_x : R^R -> R^9 ⊕ R^18`

defined by

`C_x(lambda) = (Gamma lambda, H^T lambda)`.

Thus the exact defect condition is

`ker(C_x) != {0}`.

This gives the first clean derived quantity:

`kappa_full(x) := sigma_min(C_x)`.

Interpretation:

- `kappa_full(x) = 0` iff there exists a nonzero honest kernel-linked annihilator,
- `kappa_full(x) > 0` quantitatively measures how far the decomposition is from having one.

On unit vectors `||lambda|| = 1`, the associated energy is

`E_full(x; lambda) = ||Gamma lambda||_2^2 + ||H^T lambda||_2^2`

and

`kappa_full(x)^2 = min_{||lambda||=1} E_full(x; lambda)`.

So the correct nonlinear obstruction is already visible as a single smallest-singular-value problem for the stacked operator `[Gamma ; H^T]`.

## 25b. The Intrinsic Kernel-Restricted Functional

Because the escape route in Phase 24 happens specifically by leaving the moving kernel, the more intrinsic quantity is the same energy restricted to `ker(Gamma)` itself.

Let `Q_x` be an orthonormal basis of `ker(Gamma(x))`. Then every honest kernel candidate can be written as

`lambda = Q_x y`.

Inside the moving kernel, the only remaining defect cost is

`E_ker(x; y) = ||y^T (Q_x^T H)||_2^2`

for `||y|| = 1`.

So define

`kappa_ker(x) := sigma_min(Q_x^T H_x)`.

This is exactly the quantitative version of the Phase 21 rank test:

- `kappa_ker(x) = 0` iff there exists a nonzero `lambda in ker(Gamma)` with `lambda^T H = 0`,
- `kappa_ker(x) > 0` means no honest kernel-linked annihilator exists.

Phase 21 already checked only the rank-level version of this statement. The next compute phase should track the **smallest singular value itself**, not just rank loss.

## 25c. Why Phase 24 Escapes Are Not Honest Kernel-Linked Continuations

Phase 24 froze a reference `lambda` and allowed the decomposition to move.

That was sufficient to show two things:

1. the canonical minimum-norm forcing direction is first-order transverse on AlphaTensor but tangent on the standard algorithm,
2. corrected exact paths exist, but they do so by sending the angle from `span(lambda)` to `ker(Gamma_t)` toward `90` degrees.

This means the corrected exact paths are not approximating an honest kernel-linked annihilator. They are approximating only the weaker condition

`lambda^T H_t -> 0`

while paying for it by making

`Gamma_t lambda`

large.

So the correct continuation problem must evolve `lambda` together with the decomposition rather than freezing it.

## 25d. Honest First-Order Continuation Equations

Let `x(t)` be a path of decomposition data and let `lambda(t)` be a unit vector field in term space.

To represent an honest kernel-linked annihilator path, the coupled constraints are

1. `Gamma(t) lambda(t) = 0`,
2. `lambda(t)^T H(t) = 0`,
3. `||lambda(t)||_2 = 1`.

Differentiating at `t = 0` gives the linearized system

1. `dot(Gamma) lambda + Gamma dot(lambda) = 0`,
2. `dot(lambda)^T H + lambda^T dot(H) = 0`,
3. `lambda^T dot(lambda) = 0`.

This is the correct first-order replacement for the frozen-`lambda` Phase 24 model.

It shows exactly what was missing:

- allowing `dot(lambda)` introduces the moving-kernel gauge freedom,
- but that freedom is constrained by the kernel equation and by normalization,
- so the next compute problem is a coupled constrained solve in `(dot(x), dot(lambda))`, not just in `dot(x)`.

## 25e. Derived Compute Target For Phase 25

The next compute phase should solve the following variational problem.

Given an exact decomposition `x` and a current unit vector `lambda in ker(Gamma(x))`, find the best infinitesimal step `(dot(x), dot(lambda))` that:

1. keeps the tensor residual zero to first order,
2. satisfies the moving-kernel constraint
   `dot(Gamma) lambda + Gamma dot(lambda) = 0`,
3. normalizes `lambda` by `lambda^T dot(lambda) = 0`,
4. maximally decreases the annihilator cost `||lambda^T H||` or the kernel-restricted singular value `kappa_ker(x)`.

In practice there are two natural implementations.

### Option A. Coupled constrained least squares

Solve directly for `(dot(x), dot(lambda))` under the linearized exactness and moving-kernel constraints, then optimize annihilator descent.

### Option B. Singular-value continuation

At each exact decomposition state `x`, compute the softest kernel direction as the right singular vector of `Q_x^T H_x` associated to `kappa_ker(x)`, then evolve `x` while re-updating `lambda` from the current kernel-restricted SVD.

Option B is more robust numerically and matches the actual derived object `kappa_ker(x)`.

## Overall Derivation Outcome

The narrow derivation session changes the next objective in a useful way.

The next phase should not ask:

> can we make `lambda^T H` small while staying exact?

That is too weak, because Phase 24 already showed how to do it dishonestly by leaving the kernel.

The right next question is:

> can an exact path drive the **kernel-restricted** defect `kappa_ker(x) = sigma_min(Q_x^T H_x)` toward `0` while keeping `lambda` inside the moving kernel?

That is the honest nonlinear continuation problem now exposed by Phases 21 through 24.

## 25f. First Compute Scan Along Exact Corrected Paths

After the derivation, Phase 25 implemented the stable Option B prototype: follow the exact corrected continuation seeds from Phase 24, but at each sampled time recompute the moving soft kernel mode and track

- `kappa_ker(x) = sigma_min(Q_x^T H_x)`,
- `kappa_full(x) = sigma_min([Gamma_x ; H_x^T])`,
- tensor residual,
- restricted rank inside the moving kernel.

The key point is that `lambda` is no longer frozen. At each sampled state, it is redefined as the current soft kernel direction.

### AlphaTensor Softest Path

For the AlphaTensor softest corrected path:

- at `t = 0.0`:
   - `kappa_ker = 0.6245923009701738`
   - `restricted_rank = 14`
   - tensor residual `2.49e-15`
- at `t = 0.5`:
   - `kappa_ker = 0.3168896728295838`
   - `restricted_rank = 14`
   - tensor residual `2.99e-15`
- at `t = 0.9`:
   - `kappa_ker = 0.09054210639061283`
   - `restricted_rank = 14`
   - tensor residual `7.75e-14`
- at `t = 1.0`:
   - `kappa_ker = 0.004687401734031592`
   - `restricted_rank = 14`
   - tensor residual `1.64e-13`

Interpretation:

AlphaTensor can drive the honest kernel-restricted defect very low while remaining exact to floating-point precision, but in this scan it does **not** actually hit `0`. The rank stays full at `14` all along.

So along this best sampled soft path, AlphaTensor approaches an honest kernel-linked annihilator without crossing into a true defect.

### Standard Softest Path

For the standard softest corrected path:

- at `t = 0.0`:
   - `kappa_ker = 1.0`
   - `restricted_rank = 18`
   - tensor residual `0.0`
- at `t = 0.9`:
   - `kappa_ker = 0.09999999999999981`
   - `restricted_rank = 18`
   - tensor residual `0.0`
- at `t = 1.0`:
   - `kappa_ker = 2.217730252320637e-16`
   - `restricted_rank = 4`
   - tensor residual `1.0`

Interpretation:

The standard algorithm reaches an actual honest kernel defect on the softest path only at a singular endpoint collapse. The rank drops from `18` to `4`, and exact multiplication fails catastrophically at the same endpoint.

So the standard softest path does **not** provide a smooth exact route into the defect. It reaches the defect only through structural collapse.

### Other Sampled Directions

- AlphaTensor random directions still lower `kappa_ker`, but only to about `0.0450` and `0.0649` at `t = 1`.
- Standard random directions can also lower `kappa_ker` while staying exact, reaching about `0.0820` and `0.00571` at `t = 1`.

So the soft-kernel continuation is not yet a universal separator by itself. But it does isolate a meaningful asymmetry:

- AlphaTensor's softest exact path approaches the honest defect while keeping full restricted rank,
- the standard algorithm's softest exact path reaches the honest defect only by a singular endpoint rank collapse.

### Additional Structural Observation

Along these sampled exact paths, `kappa_full(x)` and `kappa_ker(x)` agree numerically to working precision.

That indicates the ambient coupled defect operator `[Gamma_x ; H_x^T]` is being governed almost entirely by the kernel-restricted soft mode on these paths, exactly as the derivation suggested.

## Updated Interpretation

Phase 25 converts the derivation into the first honest compute test.

1. The right nonlinear quantity really is `kappa_ker(x) = sigma_min(Q_x^T H_x)`.
2. Tracking that quantity along exact corrected paths is substantially more informative than tracking a frozen `lambda^T H` alone.
3. The current evidence suggests that the standard algorithm hits the honest defect only via singular collapse on the softest path, while AlphaTensor can approach the defect closely without such rank collapse.

So the next refinement is now clearer:

- exclude singular endpoint collapse,
- distinguish smooth exact continuation from structural degeneration,
- and test whether AlphaTensor-like low-rank geometry can force `kappa_ker` toward `0` under regularity constraints that the standard algorithm cannot satisfy honestly.