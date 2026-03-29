# Phase 28 Results

Phase 28 combines the two most recent ingredients into one stricter test:

- use the coupled `(dot(x), dot(lambda))` continuation model from Phase 27,
- keep the Phase 26 regularity filter,
- and add an explicit wildcard branch at every step.

The wildcard branch is not a new theorem. It is a search safeguard.

At each continuation state, the script:

1. computes the structured coupled step,
2. samples 48 wildcard perturbations inside the same linearized admissible nullspace,
3. tests finite probe sizes `eps in {1e-3, 3e-3, 1e-2, 3e-2, 5e-2}`,
4. accepts only probes with tensor max-abs residual `<= 1e-10` and unchanged restricted rank,
5. advances by the regular probe with smallest resulting `kappa_ker`.

## 28a. AlphaTensor

AlphaTensor starts at

- base `kappa_ker = 0.6245923009701738`,
- base restricted rank `14`.

Results:

- 8 regular continuation steps were accepted,
- all 8 accepted steps came from the wildcard branch,
- `kappa_ker` decreased monotonically to `0.3655885791544301`,
- tensor residual stayed at machine scale throughout (roughly `1e-15`).

On the first step alone:

- best structured regular probe reached `kappa_ker = 0.5920644505108291`,
- best wildcard regular probe reached `kappa_ker = 0.5857724973145496`.

So the wildcard branch is genuinely stronger than the single structured step, even under the same regularity filter.

## 28b. Standard Algorithm

The standard algorithm starts at

- base `kappa_ker = 1.0`,
- base restricted rank `18`.

Results:

- 8 regular continuation steps were accepted,
- all 8 accepted steps came from the wildcard branch,
- `kappa_ker` decreased monotonically to `0.6116041118104862`,
- tensor residual also stayed at machine scale throughout.

On the first step alone:

- best structured regular probe reached `kappa_ker = 0.9479163680290742`,
- best wildcard regular probe reached `kappa_ker = 0.9399070425783675`.

Again, the wildcard branch materially outperforms the structured branch.

## 28c. Interpretation

Phase 28 changes the practical continuation picture in two ways.

1. Wildcards are necessary. The structured coupled step is not enough as a stress test; hidden admissible directions inside the same nullspace improve the finite continuation every time in this experiment.
2. The old Phase 27 conclusion was too narrow if read operationally. It was correct that the structured local solve only gives modest finite reduction, but the broader admissible cone contains better regular finite steps.
3. This does **not** yet create a universal separator. Both AlphaTensor and the standard algorithm admit sustained regular wildcard-assisted descent.

There is still a mild asymmetry: after 8 regular wildcard steps, AlphaTensor drops from about `0.6246` to `0.3656`, while the standard algorithm drops from `1.0` to `0.6116`. But this is not close to a clean obstruction.

So the next target has to be stronger again:

- either a sharper regularity criterion that forbids the wildcard escape directions,
- or a curvature / persistence invariant that distinguishes the two families even after wildcard-assisted continuation is allowed.