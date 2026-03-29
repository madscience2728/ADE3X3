# Phase 22 Results

Phase 22 implemented the next compute step suggested by Phase 21: couple the annihilator condition directly to `ker(Gamma)` instead of treating them separately.

The setup was:

1. choose `lambda` from the original `ker(Gamma)` of a known exact decomposition,
2. build a homotopy in factor space that gradually enforces `lambda^T H = 0` by channel-wise term rescaling,
3. at each homotopy time `t`, solve the best least-squares `gamma`, then measure:
   - the max residual of `lambda^T H`,
   - the principal angle between `span(lambda)` and the new `ker(Gamma_t)`,
   - the tensor residual,
   - the restricted defect dimension inside `ker(Gamma_t)`.

So the experiment directly asks: can low tensor residual coexist with both `lambda^T H approx 0` and `lambda approx ker(Gamma_t)`?

## 22a. AlphaTensor Homotopy

For the AlphaTensor rank-23 decomposition, all three tested kernel directions behaved the same qualitatively.

Using the softest kernel direction (the smallest-singular-direction of `Q^T H`) as the cleanest example:

- at `t = 0.0`:
  - angle to `ker(Gamma_t)`: `0.0` degrees
  - `lambda^T H` max residual: `0.3201972295418893`
  - tensor max-abs residual: `2.49e-15`
- at `t = 0.1`:
  - angle: `2.3916775394610275` degrees
  - `lambda^T H` max residual: `0.28817750658770036`
  - tensor max-abs residual: `0.07130410938617415`
- at `t = 0.2`:
  - angle: `5.3563441197266455` degrees
  - tensor max-abs residual: `0.15252931637611744`
- at `t = 0.5`:
  - angle: `18.918901078978116` degrees
  - tensor max-abs residual: `0.5599639002594361`
- at `t = 1.0`:
  - `lambda^T H` max residual drops to numerical zero
  - angle to `ker(Gamma_t)`: `48.67994766486628` degrees
  - tensor max-abs residual: `1.0612249246423127`

The two random kernel directions show the same pattern: as `lambda^T H` is driven to zero, the angle to the new `ker(Gamma_t)` grows sharply and the tensor residual rises to about `1.09`.

Interpretation:

For AlphaTensor, the two requirements fight each other immediately. Keeping `lambda` close to `ker(Gamma_t)` while pushing toward `lambda^T H = 0` already produces visible tensor error at very small deformation. This is the strongest computational evidence so far that the kernel-linked annihilator defect is incompatible with the AlphaTensor multiplication structure.

## 22b. Standard Homotopy

The standard rank-27 decomposition behaves differently.

- Along the softest-kernel-direction path, the tensor stays exact for most of the homotopy, and only at the fully enforced endpoint `t = 1` does the structure collapse:
  - at `t = 1.0`: `rank(H)` drops to `11`, restricted rank drops to `4`, defect dimension becomes `14`, and tensor max-abs residual jumps to `1.0`
- Along two random kernel directions, the tensor remains exact to floating-point precision across the sampled path, while the angle to `ker(Gamma_t)` can grow substantially.

Interpretation:

The standard algorithm has much more internal flexibility than AlphaTensor under this deformation. That means the obstruction cannot simply be “any kernel-linked annihilator homotopy destroys exactness” uniformly across all exact decompositions. The AlphaTensor behavior is strong evidence, but not yet a universal theorem pattern.

## Overall Interpretation

Phase 22 sharpens the picture in two ways.

1. On AlphaTensor, kernel-linked annihilator forcing looks genuinely incompatible with exact multiplication: the residual-angle tradeoff becomes bad almost immediately.
2. On the standard algorithm, the same homotopy can remain exact for long stretches, so the obstruction must be more specific than this homotopy template alone.

So the next target should not be “any kernel-linked annihilator is impossible.” It should be a sharper variant that distinguishes AlphaTensor-like saturated decompositions from the highly symmetric standard algorithm, likely by adding a minimal-rank or nonsplitting condition to the kernel-linked defect model.