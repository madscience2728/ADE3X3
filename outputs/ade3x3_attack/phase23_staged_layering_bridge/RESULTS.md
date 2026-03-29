# Phase 23 Results

Phase 23 bridges the earlier depth-2 circuit work to the newer kernel-linked obstruction diagnostics.

## 23a. What Step 70-73 already settled

The earlier circuit path split into three distinct claims:

- Step 70: the most direct recursive depth-2 attack, obtained by zero-padding `3x3` to `4x4` and recursively applying Strassen, still needs `31` surviving leaf multiplications for the top-left `3x3` target. So it does not threaten rank `23`.
- Step 72: positive local fiber dimensions in AA/BB/QQ depth-2 ansatze are only dimension signals inside the parameterization. They do not certify any exact low-rank algorithm.
- Step 73: for exact division-free depth-2 circuits, the degree-2 homogeneous truncation already captures the useful bilinear part. So exact AA/QQ depth does not escape the ordinary bilinear-rank problem.

That means the old “intermediate layered circuit” dead end was not a numerical accident. For exact division-free depth-2 circuits, layering by itself does not create new exact bilinear directions.

## 23b. What the new staged homotopies add

Phase 22 already tested the correct staged analogue on the bilinear side: instead of introducing explicit AA/QQ circuit layers, it deformed the factorization in stages while tracking the bilinear obstruction quantities directly.

The key question became:

> if we force the prospective obstruction only gradually, can the algorithm stay near exactness for a meaningful range of stages?

Using the extracted threshold summary from Phase 22:

- AlphaTensor, softest kernel direction:
  - first tensor residual above `0.05`: `t = 0.1`
  - first tensor residual above `0.10`: `t = 0.2`
  - first angle above `5` degrees: `t = 0.2`
  - first angle above `15` degrees: `t = 0.5`
  - first stage with `lambda^T H` max-abs below `0.10`: `t = 0.7`
  - by the time the annihilator condition becomes small, the tensor residual is already very large
- AlphaTensor, random kernel directions:
  - same qualitative story; residual exceeds `0.05` already at `t = 0.1`
- Standard algorithm:
  - several staged paths stay exact or near-exact for much longer, and some remain exact all the way to the endpoint for the sampled kernel direction

## 23c. Interpretation for layered operations

This clarifies the old circuit intuition.

If “adding intermediate steps” means exact division-free depth-2 layers of the Step 72/73 type, the structural theorem still blocks that route: the useful degree-2 part is already just the bilinear problem.

If “adding intermediate steps” means staged deformation inside bilinear factor space, then the new obstruction diagnostics do matter. They show that AlphaTensor-like low-rank geometry is fragile under staged forcing of the kernel-linked annihilator pattern: the failure appears early, not just at the fully enforced endpoint.

So layering does help as a diagnostic. It does not rescue the old exact depth-2 circuit route, but it does let us see where a proposed obstruction starts to bite instead of only checking the all-at-once endpoint.

## Overall Conclusion

The circuit dead end and the new staged evidence are consistent with each other.

1. Exact division-free depth-2 layering does not evade bilinear rank.
2. Staged forcing of the new obstruction pattern reveals early instability for AlphaTensor-like geometry.
3. Therefore the right next path is not “try another exact depth-2 circuit family,” but “use staged bilinear deformations to identify which obstruction patterns fail early for low-rank decompositions and which only reflect high-symmetry artifacts of the standard algorithm.”