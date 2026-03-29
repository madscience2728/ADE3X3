# Step 76: Batch of Five Quick Tests
Generated: 2026-03-28T23:54:40

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

## Task D: Division-Augmented 2x2

- Best sweep case: 6_ops_0_div
- Best sweep validation MSE: 4.301050e+00
- Any exact hit found: False
- Focus phase triggered: False

The search now runs as a staged sweep over the exported operation/division cases, with a focus phase reserved for the first case that produces any exact hit.
Output weights are still optimized by least squares rather than L-BFGS because the weighting layer is linear once the primitive operations are fixed.

## Task E: GF(9) and Frobenius

- First positive GF(9) sweep rank: None
- Any exact GF(9) hit in the ordinary bilinear model: False
- Focus phase triggered: False
- Free Frobenius was treated conservatively as outside ordinary tensor rank.

Free Frobenius yields a semilinear model over GF(3), not an ordinary GF(9)-bilinear tensor decomposition. The exact quick test therefore reports a staged ordinary-GF(9) sweep/focus proxy and leaves the semilinear Frobenius-free rank unresolved.
