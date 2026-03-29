# Phase 15 Results

## 15a. Matrix Construction

- Signal matrices constructed: 9
- Anisotropy matrices constructed: 18
- AlphaTensor null-space Q matrices constructed: 4
- Complement dimension: 14
- Signal verification on all 23 AlphaTensor terms: True
- Anisotropy verification on all 23 AlphaTensor terms: True
- Null-vector verification on all 23 AlphaTensor terms: True
- Q-form verification on all 23 AlphaTensor terms: True
- Q-vs-signal orthogonality max trace: 0.0
- Complement Gram max abs error: 1.1102230246251565e-16
- Complement-vs-null max abs inner product: 2.220446049250313e-16

## 15b-15d. Complement Basis Scan

- Sample count per basis direction: 200
- All 14 basis directions produced rank(W) = 76 and rank(sigma) = 9
- Best direction: w02
- Best projection loss: 6.648380768438399e-29
- Worst projection loss: 9.354690033997625e-29

### Direction Scores

| basis | sampled points | rank(W) | rank(sigma) | projection loss | max abs residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| w02 | 200 | 76 | 9 | 6.648380768438399e-29 | 1.1102230246251565e-15 |
| w11 | 200 | 76 | 9 | 7.14706955343313e-29 | 1.1757630490091976e-15 |
| w06 | 200 | 76 | 9 | 7.171978590605666e-29 | 9.15830553459516e-16 |
| w08 | 200 | 76 | 9 | 7.304224077005905e-29 | 1.1102230246251565e-15 |
| w13 | 200 | 76 | 9 | 7.519945452603558e-29 | 1.1102230246251565e-15 |
| w14 | 200 | 76 | 9 | 7.607138970840936e-29 | 9.483815051294025e-16 |
| w12 | 200 | 76 | 9 | 7.996125046521065e-29 | 1.1379786002407855e-15 |
| w01 | 200 | 76 | 9 | 7.997295157925338e-29 | 1.3322676295501878e-15 |
| w03 | 200 | 76 | 9 | 8.258036579670995e-29 | 1.2463824496462822e-15 |
| w10 | 200 | 76 | 9 | 8.768730587201266e-29 | 1.1102230246251565e-15 |
| w04 | 200 | 76 | 9 | 8.97737110974607e-29 | 1.5543122344752192e-15 |
| w05 | 200 | 76 | 9 | 9.100300262574379e-29 | 1.7763568394002505e-15 |
| w09 | 200 | 76 | 9 | 9.198159230135097e-29 | 1.5543122344752192e-15 |
| w07 | 200 | 76 | 9 | 9.354690033997625e-29 | 1.9984014443252818e-15 |

Interpretation: under the requested projection-loss metric, every basis direction is effectively exact. The scan therefore does not distinguish promising fifth-identity directions at all. This is the decision-point result: the current diagnostic collapses on the 14-dimensional complement and does not reproduce the expected random-baseline separation.

## 15g. Strassen Conservation Law

- Strassen 2x2 rank(H): 3
- Strassen 2x2 nullity k: 1
- Conservation check: 7 + 1 = 8 = 2^3
- Exact tensor verification: loss 0.0, max abs 0.0
- Extracted identity vector: [-1, 0, 0, -1]
- Extracted identity formula: -eta[0,0] - eta[1,1] = 0

Interpretation: every Strassen term satisfies eta[0,0] + eta[1,1] = 0, i.e. the two diagonal anisotropy coordinates cancel termwise.

## 15h. Optional Stress Test

- Standard 2x2 rank(H): 4, k = 0, so 8 + 0 = 8
- No in-repo 4x4 decomposition table was available, so the N=4 stress test remains unrun

## Status

Stopped after 15d + 15g as requested.
15e optimization was not run because the 14-direction scan produced no meaningful ranking signal; every basis direction scored essentially zero projection loss.
