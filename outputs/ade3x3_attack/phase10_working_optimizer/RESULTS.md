# Phase 10 Results Through 10c

## 10a-10b. Factored Loss and Tensor Construction

- target unfolding shape: [9, 81]
- target unfolding nonzero count: 27
- target unfolding nonzero values: [1.0]

## 10c. R=23 Baseline

- warm start final loss: 1.003277906009718e-05
- warm start final max-abs residual: 0.0004583747704334229
- warm start passes `<1e-10`: False
- warm start passes `<1e-20`: False
- cold-start runs: 24
- cold-start success rate `<1e-10`: 0.000
- cold-start success rate `<1e-20`: 0.000
- cold-start median final loss: 0.20087785384992995
- cold-start best final loss: 0.0015230691396819571
- cold-start best final max-abs residual: 0.006390291680602583
- best cold candidate rank(H)_numeric: 18
- best cold candidate quotient_gain_numeric: 0
- best cold candidate delta_in_eta_numeric: False

## Calibration Verdict

- The factored-loss baseline is not yet calibrated. Either the warm start did not return to exactness or the cold-start hit rate stayed below the 5% target, so R=22 negatives would still be untrustworthy under this optimizer.
