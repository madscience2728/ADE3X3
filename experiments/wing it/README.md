# Wing It

This folder is for direct compression attempts that do not fit the canonized proof/obstruction phases.

Current script:

- `compress19_or_lower.py`

Strategy used:

1. warm truncation from the public rank-23 AlphaTensor decomposition down to rank 19,
2. random 4-term removal masks with noisy re-optimization,
3. border-style rank-23 optimization with four penalized tail terms, then truncation to rank 19,
4. cold random baselines at ranks 19 and 18.

Outputs written here:

- `attempt_results.csv`
- `summary.json`
- `RESULTS.md`
- `best_decompositions/`

This is explicitly an experiment sandbox, not a proof artifact.