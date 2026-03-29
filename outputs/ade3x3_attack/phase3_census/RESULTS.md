# Phase 3 Results

## Summary Table

| source | count | attempts | delta-in-eta rate | rank(H) range | rank(H) mode | M sparsity range | M integrality rate |
|--------|-------|----------|-------------------|---------------|--------------|------------------|--------------------|
| perturbation | 0 | 512 | n/a | n/a | n/a | n/a | n/a |
| random_rank23 | 0 | 512 | n/a | n/a | n/a | n/a | n/a |
| symmetry_orbit | 216 | 216 | 216/216 | 14..14 | 14 | 42..80 | 216/216 |

## Verdict

delta subset eta is universal across the entire 216-element AlphaTensor symmetry orbit, but this run found no exact non-orbit rank-23 decompositions. Universality beyond the AlphaTensor equivalence class remains unclear.

## Notes

- The symmetry-orbit census is exact and exhaustive over the 216 compatible S3 x S3 x S3 actions.
- The perturbation and random-search rows count only exact kept decompositions written to the JSONL archive; all attempts are still recorded in generation_attempts.csv.
- A zero count outside the symmetry orbit means no exact decomposition met the configured residual tolerance in this run, not that nearby numerical minima were absent.
