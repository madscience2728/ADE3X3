# Composition Export: CX ∘ XC → CC

Generated: 2026-03-27 17:26:51

Explicit composition table for CX ∘ XC → CC using corrected orbit layer.

## Summary

| Property | Value |
|----------|-------|
| CX orbit count | 8 |
| XC orbit count | 8 |
| CC orbit count | 4 |
| Possible orbit-pairs | 64 |
| Realized keys | 32 |
| Deterministic | 18 |
| Mixed | 14 |

## Preview (first 10 realized keys)

| cx_orbit | xc_orbit | witnesses | cc_histogram | det |
|----------|----------|-----------|--------------|-----|
| 0 | 0 | 27 | {0: 27} | yes |
| 0 | 1 | 54 | {1: 54} | yes |
| 0 | 2 | 54 | {2: 54} | yes |
| 0 | 3 | 108 | {3: 108} | yes |
| 1 | 0 | 54 | {1: 54} | yes |
| 1 | 1 | 108 | {0: 54, 1: 54} | no |
| 1 | 2 | 108 | {3: 108} | yes |
| 1 | 3 | 216 | {2: 108, 3: 108} | no |
| 2 | 4 | 54 | {0: 54} | yes |
| 2 | 5 | 108 | {1: 108} | yes |

## Historical Note

An older type-based layer reported 256 possible keys, 64 realized,
36 deterministic, 28 mixed. Those counts used a different type system.
The current corrected orbit-based counts are shown above.

## Files

- `comp_CX_XC_to_CC.csv` — realized key summary
- `comp_CX_XC_to_CC_witnesses.csv` — full witness triples
