# Orbit-Signature Export Summary

Generated: 2026-03-27 16:34:43

Explicit orbit-signature assignments for the ADE3x3 warehouse.

Signature keys are step10 signature functions applied to the correct
canonical orbit representatives (step18 orbit computation with proper
B-action). Previous step10 orbit_metadata used orbit_id as config_id
index, which was a storage bug producing incorrect representatives.

## Summary Table

| Schema | Orbit Count | Distinct Signatures | Orbit Complete |
|--------|-------------|---------------------|----------------|
| XX | 56 | 48 | no |
| CX | 8 | 8 | yes |
| XC | 8 | 8 | yes |
| CC | 4 | 4 | yes |
| AX | 10 | 8 | no |
| BX | 10 | 8 | no |
| CXC | 50 | 50 | yes |

## Exported CSV Files

- `signatures_XX.csv`
- `signatures_CX.csv`
- `signatures_XC.csv`
- `signatures_CC.csv`
- `signatures_AX.csv`
- `signatures_BX.csv`
- `signatures_CXC.csv`

Signature keys are now explicitly recorded as warehouse artifacts.
