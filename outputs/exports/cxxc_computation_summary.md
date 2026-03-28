# CXXC Orbit Computation Summary

**Generated:** 2026-03-28 12:36:15
**Schema:** CXXC (C × X × X × C)
**Typed Arity:** 4
**Raw Arity:** 6

## Results

- Total configurations: 531,441
- Total orbits: 2744
- Distinct signatures: 784
- Orbit-complete: no

## Verification

- All orbits satisfy |orbit| × |stabilizer| = 216 ✓
- Sum of orbit sizes = 531,441 ✓

## Signature Format

Signature tuple fields:
```
(x1_live, x2_live, c1_equals_c2,
 c1_is_target_of_x1_if_live, c1_is_target_of_x2_if_live,
 c2_is_target_of_x1_if_live, c2_is_target_of_x2_if_live,
 row_quad, col_quad)
```

Where:
- `x1_live`, `x2_live`: Boolean liveness of each X atom
- `c1_equals_c2`: Whether first and last C atoms are equal
- Target flags: Whether C atoms are targets of live X atoms
- `row_quad`, `col_quad`: Canonical encoding of equality partitions

## Exported Files

- `signatures_CXXC.csv`: Complete orbit roster with signatures
- `cxxc_computation_summary.md`: This file
