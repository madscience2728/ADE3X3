# Step 53: Support-Type Representative Incidence
Generated: 2026-03-28T12:43:30

[EXACT_DERIVED]

This step keeps only support information for a single rank-1 term: which indices appear in alpha, beta, and gamma along their row/column coordinates. Those six subset supports are classified modulo the natural S3 x S3 x S3 action.

Support-type classes: 8000
Orbit sizes: 1,3,6,9,18,27,36,54,108,216
Type-0-feasible classes: 1000
Minimum forced auto-zero count among Type-0-feasible classes: 0
All-8-active Type-0-feasible classes: 216

## Orbit-0-Feasible Histogram

| active_count | auto_zero_count | n_support_classes |
|--------------|-----------------|-------------------|
| 1 | 7 | 64 |
| 2 | 6 | 288 |
| 4 | 4 | 432 |
| 8 | 0 | 216 |

## Active-Mask Summary

| active_mask | active_count | auto_zero_count | n_classes | n_orbit0_possible | n_all8_active |
|-------------|--------------|-----------------|-----------|-------------------|---------------|
| 00000000 | 0 | 8 | 6272 | 0 | 0 |
| 00000001 | 1 | 7 | 8 | 0 | 0 |
| 00000010 | 1 | 7 | 16 | 0 | 0 |
| 00000011 | 2 | 6 | 24 | 0 | 0 |
| 00000100 | 1 | 7 | 16 | 0 | 0 |
| 00000101 | 2 | 6 | 24 | 0 | 0 |
| 00001000 | 1 | 7 | 32 | 0 | 0 |
| 00001010 | 2 | 6 | 48 | 0 | 0 |
| 00001100 | 2 | 6 | 48 | 0 | 0 |
| 00001111 | 4 | 4 | 72 | 0 | 0 |
| 00010000 | 1 | 7 | 16 | 0 | 0 |
| 00010001 | 2 | 6 | 24 | 0 | 0 |
| 00100000 | 1 | 7 | 32 | 0 | 0 |
| 00100010 | 2 | 6 | 48 | 0 | 0 |
| 00110000 | 2 | 6 | 48 | 0 | 0 |
| 00110011 | 4 | 4 | 72 | 0 | 0 |

## Sample All-8-Active Support Class

- support_type_id: 2486
- active_mask: 11111111
- alpha_row_support: {0}
- alpha_col_support: {0}
- beta_row_support: {0,1}
- beta_col_support: {0}
- gamma_row_support: {0,1}
- gamma_col_support: {0,1}

[INTERPRETATION]

Step 53 closes off another support-only route. The Step 48 orbit-sum model was already too coarse; Step 53 shows that even exact support incidence against the 8 representative equation types is still vacuous. Once support is rich enough to allow the positive Type 0 equation, there are many support classes that also allow all 7 zero-RHS types. Any universal lower-bound argument must therefore use coefficient relations, quotient-space structure, or stronger algebraic constraints than support incidence alone.