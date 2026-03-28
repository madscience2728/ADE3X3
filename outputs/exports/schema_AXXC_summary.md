# Schema Registration: AXXC

Generated: 2026-03-27 19:10:47

AXXC is the second typed schema beyond the arity-3 core.

## Schema Definition

| Property | Value |
|----------|-------|
| Typed schema | A × X × X × C |
| Typed arity | 4 |
| Raw arity | 6 |
| Typed config count | 531,441 |
| Role overlay | `('A','A_X1','B_X1','A_X2','B_X2','C')` |

## Bridge Rule

Each A and C expands to 1 raw slot. Each X expands to 2 raw slots.

AXXC config (a, x1, x2, c) expands to raw tuple:
- (A_idx(a), A_idx(x1), B_idx(x1), A_idx(x2), B_idx(x2), C_idx(c))

## Export Scope

bridge_AXXC.csv contains the **full population**: all 531,441 rows.

## Preview (first 10 rows)

| axxc_config_id | axxc_readable | raw_tuple | raw_tuple_id |
|----------------|---------------|-----------|--------------|
| 0 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[0,0]] | (0,0,0,0,0,0) | 0 |
| 1 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[0,1]] | (0,0,0,0,0,1) | 1 |
| 2 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[0,2]] | (0,0,0,0,0,2) | 2 |
| 3 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[1,0]] | (0,0,0,0,0,3) | 3 |
| 4 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[1,1]] | (0,0,0,0,0,4) | 4 |
| 5 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[1,2]] | (0,0,0,0,0,5) | 5 |
| 6 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[2,0]] | (0,0,0,0,0,6) | 6 |
| 7 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[2,1]] | (0,0,0,0,0,7) | 7 |
| 8 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,0],C[2,2]] | (0,0,0,0,0,8) | 8 |
| 9 | AXXC[A[0,0],X[0,0|0,0],X[0,0|0,1],C[0,0]] | (0,0,0,0,1,0) | 9 |

Full data: see `bridge_AXXC.csv`.
