# Schema Registration: CXXC

Generated: 2026-03-27 17:29:52

CXXC is the first typed schema beyond the current arity-3 core.

## Schema Definition

| Property | Value |
|----------|-------|
| Typed schema | C × X × X × C |
| Typed arity | 4 |
| Raw arity | 6 |
| Typed config count | 531,441 |
| Role overlay | `('C','A_X1','B_X1','A_X2','B_X2','C')` |

## Bridge Rule

Each C expands to 1 raw slot. Each X expands to 2 raw slots.

CXXC config (c1, x1, x2, c2) expands to raw tuple:
- (C_idx(c1), A_idx(x1), B_idx(x1), A_idx(x2), B_idx(x2), C_idx(c2))

## Export Scope

bridge_CXXC.csv contains the **full population**: all 531,441 rows.

## Preview (first 10 rows)

| cxxc_config_id | cxxc_readable | raw_tuple | raw_tuple_id |
|----------------|---------------|-----------|--------------|
| 0 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,0]] | (0,0,0,0,0,0) | 0 |
| 1 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,1]] | (0,0,0,0,0,1) | 1 |
| 2 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,2]] | (0,0,0,0,0,2) | 2 |
| 3 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,0]] | (0,0,0,0,0,3) | 3 |
| 4 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,1]] | (0,0,0,0,0,4) | 4 |
| 5 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,2]] | (0,0,0,0,0,5) | 5 |
| 6 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[2,0]] | (0,0,0,0,0,6) | 6 |
| 7 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[2,1]] | (0,0,0,0,0,7) | 7 |
| 8 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[2,2]] | (0,0,0,0,0,8) | 8 |
| 9 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,0]] | (0,0,0,0,1,0) | 9 |

Full data: see `bridge_CXXC.csv`.
