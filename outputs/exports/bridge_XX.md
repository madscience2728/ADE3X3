# Typed/Raw Bridge: XX

Generated: 2026-03-27 17:01:00

Explicit typed/raw bridge for schema XX = X x X.

## Summary

| Property | Value |
|----------|-------|
| Typed arity | 2 |
| Raw arity | 4 |
| Typed config count | 6,561 |
| Image size | 6,561 |
| Injective | yes |
| Role overlay | `('A_X1','B_X1','A_X2','B_X2')` |

## Bridge Rule

Each X[r,s|t,u] expands to raw symbols:
- A_idx = 3*r + s
- B_idx = 3*t + u

XX config (X1, X2) expands to raw tuple:
- (A_idx(X1), B_idx(X1), A_idx(X2), B_idx(X2))

## Preview (first 10 rows)

| xx_config_id | xx_readable | raw_tuple | raw_tuple_id |
|--------------|-------------|-----------|--------------|
| 0 | XX[X[0,0|0,0],X[0,0|0,0]] | (0,0,0,0) | 0 |
| 1 | XX[X[0,0|0,0],X[0,0|0,1]] | (0,0,0,1) | 1 |
| 2 | XX[X[0,0|0,0],X[0,0|0,2]] | (0,0,0,2) | 2 |
| 3 | XX[X[0,0|0,0],X[0,0|1,0]] | (0,0,0,3) | 3 |
| 4 | XX[X[0,0|0,0],X[0,0|1,1]] | (0,0,0,4) | 4 |
| 5 | XX[X[0,0|0,0],X[0,0|1,2]] | (0,0,0,5) | 5 |
| 6 | XX[X[0,0|0,0],X[0,0|2,0]] | (0,0,0,6) | 6 |
| 7 | XX[X[0,0|0,0],X[0,0|2,1]] | (0,0,0,7) | 7 |
| 8 | XX[X[0,0|0,0],X[0,0|2,2]] | (0,0,0,8) | 8 |
| 9 | XX[X[0,0|0,0],X[0,1|0,0]] | (0,0,1,0) | 9 |

Full data: see `bridge_XX.csv`.
