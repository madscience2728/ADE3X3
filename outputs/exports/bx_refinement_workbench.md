# BX Refinement Workbench Report

**Generated:** 2026-03-28 00:19:12

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Schema | BX |
| Total orbits | 10 |
| Base distinct signatures | 8 |
| Base collisions | 4 |
| Collision groups | 2 |
| Candidate features | 9 |
| Max feature order | 2 |
| Candidates evaluated | 45 |

## 2. Collision Structure

- 2 group(s) of size 2

**Total collisions:** 4 (sum of group sizes)

## 3. Top Candidates by Resolving Power

| Rank | Features | Count | Distinct | Remaining | Reduction |
|------|----------|-------|----------|-----------|-----------|
| 1 | x_s | 1 | 10 | 0 | 4 |
| 2 | x_t | 1 | 10 | 0 | 4 |
| 3 | b_col_eq_x_u+x_s | 2 | 10 | 0 | 4 |
| 4 | b_col_eq_x_u+x_t | 2 | 10 | 0 | 4 |
| 5 | b_row_eq_x_t+x_s | 2 | 10 | 0 | 4 |
| 6 | b_row_eq_x_t+x_t | 2 | 10 | 0 | 4 |
| 7 | b_t_b+x_s | 2 | 10 | 0 | 4 |
| 8 | b_t_b+x_t | 2 | 10 | 0 | 4 |
| 9 | b_u_b+x_s | 2 | 10 | 0 | 4 |
| 10 | b_u_b+x_t | 2 | 10 | 0 | 4 |
| 11 | x_live+x_s | 2 | 10 | 0 | 4 |
| 12 | x_live+x_t | 2 | 10 | 0 | 4 |
| 13 | x_r+x_s | 2 | 10 | 0 | 4 |
| 14 | x_r+x_t | 2 | 10 | 0 | 4 |
| 15 | x_s+x_t | 2 | 10 | 0 | 4 |
| 16 | x_s+x_u | 2 | 10 | 0 | 4 |
| 17 | x_t+x_u | 2 | 10 | 0 | 4 |
| 18 | b_col_eq_x_u | 1 | 8 | 2 | 2 |
| 19 | b_row_eq_x_t | 1 | 8 | 2 | 2 |
| 20 | b_t_b | 1 | 8 | 2 | 2 |

**Best candidate** resolves 100.0% of collisions (4 / 4).

## 4. Full Resolution

**17 feature set(s) fully resolve all collisions.**

| Features | Count |
|----------|-------|
| x_s | 1 |
| x_t | 1 |
| b_col_eq_x_u+x_s | 2 |
| b_col_eq_x_u+x_t | 2 |
| b_row_eq_x_t+x_s | 2 |
| b_row_eq_x_t+x_t | 2 |
| b_t_b+x_s | 2 |
| b_t_b+x_t | 2 |
| b_u_b+x_s | 2 |
| b_u_b+x_t | 2 |
| x_live+x_s | 2 |
| x_live+x_t | 2 |
| x_r+x_s | 2 |
| x_r+x_t | 2 |
| x_s+x_t | 2 |
| x_s+x_u | 2 |
| x_t+x_u | 2 |

- Order-1 full resolvers: 2
- Order-2 full resolvers: 15

## 5. Summary

- 45 candidate feature sets evaluated (9 order-1, 36 order-2)
- 17 feature set(s) achieve full resolution
- Best overall: x_s (order 1, 4 reduction)