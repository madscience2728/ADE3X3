# AX Refinement Workbench Report

**Generated:** 2026-03-27 23:56:45

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Schema | AX |
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
| 1 | x_t | 1 | 10 | 0 | 4 |
| 2 | a_col_eq_x_s+x_t | 2 | 10 | 0 | 4 |
| 3 | a_r_a+x_t | 2 | 10 | 0 | 4 |
| 4 | a_row_eq_x_r+x_t | 2 | 10 | 0 | 4 |
| 5 | a_s_a+x_t | 2 | 10 | 0 | 4 |
| 6 | x_live+x_t | 2 | 10 | 0 | 4 |
| 7 | x_r+x_t | 2 | 10 | 0 | 4 |
| 8 | x_s+x_t | 2 | 10 | 0 | 4 |
| 9 | x_t+x_u | 2 | 10 | 0 | 4 |
| 10 | a_col_eq_x_s | 1 | 8 | 2 | 2 |
| 11 | a_r_a | 1 | 8 | 2 | 2 |
| 12 | a_row_eq_x_r | 1 | 8 | 2 | 2 |
| 13 | a_s_a | 1 | 8 | 2 | 2 |
| 14 | x_live | 1 | 8 | 2 | 2 |
| 15 | x_r | 1 | 8 | 2 | 2 |
| 16 | x_s | 1 | 8 | 2 | 2 |
| 17 | x_u | 1 | 8 | 2 | 2 |
| 18 | a_col_eq_x_s+a_r_a | 2 | 8 | 2 | 2 |
| 19 | a_col_eq_x_s+a_row_eq_x_r | 2 | 8 | 2 | 2 |
| 20 | a_col_eq_x_s+a_s_a | 2 | 8 | 2 | 2 |

**Best candidate** resolves 100.0% of collisions (4 / 4).

## 4. Full Resolution

**9 feature set(s) fully resolve all collisions.**

| Features | Count |
|----------|-------|
| x_t | 1 |
| a_col_eq_x_s+x_t | 2 |
| a_r_a+x_t | 2 |
| a_row_eq_x_r+x_t | 2 |
| a_s_a+x_t | 2 |
| x_live+x_t | 2 |
| x_r+x_t | 2 |
| x_s+x_t | 2 |
| x_t+x_u | 2 |

- Order-1 full resolvers: 1
- Order-2 full resolvers: 8

## 5. Summary

- 45 candidate feature sets evaluated (9 order-1, 36 order-2)
- 9 feature set(s) achieve full resolution
- Best overall: x_t (order 1, 4 reduction)