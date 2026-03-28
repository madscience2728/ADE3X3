# BX Orbit Verification Report

Generated: 2026-03-27 15:09:02

## Discrepancy Statement

- Earlier recorded value: 18 orbits
- Current recomputed value: 10 orbits

## Verification Result

- Raw BX configs: 729
- Recomputed orbit count: 10
- Status: **Current recomputed value 10 orbits is CORRECT**

The earlier 18-orbit value appears to have been computed with incorrect
action logic (likely using A-action instead of B-action on the B slot).

## Current BX Orbit Representatives

| orbit_id | rep_readable | size | stab |
|----------|--------------|------|------|
| 0 | BX[B[0,0],X[0,0|0,0]] | 27 | 8 |
| 1 | BX[B[0,0],X[0,0|0,1]] | 54 | 4 |
| 2 | BX[B[0,0],X[0,0|1,0]] | 54 | 4 |
| 3 | BX[B[0,0],X[0,0|1,1]] | 108 | 2 |
| 4 | BX[B[0,0],X[0,1|0,0]] | 54 | 4 |
| 5 | BX[B[0,0],X[0,1|0,1]] | 108 | 2 |
| 6 | BX[B[0,0],X[0,1|1,0]] | 54 | 4 |
| 7 | BX[B[0,0],X[0,1|1,1]] | 108 | 2 |
| 8 | BX[B[0,0],X[0,1|2,0]] | 54 | 4 |
| 9 | BX[B[0,0],X[0,1|2,1]] | 108 | 2 |

## AX vs BX Comparison

- AX orbits: 10
- BX orbits: 10
(BX uses pi_shared/pi_cB, AX uses pi_rA/pi_shared - different action)
