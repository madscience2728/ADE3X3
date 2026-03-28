# Step 45: 58-Orbit Closure + Doubly-Live Floor Core
Generated: 2026-03-28T12:43:52

[EXACT_DERIVED]

## Task 1: Closure Iteration Starting from the 58-Orbit Candidate

| closure_step | set_size | pairs_checked | pairs_inside | pairs_outside | new_escapes |
|--------------|----------|---------------|--------------|---------------|-------------|
| 1 | 58 | 56,889 | 49,113 | 7,776 | 6 |
| 2 | 64 | 86,265 | 86,265 | 0 | 0 |

Step 1 new escape ids: [185, 191, 296, 1031, 1100, 1175]
Stop reason: closed
Final checked set size: 64

## Task 2: Doubly-Live Floor Core

Doubly-live floor orbits: 28
- Same target fiber: 10 / 28
- Focused on a boundary C atom: 6 / 28

| orbit_id | stab_type | target_of_x1 | target_of_x2 | same_fiber | focused_on_boundary |
|----------|-----------|--------------|--------------|------------|---------------------|
| 0 | (Z2)^3 | C[0,0] | C[0,0] | True | True |
| 1 | Z2xZ2 | C[0,0] | C[0,0] | True | True |
| 2 | Z2xZ2 | C[0,0] | C[0,0] | True | True |
| 4 | Z2xZ2 | C[0,0] | C[0,1] | False | False |
| 5 | Z2xZ2 | C[0,0] | C[0,1] | False | False |
| 6 | Z2xZ2 | C[0,0] | C[0,1] | False | False |
| 30 | Z2xZ2 | C[0,0] | C[0,0] | True | True |
| 50 | Z2xZ2 | C[0,0] | C[1,0] | False | False |
| 52 | Z2xZ2 | C[0,0] | C[1,0] | False | False |
| 54 | Z2xZ2 | C[0,0] | C[1,0] | False | False |
| 125 | Z2xZ2 | C[0,1] | C[0,0] | False | False |
| 126 | Z2xZ2 | C[0,1] | C[0,0] | False | False |
| 127 | Z2xZ2 | C[0,1] | C[0,0] | False | False |
| 131 | Z2xZ2 | C[0,1] | C[0,1] | True | False |
| 132 | Z2xZ2 | C[0,1] | C[0,1] | True | True |
| 133 | Z2xZ2 | C[0,1] | C[0,1] | True | False |
| 137 | Z2xZ2 | C[0,1] | C[0,2] | False | False |
| 138 | Z2xZ2 | C[0,1] | C[0,2] | False | False |
| 139 | Z2xZ2 | C[0,1] | C[0,2] | False | False |
| 980 | Z2xZ2 | C[1,0] | C[0,0] | False | False |
| 982 | Z2xZ2 | C[1,0] | C[0,0] | False | False |
| 984 | Z2xZ2 | C[1,0] | C[0,0] | False | False |
| 1055 | Z2xZ2 | C[1,0] | C[1,0] | True | False |
| 1057 | Z2xZ2 | C[1,0] | C[1,0] | True | True |
| 1059 | Z2xZ2 | C[1,0] | C[1,0] | True | False |
| 1130 | Z2xZ2 | C[1,0] | C[2,0] | False | False |
| 1132 | Z2xZ2 | C[1,0] | C[2,0] | False | False |
| 1134 | Z2xZ2 | C[1,0] | C[2,0] | False | False |

## Task 2c: Doubly-Live Core Composition

| output_class | pair_count | fraction |
|--------------|------------|----------|
| doubly_live_floor | 4,563 | 0.875648 |
| doubly_live_nonfloor | 648 | 0.124352 |
| singly_live | 0 | 0.000000 |
| dead | 0 | 0.000000 |

## Task 3: Fixed-Point Subspace Dimensions on X

Computed rows: 39 Z2xZ2 floor stabilizers
- Fixed dimensions observed: [30, 36]
- Non-fixed dimensions observed: [45, 51]

[INTERPRETATION]

The live-core rows separate two geometric motifs: same-fiber pairs, where both
bilinear terms feed the same output C atom, and spread pairs, where the two live
terms feed different C atoms. The closure table shows whether the 58-orbit layer
stabilizes as a small sub-algebra or continues expanding; the fixed-space data shows
how much of the 81-dimensional X space remains rigid under each floor stabilizer.