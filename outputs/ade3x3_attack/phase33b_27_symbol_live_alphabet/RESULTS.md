# Phase 33b Results: 27-Symbol Live Alphabet

## 33b-1. L Schema Orbit Roster

[GROUND_TRUTH] The live alphabet is L[r,s,u] with local index 9*r + 3*s + u, identified with the live X atom X[r,s|s,u].

[EXACT_DERIVED] Under the independent action of S3 on row, channel, and output-column coordinates, the orbit invariants are the partition patterns of the row tuple, channel tuple, and column tuple separately. Cross-coordinate predicates such as r == u are not orbit-invariant under this action.

| schema | raw count | predicted orbits | computed orbits | distinct signatures | verified |
|--------|-----------|------------------|-----------------|---------------------|----------|
| L | 27 | 1 | 1 | 1 | True |
| LL | 729 | 8 | 8 | 8 | True |
| LLL | 19,683 | 125 | 125 | 125 | True |

### L

| orbit_id | rep_config_id | rep_readable | orbit_size | stabilizer_size | signature_key |
|----------|---------------|--------------|------------|-----------------|---------------|
| 0 | 0 | L[0,0,0] | 27 | 8 | rows=(0,); channels=(0,); cols=(0,) |

### LL

| orbit_id | rep_config_id | rep_readable | orbit_size | stabilizer_size | signature_key |
|----------|---------------|--------------|------------|-----------------|---------------|
| 0 | 0 | LL[L[0,0,0],L[0,0,0]] | 27 | 8 | rows=(0, 0); channels=(0, 0); cols=(0, 0) |
| 1 | 1 | LL[L[0,0,0],L[0,0,1]] | 54 | 4 | rows=(0, 0); channels=(0, 0); cols=(0, 1) |
| 2 | 3 | LL[L[0,0,0],L[0,1,0]] | 54 | 4 | rows=(0, 0); channels=(0, 1); cols=(0, 0) |
| 3 | 4 | LL[L[0,0,0],L[0,1,1]] | 108 | 2 | rows=(0, 0); channels=(0, 1); cols=(0, 1) |
| 4 | 9 | LL[L[0,0,0],L[1,0,0]] | 54 | 4 | rows=(0, 1); channels=(0, 0); cols=(0, 0) |
| 5 | 10 | LL[L[0,0,0],L[1,0,1]] | 108 | 2 | rows=(0, 1); channels=(0, 0); cols=(0, 1) |
| 6 | 12 | LL[L[0,0,0],L[1,1,0]] | 108 | 2 | rows=(0, 1); channels=(0, 1); cols=(0, 0) |
| 7 | 13 | LL[L[0,0,0],L[1,1,1]] | 216 | 1 | rows=(0, 1); channels=(0, 1); cols=(0, 1) |

### LLL

| orbit_id | rep_config_id | rep_readable | orbit_size | stabilizer_size | signature_key |
|----------|---------------|--------------|------------|-----------------|---------------|
| 0 | 0 | LLL[L[0,0,0],L[0,0,0],L[0,0,0]] | 27 | 8 | rows=(0, 0, 0); channels=(0, 0, 0); cols=(0, 0, 0) |
| 1 | 1 | LLL[L[0,0,0],L[0,0,0],L[0,0,1]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 0, 0); cols=(0, 0, 1) |
| 2 | 3 | LLL[L[0,0,0],L[0,0,0],L[0,1,0]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 0, 1); cols=(0, 0, 0) |
| 3 | 4 | LLL[L[0,0,0],L[0,0,0],L[0,1,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 0, 1); cols=(0, 0, 1) |
| 4 | 9 | LLL[L[0,0,0],L[0,0,0],L[1,0,0]] | 54 | 4 | rows=(0, 0, 1); channels=(0, 0, 0); cols=(0, 0, 0) |
| 5 | 10 | LLL[L[0,0,0],L[0,0,0],L[1,0,1]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 0, 0); cols=(0, 0, 1) |
| 6 | 12 | LLL[L[0,0,0],L[0,0,0],L[1,1,0]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 0, 1); cols=(0, 0, 0) |
| 7 | 13 | LLL[L[0,0,0],L[0,0,0],L[1,1,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 0, 1); cols=(0, 0, 1) |
| 8 | 27 | LLL[L[0,0,0],L[0,0,1],L[0,0,0]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 0, 0); cols=(0, 1, 0) |
| 9 | 28 | LLL[L[0,0,0],L[0,0,1],L[0,0,1]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 0, 0); cols=(0, 1, 1) |
| 10 | 29 | LLL[L[0,0,0],L[0,0,1],L[0,0,2]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 0, 0); cols=(0, 1, 2) |
| 11 | 30 | LLL[L[0,0,0],L[0,0,1],L[0,1,0]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 0, 1); cols=(0, 1, 0) |
| 12 | 31 | LLL[L[0,0,0],L[0,0,1],L[0,1,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 0, 1); cols=(0, 1, 1) |
| 13 | 32 | LLL[L[0,0,0],L[0,0,1],L[0,1,2]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 0, 1); cols=(0, 1, 2) |
| 14 | 36 | LLL[L[0,0,0],L[0,0,1],L[1,0,0]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 0, 0); cols=(0, 1, 0) |
| 15 | 37 | LLL[L[0,0,0],L[0,0,1],L[1,0,1]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 0, 0); cols=(0, 1, 1) |
| 16 | 38 | LLL[L[0,0,0],L[0,0,1],L[1,0,2]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 0, 0); cols=(0, 1, 2) |
| 17 | 39 | LLL[L[0,0,0],L[0,0,1],L[1,1,0]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 0, 1); cols=(0, 1, 0) |
| 18 | 40 | LLL[L[0,0,0],L[0,0,1],L[1,1,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 0, 1); cols=(0, 1, 1) |
| 19 | 41 | LLL[L[0,0,0],L[0,0,1],L[1,1,2]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 0, 1); cols=(0, 1, 2) |
| 20 | 81 | LLL[L[0,0,0],L[0,1,0],L[0,0,0]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 1, 0); cols=(0, 0, 0) |
| 21 | 82 | LLL[L[0,0,0],L[0,1,0],L[0,0,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 0); cols=(0, 0, 1) |
| 22 | 84 | LLL[L[0,0,0],L[0,1,0],L[0,1,0]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 1, 1); cols=(0, 0, 0) |
| 23 | 85 | LLL[L[0,0,0],L[0,1,0],L[0,1,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 1); cols=(0, 0, 1) |
| 24 | 87 | LLL[L[0,0,0],L[0,1,0],L[0,2,0]] | 54 | 4 | rows=(0, 0, 0); channels=(0, 1, 2); cols=(0, 0, 0) |
| 25 | 88 | LLL[L[0,0,0],L[0,1,0],L[0,2,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 2); cols=(0, 0, 1) |
| 26 | 90 | LLL[L[0,0,0],L[0,1,0],L[1,0,0]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 1, 0); cols=(0, 0, 0) |
| 27 | 91 | LLL[L[0,0,0],L[0,1,0],L[1,0,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 0); cols=(0, 0, 1) |
| 28 | 93 | LLL[L[0,0,0],L[0,1,0],L[1,1,0]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 1, 1); cols=(0, 0, 0) |
| 29 | 94 | LLL[L[0,0,0],L[0,1,0],L[1,1,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 1); cols=(0, 0, 1) |
| 30 | 96 | LLL[L[0,0,0],L[0,1,0],L[1,2,0]] | 108 | 2 | rows=(0, 0, 1); channels=(0, 1, 2); cols=(0, 0, 0) |
| 31 | 97 | LLL[L[0,0,0],L[0,1,0],L[1,2,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 2); cols=(0, 0, 1) |
| 32 | 108 | LLL[L[0,0,0],L[0,1,1],L[0,0,0]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 0); cols=(0, 1, 0) |
| 33 | 109 | LLL[L[0,0,0],L[0,1,1],L[0,0,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 0); cols=(0, 1, 1) |
| 34 | 110 | LLL[L[0,0,0],L[0,1,1],L[0,0,2]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 0); cols=(0, 1, 2) |
| 35 | 111 | LLL[L[0,0,0],L[0,1,1],L[0,1,0]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 1); cols=(0, 1, 0) |
| 36 | 112 | LLL[L[0,0,0],L[0,1,1],L[0,1,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 1); cols=(0, 1, 1) |
| 37 | 113 | LLL[L[0,0,0],L[0,1,1],L[0,1,2]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 1); cols=(0, 1, 2) |
| 38 | 114 | LLL[L[0,0,0],L[0,1,1],L[0,2,0]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 2); cols=(0, 1, 0) |
| 39 | 115 | LLL[L[0,0,0],L[0,1,1],L[0,2,1]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 2); cols=(0, 1, 1) |
| 40 | 116 | LLL[L[0,0,0],L[0,1,1],L[0,2,2]] | 108 | 2 | rows=(0, 0, 0); channels=(0, 1, 2); cols=(0, 1, 2) |
| 41 | 117 | LLL[L[0,0,0],L[0,1,1],L[1,0,0]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 0); cols=(0, 1, 0) |
| 42 | 118 | LLL[L[0,0,0],L[0,1,1],L[1,0,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 0); cols=(0, 1, 1) |
| 43 | 119 | LLL[L[0,0,0],L[0,1,1],L[1,0,2]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 0); cols=(0, 1, 2) |
| 44 | 120 | LLL[L[0,0,0],L[0,1,1],L[1,1,0]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 1); cols=(0, 1, 0) |
| 45 | 121 | LLL[L[0,0,0],L[0,1,1],L[1,1,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 1); cols=(0, 1, 1) |
| 46 | 122 | LLL[L[0,0,0],L[0,1,1],L[1,1,2]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 1); cols=(0, 1, 2) |
| 47 | 123 | LLL[L[0,0,0],L[0,1,1],L[1,2,0]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 2); cols=(0, 1, 0) |
| 48 | 124 | LLL[L[0,0,0],L[0,1,1],L[1,2,1]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 2); cols=(0, 1, 1) |
| 49 | 125 | LLL[L[0,0,0],L[0,1,1],L[1,2,2]] | 216 | 1 | rows=(0, 0, 1); channels=(0, 1, 2); cols=(0, 1, 2) |
| 50 | 243 | LLL[L[0,0,0],L[1,0,0],L[0,0,0]] | 54 | 4 | rows=(0, 1, 0); channels=(0, 0, 0); cols=(0, 0, 0) |
| 51 | 244 | LLL[L[0,0,0],L[1,0,0],L[0,0,1]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 0, 0); cols=(0, 0, 1) |
| 52 | 246 | LLL[L[0,0,0],L[1,0,0],L[0,1,0]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 0, 1); cols=(0, 0, 0) |
| 53 | 247 | LLL[L[0,0,0],L[1,0,0],L[0,1,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 0, 1); cols=(0, 0, 1) |
| 54 | 252 | LLL[L[0,0,0],L[1,0,0],L[1,0,0]] | 54 | 4 | rows=(0, 1, 1); channels=(0, 0, 0); cols=(0, 0, 0) |
| 55 | 253 | LLL[L[0,0,0],L[1,0,0],L[1,0,1]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 0, 0); cols=(0, 0, 1) |
| 56 | 255 | LLL[L[0,0,0],L[1,0,0],L[1,1,0]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 0, 1); cols=(0, 0, 0) |
| 57 | 256 | LLL[L[0,0,0],L[1,0,0],L[1,1,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 0, 1); cols=(0, 0, 1) |
| 58 | 261 | LLL[L[0,0,0],L[1,0,0],L[2,0,0]] | 54 | 4 | rows=(0, 1, 2); channels=(0, 0, 0); cols=(0, 0, 0) |
| 59 | 262 | LLL[L[0,0,0],L[1,0,0],L[2,0,1]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 0, 0); cols=(0, 0, 1) |
| 60 | 264 | LLL[L[0,0,0],L[1,0,0],L[2,1,0]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 0, 1); cols=(0, 0, 0) |
| 61 | 265 | LLL[L[0,0,0],L[1,0,0],L[2,1,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 0, 1); cols=(0, 0, 1) |
| 62 | 270 | LLL[L[0,0,0],L[1,0,1],L[0,0,0]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 0, 0); cols=(0, 1, 0) |
| 63 | 271 | LLL[L[0,0,0],L[1,0,1],L[0,0,1]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 0, 0); cols=(0, 1, 1) |
| 64 | 272 | LLL[L[0,0,0],L[1,0,1],L[0,0,2]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 0, 0); cols=(0, 1, 2) |
| 65 | 273 | LLL[L[0,0,0],L[1,0,1],L[0,1,0]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 0, 1); cols=(0, 1, 0) |
| 66 | 274 | LLL[L[0,0,0],L[1,0,1],L[0,1,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 0, 1); cols=(0, 1, 1) |
| 67 | 275 | LLL[L[0,0,0],L[1,0,1],L[0,1,2]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 0, 1); cols=(0, 1, 2) |
| 68 | 279 | LLL[L[0,0,0],L[1,0,1],L[1,0,0]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 0, 0); cols=(0, 1, 0) |
| 69 | 280 | LLL[L[0,0,0],L[1,0,1],L[1,0,1]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 0, 0); cols=(0, 1, 1) |
| 70 | 281 | LLL[L[0,0,0],L[1,0,1],L[1,0,2]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 0, 0); cols=(0, 1, 2) |
| 71 | 282 | LLL[L[0,0,0],L[1,0,1],L[1,1,0]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 0, 1); cols=(0, 1, 0) |
| 72 | 283 | LLL[L[0,0,0],L[1,0,1],L[1,1,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 0, 1); cols=(0, 1, 1) |
| 73 | 284 | LLL[L[0,0,0],L[1,0,1],L[1,1,2]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 0, 1); cols=(0, 1, 2) |
| 74 | 288 | LLL[L[0,0,0],L[1,0,1],L[2,0,0]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 0, 0); cols=(0, 1, 0) |
| 75 | 289 | LLL[L[0,0,0],L[1,0,1],L[2,0,1]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 0, 0); cols=(0, 1, 1) |
| 76 | 290 | LLL[L[0,0,0],L[1,0,1],L[2,0,2]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 0, 0); cols=(0, 1, 2) |
| 77 | 291 | LLL[L[0,0,0],L[1,0,1],L[2,1,0]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 0, 1); cols=(0, 1, 0) |
| 78 | 292 | LLL[L[0,0,0],L[1,0,1],L[2,1,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 0, 1); cols=(0, 1, 1) |
| 79 | 293 | LLL[L[0,0,0],L[1,0,1],L[2,1,2]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 0, 1); cols=(0, 1, 2) |
| 80 | 324 | LLL[L[0,0,0],L[1,1,0],L[0,0,0]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 1, 0); cols=(0, 0, 0) |
| 81 | 325 | LLL[L[0,0,0],L[1,1,0],L[0,0,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 0); cols=(0, 0, 1) |
| 82 | 327 | LLL[L[0,0,0],L[1,1,0],L[0,1,0]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 1, 1); cols=(0, 0, 0) |
| 83 | 328 | LLL[L[0,0,0],L[1,1,0],L[0,1,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 1); cols=(0, 0, 1) |
| 84 | 330 | LLL[L[0,0,0],L[1,1,0],L[0,2,0]] | 108 | 2 | rows=(0, 1, 0); channels=(0, 1, 2); cols=(0, 0, 0) |
| 85 | 331 | LLL[L[0,0,0],L[1,1,0],L[0,2,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 2); cols=(0, 0, 1) |
| 86 | 333 | LLL[L[0,0,0],L[1,1,0],L[1,0,0]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 1, 0); cols=(0, 0, 0) |
| 87 | 334 | LLL[L[0,0,0],L[1,1,0],L[1,0,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 0); cols=(0, 0, 1) |
| 88 | 336 | LLL[L[0,0,0],L[1,1,0],L[1,1,0]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 1, 1); cols=(0, 0, 0) |
| 89 | 337 | LLL[L[0,0,0],L[1,1,0],L[1,1,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 1); cols=(0, 0, 1) |
| 90 | 339 | LLL[L[0,0,0],L[1,1,0],L[1,2,0]] | 108 | 2 | rows=(0, 1, 1); channels=(0, 1, 2); cols=(0, 0, 0) |
| 91 | 340 | LLL[L[0,0,0],L[1,1,0],L[1,2,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 2); cols=(0, 0, 1) |
| 92 | 342 | LLL[L[0,0,0],L[1,1,0],L[2,0,0]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 1, 0); cols=(0, 0, 0) |
| 93 | 343 | LLL[L[0,0,0],L[1,1,0],L[2,0,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 0); cols=(0, 0, 1) |
| 94 | 345 | LLL[L[0,0,0],L[1,1,0],L[2,1,0]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 1, 1); cols=(0, 0, 0) |
| 95 | 346 | LLL[L[0,0,0],L[1,1,0],L[2,1,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 1); cols=(0, 0, 1) |
| 96 | 348 | LLL[L[0,0,0],L[1,1,0],L[2,2,0]] | 108 | 2 | rows=(0, 1, 2); channels=(0, 1, 2); cols=(0, 0, 0) |
| 97 | 349 | LLL[L[0,0,0],L[1,1,0],L[2,2,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 2); cols=(0, 0, 1) |
| 98 | 351 | LLL[L[0,0,0],L[1,1,1],L[0,0,0]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 0); cols=(0, 1, 0) |
| 99 | 352 | LLL[L[0,0,0],L[1,1,1],L[0,0,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 0); cols=(0, 1, 1) |
| 100 | 353 | LLL[L[0,0,0],L[1,1,1],L[0,0,2]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 0); cols=(0, 1, 2) |
| 101 | 354 | LLL[L[0,0,0],L[1,1,1],L[0,1,0]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 1); cols=(0, 1, 0) |
| 102 | 355 | LLL[L[0,0,0],L[1,1,1],L[0,1,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 1); cols=(0, 1, 1) |
| 103 | 356 | LLL[L[0,0,0],L[1,1,1],L[0,1,2]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 1); cols=(0, 1, 2) |
| 104 | 357 | LLL[L[0,0,0],L[1,1,1],L[0,2,0]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 2); cols=(0, 1, 0) |
| 105 | 358 | LLL[L[0,0,0],L[1,1,1],L[0,2,1]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 2); cols=(0, 1, 1) |
| 106 | 359 | LLL[L[0,0,0],L[1,1,1],L[0,2,2]] | 216 | 1 | rows=(0, 1, 0); channels=(0, 1, 2); cols=(0, 1, 2) |
| 107 | 360 | LLL[L[0,0,0],L[1,1,1],L[1,0,0]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 0); cols=(0, 1, 0) |
| 108 | 361 | LLL[L[0,0,0],L[1,1,1],L[1,0,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 0); cols=(0, 1, 1) |
| 109 | 362 | LLL[L[0,0,0],L[1,1,1],L[1,0,2]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 0); cols=(0, 1, 2) |
| 110 | 363 | LLL[L[0,0,0],L[1,1,1],L[1,1,0]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 1); cols=(0, 1, 0) |
| 111 | 364 | LLL[L[0,0,0],L[1,1,1],L[1,1,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 1); cols=(0, 1, 1) |
| 112 | 365 | LLL[L[0,0,0],L[1,1,1],L[1,1,2]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 1); cols=(0, 1, 2) |
| 113 | 366 | LLL[L[0,0,0],L[1,1,1],L[1,2,0]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 2); cols=(0, 1, 0) |
| 114 | 367 | LLL[L[0,0,0],L[1,1,1],L[1,2,1]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 2); cols=(0, 1, 1) |
| 115 | 368 | LLL[L[0,0,0],L[1,1,1],L[1,2,2]] | 216 | 1 | rows=(0, 1, 1); channels=(0, 1, 2); cols=(0, 1, 2) |
| 116 | 369 | LLL[L[0,0,0],L[1,1,1],L[2,0,0]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 0); cols=(0, 1, 0) |
| 117 | 370 | LLL[L[0,0,0],L[1,1,1],L[2,0,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 0); cols=(0, 1, 1) |
| 118 | 371 | LLL[L[0,0,0],L[1,1,1],L[2,0,2]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 0); cols=(0, 1, 2) |
| 119 | 372 | LLL[L[0,0,0],L[1,1,1],L[2,1,0]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 1); cols=(0, 1, 0) |
| 120 | 373 | LLL[L[0,0,0],L[1,1,1],L[2,1,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 1); cols=(0, 1, 1) |
| 121 | 374 | LLL[L[0,0,0],L[1,1,1],L[2,1,2]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 1); cols=(0, 1, 2) |
| 122 | 375 | LLL[L[0,0,0],L[1,1,1],L[2,2,0]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 2); cols=(0, 1, 0) |
| 123 | 376 | LLL[L[0,0,0],L[1,1,1],L[2,2,1]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 2); cols=(0, 1, 1) |
| 124 | 377 | LLL[L[0,0,0],L[1,1,1],L[2,2,2]] | 216 | 1 | rows=(0, 1, 2); channels=(0, 1, 2); cols=(0, 1, 2) |

## 33b-2. Bridge Maps

[EXACT_DERIVED] L -> X hits the 27 live X atoms exactly once: True. The group action commutes with the bridge: True.

[MEASURED_FROM_CODE] LL maps into 8 distinct XX orbits and 4 distinct CX orbits. The XX non-image orbit ids are [2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]. The direct LL -> CX bijection claim is therefore False, because the C projection forgets one channel coordinate. The LLLL -> CXXC orbit-key bijection verifies True across all 2,744 orbits.

| ll_orbit_id | ll_rep_readable | xx_orbit_id | xx_rep_readable | cx_orbit_id | cx_rep_readable |
|-------------|-----------------|-------------|-----------------|-------------|-----------------|
| 0 | LL[L[0,0,0],L[0,0,0]] | 0 | XX[X[0,0|0,0],X[0,0|0,0]] | 0 | CX[C[0,0],X[0,0|0,0]] |
| 1 | LL[L[0,0,0],L[0,0,1]] | 1 | XX[X[0,0|0,0],X[0,0|0,1]] | 1 | CX[C[0,0],X[0,0|0,1]] |
| 2 | LL[L[0,0,0],L[0,1,0]] | 6 | XX[X[0,0|0,0],X[0,1|1,0]] | 0 | CX[C[0,0],X[0,0|0,0]] |
| 3 | LL[L[0,0,0],L[0,1,1]] | 7 | XX[X[0,0|0,0],X[0,1|1,1]] | 1 | CX[C[0,0],X[0,0|0,1]] |
| 4 | LL[L[0,0,0],L[1,0,0]] | 10 | XX[X[0,0|0,0],X[1,0|0,0]] | 4 | CX[C[0,0],X[1,0|0,0]] |
| 5 | LL[L[0,0,0],L[1,0,1]] | 11 | XX[X[0,0|0,0],X[1,0|0,1]] | 5 | CX[C[0,0],X[1,0|0,1]] |
| 6 | LL[L[0,0,0],L[1,1,0]] | 16 | XX[X[0,0|0,0],X[1,1|1,0]] | 4 | CX[C[0,0],X[1,0|0,0]] |
| 7 | LL[L[0,0,0],L[1,1,1]] | 17 | XX[X[0,0|0,0],X[1,1|1,1]] | 5 | CX[C[0,0],X[1,0|0,1]] |

Preview of the 2,744-row LLLL -> CXXC orbit map:

| llll_orbit_id | llll_rep_readable | cxxc_orbit_id | cxxc_rep_readable |
|---------------|-------------------|---------------|-------------------|
| 0 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[0,0,0]] | 0 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,0]] |
| 1 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[0,0,1]] | 1 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,1]] |
| 2 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[0,1,0]] | 10 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[0,0]] |
| 3 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[0,1,1]] | 11 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[0,1]] |
| 4 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[1,0,0]] | 2 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,0]] |
| 5 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[1,0,1]] | 3 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,1]] |
| 6 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[1,1,0]] | 12 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[1,0]] |
| 7 | LLLL[L[0,0,0],L[0,0,0],L[0,0,0],L[1,1,1]] | 13 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[1,1]] |
| 8 | LLLL[L[0,0,0],L[0,0,0],L[0,0,1],L[0,0,0]] | 4 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,0]] |
| 9 | LLLL[L[0,0,0],L[0,0,0],L[0,0,1],L[0,0,1]] | 5 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,1]] |
| 10 | LLLL[L[0,0,0],L[0,0,0],L[0,0,1],L[0,0,2]] | 6 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,2]] |
| 11 | LLLL[L[0,0,0],L[0,0,0],L[0,0,1],L[0,1,0]] | 14 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[0,0]] |

[MEASURED_FROM_CODE] Full orbit correspondence tables are exported in CSV artifacts in this session directory.

## 33b-3. Fiber Structure

[EXACT_DERIVED] The 27 L atoms partition into 9 fibers of size 3, indexed by C[r,u]. This matches the Step 51 live-fiber split exactly: sigma is the within-fiber sum, and eta1/eta2 are the two within-fiber differences.

| c_local_id | c_name | l0_name | l1_name | l2_name |
|------------|--------|---------|---------|---------|
| 0 | C[0,0] | L[0,0,0] | L[0,1,0] | L[0,2,0] |
| 1 | C[0,1] | L[0,0,1] | L[0,1,1] | L[0,2,1] |
| 2 | C[0,2] | L[0,0,2] | L[0,1,2] | L[0,2,2] |
| 3 | C[1,0] | L[1,0,0] | L[1,1,0] | L[1,2,0] |
| 4 | C[1,1] | L[1,0,1] | L[1,1,1] | L[1,2,1] |
| 5 | C[1,2] | L[1,0,2] | L[1,1,2] | L[1,2,2] |
| 6 | C[2,0] | L[2,0,0] | L[2,1,0] | L[2,2,0] |
| 7 | C[2,1] | L[2,0,1] | L[2,1,1] | L[2,2,1] |
| 8 | C[2,2] | L[2,0,2] | L[2,1,2] | L[2,2,2] |

## 33b-4. Recovery Matrices In L Coordinates

[MEASURED_FROM_CODE] Re-extracting the Phase 33 recovery matrices in the L basis keeps the same exact containments. The hand derivation now clarifies that non-fiber-diagonality is expected: recovery is genuinely cross-output.

| label | block | rank_M | nnz_M | density_M | coefficients | fiber_diagonal | fiber_witness |
|-------|-------|--------|-------|-----------|--------------|----------------|---------------|
| alphatensor_rank23 | D_01 | 2 | 7 | 0.043209876543209874 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [1, 2], "unexpected_rows": [2]} |
| alphatensor_rank23 | D_02 | 3 | 12 | 0.07407407407407407 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [1, 2, 13], "unexpected_rows": [2, 13]} |
| alphatensor_rank23 | D_10 | 2 | 8 | 0.04938271604938271 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [0, 2], "unexpected_rows": [0, 2]} |
| alphatensor_rank23 | D_12 | 2 | 5 | 0.030864197530864196 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [10, 11, 13], "unexpected_rows": [11, 13]} |
| alphatensor_rank23 | D_20 | 1 | 8 | 0.04938271604938271 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [9, 11], "unexpected_rows": [9, 11]} |
| alphatensor_rank23 | D_21 | 2 | 6 | 0.037037037037037035 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1, 10], "actual_rows": [11], "unexpected_rows": [11]} |
| standard_rank27 | D_01 | 0 | 0 | 0.0 | none | True |  |
| standard_rank27 | D_02 | 0 | 0 | 0.0 | none | True |  |
| standard_rank27 | D_10 | 0 | 0 | 0.0 | none | True |  |
| standard_rank27 | D_12 | 0 | 0 | 0.0 | none | True |  |
| standard_rank27 | D_20 | 0 | 0 | 0.0 | none | True |  |
| standard_rank27 | D_21 | 0 | 0 | 0.0 | none | True |  |
| strassen_2x2 | D_01 | 1 | 2 | 0.125 | -1, 1 | False | {"target_col": 1, "allowed_rows": [1], "actual_rows": [0, 1], "unexpected_rows": [0]} |
| strassen_2x2 | D_10 | 1 | 2 | 0.125 | 1 | False | {"target_col": 2, "allowed_rows": [2], "actual_rows": [0, 2], "unexpected_rows": [0]} |

## 33b-5. Defect Condition Scan

[EXACT_DERIVED] The exact defect condition is: Delta containment can fail only if there exists nonzero w in ker(Gamma) such that w^T H = 0. Equivalently, with W_s = sum_k w_k (alpha_k[:,s] otimes beta_k[s,:]), one must have W_0 = W_1 = ... = W_{n-1}.

[EXACT_DERIVED] The conservation law is best read as an L-alphabet compression theorem: R + eta_nullity = n^3 = |L|.

| case | n | R | n_cubed | ker_gamma_dim | H_cols | eta_nullity | conservation |
|------|---|---|---------|---------------|--------|-------------|--------------|
| 3x3 AlphaTensor | 3 | 23 | 27 | 14 | 18 | 4 | 23+4=27 ✓ |
| 3x3 Standard | 3 | 27 | 27 | 18 | 18 | 0 | 27+0=27 ✓ |
| 2x2 Strassen | 2 | 7 | 8 | 3 | 4 | 1 | 7+1=8 ✓ |

### Known Decompositions

| label | n | R | ker_gamma_dim | exact_defect_exists | exact_defect_nullity | softest_pairwise_norms | softest_max_pairwise_norm |
|-------|---|---|---------------|---------------------|----------------------|------------------------|---------------------------|
| alphatensor_rank23 | 3 | 23 | 14 | False | 0 | W_0-W_1=0.445537; W_1-W_2=0.438848 | 0.4455365956769507 |
| standard_rank27 | 3 | 27 | 18 | False | 0 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865478 |
| strassen_2x2 | 2 | 7 | 3 | False | 0 | W_0-W_1=1.414214 | 1.414213562373095 |

### Basis Vectors In ker(Gamma)

| label | basis_vector | pairwise_norms | max_pairwise_norm |
|-------|--------------|----------------|-------------------|
| alphatensor_rank23 | b0 | W_0-W_1=2.236068; W_1-W_2=0.577350 | 2.2360679774997902 |
| alphatensor_rank23 | b1 | W_0-W_1=2.236068; W_1-W_2=2.449490 | 2.4494897427831783 |
| alphatensor_rank23 | b2 | W_0-W_1=3.000000; W_1-W_2=1.290994 | 3.0000000000000004 |
| alphatensor_rank23 | b3 | W_0-W_1=1.290994; W_1-W_2=0.816497 | 1.2909944487358058 |
| alphatensor_rank23 | b4 | W_0-W_1=2.366432; W_1-W_2=1.000000 | 2.3664319132398464 |
| alphatensor_rank23 | b5 | W_0-W_1=3.605551; W_1-W_2=1.732051 | 3.60555127546399 |
| alphatensor_rank23 | b6 | W_0-W_1=1.732051; W_1-W_2=1.000000 | 1.7320508075688772 |
| alphatensor_rank23 | b7 | W_0-W_1=2.144761; W_1-W_2=1.183216 | 2.1447610589527217 |
| alphatensor_rank23 | b8 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| alphatensor_rank23 | b9 | W_0-W_1=2.500000; W_1-W_2=0.500000 | 2.5 |
| alphatensor_rank23 | b10 | W_0-W_1=2.291288; W_1-W_2=1.000000 | 2.29128784747792 |
| alphatensor_rank23 | b11 | W_0-W_1=3.559026; W_1-W_2=1.825742 | 3.5590260840104375 |
| alphatensor_rank23 | b12 | W_0-W_1=2.061553; W_1-W_2=1.224745 | 2.0615528128088303 |
| alphatensor_rank23 | b13 | W_0-W_1=2.397916; W_1-W_2=1.224745 | 2.3979157616563596 |
| standard_rank27 | b0 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b1 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b2 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b3 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b4 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b5 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b6 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b7 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b8 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b9 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b10 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b11 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b12 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b13 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b14 | W_0-W_1=1.414214; W_1-W_2=0.707107 | 1.414213562373095 |
| standard_rank27 | b15 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b16 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| standard_rank27 | b17 | W_0-W_1=0.707107; W_1-W_2=0.707107 | 0.7071067811865475 |
| strassen_2x2 | b0 | W_0-W_1=1.414214 | 1.4142135623730951 |
| strassen_2x2 | b1 | W_0-W_1=1.414214 | 1.4142135623730951 |
| strassen_2x2 | b2 | W_0-W_1=2.943920 | 2.9439202887759492 |

### Random Controls

| case | trials | exact_defect_exists_count | min_softest_max_pairwise_norm | median_softest_max_pairwise_norm | max_softest_max_pairwise_norm |
|------|--------|---------------------------|-------------------------------|----------------------------------|-------------------------------|
| random_2x2_R7 | 8 | 0 | 0.5359758891771664 | 0.8907677300202599 | 1.4983802390309195 |
| random_3x3_R23 | 8 | 0 | 0.19570732751134798 | 0.3850742225856755 | 0.6440686814053406 |
| random_3x3_R27 | 8 | 0 | 0.013767897963925402 | 0.09512481696172324 | 0.22346452922309457 |

| label | n | R | exact_defect_exists | exact_defect_nullity | softest_pairwise_norms | softest_max_pairwise_norm |
|-------|---|---|---------------------|----------------------|------------------------|---------------------------|
| random_n2_R7_trial0 | 2 | 7 | False | 0 | W_0-W_1=0.740131 | 0.7401314232935797 |
| random_n2_R7_trial1 | 2 | 7 | False | 0 | W_0-W_1=0.831372 | 0.8313715727215717 |
| random_n2_R7_trial2 | 2 | 7 | False | 0 | W_0-W_1=1.498380 | 1.4983802390309195 |
| random_n2_R7_trial3 | 2 | 7 | False | 0 | W_0-W_1=0.588721 | 0.5887208608442264 |
| random_n2_R7_trial4 | 2 | 7 | False | 0 | W_0-W_1=1.021490 | 1.0214899444944157 |
| random_n2_R7_trial5 | 2 | 7 | False | 0 | W_0-W_1=0.950164 | 0.950163887318948 |
| random_n2_R7_trial6 | 2 | 7 | False | 0 | W_0-W_1=0.535976 | 0.5359758891771664 |
| random_n2_R7_trial7 | 2 | 7 | False | 0 | W_0-W_1=1.032562 | 1.0325620332900953 |
| random_n3_R23_trial0 | 3 | 23 | False | 0 | W_0-W_1=0.195707; W_1-W_2=0.193769 | 0.19570732751134798 |
| random_n3_R23_trial1 | 3 | 23 | False | 0 | W_0-W_1=0.476965; W_1-W_2=0.475015 | 0.4769649768375833 |
| random_n3_R23_trial2 | 3 | 23 | False | 0 | W_0-W_1=0.310705; W_1-W_2=0.311840 | 0.3118402828576111 |
| random_n3_R23_trial3 | 3 | 23 | False | 0 | W_0-W_1=0.420129; W_1-W_2=0.419288 | 0.42012881567676397 |
| random_n3_R23_trial4 | 3 | 23 | False | 0 | W_0-W_1=0.644069; W_1-W_2=0.633707 | 0.6440686814053406 |
| random_n3_R23_trial5 | 3 | 23 | False | 0 | W_0-W_1=0.347286; W_1-W_2=0.350020 | 0.3500196294945869 |
| random_n3_R23_trial6 | 3 | 23 | False | 0 | W_0-W_1=0.329423; W_1-W_2=0.329488 | 0.32948789337669926 |
| random_n3_R23_trial7 | 3 | 23 | False | 0 | W_0-W_1=0.631337; W_1-W_2=0.639451 | 0.6394507905285788 |
| random_n3_R27_trial0 | 3 | 27 | False | 0 | W_0-W_1=0.015093; W_1-W_2=0.015294 | 0.015293651774172542 |
| random_n3_R27_trial1 | 3 | 27 | False | 0 | W_0-W_1=0.050864; W_1-W_2=0.050205 | 0.050863902324874886 |
| random_n3_R27_trial2 | 3 | 27 | False | 0 | W_0-W_1=0.107814; W_1-W_2=0.108619 | 0.1086186986664356 |
| random_n3_R27_trial3 | 3 | 27 | False | 0 | W_0-W_1=0.156312; W_1-W_2=0.157146 | 0.1571458818646808 |
| random_n3_R27_trial4 | 3 | 27 | False | 0 | W_0-W_1=0.088874; W_1-W_2=0.087478 | 0.08887433353084394 |
| random_n3_R27_trial5 | 3 | 27 | False | 0 | W_0-W_1=0.223465; W_1-W_2=0.221759 | 0.22346452922309457 |
| random_n3_R27_trial6 | 3 | 27 | False | 0 | W_0-W_1=0.101375; W_1-W_2=0.101316 | 0.10137530039260255 |
| random_n3_R27_trial7 | 3 | 27 | False | 0 | W_0-W_1=0.013768; W_1-W_2=0.013652 | 0.013767897963925402 |

## 33b-6. Interpretation

[INTERPRETATION] The 27-symbol alphabet is a cleaner schema for the live part of X: its orbit theory factorizes perfectly as partition patterns in row, channel, and output-column coordinates. This gives the exact cube counts 1, 8, 125, 2744 and puts the L alphabet on the same warehouse footing as the existing typed schemas.

[INTERPRETATION] The bridge results split cleanly but not exactly as proposed. LL sits inside XX orbit-by-orbit with 8 distinct image orbits, but the stated LL -> CX bridge collapses to only 4 CX orbits because converting the first live atom to C forgets its channel label. At arity 4, the strong statement is an orbit-key bijection with CXXC: the same factorized partition signatures govern both, even though the raw typed meanings differ.

[INTERPRETATION] The recovery matrices are not fiber-diagonal on the known decompositions, and that is correct. Strassen already shows recovery must mix H columns from different output fibers, so the dead-coordinate mechanism is structurally cross-output rather than fiber-local.

[INTERPRETATION] The conservation law now reads as compression: every valid n x n decomposition carries exactly |L| = n^3 units of information, split as R live coefficients plus eta_nullity nuisance slack. For 3x3 this constant is 27, the L-alphabet size itself.

[INTERPRETATION] The reported gap values are not numerology. Because alpha and beta are integer-valued on the known exact decompositions, each channel-separation quantity ||W_s-W_t||^2 is a rational quadratic form in w restricted to ker(Gamma). The observed values 1.414214, 0.707107, 0.577350, 1.732051, 2.236068, 2.449490, and 1.290994 are therefore square roots of small rationals coming from structured spectra of those quadratic forms, not floating-point noise.

[INTERPRETATION] This reframes the obstruction target. The right object is the channel-separation quadratic form on ker(Gamma): if its minimum on the multiplication variety stays strictly positive, then no nonzero w can satisfy W_0 = W_1 = ... = W_{n-1}. In that form, Phase 34 becomes a spectral-gap proof rather than a per-term identity hunt.

[INTERPRETATION] The old per-term universal identity target stays dead: generic D_01 = H M with constant coefficients is False, and generic D_10 = H M with constant coefficients is False. Containment is collective, not per-term.

[OPEN_FRONT] The next exact gap is now sharper than before: prove that the channel-separation quadratic form on ker(Gamma) has a positive spectral gap on a valid minimum-rank multiplication decomposition, equivalently that no nonzero w in ker(Gamma) can satisfy W_0 = W_1 = ... = W_{n-1}.
