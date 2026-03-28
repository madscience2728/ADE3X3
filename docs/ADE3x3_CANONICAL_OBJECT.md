======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-27 23:06:02
Generator: generate_canon_doc.py

This is a STANDALONE canonical dossier containing ALL computed results.
This document is the ONLY artifact provided to the next team.
It must be completely self-contained with all research findings.

----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-27 23:06:02
**Generator Script:** generate_canon_doc.py
**Provenance:** Built from steps 1-32+, with orbit metadata repair (step 10b)

**IMPORTANT:** This document contains all computed results inline.
No external files are required. All research findings are here.

## 2. SCOPE AND PRINCIPLES

### Scope
[GROUND_TRUTH]

This project records and organizes exact data about the ambient bilinear
universe for 3x3 matrix multiplication. The current work is analysis and
cataloging of the object as it is.

The source of truth is the object stored in the warehouse — not a reduced
model, not a surrogate, not a compressed summary.

### Principles
[GROUND_TRUTH]

- This dossier records the object without privileging any particular use
- Structure is cataloged faithfully; no reduced view is assumed as default
- Any future view of the object must explicitly state what it omits
- Current work is object-first, not reduction-first

## 3. GROUND-TRUTH RAW OBJECT

[GROUND_TRUTH]

### Full Raw Base-9 Warehouse

The raw warehouse stores all k-tuples over the 9-symbol alphabet {0,...,8}.
Base-9 arithmetic indexing provides compact storage.

| Layer | Tuple Count |
|-------|-------------|
| 9^1   | 9           |
| 9^2   | 81          |
| 9^3   | 729         |
| 9^4   | 6,561       |
| 9^5   | 59,049      |
| 9^6   | 531,441     |
| 9^7   | 4,782,969   |
| 9^8   | 43,046,721  |
| 9^9   | 387,420,489 |

All layers support exact encode/decode, random access, and iteration.

## 4. TYPED SEMANTIC OBJECT

[GROUND_TRUTH]

### Atomic Species

- **A**: Left input basis atoms, |A| = 9, named A[r,s] where r,s in {0,1,2}
- **B**: Right input basis atoms, |B| = 9, named B[t,u] where t,u in {0,1,2}
- **C**: Output basis atoms, |C| = 9, named C[r,u] where r,u in {0,1,2}
- **X**: Ambient product atoms, |X| = 81, named X[r,s|t,u]

### Primitive Exact Rules

[GROUND_TRUTH]

1. **Live/Dead Rule**: X[r,s|t,u] is LIVE if s == t, otherwise DEAD
2. **Target Map**: Live X maps to C[r,u]
3. **Fiber Structure**: Each C[r,u] has exactly 3 live X atoms in its fiber
4. **A/B Participation**: Each X atom has left A index and right B index

### C Target-Fiber Table

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

Each output basis atom C[r,u] has exactly 3 live X atoms in its inverse fiber:

| c_idx | C atom | Fiber X0 | Fiber X1 | Fiber X2 |
|-------|--------|----------|----------|----------|
| 0 | C[0,0] | X[0,0|0,0] | X[0,1|1,0] | X[0,2|2,0] |
| 1 | C[0,1] | X[0,0|0,1] | X[0,1|1,1] | X[0,2|2,1] |
| 2 | C[0,2] | X[0,0|0,2] | X[0,1|1,2] | X[0,2|2,2] |
| 3 | C[1,0] | X[1,0|0,0] | X[1,1|1,0] | X[1,2|2,0] |
| 4 | C[1,1] | X[1,0|0,1] | X[1,1|1,1] | X[1,2|2,1] |
| 5 | C[1,2] | X[1,0|0,2] | X[1,1|1,2] | X[1,2|2,2] |
| 6 | C[2,0] | X[2,0|0,0] | X[2,1|1,0] | X[2,2|2,0] |
| 7 | C[2,1] | X[2,0|0,1] | X[2,1|1,1] | X[2,2|2,1] |
| 8 | C[2,2] | X[2,0|0,2] | X[2,1|1,2] | X[2,2|2,2] |

**Derivable Pattern**: C[r,u] ← {X[r,s|s,u] : s ∈ {0,1,2}}

### Complete X Atom Inventory (All 81)

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

All 81 X atoms with live/dead status and target mapping:

| x_idx | X atom | a_idx | b_idx | live | target |
|-------|--------|-------|-------|------|--------|
| 0 | X[0,0|0,0] | 0 | 0 | LIVE | C[0,0] |
| 1 | X[0,0|0,1] | 0 | 1 | LIVE | C[0,1] |
| 2 | X[0,0|0,2] | 0 | 2 | LIVE | C[0,2] |
| 3 | X[0,0|1,0] | 0 | 3 | DEAD |  |
| 4 | X[0,0|1,1] | 0 | 4 | DEAD |  |
| 5 | X[0,0|1,2] | 0 | 5 | DEAD |  |
| 6 | X[0,0|2,0] | 0 | 6 | DEAD |  |
| 7 | X[0,0|2,1] | 0 | 7 | DEAD |  |
| 8 | X[0,0|2,2] | 0 | 8 | DEAD |  |
| 9 | X[0,1|0,0] | 1 | 0 | DEAD |  |
| 10 | X[0,1|0,1] | 1 | 1 | DEAD |  |
| 11 | X[0,1|0,2] | 1 | 2 | DEAD |  |
| 12 | X[0,1|1,0] | 1 | 3 | LIVE | C[0,0] |
| 13 | X[0,1|1,1] | 1 | 4 | LIVE | C[0,1] |
| 14 | X[0,1|1,2] | 1 | 5 | LIVE | C[0,2] |
| 15 | X[0,1|2,0] | 1 | 6 | DEAD |  |
| 16 | X[0,1|2,1] | 1 | 7 | DEAD |  |
| 17 | X[0,1|2,2] | 1 | 8 | DEAD |  |
| 18 | X[0,2|0,0] | 2 | 0 | DEAD |  |
| 19 | X[0,2|0,1] | 2 | 1 | DEAD |  |
| 20 | X[0,2|0,2] | 2 | 2 | DEAD |  |
| 21 | X[0,2|1,0] | 2 | 3 | DEAD |  |
| 22 | X[0,2|1,1] | 2 | 4 | DEAD |  |
| 23 | X[0,2|1,2] | 2 | 5 | DEAD |  |
| 24 | X[0,2|2,0] | 2 | 6 | LIVE | C[0,0] |
| 25 | X[0,2|2,1] | 2 | 7 | LIVE | C[0,1] |
| 26 | X[0,2|2,2] | 2 | 8 | LIVE | C[0,2] |
| 27 | X[1,0|0,0] | 3 | 0 | LIVE | C[1,0] |
| 28 | X[1,0|0,1] | 3 | 1 | LIVE | C[1,1] |
| 29 | X[1,0|0,2] | 3 | 2 | LIVE | C[1,2] |
| 30 | X[1,0|1,0] | 3 | 3 | DEAD |  |
| 31 | X[1,0|1,1] | 3 | 4 | DEAD |  |
| 32 | X[1,0|1,2] | 3 | 5 | DEAD |  |
| 33 | X[1,0|2,0] | 3 | 6 | DEAD |  |
| 34 | X[1,0|2,1] | 3 | 7 | DEAD |  |
| 35 | X[1,0|2,2] | 3 | 8 | DEAD |  |
| 36 | X[1,1|0,0] | 4 | 0 | DEAD |  |
| 37 | X[1,1|0,1] | 4 | 1 | DEAD |  |
| 38 | X[1,1|0,2] | 4 | 2 | DEAD |  |
| 39 | X[1,1|1,0] | 4 | 3 | LIVE | C[1,0] |
| 40 | X[1,1|1,1] | 4 | 4 | LIVE | C[1,1] |
| 41 | X[1,1|1,2] | 4 | 5 | LIVE | C[1,2] |
| 42 | X[1,1|2,0] | 4 | 6 | DEAD |  |
| 43 | X[1,1|2,1] | 4 | 7 | DEAD |  |
| 44 | X[1,1|2,2] | 4 | 8 | DEAD |  |
| 45 | X[1,2|0,0] | 5 | 0 | DEAD |  |
| 46 | X[1,2|0,1] | 5 | 1 | DEAD |  |
| 47 | X[1,2|0,2] | 5 | 2 | DEAD |  |
| 48 | X[1,2|1,0] | 5 | 3 | DEAD |  |
| 49 | X[1,2|1,1] | 5 | 4 | DEAD |  |
| 50 | X[1,2|1,2] | 5 | 5 | DEAD |  |
| 51 | X[1,2|2,0] | 5 | 6 | LIVE | C[1,0] |
| 52 | X[1,2|2,1] | 5 | 7 | LIVE | C[1,1] |
| 53 | X[1,2|2,2] | 5 | 8 | LIVE | C[1,2] |
| 54 | X[2,0|0,0] | 6 | 0 | LIVE | C[2,0] |
| 55 | X[2,0|0,1] | 6 | 1 | LIVE | C[2,1] |
| 56 | X[2,0|0,2] | 6 | 2 | LIVE | C[2,2] |
| 57 | X[2,0|1,0] | 6 | 3 | DEAD |  |
| 58 | X[2,0|1,1] | 6 | 4 | DEAD |  |
| 59 | X[2,0|1,2] | 6 | 5 | DEAD |  |
| 60 | X[2,0|2,0] | 6 | 6 | DEAD |  |
| 61 | X[2,0|2,1] | 6 | 7 | DEAD |  |
| 62 | X[2,0|2,2] | 6 | 8 | DEAD |  |
| 63 | X[2,1|0,0] | 7 | 0 | DEAD |  |
| 64 | X[2,1|0,1] | 7 | 1 | DEAD |  |
| 65 | X[2,1|0,2] | 7 | 2 | DEAD |  |
| 66 | X[2,1|1,0] | 7 | 3 | LIVE | C[2,0] |
| 67 | X[2,1|1,1] | 7 | 4 | LIVE | C[2,1] |
| 68 | X[2,1|1,2] | 7 | 5 | LIVE | C[2,2] |
| 69 | X[2,1|2,0] | 7 | 6 | DEAD |  |
| 70 | X[2,1|2,1] | 7 | 7 | DEAD |  |
| 71 | X[2,1|2,2] | 7 | 8 | DEAD |  |
| 72 | X[2,2|0,0] | 8 | 0 | DEAD |  |
| 73 | X[2,2|0,1] | 8 | 1 | DEAD |  |
| 74 | X[2,2|0,2] | 8 | 2 | DEAD |  |
| 75 | X[2,2|1,0] | 8 | 3 | DEAD |  |
| 76 | X[2,2|1,1] | 8 | 4 | DEAD |  |
| 77 | X[2,2|1,2] | 8 | 5 | DEAD |  |
| 78 | X[2,2|2,0] | 8 | 6 | LIVE | C[2,0] |
| 79 | X[2,2|2,1] | 8 | 7 | LIVE | C[2,1] |
| 80 | X[2,2|2,2] | 8 | 8 | LIVE | C[2,2] |

**Summary**: 27 live, 54 dead

## 5. SYMMETRY/ACTION SYSTEM

[GROUND_TRUTH]

- **Compatible Symmetry Group Size**: 216
- **Group Type**: S3 x S3 x S3 with compatibility constraint
- **Action Scope**: Acts on A, B, C, and X atoms
- **Canonicalization**: All core schemas are canonicalized under group action

### Explicit Group Action Formulas

[GROUND_TRUTH]

The group is coordinatized by three S3 factors:
- `pi_rA` acts on row indices of A, C, and the left row index of X
- `pi_shared` acts on the column index of A and row index of B (shared)
- `pi_cB` acts on column indices of B, C, and the right column index of X

**Action on atomic species:**

- **A**: A[r,s] → A[pi_rA(r), pi_shared(s)]
- **B**: B[t,u] → B[pi_shared(t), pi_cB(u)]
- **C**: C[r,u] → C[pi_rA(r), pi_cB(u)]
- **X**: X[r,s|t,u] → X[pi_rA(r), pi_shared(s) | pi_shared(t), pi_cB(u)]

**Compatibility constraint**: pi_shared is the same for A-column and B-row

### Canonicalization Rule

[GROUND_TRUTH]

**Canonical Representative Selection**: For each orbit, the canonical
representative is the configuration with the **minimum config_id** under the
group action (lexicographic minimum).

**Config ID Encoding**: Configurations are encoded as base-9 integers.

**rep_config_id**: The config_id of the canonical orbit representative

## 6. TYPED SCHEMAS CURRENTLY BUILT

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

| Schema | Typed Arity | Raw Arity | Count | Orbits | Notes |
|--------|-------------|-----------|-------|--------|-------|
| A      | 1           | 1         | 9     | -      | direct embed |
| B      | 1           | 1         | 9     | -      | direct embed |
| C      | 1           | 1         | 9     | -      | direct embed |
| X      | 1           | 2         | 81    | -      | expands to 2 slots |
| CC     | 2           | 2         | 81    | 4      | See Section 8 |
| CX     | 2           | 3         | 729   | 8      | See Section 9 |
| XC     | 2           | 3         | 729   | 8      | See Section 10 |
| AX     | 2           | 3         | 729   | 10     | See Section 11 |
| BX     | 2           | 3         | 729   | 10     | See Section 12 |
| CXC    | 3           | 4         | 6,561 | 50     | See Section 13 |
| XX     | 2           | 4         | 6,561 | 56     | See Section 14 |
| CXXC   | 4           | 6         | 531,441 | 2744   | See Section 15 |

## 7. TYPED/RAW BRIDGE

[GROUND_TRUTH]

### Bridge Rules

- A[r,s] -> raw symbol idx = 3*r + s
- B[t,u] -> raw symbol idx = 3*t + u
- C[r,u] -> raw symbol idx = 3*r + u
- X[r,s|t,u] -> raw tuple (A_idx(r,s), B_idx(t,u))

### Bridge Summary

[GROUND_TRUTH]

| Schema | Typed Arity | Raw Arity | Role Overlay | Injectivity |
|--------|-------------|-----------|--------------|-------------|
| C      | 1           | 1         | (C,)         | yes         |
| CC     | 2           | 2         | (C, C)       | yes         |
| CX     | 2           | 3         | (C, A_X, B_X)| yes         |
| XC     | 2           | 3         | (A_X, B_X, C)| yes         |
| AX     | 2           | 3         | (A, A_X, B_X)| yes         |
| BX     | 2           | 3         | (B, A_X, B_X)| yes         |
| CXC    | 3           | 4         | (C, A_X, B_X, C) | yes    |
| XX     | 2           | 4         | (A_X1, B_X1, A_X2, B_X2) | yes |
| CXXC   | 4           | 6         | (C, A_X1, B_X1, A_X2, B_X2, C) | yes |

## 8. SCHEMA CC - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: CC
**Total Configurations**: 81
**Orbit Count**: 4

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | CC[C[0,0],C[0,0]] | 9 | 24 | (True, True, True) |
| 1 | 1 | CC[C[0,0],C[0,1]] | 18 | 12 | (False, True, False) |
| 2 | 3 | CC[C[0,0],C[1,0]] | 18 | 12 | (False, False, True) |
| 3 | 4 | CC[C[0,0],C[1,1]] | 36 | 6 | (False, False, False) |

**Verification**: Sum of orbit sizes = 81
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 9. SCHEMA CX - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: CX
**Total Configurations**: 729
**Orbit Count**: 8

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | CX[C[0,0],X[0,0|0,0]] | 27 | 8 | (True, True, True, True) |
| 1 | 1 | CX[C[0,0],X[0,0|0,1]] | 54 | 4 | (True, False, True, False) |
| 2 | 3 | CX[C[0,0],X[0,0|1,0]] | 54 | 4 | (False, False, True, True) |
| 3 | 4 | CX[C[0,0],X[0,0|1,1]] | 108 | 2 | (False, False, True, False) |
| 4 | 27 | CX[C[0,0],X[1,0|0,0]] | 54 | 4 | (True, False, False, True) |
| 5 | 28 | CX[C[0,0],X[1,0|0,1]] | 108 | 2 | (True, False, False, False) |
| 6 | 30 | CX[C[0,0],X[1,0|1,0]] | 108 | 2 | (False, False, False, True) |
| 7 | 31 | CX[C[0,0],X[1,0|1,1]] | 216 | 1 | (False, False, False, False) |

**Verification**: Sum of orbit sizes = 729
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 10. SCHEMA XC - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: XC
**Total Configurations**: 729
**Orbit Count**: 8

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | XC[X[0,0|0,0],C[0,0]] | 27 | 8 | (True, True, True, True) |
| 1 | 1 | XC[X[0,0|0,0],C[0,1]] | 54 | 4 | (True, False, True, False) |
| 2 | 3 | XC[X[0,0|0,0],C[1,0]] | 54 | 4 | (True, False, False, True) |
| 3 | 4 | XC[X[0,0|0,0],C[1,1]] | 108 | 2 | (True, False, False, False) |
| 4 | 27 | XC[X[0,0|1,0],C[0,0]] | 54 | 4 | (False, False, True, True) |
| 5 | 28 | XC[X[0,0|1,0],C[0,1]] | 108 | 2 | (False, False, True, False) |
| 6 | 30 | XC[X[0,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, True) |
| 7 | 31 | XC[X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False) |

**Verification**: Sum of orbit sizes = 729
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 11. SCHEMA AX - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: AX
**Total Configurations**: 729
**Orbit Count**: 10

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | AX[A[0,0],X[0,0|0,0]] | 27 | 8 | (True, True, True) |
| 1 | 3 | AX[A[0,0],X[0,0|1,0]] | 54 | 4 | (True, True, False) |
| 2 | 9 | AX[A[0,0],X[0,1|0,0]] | 54 | 4 | (True, False, False) |
| 3 | 12 | AX[A[0,0],X[0,1|1,0]] | 54 | 4 | (True, False, True) |
| 4 | 15 | AX[A[0,0],X[0,1|2,0]] | 54 | 4 | (True, False, False) |
| 5 | 27 | AX[A[0,0],X[1,0|0,0]] | 54 | 4 | (False, True, True) |
| 6 | 30 | AX[A[0,0],X[1,0|1,0]] | 108 | 2 | (False, True, False) |
| 7 | 36 | AX[A[0,0],X[1,1|0,0]] | 108 | 2 | (False, False, False) |
| 8 | 39 | AX[A[0,0],X[1,1|1,0]] | 108 | 2 | (False, False, True) |
| 9 | 42 | AX[A[0,0],X[1,1|2,0]] | 108 | 2 | (False, False, False) |

**Verification**: Sum of orbit sizes = 729
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 12. SCHEMA BX - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: BX
**Total Configurations**: 729
**Orbit Count**: 10

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | BX[B[0,0],X[0,0|0,0]] | 27 | 8 | (True, True, True) |
| 1 | 1 | BX[B[0,0],X[0,0|0,1]] | 54 | 4 | (True, False, True) |
| 2 | 3 | BX[B[0,0],X[0,0|1,0]] | 54 | 4 | (False, True, False) |
| 3 | 4 | BX[B[0,0],X[0,0|1,1]] | 108 | 2 | (False, False, False) |
| 4 | 9 | BX[B[0,0],X[0,1|0,0]] | 54 | 4 | (True, True, False) |
| 5 | 10 | BX[B[0,0],X[0,1|0,1]] | 108 | 2 | (True, False, False) |
| 6 | 12 | BX[B[0,0],X[0,1|1,0]] | 54 | 4 | (False, True, True) |
| 7 | 13 | BX[B[0,0],X[0,1|1,1]] | 108 | 2 | (False, False, True) |
| 8 | 15 | BX[B[0,0],X[0,1|2,0]] | 54 | 4 | (False, True, False) |
| 9 | 16 | BX[B[0,0],X[0,1|2,1]] | 108 | 2 | (False, False, False) |

**Verification**: Sum of orbit sizes = 729
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 13. SCHEMA CXC - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: CXC
**Total Configurations**: 6561
**Orbit Count**: 50

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | CXC[C[0,0],X[0,0|0,0],C[0,0]] | 27 | 8 | (True, True, True, True, (0, 0, 0), (0, 0, 0)) |
| 1 | 1 | CXC[C[0,0],X[0,0|0,0],C[0,1]] | 54 | 4 | (True, False, True, False, (0, 0, 0), (0, 0, 1)) |
| 2 | 3 | CXC[C[0,0],X[0,0|0,0],C[1,0]] | 54 | 4 | (True, False, True, False, (0, 0, 1), (0, 0, 0)) |
| 3 | 4 | CXC[C[0,0],X[0,0|0,0],C[1,1]] | 108 | 2 | (True, False, True, False, (0, 0, 1), (0, 0, 1)) |
| 4 | 9 | CXC[C[0,0],X[0,0|0,1],C[0,0]] | 54 | 4 | (True, True, False, False, (0, 0, 0), (0, 1, 0)) |
| 5 | 10 | CXC[C[0,0],X[0,0|0,1],C[0,1]] | 54 | 4 | (True, False, False, True, (0, 0, 0), (0, 1, 1)) |
| 6 | 11 | CXC[C[0,0],X[0,0|0,1],C[0,2]] | 54 | 4 | (True, False, False, False, (0, 0, 0), (0, 1, 2)) |
| 7 | 12 | CXC[C[0,0],X[0,0|0,1],C[1,0]] | 108 | 2 | (True, False, False, False, (0, 0, 1), (0, 1, 0)) |
| 8 | 13 | CXC[C[0,0],X[0,0|0,1],C[1,1]] | 108 | 2 | (True, False, False, False, (0, 0, 1), (0, 1, 1)) |
| 9 | 14 | CXC[C[0,0],X[0,0|0,1],C[1,2]] | 108 | 2 | (True, False, False, False, (0, 0, 1), (0, 1, 2)) |
| 10 | 27 | CXC[C[0,0],X[0,0|1,0],C[0,0]] | 54 | 4 | (False, True, False, False, (0, 0, 0), (0, 0, 0)) |
| 11 | 28 | CXC[C[0,0],X[0,0|1,0],C[0,1]] | 108 | 2 | (False, False, False, False, (0, 0, 0), (0, 0, 1)) |
| 12 | 30 | CXC[C[0,0],X[0,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, (0, 0, 1), (0, 0, 0)) |
| 13 | 31 | CXC[C[0,0],X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, (0, 0, 1), (0, 0, 1)) |
| 14 | 36 | CXC[C[0,0],X[0,0|1,1],C[0,0]] | 108 | 2 | (False, True, False, False, (0, 0, 0), (0, 1, 0)) |
| 15 | 37 | CXC[C[0,0],X[0,0|1,1],C[0,1]] | 108 | 2 | (False, False, False, False, (0, 0, 0), (0, 1, 1)) |
| 16 | 38 | CXC[C[0,0],X[0,0|1,1],C[0,2]] | 108 | 2 | (False, False, False, False, (0, 0, 0), (0, 1, 2)) |
| 17 | 39 | CXC[C[0,0],X[0,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, (0, 0, 1), (0, 1, 0)) |
| 18 | 40 | CXC[C[0,0],X[0,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, (0, 0, 1), (0, 1, 1)) |
| 19 | 41 | CXC[C[0,0],X[0,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, (0, 0, 1), (0, 1, 2)) |
| 20 | 243 | CXC[C[0,0],X[1,0|0,0],C[0,0]] | 54 | 4 | (True, True, False, False, (0, 1, 0), (0, 0, 0)) |
| 21 | 244 | CXC[C[0,0],X[1,0|0,0],C[0,1]] | 108 | 2 | (True, False, False, False, (0, 1, 0), (0, 0, 1)) |
| 22 | 246 | CXC[C[0,0],X[1,0|0,0],C[1,0]] | 54 | 4 | (True, False, False, True, (0, 1, 1), (0, 0, 0)) |
| 23 | 247 | CXC[C[0,0],X[1,0|0,0],C[1,1]] | 108 | 2 | (True, False, False, False, (0, 1, 1), (0, 0, 1)) |
| 24 | 249 | CXC[C[0,0],X[1,0|0,0],C[2,0]] | 54 | 4 | (True, False, False, False, (0, 1, 2), (0, 0, 0)) |
| 25 | 250 | CXC[C[0,0],X[1,0|0,0],C[2,1]] | 108 | 2 | (True, False, False, False, (0, 1, 2), (0, 0, 1)) |
| 26 | 252 | CXC[C[0,0],X[1,0|0,1],C[0,0]] | 108 | 2 | (True, True, False, False, (0, 1, 0), (0, 1, 0)) |
| 27 | 253 | CXC[C[0,0],X[1,0|0,1],C[0,1]] | 108 | 2 | (True, False, False, False, (0, 1, 0), (0, 1, 1)) |
| 28 | 254 | CXC[C[0,0],X[1,0|0,1],C[0,2]] | 108 | 2 | (True, False, False, False, (0, 1, 0), (0, 1, 2)) |
| 29 | 255 | CXC[C[0,0],X[1,0|0,1],C[1,0]] | 108 | 2 | (True, False, False, False, (0, 1, 1), (0, 1, 0)) |
| 30 | 256 | CXC[C[0,0],X[1,0|0,1],C[1,1]] | 108 | 2 | (True, False, False, True, (0, 1, 1), (0, 1, 1)) |
| 31 | 257 | CXC[C[0,0],X[1,0|0,1],C[1,2]] | 108 | 2 | (True, False, False, False, (0, 1, 1), (0, 1, 2)) |
| 32 | 258 | CXC[C[0,0],X[1,0|0,1],C[2,0]] | 108 | 2 | (True, False, False, False, (0, 1, 2), (0, 1, 0)) |
| 33 | 259 | CXC[C[0,0],X[1,0|0,1],C[2,1]] | 108 | 2 | (True, False, False, False, (0, 1, 2), (0, 1, 1)) |
| 34 | 260 | CXC[C[0,0],X[1,0|0,1],C[2,2]] | 108 | 2 | (True, False, False, False, (0, 1, 2), (0, 1, 2)) |
| 35 | 270 | CXC[C[0,0],X[1,0|1,0],C[0,0]] | 108 | 2 | (False, True, False, False, (0, 1, 0), (0, 0, 0)) |
| 36 | 271 | CXC[C[0,0],X[1,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, (0, 1, 0), (0, 0, 1)) |
| 37 | 273 | CXC[C[0,0],X[1,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, (0, 1, 1), (0, 0, 0)) |
| 38 | 274 | CXC[C[0,0],X[1,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, (0, 1, 1), (0, 0, 1)) |
| 39 | 276 | CXC[C[0,0],X[1,0|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, (0, 1, 2), (0, 0, 0)) |
| 40 | 277 | CXC[C[0,0],X[1,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, (0, 1, 2), (0, 0, 1)) |
| 41 | 279 | CXC[C[0,0],X[1,0|1,1],C[0,0]] | 216 | 1 | (False, True, False, False, (0, 1, 0), (0, 1, 0)) |
| 42 | 280 | CXC[C[0,0],X[1,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, (0, 1, 0), (0, 1, 1)) |
| 43 | 281 | CXC[C[0,0],X[1,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, (0, 1, 0), (0, 1, 2)) |
| 44 | 282 | CXC[C[0,0],X[1,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, (0, 1, 1), (0, 1, 0)) |
| 45 | 283 | CXC[C[0,0],X[1,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, (0, 1, 1), (0, 1, 1)) |
| 46 | 284 | CXC[C[0,0],X[1,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, (0, 1, 1), (0, 1, 2)) |
| 47 | 285 | CXC[C[0,0],X[1,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, (0, 1, 2), (0, 1, 0)) |
| 48 | 286 | CXC[C[0,0],X[1,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, (0, 1, 2), (0, 1, 1)) |
| 49 | 287 | CXC[C[0,0],X[1,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, (0, 1, 2), (0, 1, 2)) |

**Verification**: Sum of orbit sizes = 6561
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 14. SCHEMA XX - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: XX
**Total Configurations**: 6561
**Orbit Count**: 56

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | XX[X[0,0|0,0],X[0,0|0,0]] | 27 | 8 | (True, True, True, True, True, True, True, True) |
| 1 | 1 | XX[X[0,0|0,0],X[0,0|0,1]] | 54 | 4 | (True, True, True, True, True, False, True, False) |
| 2 | 3 | XX[X[0,0|0,0],X[0,0|1,0]] | 54 | 4 | (True, False, True, True, False, True, True, False) |
| 3 | 4 | XX[X[0,0|0,0],X[0,0|1,1]] | 108 | 2 | (True, False, True, True, False, False, True, False) |
| 4 | 9 | XX[X[0,0|0,0],X[0,1|0,0]] | 54 | 4 | (True, False, True, False, True, True, False, True) |
| 5 | 10 | XX[X[0,0|0,0],X[0,1|0,1]] | 108 | 2 | (True, False, True, False, True, False, False, False) |
| 6 | 12 | XX[X[0,0|0,0],X[0,1|1,0]] | 54 | 4 | (True, True, True, False, False, True, False, False) |
| 7 | 13 | XX[X[0,0|0,0],X[0,1|1,1]] | 108 | 2 | (True, True, True, False, False, False, False, False) |
| 8 | 15 | XX[X[0,0|0,0],X[0,1|2,0]] | 54 | 4 | (True, False, True, False, False, True, False, False) |
| 9 | 16 | XX[X[0,0|0,0],X[0,1|2,1]] | 108 | 2 | (True, False, True, False, False, False, False, False) |
| 10 | 27 | XX[X[0,0|0,0],X[1,0|0,0]] | 54 | 4 | (True, True, False, True, True, True, False, True) |
| 11 | 28 | XX[X[0,0|0,0],X[1,0|0,1]] | 108 | 2 | (True, True, False, True, True, False, False, False) |
| 12 | 30 | XX[X[0,0|0,0],X[1,0|1,0]] | 108 | 2 | (True, False, False, True, False, True, False, False) |
| 13 | 31 | XX[X[0,0|0,0],X[1,0|1,1]] | 216 | 1 | (True, False, False, True, False, False, False, False) |
| 14 | 36 | XX[X[0,0|0,0],X[1,1|0,0]] | 108 | 2 | (True, False, False, False, True, True, False, True) |
| 15 | 37 | XX[X[0,0|0,0],X[1,1|0,1]] | 216 | 1 | (True, False, False, False, True, False, False, False) |
| 16 | 39 | XX[X[0,0|0,0],X[1,1|1,0]] | 108 | 2 | (True, True, False, False, False, True, False, False) |
| 17 | 40 | XX[X[0,0|0,0],X[1,1|1,1]] | 216 | 1 | (True, True, False, False, False, False, False, False) |
| 18 | 42 | XX[X[0,0|0,0],X[1,1|2,0]] | 108 | 2 | (True, False, False, False, False, True, False, False) |
| 19 | 43 | XX[X[0,0|0,0],X[1,1|2,1]] | 216 | 1 | (True, False, False, False, False, False, False, False) |
| 20 | 243 | XX[X[0,0|1,0],X[0,0|0,0]] | 54 | 4 | (False, True, True, True, False, True, True, False) |
| 21 | 244 | XX[X[0,0|1,0],X[0,0|0,1]] | 108 | 2 | (False, True, True, True, False, False, True, False) |
| 22 | 246 | XX[X[0,0|1,0],X[0,0|1,0]] | 54 | 4 | (False, False, True, True, True, True, True, True) |
| 23 | 247 | XX[X[0,0|1,0],X[0,0|1,1]] | 108 | 2 | (False, False, True, True, True, False, True, False) |
| 24 | 249 | XX[X[0,0|1,0],X[0,0|2,0]] | 54 | 4 | (False, False, True, True, False, True, True, False) |
| 25 | 250 | XX[X[0,0|1,0],X[0,0|2,1]] | 108 | 2 | (False, False, True, True, False, False, True, False) |
| 26 | 252 | XX[X[0,0|1,0],X[0,1|0,0]] | 54 | 4 | (False, False, True, False, False, True, False, False) |
| 27 | 253 | XX[X[0,0|1,0],X[0,1|0,1]] | 108 | 2 | (False, False, True, False, False, False, False, False) |
| 28 | 255 | XX[X[0,0|1,0],X[0,1|1,0]] | 54 | 4 | (False, True, True, False, True, True, False, True) |
| 29 | 256 | XX[X[0,0|1,0],X[0,1|1,1]] | 108 | 2 | (False, True, True, False, True, False, False, False) |
| 30 | 258 | XX[X[0,0|1,0],X[0,1|2,0]] | 54 | 4 | (False, False, True, False, False, True, False, False) |
| 31 | 259 | XX[X[0,0|1,0],X[0,1|2,1]] | 108 | 2 | (False, False, True, False, False, False, False, False) |
| 32 | 261 | XX[X[0,0|1,0],X[0,2|0,0]] | 54 | 4 | (False, False, True, False, False, True, False, False) |
| 33 | 262 | XX[X[0,0|1,0],X[0,2|0,1]] | 108 | 2 | (False, False, True, False, False, False, False, False) |
| 34 | 264 | XX[X[0,0|1,0],X[0,2|1,0]] | 54 | 4 | (False, False, True, False, True, True, False, True) |
| 35 | 265 | XX[X[0,0|1,0],X[0,2|1,1]] | 108 | 2 | (False, False, True, False, True, False, False, False) |
| 36 | 267 | XX[X[0,0|1,0],X[0,2|2,0]] | 54 | 4 | (False, True, True, False, False, True, False, False) |
| 37 | 268 | XX[X[0,0|1,0],X[0,2|2,1]] | 108 | 2 | (False, True, True, False, False, False, False, False) |
| 38 | 270 | XX[X[0,0|1,0],X[1,0|0,0]] | 108 | 2 | (False, True, False, True, False, True, False, False) |
| 39 | 271 | XX[X[0,0|1,0],X[1,0|0,1]] | 216 | 1 | (False, True, False, True, False, False, False, False) |
| 40 | 273 | XX[X[0,0|1,0],X[1,0|1,0]] | 108 | 2 | (False, False, False, True, True, True, False, True) |
| 41 | 274 | XX[X[0,0|1,0],X[1,0|1,1]] | 216 | 1 | (False, False, False, True, True, False, False, False) |
| 42 | 276 | XX[X[0,0|1,0],X[1,0|2,0]] | 108 | 2 | (False, False, False, True, False, True, False, False) |
| 43 | 277 | XX[X[0,0|1,0],X[1,0|2,1]] | 216 | 1 | (False, False, False, True, False, False, False, False) |
| 44 | 279 | XX[X[0,0|1,0],X[1,1|0,0]] | 108 | 2 | (False, False, False, False, False, True, False, False) |
| 45 | 280 | XX[X[0,0|1,0],X[1,1|0,1]] | 216 | 1 | (False, False, False, False, False, False, False, False) |
| 46 | 282 | XX[X[0,0|1,0],X[1,1|1,0]] | 108 | 2 | (False, True, False, False, True, True, False, True) |
| 47 | 283 | XX[X[0,0|1,0],X[1,1|1,1]] | 216 | 1 | (False, True, False, False, True, False, False, False) |
| 48 | 285 | XX[X[0,0|1,0],X[1,1|2,0]] | 108 | 2 | (False, False, False, False, False, True, False, False) |
| 49 | 286 | XX[X[0,0|1,0],X[1,1|2,1]] | 216 | 1 | (False, False, False, False, False, False, False, False) |
| 50 | 288 | XX[X[0,0|1,0],X[1,2|0,0]] | 108 | 2 | (False, False, False, False, False, True, False, False) |
| 51 | 289 | XX[X[0,0|1,0],X[1,2|0,1]] | 216 | 1 | (False, False, False, False, False, False, False, False) |
| 52 | 291 | XX[X[0,0|1,0],X[1,2|1,0]] | 108 | 2 | (False, False, False, False, True, True, False, True) |
| 53 | 292 | XX[X[0,0|1,0],X[1,2|1,1]] | 216 | 1 | (False, False, False, False, True, False, False, False) |
| 54 | 294 | XX[X[0,0|1,0],X[1,2|2,0]] | 108 | 2 | (False, True, False, False, False, True, False, False) |
| 55 | 295 | XX[X[0,0|1,0],X[1,2|2,1]] | 216 | 1 | (False, True, False, False, False, False, False, False) |

**Verification**: Sum of orbit sizes = 6561
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 15. SCHEMA CXXC - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Schema**: CXXC
**Total Configurations**: 531441
**Orbit Count**: 2744

Complete orbit roster with signatures:

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|----------|---------------|----------------|------------|-----------------|---------------|
| 0 | 0 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,0]] | 27 | 8 | (True, True, True, True, True, True, True, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 1 | 1 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,1]] | 54 | 4 | (True, True, False, True, True, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 2 | 3 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,0]] | 54 | 4 | (True, True, False, True, True, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 3 | 4 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, True, True, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 4 | 9 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,0]] | 54 | 4 | (True, True, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 5 | 10 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,1]] | 54 | 4 | (True, True, False, True, False, False, True, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 6 | 11 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[0,2]] | 54 | 4 | (True, True, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 7 | 12 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 8 | 13 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 9 | 14 | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 10 | 27 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[0,0]] | 54 | 4 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 11 | 28 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 12 | 30 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 13 | 31 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 14 | 36 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 15 | 37 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 16 | 38 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[0,2]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 17 | 39 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 18 | 40 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 19 | 41 | CXXC[C[0,0],X[0,0|0,0],X[0,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 20 | 81 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,0],C[0,0]] | 54 | 4 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 21 | 82 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,0],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 22 | 84 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 23 | 85 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 24 | 90 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 25 | 91 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 26 | 92 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[0,2]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 27 | 93 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 28 | 94 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 29 | 95 | CXXC[C[0,0],X[0,0|0,0],X[0,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 30 | 108 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,0],C[0,0]] | 54 | 4 | (True, True, True, True, True, True, True, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 31 | 109 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,0],C[0,1]] | 108 | 2 | (True, True, False, True, True, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 32 | 111 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,0],C[1,0]] | 108 | 2 | (True, True, False, True, True, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 33 | 112 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, True, True, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 34 | 117 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[0,0]] | 108 | 2 | (True, True, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 35 | 118 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[0,1]] | 108 | 2 | (True, True, False, True, False, False, True, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 36 | 119 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[0,2]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 37 | 120 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 38 | 121 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 39 | 122 | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 40 | 135 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,0],C[0,0]] | 54 | 4 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 41 | 136 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,0],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 42 | 138 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 43 | 139 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 44 | 144 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 45 | 145 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[0,1]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 46 | 146 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[0,2]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 47 | 147 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 48 | 148 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 49 | 149 | CXXC[C[0,0],X[0,0|0,0],X[0,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 50 | 243 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[0,0]] | 54 | 4 | (True, True, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 51 | 244 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 52 | 246 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[1,0]] | 54 | 4 | (True, True, False, True, False, False, True, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 53 | 247 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 54 | 249 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[2,0]] | 54 | 4 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 55 | 250 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 56 | 252 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 57 | 253 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 58 | 254 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 59 | 255 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 60 | 256 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, True, False, False, True, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 61 | 257 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 62 | 258 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 63 | 259 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 64 | 260 | CXXC[C[0,0],X[0,0|0,0],X[1,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 65 | 270 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 66 | 271 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 67 | 273 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 68 | 274 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 69 | 276 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[2,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 70 | 277 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 71 | 279 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 72 | 280 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 73 | 281 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 74 | 282 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 75 | 283 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 76 | 284 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 77 | 285 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 78 | 286 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 79 | 287 | CXXC[C[0,0],X[0,0|0,0],X[1,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 80 | 324 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 81 | 325 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 82 | 327 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 83 | 328 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 84 | 330 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[2,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 85 | 331 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 86 | 333 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 87 | 334 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 88 | 335 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 89 | 336 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 90 | 337 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 91 | 338 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 92 | 339 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 93 | 340 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 94 | 341 | CXXC[C[0,0],X[0,0|0,0],X[1,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 95 | 351 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[0,0]] | 108 | 2 | (True, True, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 96 | 352 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 97 | 354 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[1,0]] | 108 | 2 | (True, True, False, True, False, False, True, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 98 | 355 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 99 | 357 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[2,0]] | 108 | 2 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 100 | 358 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 101 | 360 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 102 | 361 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 103 | 362 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 104 | 363 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 105 | 364 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, True, False, False, True, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 106 | 365 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 107 | 366 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 108 | 367 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 109 | 368 | CXXC[C[0,0],X[0,0|0,0],X[1,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 110 | 378 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[0,0]] | 108 | 2 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 111 | 379 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 112 | 381 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[1,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 113 | 382 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 114 | 384 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[2,0]] | 108 | 2 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 115 | 385 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 116 | 387 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, True, False, True, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 117 | 388 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 118 | 389 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 119 | 390 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 120 | 391 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 121 | 392 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 122 | 393 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 123 | 394 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 124 | 395 | CXXC[C[0,0],X[0,0|0,0],X[1,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, True, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 125 | 729 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[0,0]] | 54 | 4 | (True, True, True, False, True, False, True, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 126 | 730 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[0,1]] | 54 | 4 | (True, True, False, False, True, True, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 127 | 731 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[0,2]] | 54 | 4 | (True, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 128 | 732 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[1,0]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 129 | 733 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 130 | 734 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,0],C[1,2]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 131 | 738 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[0,0]] | 54 | 4 | (True, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 132 | 739 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[0,1]] | 54 | 4 | (True, True, False, False, False, True, True, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 133 | 740 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[0,2]] | 54 | 4 | (True, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 134 | 741 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 135 | 742 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 136 | 743 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 137 | 747 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[0,0]] | 54 | 4 | (True, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 138 | 748 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[0,1]] | 54 | 4 | (True, True, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 139 | 749 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[0,2]] | 54 | 4 | (True, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 140 | 750 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 141 | 751 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 142 | 752 | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,2],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 143 | 756 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 144 | 757 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 145 | 758 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 146 | 759 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 147 | 760 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 148 | 761 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 149 | 765 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 150 | 766 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 151 | 767 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 152 | 768 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 153 | 769 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 154 | 770 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 155 | 774 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 156 | 775 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 157 | 776 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 158 | 777 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 159 | 778 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 160 | 779 | CXXC[C[0,0],X[0,0|0,1],X[0,0|1,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 161 | 810 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 162 | 811 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 163 | 812 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 164 | 813 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 165 | 814 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 166 | 815 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 167 | 819 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 168 | 820 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 169 | 821 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 170 | 822 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 171 | 823 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 172 | 824 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 173 | 828 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 174 | 829 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 175 | 830 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 176 | 831 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 177 | 832 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 178 | 833 | CXXC[C[0,0],X[0,0|0,1],X[0,1|0,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 179 | 837 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[0,0]] | 108 | 2 | (True, True, True, False, True, False, True, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 180 | 838 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[0,1]] | 108 | 2 | (True, True, False, False, True, True, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 181 | 839 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[0,2]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 182 | 840 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[1,0]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 183 | 841 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 184 | 842 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,0],C[1,2]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 185 | 846 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 186 | 847 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, True, True, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 187 | 848 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 188 | 849 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 189 | 850 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 190 | 851 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 191 | 855 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 192 | 856 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[0,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 193 | 857 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 194 | 858 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 195 | 859 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 196 | 860 | CXXC[C[0,0],X[0,0|0,1],X[0,1|1,2],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 197 | 864 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 198 | 865 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 199 | 866 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 200 | 867 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 201 | 868 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 202 | 869 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 203 | 873 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 204 | 874 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 205 | 875 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 206 | 876 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 207 | 877 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 208 | 878 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 209 | 882 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 210 | 883 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[0,1]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 211 | 884 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[0,2]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 212 | 885 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 213 | 886 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 214 | 887 | CXXC[C[0,0],X[0,0|0,1],X[0,1|2,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 215 | 972 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 216 | 973 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 217 | 974 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 218 | 975 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 219 | 976 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 220 | 977 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 221 | 978 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 222 | 979 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 223 | 980 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,0],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 224 | 981 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 225 | 982 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 226 | 983 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 227 | 984 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 228 | 985 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 229 | 986 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 230 | 987 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 231 | 988 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 232 | 989 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 233 | 990 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 234 | 991 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[0,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 235 | 992 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 236 | 993 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 237 | 994 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 238 | 995 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 239 | 996 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 240 | 997 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 241 | 998 | CXXC[C[0,0],X[0,0|0,1],X[1,0|0,2],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 242 | 999 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 243 | 1000 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 244 | 1001 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 245 | 1002 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 246 | 1003 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 247 | 1004 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 248 | 1005 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 249 | 1006 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 250 | 1007 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 251 | 1008 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 252 | 1009 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 253 | 1010 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 254 | 1011 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 255 | 1012 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 256 | 1013 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 257 | 1014 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 258 | 1015 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 259 | 1016 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 260 | 1017 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 261 | 1018 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 262 | 1019 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 263 | 1020 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 264 | 1021 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 265 | 1022 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 266 | 1023 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 267 | 1024 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 268 | 1025 | CXXC[C[0,0],X[0,0|0,1],X[1,0|1,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 269 | 1053 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 270 | 1054 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 271 | 1055 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 272 | 1056 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 273 | 1057 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 274 | 1058 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 275 | 1059 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 276 | 1060 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 277 | 1061 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 278 | 1062 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 279 | 1063 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 280 | 1064 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 281 | 1065 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 282 | 1066 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 283 | 1067 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 284 | 1068 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 285 | 1069 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 286 | 1070 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 287 | 1071 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 288 | 1072 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 289 | 1073 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 290 | 1074 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 291 | 1075 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 292 | 1076 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 293 | 1077 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 294 | 1078 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 295 | 1079 | CXXC[C[0,0],X[0,0|0,1],X[1,1|0,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 296 | 1080 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 297 | 1081 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 298 | 1082 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 299 | 1083 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 300 | 1084 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 301 | 1085 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 302 | 1086 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 303 | 1087 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 304 | 1088 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,0],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 305 | 1089 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 306 | 1090 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 307 | 1091 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 308 | 1092 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 309 | 1093 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 310 | 1094 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 311 | 1095 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 312 | 1096 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 313 | 1097 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 314 | 1098 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 315 | 1099 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[0,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 316 | 1100 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 317 | 1101 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 318 | 1102 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 319 | 1103 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 320 | 1104 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 321 | 1105 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 322 | 1106 | CXXC[C[0,0],X[0,0|0,1],X[1,1|1,2],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 323 | 1107 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 324 | 1108 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 325 | 1109 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 326 | 1110 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 327 | 1111 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 328 | 1112 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 329 | 1113 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 330 | 1114 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 331 | 1115 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 332 | 1116 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 333 | 1117 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 334 | 1118 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 335 | 1119 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 336 | 1120 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 337 | 1121 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 338 | 1122 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 339 | 1123 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 340 | 1124 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 341 | 1125 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 342 | 1126 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 343 | 1127 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 344 | 1128 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 345 | 1129 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 346 | 1130 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 347 | 1131 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 348 | 1132 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 349 | 1133 | CXXC[C[0,0],X[0,0|0,1],X[1,1|2,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 350 | 2187 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,0],C[0,0]] | 54 | 4 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 351 | 2188 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 352 | 2190 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 353 | 2191 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 354 | 2196 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 355 | 2197 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 356 | 2198 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 357 | 2199 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 358 | 2200 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 359 | 2201 | CXXC[C[0,0],X[0,0|1,0],X[0,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 360 | 2214 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 361 | 2215 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 362 | 2217 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 363 | 2218 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 364 | 2223 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 365 | 2224 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 366 | 2225 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 367 | 2226 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 368 | 2227 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 369 | 2228 | CXXC[C[0,0],X[0,0|1,0],X[0,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 370 | 2241 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 371 | 2242 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 372 | 2244 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 373 | 2245 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 374 | 2250 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 375 | 2251 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 376 | 2252 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 377 | 2253 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 378 | 2254 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 379 | 2255 | CXXC[C[0,0],X[0,0|1,0],X[0,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 380 | 2268 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 381 | 2269 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 382 | 2271 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 383 | 2272 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 384 | 2277 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 385 | 2278 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 386 | 2279 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 387 | 2280 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 388 | 2281 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 389 | 2282 | CXXC[C[0,0],X[0,0|1,0],X[0,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 390 | 2295 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,0],C[0,0]] | 54 | 4 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 391 | 2296 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 392 | 2298 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 393 | 2299 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 394 | 2304 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 395 | 2305 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 396 | 2306 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 397 | 2307 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 398 | 2308 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 399 | 2309 | CXXC[C[0,0],X[0,0|1,0],X[0,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 400 | 2322 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 401 | 2323 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 402 | 2325 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 403 | 2326 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 404 | 2331 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 405 | 2332 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 406 | 2333 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 407 | 2334 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 408 | 2335 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 409 | 2336 | CXXC[C[0,0],X[0,0|1,0],X[0,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 410 | 2349 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 411 | 2350 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 412 | 2352 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 413 | 2353 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 414 | 2358 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 415 | 2359 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 416 | 2360 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 417 | 2361 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 418 | 2362 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 419 | 2363 | CXXC[C[0,0],X[0,0|1,0],X[0,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 420 | 2376 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,0],C[0,0]] | 54 | 4 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 421 | 2377 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 422 | 2379 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 423 | 2380 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 424 | 2385 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 425 | 2386 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 426 | 2387 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 427 | 2388 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 428 | 2389 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 429 | 2390 | CXXC[C[0,0],X[0,0|1,0],X[0,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 430 | 2403 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,0],C[0,0]] | 54 | 4 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 0, 0, 0)) |
| 431 | 2404 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 0, 0, 1)) |
| 432 | 2406 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 0)) |
| 433 | 2407 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 0, 0, 1)) |
| 434 | 2412 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 0)) |
| 435 | 2413 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 0, 1, 1)) |
| 436 | 2414 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 0, 1, 2)) |
| 437 | 2415 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 0)) |
| 438 | 2416 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 1)) |
| 439 | 2417 | CXXC[C[0,0],X[0,0|1,0],X[0,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 0, 1, 2)) |
| 440 | 2430 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 441 | 2431 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 442 | 2433 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 443 | 2434 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 444 | 2436 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 445 | 2437 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 446 | 2439 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 447 | 2440 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 448 | 2441 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 449 | 2442 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 450 | 2443 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 451 | 2444 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 452 | 2445 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 453 | 2446 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 454 | 2447 | CXXC[C[0,0],X[0,0|1,0],X[1,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 455 | 2457 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 456 | 2458 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 457 | 2460 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 458 | 2461 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 459 | 2463 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 460 | 2464 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 461 | 2466 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 462 | 2467 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 463 | 2468 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 464 | 2469 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 465 | 2470 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 466 | 2471 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 467 | 2472 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 468 | 2473 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 469 | 2474 | CXXC[C[0,0],X[0,0|1,0],X[1,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 470 | 2484 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 471 | 2485 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 472 | 2487 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 473 | 2488 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 474 | 2490 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 475 | 2491 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 476 | 2493 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 477 | 2494 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 478 | 2495 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 479 | 2496 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 480 | 2497 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 481 | 2498 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 482 | 2499 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 483 | 2500 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 484 | 2501 | CXXC[C[0,0],X[0,0|1,0],X[1,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 485 | 2511 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 486 | 2512 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 487 | 2514 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 488 | 2515 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 489 | 2517 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 490 | 2518 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 491 | 2520 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 492 | 2521 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 493 | 2522 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 494 | 2523 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 495 | 2524 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 496 | 2525 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 497 | 2526 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 498 | 2527 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 499 | 2528 | CXXC[C[0,0],X[0,0|1,0],X[1,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 500 | 2538 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 501 | 2539 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 502 | 2541 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 503 | 2542 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 504 | 2544 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 505 | 2545 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 506 | 2547 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 507 | 2548 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 508 | 2549 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 509 | 2550 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 510 | 2551 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 511 | 2552 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 512 | 2553 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 513 | 2554 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 514 | 2555 | CXXC[C[0,0],X[0,0|1,0],X[1,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 515 | 2565 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 516 | 2566 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 517 | 2568 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 518 | 2569 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 519 | 2571 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 520 | 2572 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 521 | 2574 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 522 | 2575 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 523 | 2576 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 524 | 2577 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 525 | 2578 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 526 | 2579 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 527 | 2580 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 528 | 2581 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 529 | 2582 | CXXC[C[0,0],X[0,0|1,0],X[1,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 530 | 2592 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 531 | 2593 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 532 | 2595 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 533 | 2596 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 534 | 2598 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 535 | 2599 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 536 | 2601 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 537 | 2602 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 538 | 2603 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 539 | 2604 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 540 | 2605 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 541 | 2606 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 542 | 2607 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 543 | 2608 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 544 | 2609 | CXXC[C[0,0],X[0,0|1,0],X[1,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 545 | 2619 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 546 | 2620 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 547 | 2622 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 548 | 2623 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 549 | 2625 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 550 | 2626 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 551 | 2628 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 552 | 2629 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 553 | 2630 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 554 | 2631 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 555 | 2632 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 556 | 2633 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 557 | 2634 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 558 | 2635 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 559 | 2636 | CXXC[C[0,0],X[0,0|1,0],X[1,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 560 | 2646 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 0)) |
| 561 | 2647 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 0, 1)) |
| 562 | 2649 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 0, 0)) |
| 563 | 2650 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 0, 1)) |
| 564 | 2652 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 0)) |
| 565 | 2653 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 0, 1)) |
| 566 | 2655 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 0)) |
| 567 | 2656 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 1)) |
| 568 | 2657 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 0, 1, 2)) |
| 569 | 2658 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 0)) |
| 570 | 2659 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 0, 1, 1)) |
| 571 | 2660 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 0, 1, 2)) |
| 572 | 2661 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 0)) |
| 573 | 2662 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 1)) |
| 574 | 2663 | CXXC[C[0,0],X[0,0|1,0],X[1,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 0, 1, 2)) |
| 575 | 2916 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 576 | 2917 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 577 | 2918 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[0,2]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 578 | 2919 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 579 | 2920 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 580 | 2921 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 581 | 2925 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 582 | 2926 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 583 | 2927 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 584 | 2928 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 585 | 2929 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 586 | 2930 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 587 | 2934 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 588 | 2935 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 589 | 2936 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 590 | 2937 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 591 | 2938 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 592 | 2939 | CXXC[C[0,0],X[0,0|1,1],X[0,0|0,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 593 | 2943 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 594 | 2944 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 595 | 2945 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 596 | 2946 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 597 | 2947 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 598 | 2948 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 599 | 2952 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 600 | 2953 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 601 | 2954 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 602 | 2955 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 603 | 2956 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 604 | 2957 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 605 | 2961 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 606 | 2962 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 607 | 2963 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 608 | 2964 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 609 | 2965 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 610 | 2966 | CXXC[C[0,0],X[0,0|1,1],X[0,0|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 611 | 2970 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 612 | 2971 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 613 | 2972 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 614 | 2973 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 615 | 2974 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 616 | 2975 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 617 | 2979 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 618 | 2980 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 619 | 2981 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 620 | 2982 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 621 | 2983 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 622 | 2984 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 623 | 2988 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 624 | 2989 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 625 | 2990 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 626 | 2991 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 627 | 2992 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 628 | 2993 | CXXC[C[0,0],X[0,0|1,1],X[0,0|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 629 | 2997 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 630 | 2998 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 631 | 2999 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 632 | 3000 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 633 | 3001 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 634 | 3002 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 635 | 3006 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 636 | 3007 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 637 | 3008 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 638 | 3009 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 639 | 3010 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 640 | 3011 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 641 | 3015 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 642 | 3016 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 643 | 3017 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 644 | 3018 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 645 | 3019 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 646 | 3020 | CXXC[C[0,0],X[0,0|1,1],X[0,1|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 647 | 3024 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 648 | 3025 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 649 | 3026 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[0,2]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 650 | 3027 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 651 | 3028 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 652 | 3029 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 653 | 3033 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 654 | 3034 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 655 | 3035 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 656 | 3036 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 657 | 3037 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 658 | 3038 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 659 | 3042 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 660 | 3043 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 661 | 3044 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 662 | 3045 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 663 | 3046 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 664 | 3047 | CXXC[C[0,0],X[0,0|1,1],X[0,1|1,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 665 | 3051 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 666 | 3052 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 667 | 3053 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 668 | 3054 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 669 | 3055 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 670 | 3056 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 671 | 3060 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 672 | 3061 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 673 | 3062 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 674 | 3063 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 675 | 3064 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 676 | 3065 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 677 | 3069 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 678 | 3070 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 679 | 3071 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 680 | 3072 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 681 | 3073 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 682 | 3074 | CXXC[C[0,0],X[0,0|1,1],X[0,1|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 683 | 3078 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 684 | 3079 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 685 | 3080 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 686 | 3081 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 687 | 3082 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 688 | 3083 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 689 | 3087 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 690 | 3088 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 691 | 3089 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 692 | 3090 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 693 | 3091 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 694 | 3092 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 695 | 3096 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 696 | 3097 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 697 | 3098 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 698 | 3099 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 699 | 3100 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 700 | 3101 | CXXC[C[0,0],X[0,0|1,1],X[0,2|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 701 | 3105 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 702 | 3106 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 703 | 3107 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 704 | 3108 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 705 | 3109 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 706 | 3110 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 707 | 3114 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 708 | 3115 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 709 | 3116 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 710 | 3117 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 711 | 3118 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 712 | 3119 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 713 | 3123 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 714 | 3124 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[0,1]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 715 | 3125 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[0,2]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 716 | 3126 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 717 | 3127 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 718 | 3128 | CXXC[C[0,0],X[0,0|1,1],X[0,2|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 719 | 3132 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 0, 0, 0), (0, 1, 0, 0)) |
| 720 | 3133 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[0,1]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 1)) |
| 721 | 3134 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[0,2]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 0, 0, 0), (0, 1, 0, 2)) |
| 722 | 3135 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 0)) |
| 723 | 3136 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 1)) |
| 724 | 3137 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 0, 0, 1), (0, 1, 0, 2)) |
| 725 | 3141 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 0)) |
| 726 | 3142 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 1, 1)) |
| 727 | 3143 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 1, 2)) |
| 728 | 3144 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 0)) |
| 729 | 3145 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 1)) |
| 730 | 3146 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 1, 2)) |
| 731 | 3150 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 0)) |
| 732 | 3151 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[0,1]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 0, 0, 0), (0, 1, 2, 1)) |
| 733 | 3152 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[0,2]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 0, 0, 0), (0, 1, 2, 2)) |
| 734 | 3153 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 0)) |
| 735 | 3154 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 1)) |
| 736 | 3155 | CXXC[C[0,0],X[0,0|1,1],X[0,2|2,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 0, 1), (0, 1, 2, 2)) |
| 737 | 3159 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 738 | 3160 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 739 | 3161 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 740 | 3162 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 741 | 3163 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 742 | 3164 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 743 | 3165 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 744 | 3166 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 745 | 3167 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 746 | 3168 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 747 | 3169 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 748 | 3170 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 749 | 3171 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 750 | 3172 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 751 | 3173 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 752 | 3174 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 753 | 3175 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 754 | 3176 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 755 | 3177 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 756 | 3178 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 757 | 3179 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 758 | 3180 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 759 | 3181 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 760 | 3182 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 761 | 3183 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 762 | 3184 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 763 | 3185 | CXXC[C[0,0],X[0,0|1,1],X[1,0|0,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 764 | 3186 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 765 | 3187 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 766 | 3188 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 767 | 3189 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 768 | 3190 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 769 | 3191 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 770 | 3192 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 771 | 3193 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 772 | 3194 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 773 | 3195 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 774 | 3196 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 775 | 3197 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 776 | 3198 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 777 | 3199 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 778 | 3200 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 779 | 3201 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 780 | 3202 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 781 | 3203 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 782 | 3204 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 783 | 3205 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 784 | 3206 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 785 | 3207 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 786 | 3208 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 787 | 3209 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 788 | 3210 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 789 | 3211 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 790 | 3212 | CXXC[C[0,0],X[0,0|1,1],X[1,0|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 791 | 3213 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 792 | 3214 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 793 | 3215 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 794 | 3216 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 795 | 3217 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 796 | 3218 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 797 | 3219 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 798 | 3220 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 799 | 3221 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 800 | 3222 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 801 | 3223 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 802 | 3224 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 803 | 3225 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 804 | 3226 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 805 | 3227 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 806 | 3228 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 807 | 3229 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 808 | 3230 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 809 | 3231 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 810 | 3232 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 811 | 3233 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 812 | 3234 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 813 | 3235 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 814 | 3236 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 815 | 3237 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 816 | 3238 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 817 | 3239 | CXXC[C[0,0],X[0,0|1,1],X[1,0|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 818 | 3240 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 819 | 3241 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 820 | 3242 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 821 | 3243 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 822 | 3244 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 823 | 3245 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 824 | 3246 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 825 | 3247 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 826 | 3248 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 827 | 3249 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 828 | 3250 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 829 | 3251 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 830 | 3252 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 831 | 3253 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 832 | 3254 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 833 | 3255 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 834 | 3256 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 835 | 3257 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 836 | 3258 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 837 | 3259 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 838 | 3260 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 839 | 3261 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 840 | 3262 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 841 | 3263 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 842 | 3264 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 843 | 3265 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 844 | 3266 | CXXC[C[0,0],X[0,0|1,1],X[1,1|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 845 | 3267 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 846 | 3268 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 847 | 3269 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 848 | 3270 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 849 | 3271 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 850 | 3272 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 851 | 3273 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 852 | 3274 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 853 | 3275 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 854 | 3276 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 855 | 3277 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 856 | 3278 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 857 | 3279 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 858 | 3280 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 859 | 3281 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 860 | 3282 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 861 | 3283 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 862 | 3284 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 863 | 3285 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 864 | 3286 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 865 | 3287 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 866 | 3288 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 867 | 3289 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 868 | 3290 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 869 | 3291 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 870 | 3292 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 871 | 3293 | CXXC[C[0,0],X[0,0|1,1],X[1,1|1,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 872 | 3294 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 873 | 3295 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 874 | 3296 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 875 | 3297 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 876 | 3298 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 877 | 3299 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 878 | 3300 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 879 | 3301 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 880 | 3302 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 881 | 3303 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 882 | 3304 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 883 | 3305 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 884 | 3306 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 885 | 3307 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 886 | 3308 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 887 | 3309 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 888 | 3310 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 889 | 3311 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 890 | 3312 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 891 | 3313 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 892 | 3314 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 893 | 3315 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 894 | 3316 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 895 | 3317 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 896 | 3318 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 897 | 3319 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 898 | 3320 | CXXC[C[0,0],X[0,0|1,1],X[1,1|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 899 | 3321 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 900 | 3322 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 901 | 3323 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 902 | 3324 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 903 | 3325 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 904 | 3326 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 905 | 3327 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 906 | 3328 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 907 | 3329 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 908 | 3330 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 909 | 3331 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 910 | 3332 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 911 | 3333 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 912 | 3334 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 913 | 3335 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 914 | 3336 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 915 | 3337 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 916 | 3338 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 917 | 3339 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 918 | 3340 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 919 | 3341 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 920 | 3342 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 921 | 3343 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 922 | 3344 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 923 | 3345 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 924 | 3346 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 925 | 3347 | CXXC[C[0,0],X[0,0|1,1],X[1,2|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 926 | 3348 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 927 | 3349 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 928 | 3350 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 929 | 3351 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 930 | 3352 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 931 | 3353 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 932 | 3354 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 933 | 3355 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 934 | 3356 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 935 | 3357 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 936 | 3358 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 937 | 3359 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 938 | 3360 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 939 | 3361 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 940 | 3362 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 941 | 3363 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 942 | 3364 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 943 | 3365 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 944 | 3366 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 945 | 3367 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 946 | 3368 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 947 | 3369 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 948 | 3370 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 949 | 3371 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 950 | 3372 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 951 | 3373 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 952 | 3374 | CXXC[C[0,0],X[0,0|1,1],X[1,2|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 953 | 3375 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 0)) |
| 954 | 3376 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 1)) |
| 955 | 3377 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 0, 2)) |
| 956 | 3378 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 0, 0)) |
| 957 | 3379 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 1)) |
| 958 | 3380 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 0, 2)) |
| 959 | 3381 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 0)) |
| 960 | 3382 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 1)) |
| 961 | 3383 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 0, 2)) |
| 962 | 3384 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 0)) |
| 963 | 3385 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 1)) |
| 964 | 3386 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 1, 2)) |
| 965 | 3387 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 0)) |
| 966 | 3388 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 1, 1)) |
| 967 | 3389 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 1, 2)) |
| 968 | 3390 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 0)) |
| 969 | 3391 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 1)) |
| 970 | 3392 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 1, 2)) |
| 971 | 3393 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 0)) |
| 972 | 3394 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 1)) |
| 973 | 3395 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 0), (0, 1, 2, 2)) |
| 974 | 3396 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 0)) |
| 975 | 3397 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 1), (0, 1, 2, 1)) |
| 976 | 3398 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 0, 1, 1), (0, 1, 2, 2)) |
| 977 | 3399 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 0)) |
| 978 | 3400 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 1)) |
| 979 | 3401 | CXXC[C[0,0],X[0,0|1,1],X[1,2|2,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 0, 1, 2), (0, 1, 2, 2)) |
| 980 | 19683 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[0,0]] | 54 | 4 | (True, True, True, False, True, False, True, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 981 | 19684 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 982 | 19686 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[1,0]] | 54 | 4 | (True, True, False, False, True, True, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 983 | 19687 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 984 | 19689 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[2,0]] | 54 | 4 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 985 | 19690 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 986 | 19692 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 987 | 19693 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 988 | 19694 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 989 | 19695 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 990 | 19696 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 991 | 19697 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 992 | 19698 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 993 | 19699 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 994 | 19700 | CXXC[C[0,0],X[1,0|0,0],X[0,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 995 | 19710 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 996 | 19711 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 997 | 19713 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 998 | 19714 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 999 | 19716 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1000 | 19717 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1001 | 19719 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1002 | 19720 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1003 | 19721 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1004 | 19722 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1005 | 19723 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1006 | 19724 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1007 | 19725 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1008 | 19726 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1009 | 19727 | CXXC[C[0,0],X[1,0|0,0],X[0,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1010 | 19764 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1011 | 19765 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1012 | 19767 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1013 | 19768 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1014 | 19770 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1015 | 19771 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1016 | 19773 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1017 | 19774 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1018 | 19775 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1019 | 19776 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1020 | 19777 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1021 | 19778 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1022 | 19779 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1023 | 19780 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1024 | 19781 | CXXC[C[0,0],X[1,0|0,0],X[0,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1025 | 19791 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[0,0]] | 108 | 2 | (True, True, True, False, True, False, True, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1026 | 19792 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1027 | 19794 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[1,0]] | 108 | 2 | (True, True, False, False, True, True, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1028 | 19795 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1029 | 19797 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[2,0]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1030 | 19798 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1031 | 19800 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1032 | 19801 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1033 | 19802 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1034 | 19803 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1035 | 19804 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1036 | 19805 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1037 | 19806 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1038 | 19807 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1039 | 19808 | CXXC[C[0,0],X[1,0|0,0],X[0,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1040 | 19818 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1041 | 19819 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1042 | 19821 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1043 | 19822 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1044 | 19824 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1045 | 19825 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1046 | 19827 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1047 | 19828 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1048 | 19829 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1049 | 19830 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1050 | 19831 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1051 | 19832 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1052 | 19833 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1053 | 19834 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1054 | 19835 | CXXC[C[0,0],X[1,0|0,0],X[0,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1055 | 19926 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[0,0]] | 54 | 4 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1056 | 19927 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1057 | 19929 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[1,0]] | 54 | 4 | (True, True, False, False, False, True, True, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1058 | 19930 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1059 | 19932 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[2,0]] | 54 | 4 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1060 | 19933 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1061 | 19935 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1062 | 19936 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1063 | 19937 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1064 | 19938 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1065 | 19939 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1066 | 19940 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1067 | 19941 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1068 | 19942 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1069 | 19943 | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1070 | 19953 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1071 | 19954 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1072 | 19956 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1073 | 19957 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1074 | 19959 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1075 | 19960 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1076 | 19962 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1077 | 19963 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1078 | 19964 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1079 | 19965 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1080 | 19966 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1081 | 19967 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1082 | 19968 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1083 | 19969 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1084 | 19970 | CXXC[C[0,0],X[1,0|0,0],X[1,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1085 | 20007 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1086 | 20008 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1087 | 20010 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1088 | 20011 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1089 | 20013 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1090 | 20014 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1091 | 20016 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1092 | 20017 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1093 | 20018 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1094 | 20019 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1095 | 20020 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1096 | 20021 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1097 | 20022 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1098 | 20023 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1099 | 20024 | CXXC[C[0,0],X[1,0|0,0],X[1,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1100 | 20034 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1101 | 20035 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1102 | 20037 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[1,0]] | 108 | 2 | (True, True, False, False, False, True, True, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1103 | 20038 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1104 | 20040 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1105 | 20041 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1106 | 20043 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1107 | 20044 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1108 | 20045 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1109 | 20046 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1110 | 20047 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1111 | 20048 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1112 | 20049 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1113 | 20050 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1114 | 20051 | CXXC[C[0,0],X[1,0|0,0],X[1,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1115 | 20061 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1116 | 20062 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1117 | 20064 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1118 | 20065 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1119 | 20067 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1120 | 20068 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1121 | 20070 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1122 | 20071 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1123 | 20072 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1124 | 20073 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1125 | 20074 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1126 | 20075 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1127 | 20076 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1128 | 20077 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1129 | 20078 | CXXC[C[0,0],X[1,0|0,0],X[1,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1130 | 20169 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[0,0]] | 54 | 4 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1131 | 20170 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1132 | 20172 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[1,0]] | 54 | 4 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1133 | 20173 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1134 | 20175 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[2,0]] | 54 | 4 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1135 | 20176 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1136 | 20178 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1137 | 20179 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1138 | 20180 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1139 | 20181 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1140 | 20182 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1141 | 20183 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1142 | 20184 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1143 | 20185 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1144 | 20186 | CXXC[C[0,0],X[1,0|0,0],X[2,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1145 | 20196 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1146 | 20197 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1147 | 20199 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1148 | 20200 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1149 | 20202 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1150 | 20203 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1151 | 20205 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1152 | 20206 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1153 | 20207 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1154 | 20208 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1155 | 20209 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1156 | 20210 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1157 | 20211 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1158 | 20212 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1159 | 20213 | CXXC[C[0,0],X[1,0|0,0],X[2,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1160 | 20250 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1161 | 20251 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1162 | 20253 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1163 | 20254 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1164 | 20256 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1165 | 20257 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1166 | 20259 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1167 | 20260 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1168 | 20261 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1169 | 20262 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1170 | 20263 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1171 | 20264 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1172 | 20265 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1173 | 20266 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1174 | 20267 | CXXC[C[0,0],X[1,0|0,0],X[2,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1175 | 20277 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1176 | 20278 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1177 | 20280 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[1,0]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1178 | 20281 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1179 | 20283 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1180 | 20284 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1181 | 20286 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1182 | 20287 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1183 | 20288 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1184 | 20289 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1185 | 20290 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1186 | 20291 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1187 | 20292 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1188 | 20293 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1189 | 20294 | CXXC[C[0,0],X[1,0|0,0],X[2,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1190 | 20304 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[0,0]] | 108 | 2 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1191 | 20305 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1192 | 20307 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[1,0]] | 108 | 2 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1193 | 20308 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1194 | 20310 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[2,0]] | 108 | 2 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1195 | 20311 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1196 | 20313 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1197 | 20314 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1198 | 20315 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1199 | 20316 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1200 | 20317 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1201 | 20318 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1202 | 20319 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1203 | 20320 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1204 | 20321 | CXXC[C[0,0],X[1,0|0,0],X[2,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1205 | 20412 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[0,0]] | 108 | 2 | (True, True, True, False, True, False, True, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 1206 | 20413 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 1207 | 20414 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[0,2]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 1208 | 20415 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[1,0]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 1209 | 20416 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, True, True, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 1210 | 20417 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[1,2]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 1211 | 20418 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[2,0]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 1212 | 20419 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 1213 | 20420 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,0],C[2,2]] | 108 | 2 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 1214 | 20421 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 1215 | 20422 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 1216 | 20423 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 1217 | 20424 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 1218 | 20425 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 1219 | 20426 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 1220 | 20427 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 1221 | 20428 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 1222 | 20429 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 1223 | 20430 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 1224 | 20431 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 1225 | 20432 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 1226 | 20433 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 1227 | 20434 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 1228 | 20435 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 1229 | 20436 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 1230 | 20437 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 1231 | 20438 | CXXC[C[0,0],X[1,0|0,1],X[0,0|0,2],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 1232 | 20439 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 1233 | 20440 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 1234 | 20441 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 1235 | 20442 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 1236 | 20443 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 1237 | 20444 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 1238 | 20445 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 1239 | 20446 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 1240 | 20447 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 1241 | 20448 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 1242 | 20449 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 1243 | 20450 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 1244 | 20451 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 1245 | 20452 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 1246 | 20453 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 1247 | 20454 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 1248 | 20455 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 1249 | 20456 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 1250 | 20457 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 1251 | 20458 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 1252 | 20459 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 1253 | 20460 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 1254 | 20461 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 1255 | 20462 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 1256 | 20463 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 1257 | 20464 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 1258 | 20465 | CXXC[C[0,0],X[1,0|0,1],X[0,0|1,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 1259 | 20493 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 1260 | 20494 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 1261 | 20495 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 1262 | 20496 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 1263 | 20497 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 1264 | 20498 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 1265 | 20499 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 1266 | 20500 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 1267 | 20501 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 1268 | 20502 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 1269 | 20503 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 1270 | 20504 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 1271 | 20505 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 1272 | 20506 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 1273 | 20507 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 1274 | 20508 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 1275 | 20509 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 1276 | 20510 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 1277 | 20511 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 1278 | 20512 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 1279 | 20513 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 1280 | 20514 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 1281 | 20515 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 1282 | 20516 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 1283 | 20517 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 1284 | 20518 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 1285 | 20519 | CXXC[C[0,0],X[1,0|0,1],X[0,1|0,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 1286 | 20520 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[0,0]] | 216 | 1 | (True, True, True, False, True, False, True, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 1287 | 20521 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 1288 | 20522 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[0,2]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 1289 | 20523 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[1,0]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 1290 | 20524 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, True, True, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 1291 | 20525 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[1,2]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 1292 | 20526 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[2,0]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 1293 | 20527 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 1294 | 20528 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,0],C[2,2]] | 216 | 1 | (True, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 1295 | 20529 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 1296 | 20530 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 1297 | 20531 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 1298 | 20532 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 1299 | 20533 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 1300 | 20534 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 1301 | 20535 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 1302 | 20536 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 1303 | 20537 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 1304 | 20538 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 1305 | 20539 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 1306 | 20540 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 1307 | 20541 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 1308 | 20542 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 1309 | 20543 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 1310 | 20544 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 1311 | 20545 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 1312 | 20546 | CXXC[C[0,0],X[1,0|0,1],X[0,1|1,2],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 1313 | 20547 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 1314 | 20548 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 1315 | 20549 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 1316 | 20550 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 1317 | 20551 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 1318 | 20552 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 1319 | 20553 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 1320 | 20554 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 1321 | 20555 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 1322 | 20556 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 1323 | 20557 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 1324 | 20558 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 1325 | 20559 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 1326 | 20560 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 1327 | 20561 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 1328 | 20562 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 1329 | 20563 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 1330 | 20564 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 1331 | 20565 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 1332 | 20566 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 1333 | 20567 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 1334 | 20568 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 1335 | 20569 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 1336 | 20570 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 1337 | 20571 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 1338 | 20572 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 1339 | 20573 | CXXC[C[0,0],X[1,0|0,1],X[0,1|2,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 1340 | 20655 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 1341 | 20656 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 1342 | 20657 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 1343 | 20658 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 1344 | 20659 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 1345 | 20660 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 1346 | 20661 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 1347 | 20662 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 1348 | 20663 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,0],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 1349 | 20664 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 1350 | 20665 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 1351 | 20666 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 1352 | 20667 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 1353 | 20668 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, True, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 1354 | 20669 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 1355 | 20670 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 1356 | 20671 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 1357 | 20672 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 1358 | 20673 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 1359 | 20674 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 1360 | 20675 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 1361 | 20676 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 1362 | 20677 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 1363 | 20678 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 1364 | 20679 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 1365 | 20680 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 1366 | 20681 | CXXC[C[0,0],X[1,0|0,1],X[1,0|0,2],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 1367 | 20682 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 1368 | 20683 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 1369 | 20684 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 1370 | 20685 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 1371 | 20686 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 1372 | 20687 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 1373 | 20688 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 1374 | 20689 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 1375 | 20690 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 1376 | 20691 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 1377 | 20692 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 1378 | 20693 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 1379 | 20694 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 1380 | 20695 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 1381 | 20696 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 1382 | 20697 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 1383 | 20698 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 1384 | 20699 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 1385 | 20700 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 1386 | 20701 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 1387 | 20702 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 1388 | 20703 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 1389 | 20704 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 1390 | 20705 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 1391 | 20706 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 1392 | 20707 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 1393 | 20708 | CXXC[C[0,0],X[1,0|0,1],X[1,0|1,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 1394 | 20736 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 1395 | 20737 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 1396 | 20738 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 1397 | 20739 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 1398 | 20740 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 1399 | 20741 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 1400 | 20742 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 1401 | 20743 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 1402 | 20744 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 1403 | 20745 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 1404 | 20746 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 1405 | 20747 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 1406 | 20748 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 1407 | 20749 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 1408 | 20750 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 1409 | 20751 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 1410 | 20752 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 1411 | 20753 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 1412 | 20754 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 1413 | 20755 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 1414 | 20756 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 1415 | 20757 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 1416 | 20758 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 1417 | 20759 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 1418 | 20760 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 1419 | 20761 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 1420 | 20762 | CXXC[C[0,0],X[1,0|0,1],X[1,1|0,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 1421 | 20763 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 1422 | 20764 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 1423 | 20765 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 1424 | 20766 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 1425 | 20767 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 1426 | 20768 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 1427 | 20769 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 1428 | 20770 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 1429 | 20771 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,0],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 1430 | 20772 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 1431 | 20773 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 1432 | 20774 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 1433 | 20775 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 1434 | 20776 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, True, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 1435 | 20777 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 1436 | 20778 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 1437 | 20779 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 1438 | 20780 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 1439 | 20781 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 1440 | 20782 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 1441 | 20783 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 1442 | 20784 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 1443 | 20785 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 1444 | 20786 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 1445 | 20787 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 1446 | 20788 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 1447 | 20789 | CXXC[C[0,0],X[1,0|0,1],X[1,1|1,2],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 1448 | 20790 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 1449 | 20791 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 1450 | 20792 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 1451 | 20793 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 1452 | 20794 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 1453 | 20795 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 1454 | 20796 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 1455 | 20797 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 1456 | 20798 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 1457 | 20799 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 1458 | 20800 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 1459 | 20801 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 1460 | 20802 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 1461 | 20803 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 1462 | 20804 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 1463 | 20805 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 1464 | 20806 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 1465 | 20807 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 1466 | 20808 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 1467 | 20809 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 1468 | 20810 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 1469 | 20811 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 1470 | 20812 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 1471 | 20813 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 1472 | 20814 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 1473 | 20815 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 1474 | 20816 | CXXC[C[0,0],X[1,0|0,1],X[1,1|2,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 1475 | 20898 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 1476 | 20899 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 1477 | 20900 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 1478 | 20901 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 1479 | 20902 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 1480 | 20903 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 1481 | 20904 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 1482 | 20905 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 1483 | 20906 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,0],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 1484 | 20907 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 1485 | 20908 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 1486 | 20909 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 1487 | 20910 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 1488 | 20911 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 1489 | 20912 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 1490 | 20913 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 1491 | 20914 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 1492 | 20915 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,1],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 1493 | 20916 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[0,0]] | 108 | 2 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 1494 | 20917 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[0,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 1495 | 20918 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[0,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 1496 | 20919 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[1,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 1497 | 20920 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[1,1]] | 108 | 2 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 1498 | 20921 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[1,2]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 1499 | 20922 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[2,0]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 1500 | 20923 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[2,1]] | 108 | 2 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 1501 | 20924 | CXXC[C[0,0],X[1,0|0,1],X[2,0|0,2],C[2,2]] | 108 | 2 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 1502 | 20925 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 1503 | 20926 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 1504 | 20927 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 1505 | 20928 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 1506 | 20929 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 1507 | 20930 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 1508 | 20931 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 1509 | 20932 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 1510 | 20933 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 1511 | 20934 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 1512 | 20935 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 1513 | 20936 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 1514 | 20937 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 1515 | 20938 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 1516 | 20939 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 1517 | 20940 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 1518 | 20941 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 1519 | 20942 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 1520 | 20943 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 1521 | 20944 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 1522 | 20945 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 1523 | 20946 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 1524 | 20947 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 1525 | 20948 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 1526 | 20949 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 1527 | 20950 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 1528 | 20951 | CXXC[C[0,0],X[1,0|0,1],X[2,0|1,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 1529 | 20979 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 1530 | 20980 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 1531 | 20981 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 1532 | 20982 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 1533 | 20983 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 1534 | 20984 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 1535 | 20985 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 1536 | 20986 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 1537 | 20987 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 1538 | 20988 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 1539 | 20989 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 1540 | 20990 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 1541 | 20991 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 1542 | 20992 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 1543 | 20993 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 1544 | 20994 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 1545 | 20995 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 1546 | 20996 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 1547 | 20997 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 1548 | 20998 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 1549 | 20999 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 1550 | 21000 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 1551 | 21001 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 1552 | 21002 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 1553 | 21003 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 1554 | 21004 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 1555 | 21005 | CXXC[C[0,0],X[1,0|0,1],X[2,1|0,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 1556 | 21006 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 1557 | 21007 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 1558 | 21008 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 1559 | 21009 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 1560 | 21010 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 1561 | 21011 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 1562 | 21012 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 1563 | 21013 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 1564 | 21014 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,0],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 1565 | 21015 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 1566 | 21016 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 1567 | 21017 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 1568 | 21018 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 1569 | 21019 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 1570 | 21020 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 1571 | 21021 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 1572 | 21022 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 1573 | 21023 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,1],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 1574 | 21024 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[0,0]] | 216 | 1 | (True, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 1575 | 21025 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[0,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 1576 | 21026 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[0,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 1577 | 21027 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[1,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 1578 | 21028 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[1,1]] | 216 | 1 | (True, True, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 1579 | 21029 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[1,2]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 1580 | 21030 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[2,0]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 1581 | 21031 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[2,1]] | 216 | 1 | (True, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 1582 | 21032 | CXXC[C[0,0],X[1,0|0,1],X[2,1|1,2],C[2,2]] | 216 | 1 | (True, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 1583 | 21033 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 1584 | 21034 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 1585 | 21035 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 1586 | 21036 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 1587 | 21037 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 1588 | 21038 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 1589 | 21039 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 1590 | 21040 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 1591 | 21041 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,0],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 1592 | 21042 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 1593 | 21043 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 1594 | 21044 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 1595 | 21045 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 1596 | 21046 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 1597 | 21047 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 1598 | 21048 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 1599 | 21049 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 1600 | 21050 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,1],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 1601 | 21051 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[0,0]] | 216 | 1 | (True, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 1602 | 21052 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[0,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 1603 | 21053 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[0,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 1604 | 21054 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[1,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 1605 | 21055 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[1,1]] | 216 | 1 | (True, False, False, False, False, True, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 1606 | 21056 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[1,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 1607 | 21057 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[2,0]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 1608 | 21058 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[2,1]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 1609 | 21059 | CXXC[C[0,0],X[1,0|0,1],X[2,1|2,2],C[2,2]] | 216 | 1 | (True, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 1610 | 21870 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1611 | 21871 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1612 | 21873 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1613 | 21874 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1614 | 21876 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[2,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1615 | 21877 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1616 | 21879 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1617 | 21880 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1618 | 21881 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1619 | 21882 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1620 | 21883 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1621 | 21884 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1622 | 21885 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1623 | 21886 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1624 | 21887 | CXXC[C[0,0],X[1,0|1,0],X[0,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1625 | 21897 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1626 | 21898 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1627 | 21900 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1628 | 21901 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1629 | 21903 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1630 | 21904 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1631 | 21906 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1632 | 21907 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1633 | 21908 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1634 | 21909 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1635 | 21910 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1636 | 21911 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1637 | 21912 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1638 | 21913 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1639 | 21914 | CXXC[C[0,0],X[1,0|1,0],X[0,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1640 | 21924 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1641 | 21925 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1642 | 21927 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1643 | 21928 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1644 | 21930 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1645 | 21931 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1646 | 21933 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1647 | 21934 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1648 | 21935 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1649 | 21936 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1650 | 21937 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1651 | 21938 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1652 | 21939 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1653 | 21940 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1654 | 21941 | CXXC[C[0,0],X[1,0|1,0],X[0,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1655 | 21951 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1656 | 21952 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1657 | 21954 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1658 | 21955 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1659 | 21957 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1660 | 21958 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1661 | 21960 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1662 | 21961 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1663 | 21962 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1664 | 21963 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1665 | 21964 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1666 | 21965 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1667 | 21966 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1668 | 21967 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1669 | 21968 | CXXC[C[0,0],X[1,0|1,0],X[0,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1670 | 21978 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1671 | 21979 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1672 | 21981 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1673 | 21982 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1674 | 21984 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[2,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1675 | 21985 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1676 | 21987 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1677 | 21988 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1678 | 21989 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1679 | 21990 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1680 | 21991 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1681 | 21992 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1682 | 21993 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1683 | 21994 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1684 | 21995 | CXXC[C[0,0],X[1,0|1,0],X[0,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1685 | 22005 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1686 | 22006 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1687 | 22008 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1688 | 22009 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1689 | 22011 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1690 | 22012 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1691 | 22014 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1692 | 22015 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1693 | 22016 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1694 | 22017 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1695 | 22018 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1696 | 22019 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1697 | 22020 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1698 | 22021 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1699 | 22022 | CXXC[C[0,0],X[1,0|1,0],X[0,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1700 | 22032 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1701 | 22033 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1702 | 22035 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1703 | 22036 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1704 | 22038 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1705 | 22039 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1706 | 22041 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1707 | 22042 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1708 | 22043 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1709 | 22044 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1710 | 22045 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1711 | 22046 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1712 | 22047 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1713 | 22048 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1714 | 22049 | CXXC[C[0,0],X[1,0|1,0],X[0,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1715 | 22059 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1716 | 22060 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1717 | 22062 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1718 | 22063 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1719 | 22065 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1720 | 22066 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1721 | 22068 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1722 | 22069 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1723 | 22070 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1724 | 22071 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1725 | 22072 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1726 | 22073 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1727 | 22074 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1728 | 22075 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1729 | 22076 | CXXC[C[0,0],X[1,0|1,0],X[0,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1730 | 22086 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[0,0]] | 108 | 2 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 0, 0, 0)) |
| 1731 | 22087 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 0, 0, 1)) |
| 1732 | 22089 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[1,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 0)) |
| 1733 | 22090 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 0, 0, 1)) |
| 1734 | 22092 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[2,0]] | 108 | 2 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 0)) |
| 1735 | 22093 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 0, 0, 1)) |
| 1736 | 22095 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 0)) |
| 1737 | 22096 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 0, 1, 1)) |
| 1738 | 22097 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 0, 1, 2)) |
| 1739 | 22098 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 0)) |
| 1740 | 22099 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 1)) |
| 1741 | 22100 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 0, 1, 2)) |
| 1742 | 22101 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 0)) |
| 1743 | 22102 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 1)) |
| 1744 | 22103 | CXXC[C[0,0],X[1,0|1,0],X[0,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 0, 1, 2)) |
| 1745 | 22113 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1746 | 22114 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1747 | 22116 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1748 | 22117 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1749 | 22119 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1750 | 22120 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1751 | 22122 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1752 | 22123 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1753 | 22124 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1754 | 22125 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1755 | 22126 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1756 | 22127 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1757 | 22128 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1758 | 22129 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1759 | 22130 | CXXC[C[0,0],X[1,0|1,0],X[1,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1760 | 22140 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1761 | 22141 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1762 | 22143 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1763 | 22144 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1764 | 22146 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1765 | 22147 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1766 | 22149 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1767 | 22150 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1768 | 22151 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1769 | 22152 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1770 | 22153 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1771 | 22154 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1772 | 22155 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1773 | 22156 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1774 | 22157 | CXXC[C[0,0],X[1,0|1,0],X[1,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1775 | 22167 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1776 | 22168 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1777 | 22170 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1778 | 22171 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1779 | 22173 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1780 | 22174 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1781 | 22176 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1782 | 22177 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1783 | 22178 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1784 | 22179 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1785 | 22180 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1786 | 22181 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1787 | 22182 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1788 | 22183 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1789 | 22184 | CXXC[C[0,0],X[1,0|1,0],X[1,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1790 | 22194 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1791 | 22195 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1792 | 22197 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1793 | 22198 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1794 | 22200 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1795 | 22201 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1796 | 22203 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1797 | 22204 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1798 | 22205 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1799 | 22206 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1800 | 22207 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1801 | 22208 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1802 | 22209 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1803 | 22210 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1804 | 22211 | CXXC[C[0,0],X[1,0|1,0],X[1,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1805 | 22221 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1806 | 22222 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1807 | 22224 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1808 | 22225 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1809 | 22227 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1810 | 22228 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1811 | 22230 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1812 | 22231 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1813 | 22232 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1814 | 22233 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1815 | 22234 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1816 | 22235 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1817 | 22236 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1818 | 22237 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1819 | 22238 | CXXC[C[0,0],X[1,0|1,0],X[1,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1820 | 22248 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1821 | 22249 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1822 | 22251 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1823 | 22252 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1824 | 22254 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1825 | 22255 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1826 | 22257 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1827 | 22258 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1828 | 22259 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1829 | 22260 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1830 | 22261 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1831 | 22262 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1832 | 22263 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1833 | 22264 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1834 | 22265 | CXXC[C[0,0],X[1,0|1,0],X[1,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1835 | 22275 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1836 | 22276 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1837 | 22278 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1838 | 22279 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1839 | 22281 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1840 | 22282 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1841 | 22284 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1842 | 22285 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1843 | 22286 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1844 | 22287 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1845 | 22288 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1846 | 22289 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1847 | 22290 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1848 | 22291 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1849 | 22292 | CXXC[C[0,0],X[1,0|1,0],X[1,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1850 | 22302 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1851 | 22303 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1852 | 22305 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1853 | 22306 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1854 | 22308 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1855 | 22309 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1856 | 22311 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1857 | 22312 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1858 | 22313 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1859 | 22314 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1860 | 22315 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1861 | 22316 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1862 | 22317 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1863 | 22318 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1864 | 22319 | CXXC[C[0,0],X[1,0|1,0],X[1,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1865 | 22329 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 0)) |
| 1866 | 22330 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 0, 1)) |
| 1867 | 22332 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 0, 0)) |
| 1868 | 22333 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 0, 1)) |
| 1869 | 22335 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 0)) |
| 1870 | 22336 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 0, 1)) |
| 1871 | 22338 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 0)) |
| 1872 | 22339 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 1)) |
| 1873 | 22340 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 0, 1, 2)) |
| 1874 | 22341 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 0)) |
| 1875 | 22342 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 0, 1, 1)) |
| 1876 | 22343 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 0, 1, 2)) |
| 1877 | 22344 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 0)) |
| 1878 | 22345 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 1)) |
| 1879 | 22346 | CXXC[C[0,0],X[1,0|1,0],X[1,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 0, 1, 2)) |
| 1880 | 22356 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1881 | 22357 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1882 | 22359 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1883 | 22360 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1884 | 22362 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1885 | 22363 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1886 | 22365 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1887 | 22366 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1888 | 22367 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1889 | 22368 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1890 | 22369 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1891 | 22370 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1892 | 22371 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1893 | 22372 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1894 | 22373 | CXXC[C[0,0],X[1,0|1,0],X[2,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1895 | 22383 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1896 | 22384 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1897 | 22386 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1898 | 22387 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1899 | 22389 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1900 | 22390 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1901 | 22392 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1902 | 22393 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1903 | 22394 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1904 | 22395 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1905 | 22396 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1906 | 22397 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1907 | 22398 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1908 | 22399 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1909 | 22400 | CXXC[C[0,0],X[1,0|1,0],X[2,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1910 | 22410 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1911 | 22411 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1912 | 22413 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1913 | 22414 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1914 | 22416 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1915 | 22417 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1916 | 22419 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1917 | 22420 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1918 | 22421 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1919 | 22422 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1920 | 22423 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1921 | 22424 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1922 | 22425 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1923 | 22426 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1924 | 22427 | CXXC[C[0,0],X[1,0|1,0],X[2,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1925 | 22437 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1926 | 22438 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1927 | 22440 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1928 | 22441 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1929 | 22443 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1930 | 22444 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1931 | 22446 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1932 | 22447 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1933 | 22448 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1934 | 22449 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1935 | 22450 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1936 | 22451 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1937 | 22452 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1938 | 22453 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1939 | 22454 | CXXC[C[0,0],X[1,0|1,0],X[2,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1940 | 22464 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1941 | 22465 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1942 | 22467 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1943 | 22468 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1944 | 22470 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1945 | 22471 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1946 | 22473 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1947 | 22474 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1948 | 22475 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1949 | 22476 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1950 | 22477 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1951 | 22478 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1952 | 22479 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1953 | 22480 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1954 | 22481 | CXXC[C[0,0],X[1,0|1,0],X[2,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1955 | 22491 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1956 | 22492 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1957 | 22494 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1958 | 22495 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1959 | 22497 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1960 | 22498 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1961 | 22500 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1962 | 22501 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1963 | 22502 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1964 | 22503 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1965 | 22504 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1966 | 22505 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1967 | 22506 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1968 | 22507 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1969 | 22508 | CXXC[C[0,0],X[1,0|1,0],X[2,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1970 | 22518 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1971 | 22519 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1972 | 22521 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1973 | 22522 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1974 | 22524 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1975 | 22525 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1976 | 22527 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1977 | 22528 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1978 | 22529 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1979 | 22530 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1980 | 22531 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1981 | 22532 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1982 | 22533 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1983 | 22534 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1984 | 22535 | CXXC[C[0,0],X[1,0|1,0],X[2,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 1985 | 22545 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[0,0]] | 108 | 2 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 1986 | 22546 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 1987 | 22548 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[1,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 1988 | 22549 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 1989 | 22551 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[2,0]] | 108 | 2 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 1990 | 22552 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 1991 | 22554 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 1992 | 22555 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 1993 | 22556 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 1994 | 22557 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 1995 | 22558 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 1996 | 22559 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 1997 | 22560 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 1998 | 22561 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 1999 | 22562 | CXXC[C[0,0],X[1,0|1,0],X[2,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 2000 | 22572 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[0,0]] | 108 | 2 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 0)) |
| 2001 | 22573 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 0, 1)) |
| 2002 | 22575 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[1,0]] | 108 | 2 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 0)) |
| 2003 | 22576 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 0, 1)) |
| 2004 | 22578 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[2,0]] | 108 | 2 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 0, 0)) |
| 2005 | 22579 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 0, 1)) |
| 2006 | 22581 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 0)) |
| 2007 | 22582 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 1)) |
| 2008 | 22583 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 0, 1, 2)) |
| 2009 | 22584 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 0)) |
| 2010 | 22585 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 1)) |
| 2011 | 22586 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 0, 1, 2)) |
| 2012 | 22587 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 0)) |
| 2013 | 22588 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 0, 1, 1)) |
| 2014 | 22589 | CXXC[C[0,0],X[1,0|1,0],X[2,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 0, 1, 2)) |
| 2015 | 22599 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[0,0]] | 216 | 1 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2016 | 22600 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2017 | 22601 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[0,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2018 | 22602 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2019 | 22603 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2020 | 22604 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2021 | 22605 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[2,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2022 | 22606 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2023 | 22607 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,0],C[2,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2024 | 22608 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2025 | 22609 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2026 | 22610 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2027 | 22611 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2028 | 22612 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2029 | 22613 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2030 | 22614 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2031 | 22615 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2032 | 22616 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2033 | 22617 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2034 | 22618 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2035 | 22619 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2036 | 22620 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2037 | 22621 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2038 | 22622 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2039 | 22623 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2040 | 22624 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2041 | 22625 | CXXC[C[0,0],X[1,0|1,1],X[0,0|0,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2042 | 22626 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2043 | 22627 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2044 | 22628 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2045 | 22629 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2046 | 22630 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2047 | 22631 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2048 | 22632 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2049 | 22633 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2050 | 22634 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2051 | 22635 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2052 | 22636 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2053 | 22637 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2054 | 22638 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2055 | 22639 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2056 | 22640 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2057 | 22641 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2058 | 22642 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2059 | 22643 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2060 | 22644 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2061 | 22645 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2062 | 22646 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2063 | 22647 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2064 | 22648 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2065 | 22649 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2066 | 22650 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2067 | 22651 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2068 | 22652 | CXXC[C[0,0],X[1,0|1,1],X[0,0|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2069 | 22653 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2070 | 22654 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2071 | 22655 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2072 | 22656 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2073 | 22657 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2074 | 22658 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2075 | 22659 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2076 | 22660 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2077 | 22661 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2078 | 22662 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2079 | 22663 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2080 | 22664 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2081 | 22665 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2082 | 22666 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2083 | 22667 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2084 | 22668 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2085 | 22669 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2086 | 22670 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2087 | 22671 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2088 | 22672 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2089 | 22673 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2090 | 22674 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2091 | 22675 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2092 | 22676 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2093 | 22677 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2094 | 22678 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2095 | 22679 | CXXC[C[0,0],X[1,0|1,1],X[0,0|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2096 | 22680 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2097 | 22681 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2098 | 22682 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2099 | 22683 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2100 | 22684 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2101 | 22685 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2102 | 22686 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2103 | 22687 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2104 | 22688 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2105 | 22689 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2106 | 22690 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2107 | 22691 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2108 | 22692 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2109 | 22693 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2110 | 22694 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2111 | 22695 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2112 | 22696 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2113 | 22697 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2114 | 22698 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2115 | 22699 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2116 | 22700 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2117 | 22701 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2118 | 22702 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2119 | 22703 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2120 | 22704 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2121 | 22705 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2122 | 22706 | CXXC[C[0,0],X[1,0|1,1],X[0,1|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2123 | 22707 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[0,0]] | 216 | 1 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2124 | 22708 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2125 | 22709 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[0,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2126 | 22710 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2127 | 22711 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2128 | 22712 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2129 | 22713 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[2,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2130 | 22714 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2131 | 22715 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,0],C[2,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2132 | 22716 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2133 | 22717 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2134 | 22718 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2135 | 22719 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2136 | 22720 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2137 | 22721 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2138 | 22722 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2139 | 22723 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2140 | 22724 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2141 | 22725 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2142 | 22726 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2143 | 22727 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2144 | 22728 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2145 | 22729 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2146 | 22730 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2147 | 22731 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2148 | 22732 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2149 | 22733 | CXXC[C[0,0],X[1,0|1,1],X[0,1|1,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2150 | 22734 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2151 | 22735 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2152 | 22736 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2153 | 22737 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2154 | 22738 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2155 | 22739 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2156 | 22740 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2157 | 22741 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2158 | 22742 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2159 | 22743 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2160 | 22744 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2161 | 22745 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2162 | 22746 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2163 | 22747 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2164 | 22748 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2165 | 22749 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2166 | 22750 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2167 | 22751 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2168 | 22752 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2169 | 22753 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2170 | 22754 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2171 | 22755 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2172 | 22756 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2173 | 22757 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2174 | 22758 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2175 | 22759 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2176 | 22760 | CXXC[C[0,0],X[1,0|1,1],X[0,1|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2177 | 22761 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2178 | 22762 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2179 | 22763 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2180 | 22764 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2181 | 22765 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2182 | 22766 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2183 | 22767 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2184 | 22768 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2185 | 22769 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2186 | 22770 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2187 | 22771 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2188 | 22772 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2189 | 22773 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2190 | 22774 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2191 | 22775 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2192 | 22776 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2193 | 22777 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2194 | 22778 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2195 | 22779 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2196 | 22780 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2197 | 22781 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2198 | 22782 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2199 | 22783 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2200 | 22784 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2201 | 22785 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2202 | 22786 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2203 | 22787 | CXXC[C[0,0],X[1,0|1,1],X[0,2|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2204 | 22788 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2205 | 22789 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2206 | 22790 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2207 | 22791 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2208 | 22792 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2209 | 22793 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2210 | 22794 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2211 | 22795 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2212 | 22796 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2213 | 22797 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2214 | 22798 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2215 | 22799 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2216 | 22800 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2217 | 22801 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2218 | 22802 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2219 | 22803 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2220 | 22804 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2221 | 22805 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2222 | 22806 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2223 | 22807 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2224 | 22808 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2225 | 22809 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2226 | 22810 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2227 | 22811 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2228 | 22812 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2229 | 22813 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2230 | 22814 | CXXC[C[0,0],X[1,0|1,1],X[0,2|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2231 | 22815 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[0,0]] | 216 | 1 | (False, True, True, False, True, False, True, (0, 1, 0, 0), (0, 1, 0, 0)) |
| 2232 | 22816 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 1)) |
| 2233 | 22817 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[0,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 0), (0, 1, 0, 2)) |
| 2234 | 22818 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[1,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 0)) |
| 2235 | 22819 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 1)) |
| 2236 | 22820 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[1,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 1), (0, 1, 0, 2)) |
| 2237 | 22821 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[2,0]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 0)) |
| 2238 | 22822 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 1)) |
| 2239 | 22823 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,0],C[2,2]] | 216 | 1 | (False, True, False, False, True, False, False, (0, 1, 0, 2), (0, 1, 0, 2)) |
| 2240 | 22824 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 0)) |
| 2241 | 22825 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 1, 1)) |
| 2242 | 22826 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 1, 2)) |
| 2243 | 22827 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 0)) |
| 2244 | 22828 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 1)) |
| 2245 | 22829 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 1, 2)) |
| 2246 | 22830 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 0)) |
| 2247 | 22831 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 1)) |
| 2248 | 22832 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 1, 2)) |
| 2249 | 22833 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 0)) |
| 2250 | 22834 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 0), (0, 1, 2, 1)) |
| 2251 | 22835 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 0, 0), (0, 1, 2, 2)) |
| 2252 | 22836 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 0)) |
| 2253 | 22837 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 1)) |
| 2254 | 22838 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 1), (0, 1, 2, 2)) |
| 2255 | 22839 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 0)) |
| 2256 | 22840 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 1)) |
| 2257 | 22841 | CXXC[C[0,0],X[1,0|1,1],X[0,2|2,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 0, 2), (0, 1, 2, 2)) |
| 2258 | 22842 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2259 | 22843 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2260 | 22844 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2261 | 22845 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2262 | 22846 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2263 | 22847 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2264 | 22848 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2265 | 22849 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2266 | 22850 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2267 | 22851 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2268 | 22852 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2269 | 22853 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2270 | 22854 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2271 | 22855 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2272 | 22856 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2273 | 22857 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2274 | 22858 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2275 | 22859 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2276 | 22860 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2277 | 22861 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2278 | 22862 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2279 | 22863 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2280 | 22864 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2281 | 22865 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2282 | 22866 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2283 | 22867 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2284 | 22868 | CXXC[C[0,0],X[1,0|1,1],X[1,0|0,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2285 | 22869 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2286 | 22870 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2287 | 22871 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2288 | 22872 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2289 | 22873 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2290 | 22874 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2291 | 22875 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2292 | 22876 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2293 | 22877 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2294 | 22878 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2295 | 22879 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2296 | 22880 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2297 | 22881 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2298 | 22882 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2299 | 22883 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2300 | 22884 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2301 | 22885 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2302 | 22886 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2303 | 22887 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2304 | 22888 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2305 | 22889 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2306 | 22890 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2307 | 22891 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2308 | 22892 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2309 | 22893 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2310 | 22894 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2311 | 22895 | CXXC[C[0,0],X[1,0|1,1],X[1,0|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2312 | 22896 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2313 | 22897 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2314 | 22898 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2315 | 22899 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2316 | 22900 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2317 | 22901 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2318 | 22902 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2319 | 22903 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2320 | 22904 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2321 | 22905 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2322 | 22906 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2323 | 22907 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2324 | 22908 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2325 | 22909 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2326 | 22910 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2327 | 22911 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2328 | 22912 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2329 | 22913 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2330 | 22914 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2331 | 22915 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2332 | 22916 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2333 | 22917 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2334 | 22918 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2335 | 22919 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2336 | 22920 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2337 | 22921 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2338 | 22922 | CXXC[C[0,0],X[1,0|1,1],X[1,0|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2339 | 22923 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2340 | 22924 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2341 | 22925 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2342 | 22926 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2343 | 22927 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2344 | 22928 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2345 | 22929 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2346 | 22930 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2347 | 22931 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2348 | 22932 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2349 | 22933 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2350 | 22934 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2351 | 22935 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2352 | 22936 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2353 | 22937 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2354 | 22938 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2355 | 22939 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2356 | 22940 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2357 | 22941 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2358 | 22942 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2359 | 22943 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2360 | 22944 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2361 | 22945 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2362 | 22946 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2363 | 22947 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2364 | 22948 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2365 | 22949 | CXXC[C[0,0],X[1,0|1,1],X[1,1|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2366 | 22950 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2367 | 22951 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2368 | 22952 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2369 | 22953 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2370 | 22954 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2371 | 22955 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2372 | 22956 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2373 | 22957 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2374 | 22958 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2375 | 22959 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2376 | 22960 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2377 | 22961 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2378 | 22962 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2379 | 22963 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2380 | 22964 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2381 | 22965 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2382 | 22966 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2383 | 22967 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2384 | 22968 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2385 | 22969 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2386 | 22970 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2387 | 22971 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2388 | 22972 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2389 | 22973 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2390 | 22974 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2391 | 22975 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2392 | 22976 | CXXC[C[0,0],X[1,0|1,1],X[1,1|1,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2393 | 22977 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2394 | 22978 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2395 | 22979 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2396 | 22980 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2397 | 22981 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2398 | 22982 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2399 | 22983 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2400 | 22984 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2401 | 22985 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2402 | 22986 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2403 | 22987 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2404 | 22988 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2405 | 22989 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2406 | 22990 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2407 | 22991 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2408 | 22992 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2409 | 22993 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2410 | 22994 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2411 | 22995 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2412 | 22996 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2413 | 22997 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2414 | 22998 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2415 | 22999 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2416 | 23000 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2417 | 23001 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2418 | 23002 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2419 | 23003 | CXXC[C[0,0],X[1,0|1,1],X[1,1|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2420 | 23004 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2421 | 23005 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2422 | 23006 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2423 | 23007 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2424 | 23008 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2425 | 23009 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2426 | 23010 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2427 | 23011 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2428 | 23012 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2429 | 23013 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2430 | 23014 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2431 | 23015 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2432 | 23016 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2433 | 23017 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2434 | 23018 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2435 | 23019 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2436 | 23020 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2437 | 23021 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2438 | 23022 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2439 | 23023 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2440 | 23024 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2441 | 23025 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2442 | 23026 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2443 | 23027 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2444 | 23028 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2445 | 23029 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2446 | 23030 | CXXC[C[0,0],X[1,0|1,1],X[1,2|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2447 | 23031 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2448 | 23032 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2449 | 23033 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2450 | 23034 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2451 | 23035 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2452 | 23036 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2453 | 23037 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2454 | 23038 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2455 | 23039 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2456 | 23040 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2457 | 23041 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2458 | 23042 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2459 | 23043 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2460 | 23044 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2461 | 23045 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2462 | 23046 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2463 | 23047 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2464 | 23048 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2465 | 23049 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2466 | 23050 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2467 | 23051 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2468 | 23052 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2469 | 23053 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2470 | 23054 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2471 | 23055 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2472 | 23056 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2473 | 23057 | CXXC[C[0,0],X[1,0|1,1],X[1,2|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2474 | 23058 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 0)) |
| 2475 | 23059 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 1)) |
| 2476 | 23060 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 0, 2)) |
| 2477 | 23061 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 0, 0)) |
| 2478 | 23062 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 1)) |
| 2479 | 23063 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 0, 2)) |
| 2480 | 23064 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 0)) |
| 2481 | 23065 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 1)) |
| 2482 | 23066 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 0, 2)) |
| 2483 | 23067 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 0)) |
| 2484 | 23068 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 1)) |
| 2485 | 23069 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 1, 2)) |
| 2486 | 23070 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 0)) |
| 2487 | 23071 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 1, 1)) |
| 2488 | 23072 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 1, 2)) |
| 2489 | 23073 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 0)) |
| 2490 | 23074 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 1)) |
| 2491 | 23075 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 1, 2)) |
| 2492 | 23076 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 0)) |
| 2493 | 23077 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 1)) |
| 2494 | 23078 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 0), (0, 1, 2, 2)) |
| 2495 | 23079 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 0)) |
| 2496 | 23080 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 1), (0, 1, 2, 1)) |
| 2497 | 23081 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 1, 1), (0, 1, 2, 2)) |
| 2498 | 23082 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 0)) |
| 2499 | 23083 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 1)) |
| 2500 | 23084 | CXXC[C[0,0],X[1,0|1,1],X[1,2|2,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 1, 2), (0, 1, 2, 2)) |
| 2501 | 23085 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2502 | 23086 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2503 | 23087 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2504 | 23088 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2505 | 23089 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2506 | 23090 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2507 | 23091 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2508 | 23092 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2509 | 23093 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2510 | 23094 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2511 | 23095 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2512 | 23096 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2513 | 23097 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2514 | 23098 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2515 | 23099 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2516 | 23100 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2517 | 23101 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2518 | 23102 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2519 | 23103 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2520 | 23104 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2521 | 23105 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2522 | 23106 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2523 | 23107 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2524 | 23108 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2525 | 23109 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2526 | 23110 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2527 | 23111 | CXXC[C[0,0],X[1,0|1,1],X[2,0|0,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2528 | 23112 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2529 | 23113 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2530 | 23114 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2531 | 23115 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2532 | 23116 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2533 | 23117 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2534 | 23118 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2535 | 23119 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2536 | 23120 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2537 | 23121 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2538 | 23122 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2539 | 23123 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2540 | 23124 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2541 | 23125 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2542 | 23126 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2543 | 23127 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2544 | 23128 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2545 | 23129 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2546 | 23130 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2547 | 23131 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2548 | 23132 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2549 | 23133 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2550 | 23134 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2551 | 23135 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2552 | 23136 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2553 | 23137 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2554 | 23138 | CXXC[C[0,0],X[1,0|1,1],X[2,0|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2555 | 23139 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2556 | 23140 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2557 | 23141 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2558 | 23142 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2559 | 23143 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2560 | 23144 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2561 | 23145 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2562 | 23146 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2563 | 23147 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2564 | 23148 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2565 | 23149 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2566 | 23150 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2567 | 23151 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2568 | 23152 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2569 | 23153 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2570 | 23154 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2571 | 23155 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2572 | 23156 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2573 | 23157 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2574 | 23158 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2575 | 23159 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2576 | 23160 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2577 | 23161 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2578 | 23162 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2579 | 23163 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2580 | 23164 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2581 | 23165 | CXXC[C[0,0],X[1,0|1,1],X[2,0|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2582 | 23166 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2583 | 23167 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2584 | 23168 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2585 | 23169 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2586 | 23170 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2587 | 23171 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2588 | 23172 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2589 | 23173 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2590 | 23174 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2591 | 23175 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2592 | 23176 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2593 | 23177 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2594 | 23178 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2595 | 23179 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2596 | 23180 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2597 | 23181 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2598 | 23182 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2599 | 23183 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2600 | 23184 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2601 | 23185 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2602 | 23186 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2603 | 23187 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2604 | 23188 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2605 | 23189 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2606 | 23190 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2607 | 23191 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2608 | 23192 | CXXC[C[0,0],X[1,0|1,1],X[2,1|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2609 | 23193 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2610 | 23194 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2611 | 23195 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2612 | 23196 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2613 | 23197 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2614 | 23198 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2615 | 23199 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2616 | 23200 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2617 | 23201 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2618 | 23202 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2619 | 23203 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2620 | 23204 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2621 | 23205 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2622 | 23206 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2623 | 23207 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2624 | 23208 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2625 | 23209 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2626 | 23210 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2627 | 23211 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2628 | 23212 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2629 | 23213 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2630 | 23214 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2631 | 23215 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2632 | 23216 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2633 | 23217 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2634 | 23218 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2635 | 23219 | CXXC[C[0,0],X[1,0|1,1],X[2,1|1,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2636 | 23220 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2637 | 23221 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2638 | 23222 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2639 | 23223 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2640 | 23224 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2641 | 23225 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2642 | 23226 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2643 | 23227 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2644 | 23228 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2645 | 23229 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2646 | 23230 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2647 | 23231 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2648 | 23232 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2649 | 23233 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2650 | 23234 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2651 | 23235 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2652 | 23236 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2653 | 23237 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2654 | 23238 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2655 | 23239 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2656 | 23240 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2657 | 23241 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2658 | 23242 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2659 | 23243 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2660 | 23244 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2661 | 23245 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2662 | 23246 | CXXC[C[0,0],X[1,0|1,1],X[2,1|2,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2663 | 23247 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2664 | 23248 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2665 | 23249 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2666 | 23250 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2667 | 23251 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2668 | 23252 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2669 | 23253 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2670 | 23254 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2671 | 23255 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2672 | 23256 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2673 | 23257 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2674 | 23258 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2675 | 23259 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2676 | 23260 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2677 | 23261 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2678 | 23262 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2679 | 23263 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2680 | 23264 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2681 | 23265 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2682 | 23266 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2683 | 23267 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2684 | 23268 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2685 | 23269 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2686 | 23270 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2687 | 23271 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2688 | 23272 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2689 | 23273 | CXXC[C[0,0],X[1,0|1,1],X[2,2|0,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2690 | 23274 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2691 | 23275 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2692 | 23276 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2693 | 23277 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2694 | 23278 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2695 | 23279 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2696 | 23280 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2697 | 23281 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2698 | 23282 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,0],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2699 | 23283 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2700 | 23284 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2701 | 23285 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2702 | 23286 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2703 | 23287 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2704 | 23288 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2705 | 23289 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2706 | 23290 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2707 | 23291 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,1],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2708 | 23292 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[0,0]] | 216 | 1 | (False, False, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2709 | 23293 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[0,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2710 | 23294 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[0,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2711 | 23295 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[1,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2712 | 23296 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[1,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2713 | 23297 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[1,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2714 | 23298 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[2,0]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2715 | 23299 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[2,1]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2716 | 23300 | CXXC[C[0,0],X[1,0|1,1],X[2,2|1,2],C[2,2]] | 216 | 1 | (False, False, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 2)) |
| 2717 | 23301 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 0)) |
| 2718 | 23302 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 1)) |
| 2719 | 23303 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 0, 2)) |
| 2720 | 23304 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 0)) |
| 2721 | 23305 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 1)) |
| 2722 | 23306 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 0, 2)) |
| 2723 | 23307 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 0, 0)) |
| 2724 | 23308 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 1)) |
| 2725 | 23309 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,0],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 0, 2)) |
| 2726 | 23310 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 0)) |
| 2727 | 23311 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 1)) |
| 2728 | 23312 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 1, 2)) |
| 2729 | 23313 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 0)) |
| 2730 | 23314 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 1)) |
| 2731 | 23315 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 1, 2)) |
| 2732 | 23316 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 0)) |
| 2733 | 23317 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 1, 1)) |
| 2734 | 23318 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,1],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 1, 2)) |
| 2735 | 23319 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[0,0]] | 216 | 1 | (False, True, True, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 0)) |
| 2736 | 23320 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[0,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 1)) |
| 2737 | 23321 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[0,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 0), (0, 1, 2, 2)) |
| 2738 | 23322 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[1,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 0)) |
| 2739 | 23323 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[1,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 1)) |
| 2740 | 23324 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[1,2]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 1), (0, 1, 2, 2)) |
| 2741 | 23325 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[2,0]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 0)) |
| 2742 | 23326 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[2,1]] | 216 | 1 | (False, True, False, False, False, False, False, (0, 1, 2, 2), (0, 1, 2, 1)) |
| 2743 | 23327 | CXXC[C[0,0],X[1,0|1,1],X[2,2|2,2],C[2,2]] | 216 | 1 | (False, True, False, False, False, False, True, (0, 1, 2, 2), (0, 1, 2, 2)) |

**Verification**: Sum of orbit sizes = 531441
**Verification**: All orbit_size × stabilizer_size = 216 (checked)

## 16. CX × XC → CC COMPOSITION GRID

[EXACT_DERIVED] / [POST-REPAIR]

Complete composition mapping for all realized orbit pairs:

| CX_orbit | XC_orbit | CC_orbits | Deterministic? | Witness Count |
|----------|----------|-----------|----------------|---------------|
| 0 | 0 | [0] | Yes | 27 |
| 0 | 1 | [1] | Yes | 54 |
| 0 | 2 | [2] | Yes | 54 |
| 0 | 3 | [3] | Yes | 108 |
| 1 | 0 | [1] | Yes | 54 |
| 1 | 1 | [0, 1] | No (Mixed) | 108 |
| 1 | 2 | [3] | Yes | 108 |
| 1 | 3 | [2, 3] | No (Mixed) | 216 |
| 2 | 4 | [0] | Yes | 54 |
| 2 | 5 | [1] | Yes | 108 |
| 2 | 6 | [2] | Yes | 108 |
| 2 | 7 | [3] | Yes | 216 |
| 3 | 4 | [1] | Yes | 108 |
| 3 | 5 | [0, 1] | No (Mixed) | 216 |
| 3 | 6 | [3] | Yes | 216 |
| 3 | 7 | [2, 3] | No (Mixed) | 432 |
| 4 | 0 | [2] | Yes | 54 |
| 4 | 1 | [3] | Yes | 108 |
| 4 | 2 | [0, 2] | No (Mixed) | 108 |
| 4 | 3 | [1, 3] | No (Mixed) | 216 |
| 5 | 0 | [3] | Yes | 108 |
| 5 | 1 | [2, 3] | No (Mixed) | 216 |
| 5 | 2 | [1, 3] | No (Mixed) | 216 |
| 5 | 3 | [0, 1, 2, 3] | No (Mixed) | 432 |
| 6 | 4 | [2] | Yes | 108 |
| 6 | 5 | [3] | Yes | 216 |
| 6 | 6 | [0, 2] | No (Mixed) | 216 |
| 6 | 7 | [1, 3] | No (Mixed) | 432 |
| 7 | 4 | [3] | Yes | 216 |
| 7 | 5 | [2, 3] | No (Mixed) | 432 |
| 7 | 6 | [1, 3] | No (Mixed) | 432 |
| 7 | 7 | [0, 1, 2, 3] | No (Mixed) | 864 |

**Summary**:
- Total realized pairs: 32
- Deterministic (single output): 18
- Mixed (multiple outputs): 14
- Possible pairs (8 × 8): 64

## 17. SIGNATURE COLLISION REFINEMENT

[EXACT_DERIVED] / [POST-REFINEMENT]

This section records the minimal additional Boolean features needed to resolve
all signature collisions in the XX, AX, and BX schemas.

### XX Schema Refinement

**Base collisions:** 4 groups (8 orbits)

Refinement features: `(s2, t2, u2, r2)`

Where for XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]], the features are coordinates
of the second X atom that vary within collision groups.

**Result:** 56 distinct refined signatures - all collisions resolved

### AX Schema Refinement

**Base collisions:** 2 groups (4 orbits)

Refinement features: `(t, u)`

Where for AX[A[r_a,s_a], X[r,s|t,u]], the features are the row and column
indices of the X atom's right part.

**Result:** 10 distinct refined signatures - all collisions resolved

### BX Schema Refinement

**Base collisions:** 2 groups (4 orbits)

Refinement features: `(s, r)`

Where for BX[B[t_b,u_b], X[r,s|t,u]], the features are the row and column
indices of the X atom's left part.

**Result:** 10 distinct refined signatures - all collisions resolved

**Exported Files:**
- `signatures_XX_refined.csv`
- `signatures_AX_refined.csv`
- `signatures_BX_refined.csv`
- `signature_refinement_summary.md`

## 18. CXXC MARGINAL PROJECTION ANALYSIS

[EXACT_DERIVED]

Analysis of how CXXC projects onto its natural marginal schemas:

| Projection | Target Schema | Realized Orbits | Total Orbits | Coverage |
|------------|---------------|-----------------|--------------|----------|
| CXC_left (C1,X1,C2) | CXC | 50 | 50 | 100.0% |
| CXC_right (C1,X2,C2) | CXC | 50 | 50 | 100.0% |
| XX (X1,X2) | XX | 56 | 56 | 100.0% |
| CX_left (C1,X1) | CX | 8 | 8 | 100.0% |
| CX_right (C1,X2) | CX | 8 | 8 | 100.0% |
| XC_left (X1,C2) | XC | 8 | 8 | 100.0% |
| XC_right (X2,C2) | XC | 8 | 8 | 100.0% |

**Key Result:** CXXC achieves 100% coverage on all marginal projections.
Every orbit of each lower-arity schema appears in at least one CXXC configuration.

**Exported File:** `cxxc_marginal_analysis.md`

## 19. SIGNATURE FORMAT DEFINITIONS

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

Signature tuple fields for each schema:

### CC Signature
Fields: (same_cell, same_row, same_col)

### CX Signature
Fields: (x_live, c_is_target_of_x_if_live, c_row_equals_x_row, c_col_equals_x_output_col)

### XC Signature
Fields: (x_live, c_is_target_of_x_if_live, x_row_equals_c_row, x_output_col_equals_c_col)

### AX Signature
Fields: (a_row_equals_x_left_row, a_col_equals_x_left_col, x_live)

### BX Signature
Fields: (b_row_equals_x_right_row, b_col_equals_x_right_col, x_live)

### CXC Signature
Fields: (x_live, c1_equals_c2, c1_is_target_if_live, c2_is_target_if_live,
         row_triple(c1,x,c2), col_triple(c1,x,c2))

**row_triple and col_triple encoding**: Encodes the equality partition of three
indices as a canonical representative tuple. Examples:
- (0,0,0) = all three equal
- (0,0,1) = first two equal, third distinct
- (0,1,0) = first and third equal, second distinct
- (0,1,2) = all three distinct

### XX Signature
Fields: (x1_live, x2_live, same_r, same_s, same_t, same_u, same_A_atom, same_B_atom)

**Field semantics**: For XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]]:
- same_A_atom ≡ (r1==r2 ∧ s1==s2)
- same_B_atom ≡ (t1==t2 ∧ u1==u2)

### CXXC Signature
Fields: (x1_live, x2_live, c1_equals_c2, c1_is_target_of_x1_if_live,
         c1_is_target_of_x2_if_live, c2_is_target_of_x1_if_live,
         c2_is_target_of_x2_if_live, row_quad, col_quad)

**Field semantics**: For CXXC[C[r1,u1], X[r2,s2|t2,u2], X[r4,s4|t4,u4], C[r3,u3]]:
- x1_live, x2_live: Boolean liveness of each X atom
- c1_equals_c2: Whether first and last C atoms are equal
- Target flags: Whether C atoms are targets of live X atoms
- row_quad, col_quad: Canonical encoding of equality partitions of (r1,r2,r4,r3) and (u1,u2,u4,u3)

## 20. SIGNATURE COLLISION ANALYSIS

[MEASURED_FROM_CODE] / [POST-REPAIR]

### Orbit-Complete Schemas

These schemas have unique signatures for every orbit:
- **CC**: 4 orbits → 4 signatures
- **CX**: 8 orbits → 8 signatures
- **XC**: 8 orbits → 8 signatures
- **CXC**: 50 orbits → 50 signatures

### Schemas with Signature Collisions

**XX Schema** (56 orbits → 48 distinct signatures):
- 4 signature groups contain 3 orbits each (8 collisions total)
- Collision indicates orbits share the same 8-tuple boolean signature

Collision groups:
- (False, False, False, False, False, False, False, False) → orbits {45, 49, 51}
- (False, False, False, False, False, True, False, False) → orbits {44, 48, 50}
- (False, False, True, False, False, False, False, False) → orbits {27, 31, 33}
- (False, False, True, False, False, True, False, False) → orbits {26, 30, 32}

**AX Schema** (10 orbits → 8 distinct signatures):
- 2 signature groups contain 2 orbits each

Collision groups:
- (True, False, False) → orbits {2, 4}
- (False, False, False) → orbits {7, 9}

**BX Schema** (10 orbits → 8 distinct signatures):
- 2 signature groups contain 2 orbits each

Collision groups:
- (False, True, False) → orbits {2, 8}
- (False, False, False) → orbits {3, 9}

## 21. OBJECT VS LENS DISTINCTION

[GROUND_TRUTH]

### The Object

Core structure that defines the mathematical object:
- Full raw base-9 warehouse (all 9^k tuples)
- Typed species A, B, C, X
- Typed schema definitions (including CXXC)
- Primitive exact rules (live/dead, target map, fibers)
- Typed/raw bridge embeddings

### Organizational Lenses (Not the Object)

Computed views that organize but don't change the object:
- Orbit metadata caches
- Signature caches
- Stabilizer element lists
- Canonicalization tables
- Composition caches
- Cross-schema alignment tables

**Critical**: A lens update does not change the underlying object.

## 22. PROVENANCE / EVIDENCE LABELS

[GROUND_TRUTH]

Evidence taxonomy used in this document:

- [GROUND_TRUTH] - Core definition, object structure
- [EXACT_DERIVED] - Results derived from ground truth by exact computation
- [MEASURED_FROM_CODE] - Measured numeric outputs from executed code
- [POST-REPAIR] - Computed after orbit metadata bug fix (step 10b)
- [INTERPRETATION] - Analysis or interpretation, not ground truth
- [SUPERSEDED] - Historical result replaced by corrected measurement
- [OPEN_FRONT] - Known gaps or incomplete areas

## 23. CORRECTION NOTE: ORBIT METADATA REPAIR

[POST-REPAIR]

### Bug

The original step10 orbit metadata cache keyed records by orbit_id and
implicitly used orbit_id as a config_id index into the configs list.
Since orbit_id != canonical representative config_id for most orbits,
every signature was computed from the wrong representative config.

### Impact

Old cached signature counts were partly wrong:

| Schema | Old (buggy) | Repaired | Changed |
|--------|-------------|----------|---------|
| XX     | 20          | 48       | yes     |
| CX     | 4           | 8        | yes     |
| XC     | 4           | 8        | yes     |
| CC     | 3           | 4        | yes     |
| AX     | 3           | 8        | yes     |
| BX     | 8           | 8        | no      |
| CXC    | 50          | 50       | no      |

### Resolution

- Repaired orbit metadata now uses actual canonical orbit representatives
- Signatures are computed from the correct representative config
- All orbit rosters in this document are POST-REPAIR

### Superseded Results

[SUPERSEDED]

The following results from earlier project versions are replaced:
- Old composition: 256 possible/64 realized/36 deterministic/28 mixed
  (type-based layer, now superseded by orbit-based: 64/32/18/14)
- Old BX orbit count: 18 (bug in B-action, corrected to 10)

## 24. CURRENT GAPS / OPEN FRONTS

[OPEN_FRONT]

**Completed in this session:**
- CXXC schema (arity-4): ✓ 2744 orbits computed with signatures
- XX, AX, BX signature collisions: ✓ All resolved with minimal refinement features
- CXXC marginal projections: ✓ Full coverage analysis completed

**Remaining open fronts:**
- CXXC signature collisions: 2744 orbits → 784 signatures (1960 collisions)
  Refinement for CXXC would require arity-4 specific features
- Additional arity-4 schemas: XCXC, XCCX, XXXC, XXX not yet explored
- Refinement engine: Not yet rerun on corrected composition (14 mixed keys)
- Higher arity layers: Arity 5+ unexplored

These are genuine incompletions, not promises.

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------

This canonical dossier contains all computed orbit rosters, the complete
composition grid, and all ground-truth structural data.

No external files are required. This document is standalone and complete.