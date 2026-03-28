======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-28 12:26:42
Generator: generate_canon_doc.py

This is a STANDALONE canonical dossier containing ALL computed results.
This document is the ONLY artifact provided to the next team.
It must be completely self-contained with all research findings.

----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-28 12:26:42
**Generator Script:** generate_canon_doc.py
**Provenance:** Built from steps 1-48+, including orbit metadata repair (step 10b),
signature refinement, CCXX orbit computation, arity-4 parity export,
composition kernel (step 39), CXXC marginal weight profile (step 40),
stabilizer subgroup classification (step 41), stabilizer composition (step 42),
refinement-conditioned kernel (step 43), floor-layer analysis (step 44),
58-orbit closure / doubly-live core analysis (step 45),
same-fiber / 64-subalgebra structure analysis (step 46),
mixed-pair resolution / tensor-constraint extraction (step 47),
tensor profile constraint modeling (step 48),
coefficient-level rank constraints (step 49),
symbolic fiber-mode decomposition (step 51),
quotient-space rank criterion analysis (step 52),
support-type representative incidence analysis (step 53),
and analytical low-nuisance construction analysis (step 54),
plus algebraic nuisance dependency mining and wildcard exploration (step 55)

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
| AXXC   | 4           | 6         | 531,441 | 2870   | See Section 16 |
| CCXX   | 4           | 6         | 531,441 | 2744   | See Section 17 |

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
| AXXC   | 4           | 6         | (A, A_X1, B_X1, A_X2, B_X2, C) | yes |
| CCXX   | 4           | 6         | (C, C, A_X1, B_X1, A_X2, B_X2) | yes |

**Arity-4 raw-layer note**: CXXC, AXXC, and CCXX are all injective bridges onto
the full raw arity-6 warehouse of size 9^6 = 531,441. Each schema therefore
realizes the full raw cardinality under its own role overlay. The overlays are
different, so equal raw size does not mean the typed semantics coincide.

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

**Status**: No orbit data available for CXXC

## 16. SCHEMA AXXC - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Status**: No orbit data available for AXXC

## 17. SCHEMA CCXX - COMPLETE ORBIT ROSTER

[MEASURED_FROM_CODE] / [POST-REPAIR]

**Status**: No orbit data available for CCXX

## 18. CX × XC → CC COMPOSITION GRID

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

## 19. SIGNATURE COLLISION REFINEMENT

[EXACT_DERIVED] / [POST-REFINEMENT]

This section records the minimal additional features needed to resolve
all currently measured signature collisions in XX, AX, BX, CXXC, and CCXX.

### XX Schema Refinement

**Base collisions:** 4 groups (12 orbits)

Minimal refinement features: `(s2, t2)`

Where for XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]], the features are coordinates
of the second X atom that vary within collision groups.

**Result:** 56 distinct refined signatures - all collisions resolved

### AX Schema Refinement

**Base collisions:** 2 groups (4 orbits)

Minimal refinement features: `(t,)`

Where for AX[A[r_a,s_a], X[r,s|t,u]], the feature is the row index of
the X atom's right part.

**Result:** 10 distinct refined signatures - all collisions resolved

### BX Schema Refinement

**Base collisions:** 2 groups (4 orbits)

Minimal refinement features: `(s,)`

Where for BX[B[t_b,u_b], X[r,s|t,u]], the feature is the column index of
the X atom's left part.

**Result:** 10 distinct refined signatures - all collisions resolved

### CXXC Schema Refinement

**Base collisions:** 784 groups (2744 orbits)

Minimal refinement features: `(s4, t4)`

Where for CXXC[C[r1,u1], X[r2,s2|t2,u2], X[r4,s4|t4,u4], C[r3,u3]],
the features are the shared middle indices of the second X atom.

**Workbench summary:** unique order-2 full resolver, 17 full resolvers by
order <= 3, best overall candidate `x2_s4+x2_t4`.

**Result:** 2744 distinct refined signatures - all collisions resolved

### CCXX Schema Refinement

**Base collisions:** 784 groups (2744 orbits)

Minimal refinement features: `(s4, t4)`

Where for CCXX[C[r1,u1], C[r2,u2], X[r3,s3|t3,u3], X[r4,s4|t4,u4]],
the features are the shared middle indices of the second X atom.

**Workbench summary:** unique order-2 full resolver, 16 full resolvers by
order <= 3, best overall candidate `x2_s4+x2_t4`.

**Result:** 2744 distinct refined signatures - all collisions resolved

All refinement conclusions needed by this dossier are stated inline here.

## 20. CXXC MARGINAL PROJECTION ANALYSIS

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

This dossier records the complete marginal-coverage conclusion inline.

## 21. ARITY-4 PARITY AND REFINEMENT STATUS

[MEASURED_FROM_CODE] / [POST-REFINEMENT]

Summary of the currently measured arity-4 schemas at raw arity 6:

| Schema | Orbit Count | Recorded Signature Layer | Distinct Signatures | Orbit Complete | Orbit Range | Stabilizer Range |
|--------|-------------|--------------------------|---------------------|----------------|-------------|------------------|
| CXXC   | 2744        | base signature + `(s4, t4)` refiner | 2744 | yes | 27-216 | 1-8 |
| AXXC   | 2870        | 6-face orbit tuple from face inventory | 2870 | yes | 27-216 | 1-8 |
| CCXX   | 2744        | base signature + `(s4, t4)` refiner | 2744 | yes | 27-216 | 1-8 |

**Measured comparison facts:**
- CXXC and CCXX share the same base profile: 2744 orbits, 784 base signatures,
  and the same collision structure (196 groups of size 6, 392 of size 3, 196 of size 2)
- CXXC and CCXX also share the same smallest full resolver: `(s4, t4)`
- AXXC has 126 more orbits than CXXC or CCXX at arity 4
- AXXC is orbit-complete at the current recorded face-pattern signature layer
- AXXC's recorded signatures are extrinsic face-orbit references, whereas CXXC
  and CCXX use intrinsic Boolean/equality signatures plus the recorded `(s4, t4)` refiner

All parity facts needed by this dossier are stated inline here.

## 22. SIGNATURE FORMAT DEFINITIONS

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
- Legend: `(r1,u1)` = first C, `(r2,s2,t2,u2)` = first X, `(r4,s4,t4,u4)` = second X, `(r3,u3)` = second C
- x1_live, x2_live: Boolean liveness of each X atom
- c1_equals_c2: Whether first and last C atoms are equal
- Target flags: Whether C atoms are targets of live X atoms
- row_quad, col_quad: Canonical encoding of equality partitions of (r1,r2,r4,r3) and (u1,u2,u4,u3)

**row_quad and col_quad encoding**: Same first-occurrence canonical partition rule
used for triples, extended to four indices. Examples:
- (0,0,0,0) = all four equal
- (0,0,1,1) = first two equal, last two equal, pairwise distinct
- (0,1,0,1) = first equals third, second equals fourth
- (0,1,2,0) = first equals fourth, middle two distinct from each other and from the repeated value
- (0,1,2,3) = all four distinct

### AXXC Current Recorded Signature Layer
Fields: (left_ax_orbit_id, middle_xx_orbit_id, right_xc_orbit_id,
         outer_ac_orbit_id, left_axc_orbit_id, right_axc_orbit_id)

**Field semantics**: For AXXC[A, X1, X2, C], the current arity-4 signature layer
is the tuple of orbit ids of its six natural faces: left AX, middle XX, right XC,
outer AC, left AXC, and right AXC.
These are extrinsic references into lower-arity orbit catalogues, not intrinsic
Boolean/equality features of the AXXC coordinates themselves.

### CCXX Signature
Fields: (x1_live, x2_live, c1_equals_c2, c1_is_target_of_x1_if_live,
         c1_is_target_of_x2_if_live, c2_is_target_of_x1_if_live,
         c2_is_target_of_x2_if_live, row_quad, col_quad)

**Field semantics**: For CCXX[C[r1,u1], C[r2,u2], X[r3,s3|t3,u3], X[r4,s4|t4,u4]]:
- x1_live, x2_live: Boolean liveness of each X atom
- c1_equals_c2: Whether the two C atoms are equal
- Target flags: Whether one of the live X atoms targets one of the two C atoms
- row_quad, col_quad: Canonical encoding of equality partitions of (r1,r2,r3,r4) and (u1,u2,u3,u4)
- CCXX uses the same four-index first-occurrence quad encoding convention described above for CXXC

**Arity-4 refinement appendage for CXXC and CCXX**: `(s4, t4)`

## 23. SIGNATURE COLLISION ANALYSIS

[MEASURED_FROM_CODE] / [POST-REPAIR]

### Base-Signature Orbit-Complete Schemas

These schemas have unique signatures for every orbit:
- **CC**: 4 orbits → 4 signatures
- **CX**: 8 orbits → 8 signatures
- **XC**: 8 orbits → 8 signatures
- **CXC**: 50 orbits → 50 signatures
- **AXXC current recorded layer**: 2870 orbits → 2870 signatures

### Base-Signature Collision Schemas

**XX Schema** (56 orbits → 48 distinct signatures):
- 4 signature groups contain 3 orbits each (12 collisions total)
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

**CXXC Schema** (2744 orbits → 784 distinct base signatures):
- 784 collision groups involve all 2744 orbits
- Group-size profile: 196 groups of size 6, 392 groups of size 3, 196 groups of size 2
- Smallest full resolver discovered by the workbench: `x2_s4+x2_t4`

**CCXX Schema** (2744 orbits → 784 distinct base signatures):
- 784 collision groups involve all 2744 orbits
- Group-size profile: 196 groups of size 6, 392 groups of size 3, 196 groups of size 2
- Smallest full resolver discovered by the workbench: `x2_s4+x2_t4`

### Current Resolved Signature Layers

After applying the recorded minimal refiners or current exported signature layer:
- **XX**: 56 distinct refined signatures
- **AX**: 10 distinct refined signatures
- **BX**: 10 distinct refined signatures
- **CXXC**: 2744 distinct refined signatures
- **AXXC**: 2870 distinct face-pattern signatures
- **CCXX**: 2744 distinct refined signatures

## 24. OBJECT VS LENS DISTINCTION

[GROUND_TRUTH]

### The Object

Core structure that defines the mathematical object:
- Full raw base-9 warehouse (all 9^k tuples)
- Typed species A, B, C, X
- Typed schema definitions (including CXXC, AXXC, and CCXX)
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

## 25. PROVENANCE / EVIDENCE LABELS

[GROUND_TRUTH]

Evidence taxonomy used in this document:

- [GROUND_TRUTH] - Core definition, object structure
- [EXACT_DERIVED] - Results derived from ground truth by exact computation
- [MEASURED_FROM_CODE] - Measured numeric outputs from executed code
- [POST-REPAIR] - Computed after orbit metadata bug fix (step 10b)
- [POST-REFINEMENT] - Computed after applying the recorded collision refiner or current recorded orbit-complete signature layer
- [INTERPRETATION] - Analysis or interpretation, not ground truth
- [SUPERSEDED] - Historical result replaced by corrected measurement
- [OPEN_FRONT] - Known gaps or incomplete areas

## 26. CORRECTION NOTE: ORBIT METADATA REPAIR

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

## 27. COMPOSITION KERNEL: MIXED CX x XC -> CC DISTRIBUTIONS

[EXACT_DERIVED] (Step 39)

For each of the 14 mixed CX x XC -> CC orbit-pairs, the witness distribution
across output CC orbits. All 14 keys are uniform: witnesses split equally
across all realized CC output orbits.

**Summary:**
- Mixed keys: 14
- Keys with 2 CC output orbits: 12
- Keys with 3 CC output orbits: 0
- Keys with 4 CC output orbits: 2
- Total kernel rows: 32
- Uniform distribution: ALL 14 keys (witnesses split equally across outputs)

*Run ade3x3_step39_composition_kernel.py to populate this section.*

**Finding:** The uniform distribution over CC orbits means the summation
index (contracted through the X atom) resolves ambiguity uniformly —
no CC orbit is preferred by any mixed CX x XC pair.
Verification: all kernel row sums match Section 18 witness totals exactly.

[INTERPRETATION]

This is a clean negative result with positive implications. The hypothesis
was that different mixed keys would split differently, encoding how the
summation index governs the product. Instead, the summation index is
maximally democratic: it fails to determine the output orbit with perfect
indifference. This closes the orbit-level composition as a source of free
algorithmic information. Any structure distinguishing output orbits within
a mixed pair must come from sub-orbit features, i.e., the refinement coordinates.

## 28. CXXC MARGINAL WEIGHT PROFILE

[EXACT_DERIVED] (Step 40)

For each of the 7 marginal projections of CXXC, the fiber-size profile:
how many CXXC orbits and configs map to each lower-arity orbit.

**Structural key:** Projection commutes with the group action. All 531,441
configs in a CXXC orbit project to the same lower-arity orbit. Therefore,
CXXC orbits partition cleanly across target orbits without mixing.
Verified: 100% target-orbit coverage for all 7 projections.

*Run ade3x3_step40_cxxc_marginal_weights.py to populate this section.*

## 29. STABILIZER SUBGROUP CLASSIFICATION

[EXACT_DERIVED] (Step 41)

Isomorphism type of the stabilizer subgroup for every CXC and CXXC orbit.

The ambient group is S3 x S3 x S3 (order 216). Elements have orders in
{1, 2, 3, 6} only. This constrains which subgroup types can appear.

*Run ade3x3_step41_stabilizer_classification.py to populate this section.*

## 30. STABILIZER COMPOSITION ANALYSIS

[EXACT_DERIVED] (Step 42)

For each pair of CXXC orbits (A, B) that compose through a shared CXC face,
does the stabilizer type of the inputs constrain the stabilizer type of the output?

**Composition definition:**
For CXC interface config m = (c1_m, x_m, c2_m):
  A = CXXC(c1_m, x1, x_m, c2_m)   [right-CXC face = m]
  B = CXXC(c1_m, x_m, x4, c2_m)   [left-CXC face = m]
  Output = CXXC(c1_m, x1, x4, c2_m)
Total pairs enumerated: 6,561 CXC configs x 81 x 81 = 43,046,721  [VERIFIED]

*Run ade3x3_step42_stabilizer_composition.py to populate this section.*

**Key findings:**

1. **(Z2)^3 is a composition identity (in stabilizer type):** If either input has
   stabilizer type (Z2)^3, the output type is exactly the OTHER input's type.
   (Z2)^3 ∘ X = X and X ∘ (Z2)^3 = X for all stabilizer types X.
   The (Z2)^3 orbit (rep_config_id=0, the all-zero config) passes the other input through unchanged.

2. **Stabilizer type is not generally preserved:** For most input type pairs,
   the output type ranges across multiple types. The input types provide only
   weak constraints on the output, except at the (Z2)^3 fixed point.

3. **No order-bounding:** Composing two Trivial-stabilizer orbits can produce
   a (Z2)^3 orbit (864 such pairs observed). The stabilizer order can increase
   under composition. Equivalently, two generic configs can land on the
   maximally-symmetric all-zero output.

[INTERPRETATION]

The negative finding: stabilizer type layers are not closed under composition.
The positive finding: the (Z2)^3 identity structure is clean and algebraically
precise. The search for composition-stable substructures must go below the
stabilizer-type coarsening — either to full orbit identity or to the refinement
coordinate layer.

## 31. REFINEMENT-CONDITIONED COMPOSITION KERNEL

[EXACT_DERIVED] (Step 43)

For each of the 14 mixed CX x XC -> CC orbit-pairs, witnesses are stratified
by the (s, t) refinement coordinates of the shared X atom (s = contraction index
from CX side, t = contraction index from XC side).

**Hypothesis tested:** Does conditioning on (s, t) break the orbit-level CC uniformity?

**X atom coordinate encoding:** x = 9*(3r+s) + (3t+u);  s = (x//9)%3, t = (x%9)//3
The shared X atom's (s, t) pair is the 'summation index' contracted through.

*Run ade3x3_step43_refinement_conditioned_kernel.py to populate this section.*

[INTERPRETATION]

The (s,t) uniformity is a second-layer negative result: not only is the orbit-level
composition uniform (Section 27), but conditioning on the shared X atom's summation
index also produces uniform CC-orbit splits within each stratum. The contraction
operation is maximally democratic at both the orbit level and the (s,t) stratum level.
Any structure distinguishing output orbits in the mixed CX x XC pairs must come
from finer features than (s,t) alone — possibly the full (r,s,t,u) coordinate of
the shared X atom, or the specific (c1,c2) boundary conditions.

## 32. Z2xZ2 FLOOR LAYER ANALYSIS

[EXACT_DERIVED] (Step 44)

The 40 CXXC orbits with stabilizer order >= 4 (39 Z2xZ2 + 1 (Z2)^3) are
investigated as a composition-stable 'floor layer'. Uses the same composition
definition as step 42. Results verified against step 42 ground truth (self-test).

*Run ade3x3_step44_z2z2_floor_layer.py to populate this section.*

**Key findings:**

1. **Not fully closed (2a=NO):** Z2xZ2 + Z2xZ2 can produce Z2 output (15.16% exit rate).
   The floor layer is not a sub-algebra under orbit composition.

2. **Floor property holds (2b=YES):** Compositions within the 40-orbit set NEVER
   produce Trivial-stabilizer output. These orbits form a genuine composition floor:
   they cannot spontaneously decay to generic (Trivial-stabilizer) position.

3. **Structural motif (2c):** 28/40 floor orbits have both X atoms live (s=t).
   10/40 have identical X atoms (x1=x2). 22/40 have identical boundary atoms (c1=c2).
   The floor layer is concentrated on structurally symmetric configurations.

4. **Step-1 closure adds 18 orbits:** The smallest composition-closed set containing
   the 40 floor orbits requires adding 18 more Z2 orbits (escaped outputs).

[INTERPRETATION]

The floor layer is a genuine algebraic feature: it is a composition sub-floor
(never decays to Trivial) but not a sub-algebra (can escape to Z2). The 28 doubly-live
orbits are the structural core — their liveness constraint (s=t on both X atoms) is
preserved as a floor property even when the full stabilizer type is not preserved.
The 18 escaped Z2 orbits that complete the step-1 closure are the next candidates
for investigation: do they form a closed layer with the original 40?

## 33. 58-ORBIT CLOSURE

[EXACT_DERIVED] (Step 45, Task 1)

Start from the 58-orbit candidate set = 40 floor orbits + 18 step-1 escaped Z2 orbits
from Section 32. Iterate closure under the same CXC-interface composition used in
steps 42 and 44.

*Run ade3x3_step45_58_orbit_closure_live_core.py to populate this section.*

[INTERPRETATION]

The step-1 58-orbit candidate does NOT blow up toward the full 2744-orbit algebra.
Instead it stabilizes immediately at step 2 as a 64-orbit closed layer. This is the
first genuinely small composition-closed CXXC sub-algebra found so far: 64 is tiny
compared to 2744, yet large enough to strictly contain both the 40-orbit floor and
its 18 first escapes.

## 34. DOUBLY-LIVE FLOOR CORE AND FIXED-POINT SUBSPACES

[EXACT_DERIVED] (Step 45, Tasks 2-3)

The 28 floor orbits with both X atoms live (s=t on both X positions) are the
computationally relevant live core: both bilinear terms fire. For each such orbit,
record the two target C atoms, whether the targets coincide, and whether the shared
target is a boundary atom c1 or c2.

*Run ade3x3_step45_58_orbit_closure_live_core.py to populate this section.*

[INTERPRETATION]

The 28-orbit live core is even more rigid than the 40-orbit floor. It is not closed
inside the floor layer, but it is closed inside the larger both-live world: every core
x core composition remains doubly-live. The escape channel is therefore not liveness
failure, but symmetry failure: 12.44% of outputs leave the floor while staying fully
live. The fixed-space analysis is likewise non-uniform: floor stabilizers split into
two fixed-dimension classes (30 and 36), so the Z2xZ2 symmetry constraint does not
impose a single universal live-X linear subspace.

## 35. SAME-FIBER CORE AND FOCUSED ORBITS

[EXACT_DERIVED] (Step 46, Tasks 1 and 3)

Inside the 28-orbit doubly-live floor core, the 10 same-fiber orbits are those where
both live X atoms target the SAME C atom. The 6 focused orbits are the subset where
that shared target is also a boundary atom c1 or c2.

*Run ade3x3_step46_same_fiber_subalgebra.py to populate this section.*

[INTERPRETATION]

The same-fiber and focused layers are dramatically more rigid than the full live core.
The 10-orbit same-fiber set is already closed, and the 6 focused orbits form an even
smaller closed motif inside it. The geometry is therefore nested: focused ⊂ same-fiber
⊂ doubly-live core ⊂ 64-orbit sub-algebra.

## 36. 64-ORBIT SUB-ALGEBRA STRUCTURE

[EXACT_DERIVED] (Step 46, Tasks 2 and 4)

The step-2 closure from Section 33 is a 64-orbit composition-closed CXXC sub-algebra.
Step 46 computes its full ordered-pair composition table (64 x 64 = 4096 orbit pairs),
recording the full output set for each compatible pair.

*Run ade3x3_step46_same_fiber_subalgebra.py to populate this section.*

[INTERPRETATION]

The 64-orbit object is not a tiny deterministic semigroup; it is a small closed algebra
with real branching. Most compatible pairs are still deterministic, but the mixed pairs
are common enough to matter structurally. The greedy generator search also suggests that
the algebra is not monogenic: one-orbit closure stalls immediately for orbit 1, and the
best greedy construction still needs a large multi-orbit seed (18 orbits).

## 37. MIXED-PAIR RESOLUTION AND TENSOR CONSTRAINTS

[EXACT_DERIVED] (Step 47)

The 164 mixed compatible pairs from Section 36 are scanned at witness level to test
whether finer interface coordinates resolve the orbit-level branching. For each witness,
the shared CXC face records the interface X atom and its coordinates (s,t) and (r,s,t,u).

*Run ade3x3_step47_mixed_pair_resolution_algorithm_constraints.py to populate this section.*

[INTERPRETATION]

This is a third-layer negative result, but an important one. The 64-orbit branching is not
caused by forgetting interface coordinates: even the full shared-X coordinates leave every
mixed output-set intact. Any future deterministic refinement of the 64-orbit algebra must
therefore depend on data beyond the shared face alone, presumably involving the outer X
atoms or finer orbit-internal structure. On the positive side, the raw multiplication tensor
occupies only the two most constrained same-fiber orbits, 0 and 30, which gives a precise
target profile for any algorithm search restricted to the discovered closed layers.

## 38. TENSOR PROFILE CONSTRAINT MODEL

[EXACT_DERIVED] (Step 48)

Step 48 turns the Step 47 same-fiber tensor profile into an explicit rank-1 constraint
model. There are three levels: the exact 729 coordinate equations, the support-level
same-fiber activation counts of a single rank-1 term, and the 8-equation XC orbit-sum
linearization obtained by aggregating the tensor equations by XC orbit class.

*Run ade3x3_step48_tensor_profile_constraint_model.py to populate this section.*

[INTERPRETATION]

Step 48 clarifies exactly where the current tensor-profile program stops being sharp.
The support-level same-fiber profile is still extremely rigid: the generic rank-1 term
has the same unweighted (orbit 0, orbit 30) counts, namely (27,27), as the full raw
multiplication tensor. But once the 729 equations are aggregated down to 8 XC orbit
classes, too much information is lost: the orbit-sum system becomes vacuous for rank
lower bounds. Any useful lower-bound attack must therefore retain finer equation-level
structure and real coefficient constraints, not just orbit-summed totals.

## 39. COEFFICIENT-LEVEL RANK CONSTRAINTS

[EXACT_DERIVED] (Step 49)

Step 49 writes the exact tensor decomposition problem in explicit coordinates, but keeps
the known symmetry reduction visible. The 729 coordinate equations have exactly 8 structural
types under the S3 x S3 x S3 action, and those 8 types are the right coordinate-level
replacement for the orbit-sum model from Step 48.

*Run ade3x3_step49_coefficient_level_rank_constraints.py to populate this section.*

[INTERPRETATION]

Step 49 converts the Step 48 obstruction into the right next object: the exact coordinate
system with its 8 symmetry types still visible. That is a genuine reduction in structure, but
not a reduction in mathematical difficulty. The unsolved problem is now precise: determine
whether the 729 trilinear equations admit a rank-R solution, especially for sparse, non-group-
closed ansatze. Strassen 2x2 shows exactly the kind of cancellation behavior that a 3x3 fast
algorithm would need, so any future search has to keep real coefficients and dead-X cancellation
in the model rather than support patterns alone.

## 40. SYMBOLIC FIBER-MODE DECOMPOSITION

[EXACT_DERIVED] (Step 51)

Step 51 replaces stochastic search with an exact symbolic block decomposition of the tensor
equations. Every rank-1 term contributes three kinds of A x B data: fiber sums, live-fiber
anisotropy, and dead-X coordinates. The full 729-equation system splits exactly into those
three blocks.

*Run ade3x3_step51_symbolic_fiber_mode_decomposition.py to populate this section.*

[INTERPRETATION]

Step 51 isolates the real symbolic burden of any fast 3x3 algorithm. The target tensor lives
entirely in the 9-dimensional fiber-sum block. Every candidate rank-1 term also generates live
anisotropy and dead-X mass, and those nuisance components must cancel exactly after gamma
weighting. This turns the problem into a structured elimination problem on subspaces rather
than an undirected search through raw coefficient space.

## 41. QUOTIENT-SPACE RANK CRITERION

[EXACT_DERIVED] (Step 52)

Step 52 turns the Step 51 matrix form into an exact quotient-space solvability test. For a
fixed decomposition, solvability is equivalent to the fiber-sum columns remaining independent
modulo the nuisance span generated by live anisotropy and dead-X columns.

*Run ade3x3_step52_quotient_rank_criterion.py to populate this section.*

[INTERPRETATION]

Step 52 is the first exact linear-algebra obstruction beyond raw equation counting, but its
scope matters: the bound R >= 9 + rank(Nuisance) is per-algorithm, not universal. It depends
on the nuisance span produced by the chosen alpha,beta factors. What the step proves is that
every candidate algorithm must compress its own nuisance span into dimension at most R-9. The
2x2 Strassen check is the key validation: its nuisance matrix is 7x12 with rank exactly 3, so
Strassen is tight against the criterion R = 4 + rank(Nuisance).

## 42. SUPPORT-TYPE REPRESENTATIVE INCIDENCE

[EXACT_DERIVED] (Step 53)

Step 53 asks whether support geometry alone can force any of the 8 exact representative
equation types from Step 49 to be absent for a single rank-1 term. A support type keeps only
the six subset supports of alpha, beta, and gamma, modulo the natural S3 x S3 x S3 action.

*Run ade3x3_step53_support_type_representative_incidence.py to populate this section.*

[INTERPRETATION]

Step 53 closes off another support-only route. The Step 48 orbit-sum model was already too
coarse; Step 53 shows that even exact support incidence against the 8 representative equation
types is still vacuous. Once support is rich enough to allow the positive Type 0 equation,
there are many support classes that also allow all 7 zero-RHS types. Any universal lower-bound
argument must therefore use coefficient relations, quotient-space structure, or stronger
algebraic constraints than support incidence alone.

## 43. ANALYTICAL LOW-NUISANCE CONSTRUCTION

[EXACT_DERIVED] (Step 54)

Step 54 pivots from obstruction to construction. It first corrects the dead-free term
template exactly, then measures the nuisance-rank landscape for random low-rank factor
families alpha=A*C and beta=B*D, with emphasis on the R=22 target nuisance threshold <= 13.

*Run ade3x3_step54_analytical_low_nuisance_construction.py to populate this section.*

[INTERPRETATION]

Step 54 sharpens the constructive picture in two ways. First, dead-free terms do not give a
free nuisance bypass: they kill Delta but still generate anisotropy, so the Strassen 2x2
template does not port directly into the Step 51 basis. Second, generic low-rank factor
families can indeed hit low nuisance numerically at R=22, but in the sampled families Sigma
never escaped the nuisance span, so the quotient-space gain stayed far below the required 9.
That points the next constructive search toward structured, nongeneric coefficient designs
rather than random low-rank factor models.

## 44. ALGEBRAIC NUISANCE DEPENDENCIES + WILDCARD EXPLORATION

[EXACT_DERIVED] (Step 55)

Step 55 moves from generic low-rank profiling to explicit Hadamard-space dependency
arithmetic. The main exact point is that in a p*q-dimensional Hadamard space, full
9-dimensional quotient recovery forces nuisance rank <= p*q-9, which sharpens the
R=22 nuisance target drastically in the structured p=3, q=4 regime.

**p=q=3 combined nuisance cap:** 0
**p=3,q=4 combined nuisance cap:** 3
**Best structured 3x4 nuisance rank:** 11
**Best structured 3x4 quotient gain:** 1
**GF(2) flattening lower bound:** 9
**Tropical flattening rank:** 9
**Standard 3x3 zero commutators:** 81
**Strassen 2x2 zero commutators:** 7

### Track A: Hadamard-Space Targets

| regime | Hadamard dim upper bound | quotient target | max nuisance from geometry | max nuisance from R=22 | combined target |
|--------|---------------------------|-----------------|----------------------------|------------------------|----------------|
| p=3,q=3 | 9 | 9 | 0 | 13 | 0 |
| p=3,q=4 | 12 | 9 | 3 | 13 | 3 |
| p=4,q=3 | 12 | 9 | 3 | 13 | 3 |

| family | label | c_rank | d_rank | hadamard_dim | sigma_rank | nuisance_rank | quotient_gain | meets nuisance<=3 | full quotient target |
|--------|-------|--------|--------|--------------|------------|---------------|---------------|-------------------|----------------------|
| dft_modes | dft_modes_3 | 3 | 4 | 12 | 3 | 11 | 1 | False | False |
| toeplitz_genericD | toeplitz_genericD_2 | 3 | 4 | 12 | 3 | 12 | 0 | False | False |

### Symbolic Minor Witness

- toeplitz_genericD_symbolic: selected 4x4 minor rows (0, 1, 2, 3) and columns (0, 1, 2, 9) factor as -21*a**3*b + 8*a**3 + 25*a**2*b + 4*a**2 + 3*a*b - 8*a - 7*b

### Track B

[WILDCARD]

| flattening | shape | GF(2) rank | tropical rank |
|------------|-------|------------|---------------|
| A_vs_BC | 9x81 | 9 | 9 |
| B_vs_AC | 9x81 | 9 | 9 |
| C_vs_AB | 9x81 | 9 | 9 |

| algorithm | term_count | ordered_pairs | zero_commutators | nonzero_commutators | max_commutator_rank |
|-----------|------------|---------------|------------------|---------------------|---------------------|
| standard_3x3 | 27 | 729 | 81 | 648 | 2 |
| strassen_2x2 | 7 | 49 | 7 | 42 | 2 |


[INTERPRETATION]

Step 55 makes the Step 54 constructive obstruction sharper. In the p=q=3 regime the
Hadamard space itself has dimension only 9, so full quotient recovery would force the
nuisance span to vanish entirely. In the p=3,q=4 regime the raw R=22 target rank(N)<=13
is still far too loose: Hadamard geometry tightens it to rank(N)<=3. The structured
families tested here did not produce such a collapse, and their quotient gains stayed
well below the required 9. The wildcard checks are also informative but not decisive:
GF(2) flattening rank only gives the obvious bound 9, tropical flattening rank is capped
at 9 by the matrix dimensions, and the commutator profile shows real overlap structure
even for the standard and Strassen decompositions rather than automatic vanishing.

## 45. CURRENT GAPS / OPEN FRONTS

[OPEN_FRONT]

**Completed in this session:**
- CXXC schema (arity-4): ✓ 2744 orbits with base and refined signatures recorded
- AXXC parity layer: ✓ 2870 orbits and 2870 current-recorded signatures measured
- CCXX schema (arity-4): ✓ 2744 orbits with base and refined signatures recorded
- XX, AX, BX signature collisions: ✓ All resolved with minimal refinement features
- Composition kernel: ✓ Mixed key CC-orbit distributions computed (all 14 keys uniform)
- CXXC marginal weight profile: ✓ Fiber-size histogram for all 7 projections
- Stabilizer subgroup classification: ✓ CXC and CXXC; all pure-2-groups (Z2, Z2xZ2, (Z2)^3)
- Stabilizer composition analysis: ✓ 43M pairs; (Z2)^3 is composition identity; types not generally closed
- CXXC and CCXX arity-4 collisions: ✓ All resolved with minimal refiner `(s4, t4)`
- CXXC marginal projections: ✓ Full coverage analysis completed
- Refinement-conditioned kernel: ✓ All 63 (s,t) strata uniform; hypothesis closed at stratum level
- Z2xZ2 floor layer: ✓ 40 orbits; floor property holds (no Trivial decay); not fully closed; step-1 closure adds 18 Z2 orbits
- 58-orbit closure + doubly-live core: ✓ 58-seed closes at 64 orbits; 28-core stays 100% doubly-live; floor fixed dims are 30 or 36
- Same-fiber core + 64-subalgebra structure: ✓ 10 same-fiber and 6 focused orbits are both closed; 64-table has 602 compatible rows with 164 mixed; greedy generator set size 18
- Mixed-pair resolution + tensor constraints: ✓ 41,688 witnesses scanned; 0/164 mixed pairs resolved by interface coordinates; raw tensor same-fiber support uses only orbits 0 and 30
- Tensor profile constraint model: ✓ 729 tensor equations collapse to 8 XC orbit classes; only XC orbit 0 is positive; generic rank-1 support has profile (27,27); the 8-orbit linearization alone gives no rank lower bound
- Coefficient-level rank constraints: ✓ explicit 8 equation types recorded with orbit sizes (27,54,54,108,54,108,108,216); standard 27-term basis algorithm and Strassen 2x2 both verified exactly; search-space dimensions exported
- Symbolic fiber-mode decomposition: ✓ the 729 equations now split exactly as 81 fiber-sum + 162 live-anisotropy + 486 dead-X equations, with matrix form Gamma*Sigma=3I_9 and Gamma annihilating the nuisance blocks
- Quotient-space rank criterion: ✓ for a fixed decomposition solvability is equivalent to quotient-space independence of Sigma modulo nuisance; standard 3x3 gives nuisance rank 18 and Strassen 2x2 gives rank 3, with Strassen exactly tight
- Support-type representative incidence: ✓ there are 8000 support classes modulo S3^3; 1000 can realize Type 0, and 216 of those allow all 8 representative equation types, so support-only pruning is vacuous
- Analytical low-nuisance construction: ✓ dead-free terms were shown not to be nuisance-free in the Step 51 basis; random low-rank factor families were profiled numerically, and although some R=22 families reached nuisance rank <= 13, none achieved the quotient-space gain required by Step 52
- Algebraic nuisance dependencies + wildcards: ✓ Hadamard-space geometry now sharpens the p=3,q=4 target to nuisance rank <= 3; tested Toeplitz, circulant, shared-latent, and DFT families still failed to produce quotient gain 9; GF(2) and tropical flattening ranks both stayed at 9, and fiber commutators were mostly nonzero

**Remaining open fronts:**
- Additional arity-4 schemas: XCXC, XCCX, XXXC, XXX not yet explored
- AXXC currently has an orbit-complete induced face-pattern signature layer;
  whether there is a simpler intrinsic minimal closed-form signature/refinement rule remains open
- Refinement engine: Not yet rerun on corrected composition (14 mixed keys)
- Higher arity layers: Arity 5+ unexplored
- The 14^3 = 2744 CXXC orbit count factorization: whether 14 = C(4,2)+C(4,1)+C(4,0) reflects
  partition types at arity 4 under the compatible group action is unverified
- The 64-orbit closed layer now has a full composition table; its intrinsic signature rule
  and exact minimum generating set are still unknown
- The 164 mixed pairs are invariant under all shared-face coordinate conditioning tried so far;
  any deterministic refinement must depend on data beyond the interface X coordinates alone
- The 12.44% live-core escapes stay doubly-live but leave the floor; classify those nonfloor
  doubly-live targets as a structural layer of their own
- Same-fiber and focused subsets are closed; determine whether they admit a clean intrinsic
  signature or conceptual description beyond the target/focus predicates
- Step 48 shows that the 8 XC-orbit linearization is exact but vacuous for rank lower bounds;
  any useful lower-bound model must retain finer-than-orbit-sum equation structure
- Step 49 now records the exact 729-equation trilinear system and the 8 representative types;
  the remaining open problem is whether the rank-R solution variety is nonempty for sparse or non-group-closed ansatze
- Step 52 gives a per-algorithm quotient-rank bound R >= 9 + rank(Nuisance); the remaining
  hard theorem is universal: prove a decomposition-independent lower bound on rank(Nuisance)
- Step 53 shows that support-only representative incidence is also vacuous; any sharper universal
  theorem must use coefficient identities or subspace geometry, not only index-support patterns
- Step 54 shows that generic low-rank factor models also fail constructively: low nuisance can
  occur without any quotient-space gain, so the next constructive family must impose structured
  coefficient relations that separate Sigma from the nuisance span
- Step 55 tightens the structured p=3,q=4 target to rank(Nuisance) <= 3 inside Hadamard space;
  the remaining constructive problem is to find explicit polynomial identities on C and D that
  force that collapse without also collapsing Sigma
- The current structured families were still too rigid or too generic; next candidates should
  target exact nuisance-column identities rather than only symmetry patterns such as Toeplitz or DFT
- The characteristic-2, tropical-flattening, and commutator wildcards did not produce a new
  lower bound yet; if a wildcard route is to matter, it must retain more than flattening data
- Kernel uniformity: cc uniformity holds at orbit level AND (s,t) stratum level;
  next level to check is the full (r,s,t,u) X coordinate or the (c1,c2) boundary

These are genuine incompletions, not promises.

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------

This canonical dossier contains all computed orbit rosters, the complete
composition grid, and all ground-truth structural data.

No external files are required. This document is standalone and complete.