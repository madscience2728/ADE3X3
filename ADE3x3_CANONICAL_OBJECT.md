======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-28 00:12:51
Generator: ade3x3_step13e_update_md_dossier_generator.py
Source: Current warehouse state (steps 1-31, repairs 10b/13b, exports 14-31)

This is a canonical technical dossier of the current object state.
----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-28 00:12:51
**Generator Script:** ade3x3_step13e_update_md_dossier_generator.py
**Provenance:** Built from steps 1-31, with orbit metadata repair (step 10b)

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
Role overlay mechanism is ready for typed schema attachment.

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


### Atomic Index Tables

[GROUND_TRUTH]

#### A index table

| A_local_id | A_name | row | col | raw_idx |
|---:|---|---:|---:|---:|
| 0 | A[0,0] | 0 | 0 | 0 |
| 1 | A[0,1] | 0 | 1 | 1 |
| 2 | A[0,2] | 0 | 2 | 2 |
| 3 | A[1,0] | 1 | 0 | 3 |
| 4 | A[1,1] | 1 | 1 | 4 |
| 5 | A[1,2] | 1 | 2 | 5 |
| 6 | A[2,0] | 2 | 0 | 6 |
| 7 | A[2,1] | 2 | 1 | 7 |
| 8 | A[2,2] | 2 | 2 | 8 |

#### B index table

| B_local_id | B_name | row | col | raw_idx |
|---:|---|---:|---:|---:|
| 0 | B[0,0] | 0 | 0 | 0 |
| 1 | B[0,1] | 0 | 1 | 1 |
| 2 | B[0,2] | 0 | 2 | 2 |
| 3 | B[1,0] | 1 | 0 | 3 |
| 4 | B[1,1] | 1 | 1 | 4 |
| 5 | B[1,2] | 1 | 2 | 5 |
| 6 | B[2,0] | 2 | 0 | 6 |
| 7 | B[2,1] | 2 | 1 | 7 |
| 8 | B[2,2] | 2 | 2 | 8 |

#### C index table

| C_local_id | C_name | row | col | raw_idx |
|---:|---|---:|---:|---:|
| 0 | C[0,0] | 0 | 0 | 0 |
| 1 | C[0,1] | 0 | 1 | 1 |
| 2 | C[0,2] | 0 | 2 | 2 |
| 3 | C[1,0] | 1 | 0 | 3 |
| 4 | C[1,1] | 1 | 1 | 4 |
| 5 | C[1,2] | 1 | 2 | 5 |
| 6 | C[2,0] | 2 | 0 | 6 |
| 7 | C[2,1] | 2 | 1 | 7 |
| 8 | C[2,2] | 2 | 2 | 8 |

### C Fiber Table

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

| c_local_id | c_name | x0_name | x1_name | x2_name |
|---:|---|---|---|---|
| 0 | C[0,0] | X[0,0|0,0] | X[0,1|1,0] | X[0,2|2,0] |
| 1 | C[0,1] | X[0,0|0,1] | X[0,1|1,1] | X[0,2|2,1] |
| 2 | C[0,2] | X[0,0|0,2] | X[0,1|1,2] | X[0,2|2,2] |
| 3 | C[1,0] | X[1,0|0,0] | X[1,1|1,0] | X[1,2|2,0] |
| 4 | C[1,1] | X[1,0|0,1] | X[1,1|1,1] | X[1,2|2,1] |
| 5 | C[1,2] | X[1,0|0,2] | X[1,1|1,2] | X[1,2|2,2] |
| 6 | C[2,0] | X[2,0|0,0] | X[2,1|1,0] | X[2,2|2,0] |
| 7 | C[2,1] | X[2,0|0,1] | X[2,1|1,1] | X[2,2|2,1] |
| 8 | C[2,2] | X[2,0|0,2] | X[2,1|1,2] | X[2,2|2,2] |

### X Atom Table (Inline)

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

| x_local_id | x_name | r | s | t | u | live | a_idx | b_idx | target_c |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | X[0,0|0,0] | 0 | 0 | 0 | 0 | 1 | 0 | 0 | C[0,0] |
| 1 | X[0,0|0,1] | 0 | 0 | 0 | 1 | 1 | 0 | 1 | C[0,1] |
| 2 | X[0,0|0,2] | 0 | 0 | 0 | 2 | 1 | 0 | 2 | C[0,2] |
| 3 | X[0,0|1,0] | 0 | 0 | 1 | 0 | 0 | 0 | 3 | - |
| 4 | X[0,0|1,1] | 0 | 0 | 1 | 1 | 0 | 0 | 4 | - |
| 5 | X[0,0|1,2] | 0 | 0 | 1 | 2 | 0 | 0 | 5 | - |
| 6 | X[0,0|2,0] | 0 | 0 | 2 | 0 | 0 | 0 | 6 | - |
| 7 | X[0,0|2,1] | 0 | 0 | 2 | 1 | 0 | 0 | 7 | - |
| 8 | X[0,0|2,2] | 0 | 0 | 2 | 2 | 0 | 0 | 8 | - |
| 9 | X[0,1|0,0] | 0 | 1 | 0 | 0 | 0 | 1 | 0 | - |
| 10 | X[0,1|0,1] | 0 | 1 | 0 | 1 | 0 | 1 | 1 | - |
| 11 | X[0,1|0,2] | 0 | 1 | 0 | 2 | 0 | 1 | 2 | - |
| 12 | X[0,1|1,0] | 0 | 1 | 1 | 0 | 1 | 1 | 3 | C[0,0] |
| 13 | X[0,1|1,1] | 0 | 1 | 1 | 1 | 1 | 1 | 4 | C[0,1] |
| 14 | X[0,1|1,2] | 0 | 1 | 1 | 2 | 1 | 1 | 5 | C[0,2] |
| 15 | X[0,1|2,0] | 0 | 1 | 2 | 0 | 0 | 1 | 6 | - |
| 16 | X[0,1|2,1] | 0 | 1 | 2 | 1 | 0 | 1 | 7 | - |
| 17 | X[0,1|2,2] | 0 | 1 | 2 | 2 | 0 | 1 | 8 | - |
| 18 | X[0,2|0,0] | 0 | 2 | 0 | 0 | 0 | 2 | 0 | - |
| 19 | X[0,2|0,1] | 0 | 2 | 0 | 1 | 0 | 2 | 1 | - |
| 20 | X[0,2|0,2] | 0 | 2 | 0 | 2 | 0 | 2 | 2 | - |
| 21 | X[0,2|1,0] | 0 | 2 | 1 | 0 | 0 | 2 | 3 | - |
| 22 | X[0,2|1,1] | 0 | 2 | 1 | 1 | 0 | 2 | 4 | - |
| 23 | X[0,2|1,2] | 0 | 2 | 1 | 2 | 0 | 2 | 5 | - |
| 24 | X[0,2|2,0] | 0 | 2 | 2 | 0 | 1 | 2 | 6 | C[0,0] |
| 25 | X[0,2|2,1] | 0 | 2 | 2 | 1 | 1 | 2 | 7 | C[0,1] |
| 26 | X[0,2|2,2] | 0 | 2 | 2 | 2 | 1 | 2 | 8 | C[0,2] |
| 27 | X[1,0|0,0] | 1 | 0 | 0 | 0 | 1 | 3 | 0 | C[1,0] |
| 28 | X[1,0|0,1] | 1 | 0 | 0 | 1 | 1 | 3 | 1 | C[1,1] |
| 29 | X[1,0|0,2] | 1 | 0 | 0 | 2 | 1 | 3 | 2 | C[1,2] |
| 30 | X[1,0|1,0] | 1 | 0 | 1 | 0 | 0 | 3 | 3 | - |
| 31 | X[1,0|1,1] | 1 | 0 | 1 | 1 | 0 | 3 | 4 | - |
| 32 | X[1,0|1,2] | 1 | 0 | 1 | 2 | 0 | 3 | 5 | - |
| 33 | X[1,0|2,0] | 1 | 0 | 2 | 0 | 0 | 3 | 6 | - |
| 34 | X[1,0|2,1] | 1 | 0 | 2 | 1 | 0 | 3 | 7 | - |
| 35 | X[1,0|2,2] | 1 | 0 | 2 | 2 | 0 | 3 | 8 | - |
| 36 | X[1,1|0,0] | 1 | 1 | 0 | 0 | 0 | 4 | 0 | - |
| 37 | X[1,1|0,1] | 1 | 1 | 0 | 1 | 0 | 4 | 1 | - |
| 38 | X[1,1|0,2] | 1 | 1 | 0 | 2 | 0 | 4 | 2 | - |
| 39 | X[1,1|1,0] | 1 | 1 | 1 | 0 | 1 | 4 | 3 | C[1,0] |
| 40 | X[1,1|1,1] | 1 | 1 | 1 | 1 | 1 | 4 | 4 | C[1,1] |
| 41 | X[1,1|1,2] | 1 | 1 | 1 | 2 | 1 | 4 | 5 | C[1,2] |
| 42 | X[1,1|2,0] | 1 | 1 | 2 | 0 | 0 | 4 | 6 | - |
| 43 | X[1,1|2,1] | 1 | 1 | 2 | 1 | 0 | 4 | 7 | - |
| 44 | X[1,1|2,2] | 1 | 1 | 2 | 2 | 0 | 4 | 8 | - |
| 45 | X[1,2|0,0] | 1 | 2 | 0 | 0 | 0 | 5 | 0 | - |
| 46 | X[1,2|0,1] | 1 | 2 | 0 | 1 | 0 | 5 | 1 | - |
| 47 | X[1,2|0,2] | 1 | 2 | 0 | 2 | 0 | 5 | 2 | - |
| 48 | X[1,2|1,0] | 1 | 2 | 1 | 0 | 0 | 5 | 3 | - |
| 49 | X[1,2|1,1] | 1 | 2 | 1 | 1 | 0 | 5 | 4 | - |
| 50 | X[1,2|1,2] | 1 | 2 | 1 | 2 | 0 | 5 | 5 | - |
| 51 | X[1,2|2,0] | 1 | 2 | 2 | 0 | 1 | 5 | 6 | C[1,0] |
| 52 | X[1,2|2,1] | 1 | 2 | 2 | 1 | 1 | 5 | 7 | C[1,1] |
| 53 | X[1,2|2,2] | 1 | 2 | 2 | 2 | 1 | 5 | 8 | C[1,2] |
| 54 | X[2,0|0,0] | 2 | 0 | 0 | 0 | 1 | 6 | 0 | C[2,0] |
| 55 | X[2,0|0,1] | 2 | 0 | 0 | 1 | 1 | 6 | 1 | C[2,1] |
| 56 | X[2,0|0,2] | 2 | 0 | 0 | 2 | 1 | 6 | 2 | C[2,2] |
| 57 | X[2,0|1,0] | 2 | 0 | 1 | 0 | 0 | 6 | 3 | - |
| 58 | X[2,0|1,1] | 2 | 0 | 1 | 1 | 0 | 6 | 4 | - |
| 59 | X[2,0|1,2] | 2 | 0 | 1 | 2 | 0 | 6 | 5 | - |
| 60 | X[2,0|2,0] | 2 | 0 | 2 | 0 | 0 | 6 | 6 | - |
| 61 | X[2,0|2,1] | 2 | 0 | 2 | 1 | 0 | 6 | 7 | - |
| 62 | X[2,0|2,2] | 2 | 0 | 2 | 2 | 0 | 6 | 8 | - |
| 63 | X[2,1|0,0] | 2 | 1 | 0 | 0 | 0 | 7 | 0 | - |
| 64 | X[2,1|0,1] | 2 | 1 | 0 | 1 | 0 | 7 | 1 | - |
| 65 | X[2,1|0,2] | 2 | 1 | 0 | 2 | 0 | 7 | 2 | - |
| 66 | X[2,1|1,0] | 2 | 1 | 1 | 0 | 1 | 7 | 3 | C[2,0] |
| 67 | X[2,1|1,1] | 2 | 1 | 1 | 1 | 1 | 7 | 4 | C[2,1] |
| 68 | X[2,1|1,2] | 2 | 1 | 1 | 2 | 1 | 7 | 5 | C[2,2] |
| 69 | X[2,1|2,0] | 2 | 1 | 2 | 0 | 0 | 7 | 6 | - |
| 70 | X[2,1|2,1] | 2 | 1 | 2 | 1 | 0 | 7 | 7 | - |
| 71 | X[2,1|2,2] | 2 | 1 | 2 | 2 | 0 | 7 | 8 | - |
| 72 | X[2,2|0,0] | 2 | 2 | 0 | 0 | 0 | 8 | 0 | - |
| 73 | X[2,2|0,1] | 2 | 2 | 0 | 1 | 0 | 8 | 1 | - |
| 74 | X[2,2|0,2] | 2 | 2 | 0 | 2 | 0 | 8 | 2 | - |
| 75 | X[2,2|1,0] | 2 | 2 | 1 | 0 | 0 | 8 | 3 | - |
| 76 | X[2,2|1,1] | 2 | 2 | 1 | 1 | 0 | 8 | 4 | - |
| 77 | X[2,2|1,2] | 2 | 2 | 1 | 2 | 0 | 8 | 5 | - |
| 78 | X[2,2|2,0] | 2 | 2 | 2 | 0 | 1 | 8 | 6 | C[2,0] |
| 79 | X[2,2|2,1] | 2 | 2 | 2 | 1 | 1 | 8 | 7 | C[2,1] |
| 80 | X[2,2|2,2] | 2 | 2 | 2 | 2 | 1 | 8 | 8 | C[2,2] |

## 5. SYMMETRY/ACTION SYSTEM

[GROUND_TRUTH]

- **Compatible Symmetry Group Size**: 216
- **Group Type**: S3 x S3 x S3 with compatibility constraint pi_cA == pi_rB
- **Action Scope**: Acts on A, B, C, and X atoms
- **Canonicalization**: All core schemas are canonicalized under group action

## 6. TYPED SCHEMAS CURRENTLY BUILT

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

| Schema | Typed Arity | Raw Arity | Count | Orbits | Bridged | Notes |
|--------|-------------|-----------|-------|--------|---------|-------|
| A      | 1           | 1         | 9     | -      | Yes     | direct embed |
| B      | 1           | 1         | 9     | -      | Yes     | direct embed |
| C      | 1           | 1         | 9     | -      | Yes     | direct embed |
| X      | 1           | 2         | 81    | -      | Yes*    | expands to 2 slots |
| CC     | 2           | 2         | 81    | 4      | Yes     | direct embed |
| CX     | 2           | 3         | 729   | 8      | Yes     | X expands |
| XC     | 2           | 3         | 729   | 8      | Yes     | X expands |
| AX     | 2           | 3         | 729   | 10     | Yes     | X expands |
| BX     | 2           | 3         | 729   | 10     | Yes     | X expands |
| CXC    | 3           | 4         | 6,561 | 50     | Yes     | X expands |
| XX     | 2           | 4         | 6,561 | 56     | Yes     | X x X, each X expands to 2 slots |
| CXXC   | 4           | 6         | 531,441 | 2,744  | Yes     | first arity-4 schema |
| AXXC   | 4           | 6         | 531,441 | 2,870  | Yes     | second arity-4 schema |

Note: All typed schemas including XX are now bridged into the raw warehouse.
X bridge marked Yes* because X expands to 2 raw slots (A_idx, B_idx), per Section 7 bridge rules.
CXXC and AXXC are the first typed schemas beyond the arity-3 core.

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

[INTERPRETATION]
Raw arity 3 occupancy: When summing overlay image counts across CX, XC, AX, BX,
the total is 2916 (729 x 4 schemas). Divided by unique raw tuples (729), this gives
400%. This is NOT a physical occupancy claim; it reflects aggregate overlay images
across multiple schemas with different role overlays occupying the same positions.
See Section 15 for the explicit raw-arity-3 cross-schema alignment export.

## 8. WAREHOUSE INFRASTRUCTURE STATUS

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

### Current Infrastructure

- In-memory object DB with atomic species
- Raw config bulk-loading for arity-2 and arity-3 core schemas
- Symmetry-aware canonicalization for all typed schemas
- Orbit metadata cache for all schemas (repaired, step 10b)
- Signature caches per orbit (repaired, step 10b)
- Stabilizer element lists exported for all schemas
- Full raw base-9 warehouse through arity 9
- Typed/raw bridge for all schemas including XX and CXXC
- Orbit-signature export tables
- Cross-schema raw alignment export
- Corrected orbit-based composition export
- First two arity-4 typed schemas (CXXC, AXXC) registered and bridged
- CXXC face inventory, face-pattern summary, middle-XX marginals, left/right CXC marginals, and joint interior co-occurrence export
- AXXC face inventory, face-pattern summary, middle-XX marginals, and joint interior co-occurrence export
- Raw arity-6 alignment export for CXXC and AXXC
- Arity-4 orbit/signature/stabilizer parity export for CXXC and AXXC

## 9. EXACT MEASURED COUNTS AND MEMORY

[MEASURED_FROM_CODE]

### Memory Footprints

| Component | Approx Size |
|-----------|-------------|
| Full raw warehouse | ~26.8 MB |
| Object DB base | ~387 KB |
| Bulk core configs | ~881.5 KB |
| Bridge tables | ~870.4 KB |
| Orbit metadata cache | ~8.4 KB |

### Orbit Summaries

| Schema | Raw Configs | Orbits | Avg Size | Stabilizer Range |
|--------|-------------|--------|----------|------------------|
| XX     | 6,561       | 56     | 117.2    | 1-8              |
| CX     | 729         | 8      | 91.1     | 1-8              |
| XC     | 729         | 8      | 91.1     | 1-8              |
| CC     | 81          | 4      | 20.2     | 6-24             |
| AX     | 729         | 10     | 72.9     | 2-8              |
| BX     | 729         | 10     | 72.9     | 2-8              |
| CXC    | 6,561       | 50     | 131.2    | 1-8              |
| CXXC   | 531,441     | 2,744  | 193.7    | 1-8              |
| AXXC   | 531,441     | 2,870  | 185.2    | 1-8              |

All orbit x stabilizer = 216 verified.
Stabilizer element sets (not just sizes) are now explicitly exported.

### Repaired Signature Summary

[MEASURED_FROM_CODE] / [REPAIRED]

| Schema | Orbit Count | Distinct Signatures | Orbit Complete |
|--------|-------------|---------------------|----------------|
| XX     |          56 |                  48 |             no |
| CX     |           8 |                   8 |            yes |
| XC     |           8 |                   8 |            yes |
| CC     |           4 |                   4 |            yes |
| AX     |          10 |                   8 |             no |
| BX     |          10 |                   8 |             no |
| CXC    |          50 |                  50 |            yes |
| CXXC   |       2,744 |               2,744 |            yes |
| AXXC   |       2,870 |               2,870 |            yes |

## 10. EXACT DERIVED RESULTS CURRENTLY KNOWN

[EXACT_DERIVED]

### Dead-Dead Pair Classification

- 24 orbits classified exactly
- Classification by: row equality, column equality, one of 6 shared-index patterns
- Full resolution with refined signature tuple (row_eq, col_eq, shared_pattern)

### CX x XC -> CC Composition Result (Corrected Orbit-Based)

[EXACT_DERIVED] / [REPAIRED]

- CX orbits = 8, XC orbits = 8, CC orbits = 4
- Possible orbit-pairs = 64
- Realized keys = 32
- Deterministic = 18
- Mixed = 14

An older type-based layer reported 256/64/36/28. Those counts used a different
type system and are now superseded by the corrected orbit-based counts above.


### Full Corrected 8×8 Orbit-Based Composition Grid

[MEASURED_FROM_CODE] / [REPAIRED]

| CX_orbit | XC_orbit | realized | cc_orbit_ids | deterministic | witness_count |
|---:|---:|---:|---|---:|---:|
| 0 | 0 | 1 | [0] | 1 | 27 |
| 0 | 1 | 1 | [1] | 1 | 54 |
| 0 | 2 | 1 | [2] | 1 | 54 |
| 0 | 3 | 1 | [3] | 1 | 108 |
| 0 | 4 | 0 | [] | 0 | 0 |
| 0 | 5 | 0 | [] | 0 | 0 |
| 0 | 6 | 0 | [] | 0 | 0 |
| 0 | 7 | 0 | [] | 0 | 0 |
| 1 | 0 | 1 | [1] | 1 | 54 |
| 1 | 1 | 1 | [0, 1] | 0 | 108 |
| 1 | 2 | 1 | [3] | 1 | 108 |
| 1 | 3 | 1 | [2, 3] | 0 | 216 |
| 1 | 4 | 0 | [] | 0 | 0 |
| 1 | 5 | 0 | [] | 0 | 0 |
| 1 | 6 | 0 | [] | 0 | 0 |
| 1 | 7 | 0 | [] | 0 | 0 |
| 2 | 0 | 0 | [] | 0 | 0 |
| 2 | 1 | 0 | [] | 0 | 0 |
| 2 | 2 | 0 | [] | 0 | 0 |
| 2 | 3 | 0 | [] | 0 | 0 |
| 2 | 4 | 1 | [0] | 1 | 54 |
| 2 | 5 | 1 | [1] | 1 | 108 |
| 2 | 6 | 1 | [2] | 1 | 108 |
| 2 | 7 | 1 | [3] | 1 | 216 |
| 3 | 0 | 0 | [] | 0 | 0 |
| 3 | 1 | 0 | [] | 0 | 0 |
| 3 | 2 | 0 | [] | 0 | 0 |
| 3 | 3 | 0 | [] | 0 | 0 |
| 3 | 4 | 1 | [1] | 1 | 108 |
| 3 | 5 | 1 | [0, 1] | 0 | 216 |
| 3 | 6 | 1 | [3] | 1 | 216 |
| 3 | 7 | 1 | [2, 3] | 0 | 432 |
| 4 | 0 | 1 | [2] | 1 | 54 |
| 4 | 1 | 1 | [3] | 1 | 108 |
| 4 | 2 | 1 | [0, 2] | 0 | 108 |
| 4 | 3 | 1 | [1, 3] | 0 | 216 |
| 4 | 4 | 0 | [] | 0 | 0 |
| 4 | 5 | 0 | [] | 0 | 0 |
| 4 | 6 | 0 | [] | 0 | 0 |
| 4 | 7 | 0 | [] | 0 | 0 |
| 5 | 0 | 1 | [3] | 1 | 108 |
| 5 | 1 | 1 | [2, 3] | 0 | 216 |
| 5 | 2 | 1 | [1, 3] | 0 | 216 |
| 5 | 3 | 1 | [0, 1, 2, 3] | 0 | 432 |
| 5 | 4 | 0 | [] | 0 | 0 |
| 5 | 5 | 0 | [] | 0 | 0 |
| 5 | 6 | 0 | [] | 0 | 0 |
| 5 | 7 | 0 | [] | 0 | 0 |
| 6 | 0 | 0 | [] | 0 | 0 |
| 6 | 1 | 0 | [] | 0 | 0 |
| 6 | 2 | 0 | [] | 0 | 0 |
| 6 | 3 | 0 | [] | 0 | 0 |
| 6 | 4 | 1 | [2] | 1 | 108 |
| 6 | 5 | 1 | [3] | 1 | 216 |
| 6 | 6 | 1 | [0, 2] | 0 | 216 |
| 6 | 7 | 1 | [1, 3] | 0 | 432 |
| 7 | 0 | 0 | [] | 0 | 0 |
| 7 | 1 | 0 | [] | 0 | 0 |
| 7 | 2 | 0 | [] | 0 | 0 |
| 7 | 3 | 0 | [] | 0 | 0 |
| 7 | 4 | 1 | [3] | 1 | 216 |
| 7 | 5 | 1 | [2, 3] | 0 | 432 |
| 7 | 6 | 1 | [1, 3] | 0 | 432 |
| 7 | 7 | 1 | [0, 1, 2, 3] | 0 | 864 |

### Refinement Engine Result

[MEASURED_FROM_CODE]

The following separator results come from a tested subset of simple coordinate-alignment predicates on the older composition layer; they are not claimed to exhaust all possible separator families.

- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)
- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target
- Best 2-tuple within the tested subset: [r1_eq_r, u1_eq_u]

### Orbit-Complete Schemas (Repaired)

[EXACT_DERIVED] / [REPAIRED]

- CX: 8 orbits, 8 distinct signatures
- XC: 8 orbits, 8 distinct signatures
- CC: 4 orbits, 4 distinct signatures
- CXC: 50 orbits, 50 distinct signatures

### Schemas with Coarse Signatures (Repaired)

[EXACT_DERIVED] / [REPAIRED]

- XX: 56 orbits, 48 distinct signatures
- AX: 10 orbits, 8 distinct signatures
- BX: 10 orbits, 8 distinct signatures

### Cross-Schema Raw Arity-3 Alignment

[MEASURED_FROM_CODE]

- Schemas CX, XC, AX, BX all bridge to raw arity 3
- Each image size = 729
- Distinct raw tuples occupied = 729
- Aggregate image count = 2,916
- Occupancy ratio = 4.0 (all four images coincide exactly)


### Arity-4 Interior Determination in CXXC

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

- CXXC full population = 531,441 rows
- Distinct 6-face orbit-patterns = 2,744
- Distinct joint keys (left CXC, middle XX, right CXC) = 2,744
- Therefore the interior 3-tuple (left CXC orbit, middle XX orbit, right CXC orbit)
  fully determines the exported 6-face orbit-pattern for CXXC at the current layer
- Pattern multiplicities range from 27 to 216

### Arity-4 Interaction Inventories

[MEASURED_FROM_CODE]

- CXXC face inventory exported: full population 531,441 rows with faces CX, XX, XC, CC, left CXC, right CXC
- CXXC distinct face-patterns: 2,744, with multiplicities 27–216
- CXXC conditioned on middle XX orbit: 56 rows, counts 2,187–17,496, distinct face-patterns 25–81
- CXXC conditioned on left CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81
- CXXC conditioned on right CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81
- CXXC joint interior table (left CXC, middle XX, right CXC): 2,744 distinct joint keys, exactly matching the 2,744 full face-patterns
- AXXC face inventory exported: full population 531,441 rows with faces AX, XX, XC, AC, left AXC, right AXC
- AXXC distinct face-patterns: 2,870, with multiplicities 27–216
- AXXC conditioned on middle XX orbit: 56 rows, counts 2,187–17,496, distinct face-patterns 20–81
- AXXC joint interior table (left AXC, middle XX, right AXC): 2,870 distinct joint keys, exactly matching the 2,870 full face-patterns

### Arity-4 Orbit / Signature / Stabilizer Parity

[MEASURED_FROM_CODE]

- CXXC: 2,744 orbits, 2,744 distinct signatures, orbit-complete at the current arity-4 signature layer
- AXXC: 2,870 orbits, 2,870 distinct signatures, orbit-complete at the current arity-4 signature layer
- Orbit size range for both schemas: 27–216
- Stabilizer size range for both schemas: 1–8
## 11. OBJECT VS LENS DISTINCTION

[GROUND_TRUTH]

### The Object

- Full raw base-9 warehouse (all 9^k tuples)
- Typed species A, B, C, X
- Typed schema definitions currently instantiated and bridged in the warehouse (including CXXC and AXXC, with arity-4 orbit/signature/stabilizer parity now recorded)
- Primitive exact rules (live/dead, fibers)
- Typed/raw bridge embeddings for all schemas

### Organizational Lenses (Not the Object)

- Orbit metadata caches
- Signature caches
- Stabilizer element lists
- Canonicalization tables
- Projection metadata
- Composition caches
- Cross-schema alignment tables

**Critical:** Object facts and lens facts must be kept distinct.
A lens update does not change the underlying object.

In this dossier, once a typed schema has been explicitly registered and injectively bridged into the warehouse, it is treated as part of the built object structure. Orbit tables, signature caches, marginal summaries, and alignment summaries remain organizational lenses over that built structure.

## 12. PROVENANCE / EVIDENCE LABELS

[GROUND_TRUTH]

Evidence taxonomy used in this document:

- [GROUND_TRUTH] - Core definition, object structure
- [EXACT_DERIVED] - Results derived from ground truth by exact computation
- [MEASURED_FROM_CODE] - Measured numeric outputs from executed code
- [REPAIRED] - Corrected after bug fix in orbit metadata cache
- [INTERPRETATION] - Analysis or interpretation, not ground truth
- [OPEN_FRONT] - Known gaps or incomplete areas

## 13. CURRENT GAPS / OPEN FRONTS

[OPEN_FRONT]

- First two arity-4 typed schemas exist (CXXC, AXXC); broader higher-arity typed schema
  family expansion remains open
- Full orbit-complete signatures not yet known for XX, AX, BX schemas
- Unified raw-backed composition caches not yet built
- Closure/refinement not yet globally rerun inside unified raw-backed warehouse

These are genuine incompletions, not promises.

## 14. HOW TO READ THIS DOSSIER

[GROUND_TRUTH]

### Reading Rules

1. **Distinguish object facts from lens facts.** The object is the raw
   warehouse, typed species, schemas, rules, and bridge embeddings.
   Caches, orbits, signatures, and alignment tables are organizational
   lenses, not the object itself.
2. **Cite sections precisely.** Any reference to facts in this dossier
   should name the section and the specific claim used.
3. **Do not confuse a lens with the ground truth.** Orbit metadata is a
   computed view. Signature counts are a computed view. Neither replaces
   the raw config data.
4. **Any partial or reorganized view must state what it omits or re-indexes.** If a future document
   presents a simplified or reorganized picture of the object, it must say explicitly
   which raw distinctions have been dropped, merged, or re-keyed.

### Scope Reminder

This dossier records the object as built. It does not prescribe how the
object should be viewed, used, or simplified. Those are separate decisions
that must be made consciously and stated explicitly.

## 15. CROSS-SCHEMA RAW ALIGNMENT

[MEASURED_FROM_CODE]

### Raw Arity-3 Alignment (CX, XC, AX, BX)

All four schemas bridge to raw arity 3 under different role overlays:

| Schema | Role Overlay |
|--------|--------------|
| CX | (C, A_X, B_X) |
| XC | (A_X, B_X, C) |
| AX | (A, A_X, B_X) |
| BX | (B, A_X, B_X) |

Alignment facts:

- Each schema image size = 729
- Distinct raw tuples occupied = 729
- Aggregate image count = 2,916
- Occupancy ratio = 4.0
- All four schema images coincide exactly on raw tuple support

Supporting artifacts: exports/raw3_alignment.csv, exports/raw3_alignment.md


### Raw Arity-6 Alignment (CXXC, AXXC)

[MEASURED_FROM_CODE]

Both arity-4 schemas bridge to raw arity 6:

| Schema | Role Overlay | Image Size |
|--------|--------------|-----------:|
| CXXC | (C, A_X1, B_X1, A_X2, B_X2, C) | 531,441 |
| AXXC | (A, A_X1, B_X1, A_X2, B_X2, C) | 531,441 |

Alignment facts:

- Distinct raw arity-6 tuples occupied by CXXC = 531,441
- Distinct raw arity-6 tuples occupied by AXXC = 531,441
- Overlap size = 531,441
- The two schema images coincide exactly on raw arity-6 support

Supporting artifacts: exports/raw6_alignment_CXXC_AXXC.csv, exports/raw6_alignment_CXXC_AXXC.md

## 16. CORRECTION NOTE: ORBIT METADATA REPAIR

[REPAIRED]

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

Additionally, the earlier BX orbit count of 18 was wrong due to a
separate bug (identity B-action). The corrected BX orbit count is 10.

### Resolution

- Repaired orbit metadata now uses actual canonical orbit representatives
- Signatures are computed from the correct representative config
- See exports/orbit_metadata_repair_report.md for full details
- Old cached signature counts are superseded

## 17. SUPPORTING EXPORT ARTIFACTS

[MEASURED_FROM_CODE]

The following evidence tables have been exported alongside this dossier:

**Atomic / Raw Tables:**
- `exports/X_atoms.csv`
- `exports/X_atoms.md`
- `exports/C_fibers.csv`
- `exports/C_fibers.md`
- `exports/X_live.csv`
- `exports/X_dead.csv`
- `exports/X_partition.md`

**Action Tables:**
- `exports/actions_A.csv`
- `exports/actions_B.csv`
- `exports/actions_C.csv`
- `exports/actions_X.csv`
- `exports/actions_summary.csv`
- `exports/actions_summary.md`

**Orbit Tables:**
- `exports/orbits_XX.csv`
- `exports/orbits_CX.csv`
- `exports/orbits_XC.csv`
- `exports/orbits_CC.csv`
- `exports/orbits_AX.csv`
- `exports/orbits_BX.csv`
- `exports/orbits_CXC.csv`
- `exports/orbits_summary.md`

**Signature Tables:**
- `exports/signatures_XX.csv`
- `exports/signatures_CX.csv`
- `exports/signatures_XC.csv`
- `exports/signatures_CC.csv`
- `exports/signatures_AX.csv`
- `exports/signatures_BX.csv`
- `exports/signatures_CXC.csv`
- `exports/signature_summary.md`

**Stabilizer Tables:**
- `exports/stabilizers_XX.csv`
- `exports/stabilizers_CX.csv`
- `exports/stabilizers_XC.csv`
- `exports/stabilizers_CC.csv`
- `exports/stabilizers_AX.csv`
- `exports/stabilizers_BX.csv`
- `exports/stabilizers_CXC.csv`
- `exports/stabilizers_summary.md`

**Repair Reports:**
- `exports/orbit_metadata_repair_report.md`
- `exports/orbit_metadata_repaired_summary.csv`

**Typed/Raw Bridges:**
- `exports/bridge_XX.csv`
- `exports/bridge_XX.md`

**Cross-Schema Alignment:**
- `exports/raw3_alignment.csv`
- `exports/raw3_alignment.md`
- `exports/raw6_alignment_CXXC_AXXC.csv`
- `exports/raw6_alignment_CXXC_AXXC.md`
- `exports/orbits_CXXC.csv`
- `exports/orbits_AXXC.csv`
- `exports/arity4_orbit_signature_stabilizer_summary.md`

**Composition Exports:**
- `exports/comp_CX_XC_to_CC.csv`
- `exports/comp_CX_XC_to_CC.md`
- `exports/comp_CX_XC_to_CC_witnesses.csv`

**Higher-Arity Schema Exports:**
- `exports/schema_CXXC_summary.md`
- `exports/bridge_CXXC.csv`
- `exports/schema_AXXC_summary.md`
- `exports/bridge_AXXC.csv`

**Arity-4 Interaction Inventories:**
- `exports/CXXC_face_inventory.csv`
- `exports/CXXC_face_inventory.md`
- `exports/CXXC_face_patterns.csv`
- `exports/CXXC_face_patterns.md`
- `exports/CXXC_xx_marginals.csv`
- `exports/CXXC_xx_marginals.md`
- `exports/CXXC_left_CXC_marginals.csv`
- `exports/CXXC_right_CXC_marginals.csv`
- `exports/CXXC_cxc_marginals.md`
- `exports/CXXC_joint_CXC_XX_CXC.csv`
- `exports/CXXC_joint_CXC_XX_CXC.md`
- `exports/AXXC_face_inventory.csv`
- `exports/AXXC_face_inventory.md`
- `exports/AXXC_face_patterns.csv`
- `exports/AXXC_face_patterns.md`
- `exports/AXXC_xx_marginals.csv`
- `exports/AXXC_xx_marginals.md`
- `exports/AXXC_joint_AXC_XX_AXC.csv`
- `exports/AXXC_joint_AXC_XX_AXC.md`


## 18. ARITY-4 TYPED SCHEMAS: CXXC AND AXXC

[MEASURED_FROM_CODE]

The current warehouse contains two fully bridged arity-4 schemas.

| Property | CXXC | AXXC |
|----------|------|------|
| Schema | C × X × X × C | A × X × X × C |
| Typed arity | 4 | 4 |
| Raw arity | 6 | 6 |
| Typed config count | 531,441 | 531,441 |
| Orbit count | 2,744 | 2,870 |
| Distinct signatures | 2,744 | 2,870 |
| Orbit-complete at current layer | yes | yes |
| Bridge export | full population | full population |
| Injective | yes | yes |
| Role overlay | (C, A_X1, B_X1, A_X2, B_X2, C) | (A, A_X1, B_X1, A_X2, B_X2, C) |
| Distinct face-patterns | 2,744 | 2,870 |
| Joint interior keys | 2,744 (CXC–XX–CXC) | 2,870 (AXC–XX–AXC) |

Supporting artifacts:
- CXXC: exports/schema_CXXC_summary.md, exports/bridge_CXXC.csv, exports/CXXC_face_inventory.csv, exports/CXXC_face_patterns.csv, exports/CXXC_xx_marginals.csv, exports/CXXC_joint_CXC_XX_CXC.csv
- AXXC: exports/schema_AXXC_summary.md, exports/bridge_AXXC.csv, exports/AXXC_face_inventory.csv, exports/AXXC_face_patterns.csv, exports/AXXC_xx_marginals.csv, exports/AXXC_joint_AXC_XX_AXC.csv

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------