======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-27 18:29:56
Generator: ade3x3_step13d_debias_md_dossier_generator.py
Source: Current warehouse state (steps 1-24, repairs 10b/13b, exports 14-24)

This is a canonical technical dossier of the current object state.
----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-27 18:29:56
**Generator Script:** ade3x3_step13d_debias_md_dossier_generator.py
**Provenance:** Built from steps 1-24, with orbit metadata repair (step 10b)

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
| CXXC   | 4           | 6         | 531,441 | -    | Yes     | first arity-4 schema |

Note: All typed schemas including XX are now bridged into the raw warehouse.
X bridge marked Yes* because X expands to 2 raw slots (A_idx, B_idx), per Section 7 bridge rules.
CXXC is the first typed schema beyond the arity-3 core.

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
- First arity-4 typed schema (CXXC) registered and bridged

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

### Refinement Engine Result

[MEASURED_FROM_CODE]

- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)
- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target
- Best 2-tuple: [r1_eq_r, u1_eq_u]

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

## 11. OBJECT VS LENS DISTINCTION

[GROUND_TRUTH]

### The Object

- Full raw base-9 warehouse (all 9^k tuples)
- Typed species A, B, C, X
- Typed schema definitions including first arity-4 schema CXXC
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

- First arity-4 typed schema exists (CXXC); broader higher-arity typed schema
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
4. **Any reduced view must state what it omits.** If a future document
   presents a simplified picture of the object, it must say explicitly
   which raw distinctions have been dropped.

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

**Composition Exports:**
- `exports/comp_CX_XC_to_CC.csv`
- `exports/comp_CX_XC_to_CC.md`
- `exports/comp_CX_XC_to_CC_witnesses.csv`

**Higher-Arity Schema Exports:**
- `exports/schema_CXXC_summary.md`
- `exports/bridge_CXXC.csv`


## 18. FIRST ARITY-4 TYPED SCHEMA: CXXC

[MEASURED_FROM_CODE]

Typed schema growth beyond the arity-3 core has begun.

| Property | Value |
|----------|-------|
| Schema | C × X × X × C |
| Typed arity | 4 |
| Raw arity | 6 |
| Typed config count | 531,441 |
| Bridge export | full population |
| Injective | yes |
| Role overlay | (C, A_X1, B_X1, A_X2, B_X2, C) |

Supporting artifacts: exports/schema_CXXC_summary.md, exports/bridge_CXXC.csv

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------

## 4A. INLINE LIVE/DEAD AND INVERSE-FIBER TABLES

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

### X Live/Dead Summary

- |X_live| = 27
- |X_dead| = 54

Live-count cross-table by (s,t):

| s\t | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 9 | 0 | 0 |
| 1 | 0 | 9 | 0 |
| 2 | 0 | 0 | 9 |

### A Inverse-Fiber Table

| a_local_id | A atom | X atoms with this a_idx |
|---|---|---|
| 0 | A[0,0] | X[0,0|0,0], X[0,0|0,1], X[0,0|0,2], X[0,0|1,0], X[0,0|1,1], X[0,0|1,2], X[0,0|2,0], X[0,0|2,1], X[0,0|2,2] |
| 1 | A[0,1] | X[0,1|0,0], X[0,1|0,1], X[0,1|0,2], X[0,1|1,0], X[0,1|1,1], X[0,1|1,2], X[0,1|2,0], X[0,1|2,1], X[0,1|2,2] |
| 2 | A[0,2] | X[0,2|0,0], X[0,2|0,1], X[0,2|0,2], X[0,2|1,0], X[0,2|1,1], X[0,2|1,2], X[0,2|2,0], X[0,2|2,1], X[0,2|2,2] |
| 3 | A[1,0] | X[1,0|0,0], X[1,0|0,1], X[1,0|0,2], X[1,0|1,0], X[1,0|1,1], X[1,0|1,2], X[1,0|2,0], X[1,0|2,1], X[1,0|2,2] |
| 4 | A[1,1] | X[1,1|0,0], X[1,1|0,1], X[1,1|0,2], X[1,1|1,0], X[1,1|1,1], X[1,1|1,2], X[1,1|2,0], X[1,1|2,1], X[1,1|2,2] |
| 5 | A[1,2] | X[1,2|0,0], X[1,2|0,1], X[1,2|0,2], X[1,2|1,0], X[1,2|1,1], X[1,2|1,2], X[1,2|2,0], X[1,2|2,1], X[1,2|2,2] |
| 6 | A[2,0] | X[2,0|0,0], X[2,0|0,1], X[2,0|0,2], X[2,0|1,0], X[2,0|1,1], X[2,0|1,2], X[2,0|2,0], X[2,0|2,1], X[2,0|2,2] |
| 7 | A[2,1] | X[2,1|0,0], X[2,1|0,1], X[2,1|0,2], X[2,1|1,0], X[2,1|1,1], X[2,1|1,2], X[2,1|2,0], X[2,1|2,1], X[2,1|2,2] |
| 8 | A[2,2] | X[2,2|0,0], X[2,2|0,1], X[2,2|0,2], X[2,2|1,0], X[2,2|1,1], X[2,2|1,2], X[2,2|2,0], X[2,2|2,1], X[2,2|2,2] |

### B Inverse-Fiber Table

| b_local_id | B atom | X atoms with this b_idx |
|---|---|---|
| 0 | B[0,0] | X[0,0|0,0], X[0,1|0,0], X[0,2|0,0], X[1,0|0,0], X[1,1|0,0], X[1,2|0,0], X[2,0|0,0], X[2,1|0,0], X[2,2|0,0] |
| 1 | B[0,1] | X[0,0|0,1], X[0,1|0,1], X[0,2|0,1], X[1,0|0,1], X[1,1|0,1], X[1,2|0,1], X[2,0|0,1], X[2,1|0,1], X[2,2|0,1] |
| 2 | B[0,2] | X[0,0|0,2], X[0,1|0,2], X[0,2|0,2], X[1,0|0,2], X[1,1|0,2], X[1,2|0,2], X[2,0|0,2], X[2,1|0,2], X[2,2|0,2] |
| 3 | B[1,0] | X[0,0|1,0], X[0,1|1,0], X[0,2|1,0], X[1,0|1,0], X[1,1|1,0], X[1,2|1,0], X[2,0|1,0], X[2,1|1,0], X[2,2|1,0] |
| 4 | B[1,1] | X[0,0|1,1], X[0,1|1,1], X[0,2|1,1], X[1,0|1,1], X[1,1|1,1], X[1,2|1,1], X[2,0|1,1], X[2,1|1,1], X[2,2|1,1] |
| 5 | B[1,2] | X[0,0|1,2], X[0,1|1,2], X[0,2|1,2], X[1,0|1,2], X[1,1|1,2], X[1,2|1,2], X[2,0|1,2], X[2,1|1,2], X[2,2|1,2] |
| 6 | B[2,0] | X[0,0|2,0], X[0,1|2,0], X[0,2|2,0], X[1,0|2,0], X[1,1|2,0], X[1,2|2,0], X[2,0|2,0], X[2,1|2,0], X[2,2|2,0] |
| 7 | B[2,1] | X[0,0|2,1], X[0,1|2,1], X[0,2|2,1], X[1,0|2,1], X[1,1|2,1], X[1,2|2,1], X[2,0|2,1], X[2,1|2,1], X[2,2|2,1] |
| 8 | B[2,2] | X[0,0|2,2], X[0,1|2,2], X[0,2|2,2], X[1,0|2,2], X[1,1|2,2], X[1,2|2,2], X[2,0|2,2], X[2,1|2,2], X[2,2|2,2] |


## 5A. GROUP ACTION SPECIFICATION

[GROUND_TRUTH]

The symmetry group is coordinatized by a chosen labeling convention for its three S3 factors:
- `pi_rA` acts on row indices of A, C, and the left row index of X
- `pi_shared` acts on the column index of A and both shared middle indices of X and B
- `pi_cB` acts on column indices of B, C, and the right column index of X

Compatibility is enforced by using the same `pi_shared` on the A-column / B-row interface.

Generic action on an X atom:

`X[r,s|t,u] -> X[pi_rA(r), pi_shared(s) | pi_shared(t), pi_cB(u)]`

This coordinatization is a naming convention for the three factors, not additional structure beyond the action itself.


## 20. SIGNATURE FORMAT DEFINITIONS

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

### CC Signature Definition

Signature fields: (same_cell, same_row, same_col).

### CX Signature Definition

Signature fields: (x_live, c_is_target_of_x_if_live, c_row_equals_x_row, c_col_equals_x_output_col).

### XC Signature Definition

Signature fields: (x_live, c_is_target_of_x_if_live, x_row_equals_c_row, x_output_col_equals_c_col).

### AX Signature Definition

Signature fields: (a_row_equals_x_left_row, a_col_equals_x_left_col, x_live).

### BX Signature Definition

Signature fields: (b_row_equals_x_right_row, b_col_equals_x_right_col, x_live).

### XX Signature Definition

Signature fields: (x1_live, x2_live, same_r, same_s, same_t, same_u, same_A_atom, same_B_atom).

### CXC Signature Definition

Signature fields: (x_live, c1_equals_c2, c1_is_target_if_live, c2_is_target_if_live, row_triple(c1,x,c2), col_triple(c1,x,c2)).


## 21. INLINE ORBIT REPRESENTATIVE ROSTERS

[MEASURED_FROM_CODE]

One row per orbit for the core arity-2 and arity-3 schemas.
### CC Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
| 0 | 0 | CC[C[0,0],C[0,0]] | 9 | 24 | (True, True, True) |
| 1 | 1 | CC[C[0,0],C[0,1]] | 18 | 12 | (False, True, False) |
| 2 | 3 | CC[C[0,0],C[1,0]] | 18 | 12 | (False, False, True) |
| 3 | 4 | CC[C[0,0],C[1,1]] | 36 | 6 | (False, False, False) |

### CX Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
| 0 | 0 | CX[C[0,0],X[0,0|0,0]] | 27 | 8 | (True, True, True, True) |
| 1 | 1 | CX[C[0,0],X[0,0|0,1]] | 54 | 4 | (True, False, True, False) |
| 2 | 3 | CX[C[0,0],X[0,0|1,0]] | 54 | 4 | (False, False, True, True) |
| 3 | 4 | CX[C[0,0],X[0,0|1,1]] | 108 | 2 | (False, False, True, False) |
| 4 | 27 | CX[C[0,0],X[1,0|0,0]] | 54 | 4 | (True, False, False, True) |
| 5 | 28 | CX[C[0,0],X[1,0|0,1]] | 108 | 2 | (True, False, False, False) |
| 6 | 30 | CX[C[0,0],X[1,0|1,0]] | 108 | 2 | (False, False, False, True) |
| 7 | 31 | CX[C[0,0],X[1,0|1,1]] | 216 | 1 | (False, False, False, False) |

### XC Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
| 0 | 0 | XC[X[0,0|0,0],C[0,0]] | 27 | 8 | (True, True, True, True) |
| 1 | 1 | XC[X[0,0|0,0],C[0,1]] | 54 | 4 | (True, False, True, False) |
| 2 | 3 | XC[X[0,0|0,0],C[1,0]] | 54 | 4 | (True, False, False, True) |
| 3 | 4 | XC[X[0,0|0,0],C[1,1]] | 108 | 2 | (True, False, False, False) |
| 4 | 27 | XC[X[0,0|1,0],C[0,0]] | 54 | 4 | (False, False, True, True) |
| 5 | 28 | XC[X[0,0|1,0],C[0,1]] | 108 | 2 | (False, False, True, False) |
| 6 | 30 | XC[X[0,0|1,0],C[1,0]] | 108 | 2 | (False, False, False, True) |
| 7 | 31 | XC[X[0,0|1,0],C[1,1]] | 216 | 1 | (False, False, False, False) |

### AX Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
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

### BX Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
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

### XX Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
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

### CXC Orbit Representatives

| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |
|---|---|---|---|---|---|
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


## 16A. SUPSERSEDED / REPAIRED STATUS NOTES

[REPAIRED] / [MEASURED_FROM_CODE]

The older type-based composition summary `256 / 64 / 36 / 28` is a superseded historical result from an older layer and should not be used as the current canonical composition fact.

The earlier BX orbit count `18` is superseded; the corrected current value is `10` after repair of the B-action path.

