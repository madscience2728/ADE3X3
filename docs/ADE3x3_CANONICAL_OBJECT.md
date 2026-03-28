======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-27 21:56:23
Generator: generate_canon_doc.py
Source: Current warehouse state with inline data tables

This is a canonical technical dossier of the current object state.
----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-27 21:56:23
**Generator Script:** generate_canon_doc.py
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

### C Target-Fiber Table

[GROUND_TRUTH]

Each output basis atom C[r,u] has exactly 3 live X atoms in its inverse fiber:

| c_idx | C atom | Target fiber (3 live X atoms) |
|-------|--------|-------------------------------|
| 0 | C[0,0] | X[0,0|0,0], X[0,1|1,0], X[0,2|2,0] |
| 1 | C[0,1] | X[0,0|0,1], X[0,1|1,1], X[0,2|2,1] |
| 2 | C[0,2] | X[0,0|0,2], X[0,1|1,2], X[0,2|2,2] |
| 3 | C[1,0] | X[1,0|0,0], X[1,1|1,0], X[1,2|2,0] |
| 4 | C[1,1] | X[1,0|0,1], X[1,1|1,1], X[1,2|2,1] |
| 5 | C[1,2] | X[1,0|0,2], X[1,1|1,2], X[1,2|2,2] |
| 6 | C[2,0] | X[2,0|0,0], X[2,1|1,0], X[2,2|2,0] |
| 7 | C[2,1] | X[2,0|0,1], X[2,1|1,1], X[2,2|2,1] |
| 8 | C[2,2] | X[2,0|0,2], X[2,1|1,2], X[2,2|2,2] |

**Derivable Pattern**: C[r,u] ← {X[r,s|s,u] : s ∈ {0,1,2}} (all live X with matching row r and column u)

### Complete X Atom Inventory

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

All 81 X atoms with live/dead status and target mapping:

| x_idx | X atom | a_idx | b_idx | live | target |
|-------|--------|-------|-------|------|--------|
| 0 | X[0,0|0,0] | 0 | 0 | LIVE | C[0,0] |
| 1 | X[0,0|0,1] | 0 | 1 | LIVE | C[0,1] |
| 2 | X[0,0|0,2] | 0 | 2 | LIVE | C[0,2] |
| 3 | X[0,0|1,0] | 0 | 3 | DEAD | none |
| 4 | X[0,0|1,1] | 0 | 4 | DEAD | none |
| 5 | X[0,0|1,2] | 0 | 5 | DEAD | none |
| 6 | X[0,0|2,0] | 0 | 6 | DEAD | none |
| 7 | X[0,0|2,1] | 0 | 7 | DEAD | none |
| 8 | X[0,0|2,2] | 0 | 8 | DEAD | none |
| 9 | X[0,1|0,0] | 1 | 0 | DEAD | none |
| 10 | X[0,1|0,1] | 1 | 1 | DEAD | none |
| 11 | X[0,1|0,2] | 1 | 2 | DEAD | none |
| 12 | X[0,1|1,0] | 1 | 3 | LIVE | C[0,0] |
| 13 | X[0,1|1,1] | 1 | 4 | LIVE | C[0,1] |
| 14 | X[0,1|1,2] | 1 | 5 | LIVE | C[0,2] |
| ... | ... | ... | ... | ... | ... |
| 66 | X[2,1|1,0] | 7 | 3 | LIVE | C[2,0] |
| 67 | X[2,1|1,1] | 7 | 4 | LIVE | C[2,1] |
| 68 | X[2,1|1,2] | 7 | 5 | LIVE | C[2,2] |
| 69 | X[2,1|2,0] | 7 | 6 | DEAD | none |
| 70 | X[2,1|2,1] | 7 | 7 | DEAD | none |
| 71 | X[2,1|2,2] | 7 | 8 | DEAD | none |
| 72 | X[2,2|0,0] | 8 | 0 | DEAD | none |
| 73 | X[2,2|0,1] | 8 | 1 | DEAD | none |
| 74 | X[2,2|0,2] | 8 | 2 | DEAD | none |
| 75 | X[2,2|1,0] | 8 | 3 | DEAD | none |
| 76 | X[2,2|1,1] | 8 | 4 | DEAD | none |
| 77 | X[2,2|1,2] | 8 | 5 | DEAD | none |
| 78 | X[2,2|2,0] | 8 | 6 | LIVE | C[2,0] |
| 79 | X[2,2|2,1] | 8 | 7 | LIVE | C[2,1] |
| 80 | X[2,2|2,2] | 8 | 8 | LIVE | C[2,2] |

**Summary**: 27 live, 54 dead

## 5. SYMMETRY/ACTION SYSTEM

[GROUND_TRUTH]

- **Compatible Symmetry Group Size**: 216
- **Group Type**: S3 x S3 x S3 with compatibility constraint pi_cA == pi_rB
- **Action Scope**: Acts on A, B, C, and X atoms
- **Canonicalization**: All core schemas are canonicalized under group action

### Explicit Group Action Formulas

[GROUND_TRUTH]

The group is coordinatized by three S3 factors:
- `pi_rA` acts on row indices of A, C, and the left row index of X
- `pi_shared` acts on the column index of A and row index of B (shared middle indices)
- `pi_cB` acts on column indices of B, C, and the right column index of X

**Action on atomic species:**

- **A**: A[r,s] → A[pi_rA(r), pi_shared(s)]
- **B**: B[t,u] → B[pi_shared(t), pi_cB(u)]
- **C**: C[r,u] → C[pi_rA(r), pi_cB(u)]
- **X**: X[r,s|t,u] → X[pi_rA(r), pi_shared(s) | pi_shared(t), pi_cB(u)]

**Compatibility constraint**: pi_shared must be the same permutation for both A-column and B-row

### Canonicalization Rule

[GROUND_TRUTH]

**Canonical Representative Selection**: For each orbit, the canonical representative
is the configuration with the **minimum config_id** under the group action.

**Config ID Encoding**: Configurations are encoded as base-9 integers where each
position holds an atom index from {0,...,8}. For example:
- CX[C[0,0], X[0,0|0,0]] → (c_idx=0, a_idx=0, b_idx=0) → config_id = 0
- CX[C[0,1], X[0,0|0,0]] → (c_idx=1, a_idx=0, b_idx=0) → config_id = 81

**rep_config_id**: The config_id of the canonical orbit representative

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

### Signature Collision Tables

[MEASURED_FROM_CODE]

**XX Schema** (56 orbits, 48 distinct signatures - 8 colliding signature groups):
- Note: Collision resolution requires additional features beyond base signature
- See exports/signatures_XX.csv for full collision mapping

**AX Schema** (10 orbits, 8 distinct signatures - 2 colliding signature groups):
- Orbits 2, 4, 7 share signature: (True, False, False)
- Resolution: check if X is live or specific shared-index pattern

**BX Schema** (10 orbits, 8 distinct signatures - 2 colliding signature groups):
- Similar collision pattern to AX due to symmetry
- Resolution: check if X is live or specific shared-index pattern

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

[SUPERSEDED]

Note: The following result was measured on the old type-based composition (28 mixed keys):
- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)
- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target
- Best 2-tuple: [r1_eq_r, u1_eq_u]

**Status**: Refinement engine has NOT been rerun on the corrected 14 mixed keys.

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
- [SUPERSEDED] - Historical result replaced by corrected measurement
- [OPEN_FRONT] - Known gaps or incomplete areas

## 13. CURRENT GAPS / OPEN FRONTS

[OPEN_FRONT]

- First arity-4 typed schema exists (CXXC); orbit/signature data not yet computed
- Broader higher-arity typed schema family expansion remains open
- Full orbit-complete signatures not yet known for XX, AX, BX schemas
- Unified raw-backed composition caches not yet built
- Closure/refinement not yet globally rerun inside unified raw-backed warehouse
- Composition grid for CX × XC → CC not yet inlined (see exports/comp_CX_XC_to_CC.csv)

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

## 15. SUPPORTING EXPORT ARTIFACTS

[MEASURED_FROM_CODE]

The following evidence tables have been exported alongside this dossier:

**Atomic / Raw Tables:**
- `outputs/exports/X_atoms.csv` (all 81 X atoms)
- `outputs/exports/X_atoms.md`
- `outputs/exports/C_fibers.csv` (9 C fibers)
- `outputs/exports/C_fibers.md`
- `outputs/exports/X_live.csv` (27 live X)
- `outputs/exports/X_dead.csv` (54 dead X)
- `outputs/exports/X_partition.md`

**Action Tables:**
- `outputs/exports/actions_A.csv`
- `outputs/exports/actions_B.csv`
- `outputs/exports/actions_C.csv`
- `outputs/exports/actions_X.csv`
- `outputs/exports/actions_summary.csv`
- `outputs/exports/actions_summary.md`

**Orbit Tables:**
- `outputs/exports/orbits_XX.csv`
- `outputs/exports/orbits_CX.csv`
- `outputs/exports/orbits_XC.csv`
- `outputs/exports/orbits_CC.csv`
- `outputs/exports/orbits_AX.csv`
- `outputs/exports/orbits_BX.csv`
- `outputs/exports/orbits_CXC.csv`
- `outputs/exports/orbits_summary.md`

**Composition Exports:**
- `outputs/exports/comp_CX_XC_to_CC.csv` (32 realized orbit pairs)
- `outputs/exports/comp_CX_XC_to_CC.md`
- `outputs/exports/comp_CX_XC_to_CC_witnesses.csv`

For complete artifact listing, see `outputs/exports/` directory.

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------

Note: This canonical dossier contains all critical inline tables for single-file sufficiency.
Additional orbit rosters and extended data available in exports directory.