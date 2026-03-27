======================================================================
ADE3x3 CANONICAL OBJECT DOSSIER
======================================================================

Generated: 2026-03-27 14:15:24
Generator: ade3x3_step13_md_dossier_generator.py
Source: Current warehouse state (steps 1-12)

This is a canonical technical dossier of the current object state.
----------------------------------------------------------------------

## 1. TITLE AND GENERATION METADATA

**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication
**Dossier Type:** Canonical Object Technical Dossier
**Generated:** 2026-03-27 14:15:24
**Generator Script:** step13_md_dossier_generator.py
**Provenance:** Built from steps 1-12 of the ADE3x3 project

## 2. MISSION AND NON-GOALS

### Mission
[GROUND_TRUTH]

This project is gathering full-object exact data about the ambient bilinear
universe for 3x3 matrix multiplication structure discovery.

The current source of truth is the full object as stored in the warehouse,
not a reduced model or compressed surrogate.

### Non-Goals
[GROUND_TRUTH]

- This dossier is NOT a compression-first document
- This is NOT a low-rank analysis document
- This is NOT an optimization target specification
- Downstream derivations may reduce later, but this preserves fidelity first

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
- **Group Type**: S3 × S3 × S3 with compatibility constraint pi_cA == pi_rB
- **Action Scope**: Acts on A, B, C, and X atoms
- **Canonicalization**: All core schemas are canonicalized under group action

## 6. TYPED SCHEMAS CURRENTLY BUILT

[GROUND_TRUTH]

| Schema | Typed Arity | Raw Arity | Count | Orbits | Bridged |
|--------|-------------|-----------|-------|--------|---------|
| A      | 1           | 1         | 9     | -      | -       |
| B      | 1           | 1         | 9     | -      | -       |
| C      | 1           | 1         | 9     | -      | Yes     |
| X      | 1           | 2         | 81    | -      | -       |
| CC     | 2           | 2         | 81    | 4      | Yes     |
| CX     | 2           | 3         | 729   | 8      | Yes     |
| XC     | 2           | 3         | 729   | 8      | Yes     |
| AX     | 2           | 3         | 729   | 10     | Yes     |
| BX     | 2           | 3         | 729   | 18     | Yes     |
| CXC    | 3           | 4         | 6,561 | 50     | Yes     |

## 7. TYPED/RAW BRIDGE

[GROUND_TRUTH]

### Bridge Rules

- A[r,s] -> raw symbol idx = 3*r + s
- B[t,u] -> raw symbol idx = 3*t + u
- C[r,u] -> raw symbol idx = 3*r + u
- X[r,s|t,u] -> raw tuple (A_idx(r,s), B_idx(t,u))

### Bridge Summary

| Schema | Typed Arity | Raw Arity | Role Overlay | Injectivity |
|--------|-------------|-----------|--------------|-------------|
| C      | 1           | 1         | (C,)         | yes         |
| CC     | 2           | 2         | (C, C)       | yes         |
| CX     | 2           | 3         | (C, A_X, B_X)| yes         |
| XC     | 2           | 3         | (A_X, B_X, C)| yes         |
| AX     | 2           | 3         | (A, A_X, B_X)| yes         |
| BX     | 2           | 3         | (B, A_X, B_X)| yes         |
| CXC    | 3           | 4         | (C, A_X, B_X, C) | yes    |

Note: Raw arity 3 shows 400% occupancy across multiple schemas because
different role overlays (CX, XC, AX, BX) occupy the same raw tuple positions.

## 8. WAREHOUSE INFRASTRUCTURE STATUS

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

### Current Infrastructure

- In-memory object DB with atomic species
- Raw config bulk-loading for arity-2 and arity-3 core schemas
- Symmetry-aware canonicalization for all typed schemas
- Orbit metadata cache for all schemas
- Signature caches per orbit
- Full raw 9-slot warehouse (steps 1-11)
- Typed/raw bridge (step 12)

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
| CX     | 729         | 8      | 91.1     | 2-8              |
| XC     | 729         | 8      | 91.1     | 2-8              |
| CC     | 81          | 4      | 20.2     | 12-24            |
| AX     | 729         | 10     | 72.9     | 4-8              |
| BX     | 729         | 18     | 40.5     | 4-8              |
| CXC    | 6,561       | 50     | 131.2    | 1-8              |

All orbit × stabilizer = 216 verified.

## 10. EXACT DERIVED RESULTS CURRENTLY KNOWN

[EXACT_DERIVED]

### Dead-Dead Pair Classification

- 24 orbits classified exactly
- Classification by: row equality, column equality, one of 6 shared-index patterns
- Full resolution with refined signature tuple (row_eq, col_eq, shared_pattern)

### CX x XC -> CC Composition Result

- 256 possible (alpha, beta) pairs
- 64 realized
- 36 deterministic (single gamma result)
- 28 mixed (multiple gamma results)

### Refinement Engine Result

- Single separator 'r1_eq_r' resolves ALL 28 mixed composition keys
- Also resolved by 'u1_eq_u', 'x_live', 'c1_equals_target', 'c2_equals_target', 'contracted'
- Best 2-tuple: [r1_eq_r, u1_eq_u]

### CXC Orbit Signatures

- 50 orbits -> 50 distinct signatures
- Signatures are orbit-complete at current cache level

### XX Orbit/Signature Gap

- 56 orbits collapse to 20 current cached signatures
- Signatures do not fully separate XX orbits

## 11. OBJECT VS LENS DISTINCTION

[GROUND_TRUTH]

### The Object (Ground Truth)

- Full raw base-9 warehouse (all 9^k tuples)
- Typed species A, B, C, X
- Typed schema definitions
- Primitive exact rules (live/dead, fibers)
- Typed/raw bridge embeddings

### Lens/Infrastructure (Not the Object)

- Orbit metadata caches
- Signature caches
- Canonicalization tables
- Projection metadata
- Composition caches

**Critical:** Downstream agents must distinguish object facts from lens facts.
Lens updates do not change the underlying object.

## 12. PROVENANCE / EVIDENCE LABELS

[GROUND_TRUTH]

Evidence taxonomy used in this document:

- `[GROUND_TRUTH]` - Core definition, object structure
- `[EXACT_DERIVED]` - Results derived from ground truth by exact computation
- `[MEASURED_FROM_CODE]` - Measured numeric outputs from executed code
- `[INTERPRETATION]` - Analysis or interpretation, not ground truth
- `[OPEN_FRONT]` - Known gaps or incomplete areas

## 13. CURRENT GAPS / OPEN FRONTS

[OPEN_FRONT]

- Full orbit-complete signatures not yet known for XX schema
- Unified raw-backed composition caches not yet built
- Higher-arity typed overlays beyond current core schemas not yet embedded
- Closure/refinement not yet globally rerun inside unified raw-backed warehouse

These are genuine incompletions, not promises.

## 14. INSTRUCTIONS FOR DOWNSTREAM AGENTS

[GROUND_TRUTH]

### Usage Guidelines

1. **Do NOT replace the full object with a reduced surrogate**
2. **Distinguish object facts from lens facts** in any derivation
3. **Cite exact sections/facts** used in any derivation
4. **If proposing compression** or quotients, state explicitly what raw distinctions are being discarded
5. **Prefer additions** that preserve fidelity unless reduction is explicitly the task

### Basis for Reasoning

This dossier provides the canonical basis. Any downstream derivation should
explicitly reference which sections are being used and which are being assumed.

## 15. APPENDIX: COMPACT FACT TABLES

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

### Raw Layer Counts

| k | 9^k |
|---|------|
|1|9|
|2|81|
|3|729|
|4|6561|
|5|59049|
|6|531441|
|7|4782969|
|8|43046721|
|9|387420489|

### Current Schema Counts

| Schema | Count |
|--------|-------|
| A | 9 |
| B | 9 |
| C | 9 |
| X | 81 |
| CC | 81 |
| CX | 729 |
| XC | 729 |
| AX | 729 |
| BX | 729 |
| CXC | 6561 |

### Bridge Summary

All bridges are injective and support exact roundtrip.

### Memory Summary

Total warehouse footprint: ~30 MB

----------------------------------------------------------------------
END OF DOSSIER
----------------------------------------------------------------------