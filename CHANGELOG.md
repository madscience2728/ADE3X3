# ADE3x3 Changelog

## Session: 2026-03-27

### Summary

Completed priority tasks from continuation prompt: signature collision refinement and CXXC arity-4 schema computation.

### Major Accomplishments

#### 1. Signature Collision Refinement (Priority 1 - Small)
**Status:** ✓ Complete

- **XX Schema**: Resolved 4 collision groups (8 orbits total)
  - Base: 56 orbits → 48 distinct signatures
  - Refined: 56 orbits → 56 distinct signatures
  - Refinement features: `(s2, t2, u2, r2)`

- **AX Schema**: Resolved 2 collision groups (4 orbits total)
  - Base: 10 orbits → 8 distinct signatures
  - Refined: 10 orbits → 10 distinct signatures
  - Refinement features: `(t, u)`

- **BX Schema**: Resolved 2 collision groups (4 orbits total)
  - Base: 10 orbits → 8 distinct signatures
  - Refined: 10 orbits → 10 distinct signatures
  - Refinement features: `(s, r)`

**Files Created:**
- `src/ade3x3/steps/ade3x3_step33_refine_signature_collisions.py`
- `outputs/exports/signatures_XX_refined.csv`
- `outputs/exports/signatures_AX_refined.csv`
- `outputs/exports/signatures_BX_refined.csv`
- `outputs/exports/signature_refinement_summary.md`

#### 2. CXXC Orbit Computation (Priority 2 - Medium)
**Status:** ✓ Complete

- **Total configurations:** 531,441 (9 × 81 × 81 × 9)
- **Orbits computed:** 2,744
- **Distinct signatures:** 784
- **Computation time:** 2.6 minutes
- **Hardware:** Ryzen 5900X, 80 GB RAM

**Key Results:**
- All orbits verify |orbit| × |stabilizer| = 216
- Sum of orbit sizes = 531,441 ✓
- First arity-4 schema fully cataloged

**Signature Format:**
```
(x1_live, x2_live, c1_equals_c2,
 c1_is_target_of_x1_if_live, c1_is_target_of_x2_if_live,
 c2_is_target_of_x1_if_live, c2_is_target_of_x2_if_live,
 row_quad, col_quad)
```

**Files Created:**
- `src/ade3x3/steps/ade3x3_step34_compute_cxxc_orbits.py`
- `outputs/exports/signatures_CXXC.csv` (2744 orbits)
- `outputs/exports/cxxc_computation_summary.md`

#### 3. CXXC Composition Analysis (Priority 3 - Medium)
**Status:** ✓ Complete

Analyzed marginal projections of CXXC onto all lower-arity schemas:

| Projection | Target Schema | Coverage |
|------------|---------------|----------|
| CXC_left (C1,X1,C2) | CXC | 100.0% (50/50) |
| CXC_right (C1,X2,C2) | CXC | 100.0% (50/50) |
| XX (X1,X2) | XX | 100.0% (56/56) |
| CX_left (C1,X1) | CX | 100.0% (8/8) |
| CX_right (C1,X2) | CX | 100.0% (8/8) |
| XC_left (X1,C2) | XC | 100.0% (8/8) |
| XC_right (X2,C2) | XC | 100.0% (8/8) |

**Key Finding:** CXXC achieves 100% coverage on all marginal projections - every orbit of each lower-arity schema appears in at least one CXXC configuration.

**Files Created:**
- `src/ade3x3/steps/ade3x3_step35_analyze_cxxc_marginals.py`
- `outputs/exports/cxxc_marginal_analysis.md`

#### 4. Canonical Dossier Update
**Status:** ✓ Complete

Updated [`docs/ADE3x3_CANONICAL_OBJECT.md`](docs/ADE3x3_CANONICAL_OBJECT.md):
- **New size:** 3,609 lines (447 KB) - up from 760 lines
- **New sections added:**
  - Section 15: SCHEMA CXXC - COMPLETE ORBIT ROSTER (2744 orbits inline)
  - Section 17: SIGNATURE COLLISION REFINEMENT
  - Section 18: CXXC MARGINAL PROJECTION ANALYSIS
  - Updated Section 6: CXXC orbit count (? → 2744)
  - Updated Section 19: CXXC signature format
  - Updated Section 24: Open fronts (marked completed items)

**Modified Files:**
- `generate_canon_doc.py` (added CXXC support and new sections)

### Technical Details

**Provenance Labels Used:**
- `[EXACT_DERIVED]` - Computed results from ground truth
- `[POST-REFINEMENT]` - After signature collision refinement
- `[MEASURED_FROM_CODE]` - Direct code output

**Computation Performance:**
- CXXC orbit computation: 531,441 configs in 154.9 seconds
- No memory issues (80 GB available, ~43M config limit)
- Parallelization: Not implemented (single-threaded orbit canonicalization)

### Remaining Open Fronts

As documented in dossier Section 24:

1. **CXXC signature collisions:** 2744 orbits → 784 signatures (1960 collisions)
   - Would require arity-4 specific refinement features

2. **Additional arity-4 schemas:** XCXC, XCCX, XXXC, XXX not yet explored
   - Each would require bridge definition and orbit computation

3. **Refinement engine:** Not yet rerun on corrected composition (14 mixed keys)
   - Could yield additional structural insights

4. **Higher arity layers:** Arity 5+ unexplored
   - Hardware limit at raw arity 10 (~3.5B configs)

### Files Modified

1. `generate_canon_doc.py` - Extended to include CXXC and refinement sections
2. `docs/ADE3x3_CANONICAL_OBJECT.md` - Regenerated with all new data

### Files Created

New computation scripts:
1. `src/ade3x3/steps/ade3x3_step33_refine_signature_collisions.py`
2. `src/ade3x3/steps/ade3x3_step34_compute_cxxc_orbits.py`
3. `src/ade3x3/steps/ade3x3_step35_analyze_cxxc_marginals.py`

New export files:
1. `outputs/exports/signatures_XX_refined.csv`
2. `outputs/exports/signatures_AX_refined.csv`
3. `outputs/exports/signatures_BX_refined.csv`
4. `outputs/exports/signature_refinement_summary.md`
5. `outputs/exports/signatures_CXXC.csv` (2744 rows)
6. `outputs/exports/cxxc_computation_summary.md`
7. `outputs/exports/cxxc_marginal_analysis.md`

### Verification

All results verified:
- ✓ Orbit-stabilizer theorem: |orbit| × |stabilizer| = 216 for all orbits
- ✓ Configuration coverage: Sum of orbit sizes equals total configs
- ✓ Signature uniqueness: All refined signatures are distinct (XX, AX, BX)
- ✓ Marginal coverage: 100% on all CXXC projections

### Next Steps (Optional - Priority 4)

If continuing:
1. Explore additional arity-4 schemas (XCXC, XCCX, XXXC, XXX)
2. Refine CXXC signatures to resolve 1960 collisions
3. Investigate arity-5 schemas (within hardware limits)
4. Develop composition algebra for higher-arity operations

---

**Session completed:** 2026-03-27 23:06:02
**Computation wall time:** ~10 minutes (CXXC orbit computation + analysis)
**Documentation time:** ~15 minutes (script writing + dossier updates)
