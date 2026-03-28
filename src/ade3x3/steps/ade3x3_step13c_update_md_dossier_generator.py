"""
ade3x3_step13c_update_md_dossier_generator.py

Update canonical dossier with:
- stabilizer element exports
- XX bridge
- raw3 alignment
- corrected orbit-based composition
- CXXC first arity-4 schema
"""

from datetime import datetime
from pathlib import Path

ORBIT_COUNTS = {'XX':56,'CX':8,'XC':8,'CC':4,'AX':10,'BX':10,'CXC':50}
SIG_COUNTS   = {'XX':48,'CX':8,'XC':8,'CC':4,'AX':8,'BX':8,'CXC':50}
ORBIT_COMPLETE = {'CX','XC','CC','CXC'}

EXPORT_ARTIFACTS = {
    'Atomic / Raw Tables': [
        'exports/X_atoms.csv', 'exports/X_atoms.md',
        'exports/C_fibers.csv', 'exports/C_fibers.md',
        'exports/X_live.csv', 'exports/X_dead.csv',
        'exports/X_partition.md',
    ],
    'Action Tables': [
        'exports/actions_A.csv', 'exports/actions_B.csv',
        'exports/actions_C.csv', 'exports/actions_X.csv',
        'exports/actions_summary.csv', 'exports/actions_summary.md',
    ],
    'Orbit Tables': [
        'exports/orbits_XX.csv', 'exports/orbits_CX.csv',
        'exports/orbits_XC.csv', 'exports/orbits_CC.csv',
        'exports/orbits_AX.csv', 'exports/orbits_BX.csv',
        'exports/orbits_CXC.csv', 'exports/orbits_summary.md',
    ],
    'Signature Tables': [
        'exports/signatures_XX.csv', 'exports/signatures_CX.csv',
        'exports/signatures_XC.csv', 'exports/signatures_CC.csv',
        'exports/signatures_AX.csv', 'exports/signatures_BX.csv',
        'exports/signatures_CXC.csv', 'exports/signature_summary.md',
    ],
    'Stabilizer Tables': [
        'exports/stabilizers_XX.csv', 'exports/stabilizers_CX.csv',
        'exports/stabilizers_XC.csv', 'exports/stabilizers_CC.csv',
        'exports/stabilizers_AX.csv', 'exports/stabilizers_BX.csv',
        'exports/stabilizers_CXC.csv', 'exports/stabilizers_summary.md',
    ],
    'Repair Reports': [
        'exports/orbit_metadata_repair_report.md',
        'exports/orbit_metadata_repaired_summary.csv',
    ],
    'Typed/Raw Bridges': [
        'exports/bridge_XX.csv', 'exports/bridge_XX.md',
    ],
    'Cross-Schema Alignment': [
        'exports/raw3_alignment.csv', 'exports/raw3_alignment.md',
    ],
    'Composition Exports': [
        'exports/comp_CX_XC_to_CC.csv', 'exports/comp_CX_XC_to_CC.md',
        'exports/comp_CX_XC_to_CC_witnesses.csv',
    ],
    'Higher-Arity Schema Exports': [
        'exports/schema_CXXC_summary.md', 'exports/bridge_CXXC.csv',
    ],
}


def generate():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    def w(s=""): L.append(s)

    w("=" * 70)
    w("ADE3x3 CANONICAL OBJECT DOSSIER")
    w("=" * 70)
    w()
    w(f"Generated: {ts}")
    w("Generator: ade3x3_step13c_update_md_dossier_generator.py")
    w("Source: Current warehouse state (steps 1-12, repairs 10b/13b, exports 14-24)")
    w()
    w("This is a canonical technical dossier of the current object state.")
    w("-" * 70)

    # ── 1 ──
    w()
    w("## 1. TITLE AND GENERATION METADATA")
    w()
    w("**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication")
    w("**Dossier Type:** Canonical Object Technical Dossier")
    w(f"**Generated:** {ts}")
    w("**Generator Script:** ade3x3_step13c_update_md_dossier_generator.py")
    w("**Provenance:** Built from steps 1-24, with orbit metadata repair (step 10b)")

    # ── 2 ──
    w()
    w("## 2. MISSION AND NON-GOALS")
    w()
    w("### Mission")
    w("[GROUND_TRUTH]")
    w()
    w("This project is gathering full-object exact data about the ambient bilinear")
    w("universe for 3x3 matrix multiplication structure discovery.")
    w()
    w("The current source of truth is the full object as stored in the warehouse,")
    w("not a reduced model or compressed surrogate.")
    w()
    w("### Non-Goals")
    w("[GROUND_TRUTH]")
    w()
    w("- This dossier is NOT a compression-first document")
    w("- This is NOT a low-rank analysis document")
    w("- This is NOT an optimization target specification")
    w("- Downstream derivations may reduce later, but this preserves fidelity first")

    # ── 3 ──
    w()
    w("## 3. GROUND-TRUTH RAW OBJECT")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Full Raw Base-9 Warehouse")
    w()
    w("The raw warehouse stores all k-tuples over the 9-symbol alphabet {0,...,8}.")
    w("Base-9 arithmetic indexing provides compact storage.")
    w()
    w("| Layer | Tuple Count |")
    w("|-------|-------------|")
    w("| 9^1   | 9           |")
    w("| 9^2   | 81          |")
    w("| 9^3   | 729         |")
    w("| 9^4   | 6,561       |")
    w("| 9^5   | 59,049      |")
    w("| 9^6   | 531,441     |")
    w("| 9^7   | 4,782,969   |")
    w("| 9^8   | 43,046,721  |")
    w("| 9^9   | 387,420,489 |")
    w()
    w("All layers support exact encode/decode, random access, and iteration.")
    w("Role overlay mechanism is ready for typed schema attachment.")

    # ── 4 ──
    w()
    w("## 4. TYPED SEMANTIC OBJECT")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Atomic Species")
    w()
    w("- **A**: Left input basis atoms, |A| = 9, named A[r,s] where r,s in {0,1,2}")
    w("- **B**: Right input basis atoms, |B| = 9, named B[t,u] where t,u in {0,1,2}")
    w("- **C**: Output basis atoms, |C| = 9, named C[r,u] where r,u in {0,1,2}")
    w("- **X**: Ambient product atoms, |X| = 81, named X[r,s|t,u]")
    w()
    w("### Primitive Exact Rules")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("1. **Live/Dead Rule**: X[r,s|t,u] is LIVE if s == t, otherwise DEAD")
    w("2. **Target Map**: Live X maps to C[r,u]")
    w("3. **Fiber Structure**: Each C[r,u] has exactly 3 live X atoms in its fiber")
    w("4. **A/B Participation**: Each X atom has left A index and right B index")

    # ── 5 ──
    w()
    w("## 5. SYMMETRY/ACTION SYSTEM")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("- **Compatible Symmetry Group Size**: 216")
    w("- **Group Type**: S3 x S3 x S3 with compatibility constraint pi_cA == pi_rB")
    w("- **Action Scope**: Acts on A, B, C, and X atoms")
    w("- **Canonicalization**: All core schemas are canonicalized under group action")

    # ── 6 ── (UPDATED: XX bridged, CXXC added)
    w()
    w("## 6. TYPED SCHEMAS CURRENTLY BUILT")
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("| Schema | Typed Arity | Raw Arity | Count | Orbits | Bridged | Notes |")
    w("|--------|-------------|-----------|-------|--------|---------|-------|")
    w("| A      | 1           | 1         | 9     | -      | Yes     | direct embed |")
    w("| B      | 1           | 1         | 9     | -      | Yes     | direct embed |")
    w("| C      | 1           | 1         | 9     | -      | Yes     | direct embed |")
    w("| X      | 1           | 2         | 81    | -      | Yes*    | expands to 2 slots |")
    w("| CC     | 2           | 2         | 81    | 4      | Yes     | direct embed |")
    w("| CX     | 2           | 3         | 729   | 8      | Yes     | X expands |")
    w("| XC     | 2           | 3         | 729   | 8      | Yes     | X expands |")
    w("| AX     | 2           | 3         | 729   | 10     | Yes     | X expands |")
    w("| BX     | 2           | 3         | 729   | 10     | Yes     | X expands |")
    w("| CXC    | 3           | 4         | 6,561 | 50     | Yes     | X expands |")
    w("| XX     | 2           | 4         | 6,561 | 56     | Yes     | X x X, each X expands to 2 slots |")
    w("| CXXC   | 4           | 6         | 531,441 | -    | Yes     | first arity-4 schema |")
    w()
    w("Note: All typed schemas including XX are now bridged into the raw warehouse.")
    w("X bridge marked Yes* because X expands to 2 raw slots (A_idx, B_idx), per Section 7 bridge rules.")
    w("CXXC is the first typed schema beyond the arity-3 core.")

    # ── 7 ──
    w()
    w("## 7. TYPED/RAW BRIDGE")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Bridge Rules")
    w()
    w("- A[r,s] -> raw symbol idx = 3*r + s")
    w("- B[t,u] -> raw symbol idx = 3*t + u")
    w("- C[r,u] -> raw symbol idx = 3*r + u")
    w("- X[r,s|t,u] -> raw tuple (A_idx(r,s), B_idx(t,u))")
    w()
    w("### Bridge Summary")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("| Schema | Typed Arity | Raw Arity | Role Overlay | Injectivity |")
    w("|--------|-------------|-----------|--------------|-------------|")
    w("| C      | 1           | 1         | (C,)         | yes         |")
    w("| CC     | 2           | 2         | (C, C)       | yes         |")
    w("| CX     | 2           | 3         | (C, A_X, B_X)| yes         |")
    w("| XC     | 2           | 3         | (A_X, B_X, C)| yes         |")
    w("| AX     | 2           | 3         | (A, A_X, B_X)| yes         |")
    w("| BX     | 2           | 3         | (B, A_X, B_X)| yes         |")
    w("| CXC    | 3           | 4         | (C, A_X, B_X, C) | yes    |")
    w("| XX     | 2           | 4         | (A_X1, B_X1, A_X2, B_X2) | yes |")
    w("| CXXC   | 4           | 6         | (C, A_X1, B_X1, A_X2, B_X2, C) | yes |")
    w()
    w("[INTERPRETATION]")
    w("Raw arity 3 occupancy: When summing overlay image counts across CX, XC, AX, BX,")
    w("the total is 2916 (729 x 4 schemas). Divided by unique raw tuples (729), this gives")
    w("400%. This is NOT a physical occupancy claim; it reflects aggregate overlay images")
    w("across multiple schemas with different role overlays occupying the same positions.")
    w("See Section 15 for the explicit raw-arity-3 cross-schema alignment export.")

    # ── 8 ──
    w()
    w("## 8. WAREHOUSE INFRASTRUCTURE STATUS")
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("### Current Infrastructure")
    w()
    w("- In-memory object DB with atomic species")
    w("- Raw config bulk-loading for arity-2 and arity-3 core schemas")
    w("- Symmetry-aware canonicalization for all typed schemas")
    w("- Orbit metadata cache for all schemas (repaired, step 10b)")
    w("- Signature caches per orbit (repaired, step 10b)")
    w("- Stabilizer element lists exported for all schemas")
    w("- Full raw base-9 warehouse through arity 9")
    w("- Typed/raw bridge for all schemas including XX and CXXC")
    w("- Orbit-signature export tables")
    w("- Cross-schema raw alignment export")
    w("- Corrected orbit-based composition export")
    w("- First arity-4 typed schema (CXXC) registered and bridged")

    # ── 9 ── (PATCHED: BX, signatures, stabilizer note)
    w()
    w("## 9. EXACT MEASURED COUNTS AND MEMORY")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("### Memory Footprints")
    w()
    w("| Component | Approx Size |")
    w("|-----------|-------------|")
    w("| Full raw warehouse | ~26.8 MB |")
    w("| Object DB base | ~387 KB |")
    w("| Bulk core configs | ~881.5 KB |")
    w("| Bridge tables | ~870.4 KB |")
    w("| Orbit metadata cache | ~8.4 KB |")
    w()
    w("### Orbit Summaries")
    w()
    w("| Schema | Raw Configs | Orbits | Avg Size | Stabilizer Range |")
    w("|--------|-------------|--------|----------|------------------|")
    w("| XX     | 6,561       | 56     | 117.2    | 1-8              |")
    w("| CX     | 729         | 8      | 91.1     | 1-8              |")
    w("| XC     | 729         | 8      | 91.1     | 1-8              |")
    w("| CC     | 81          | 4      | 20.2     | 6-24             |")
    w("| AX     | 729         | 10     | 72.9     | 2-8              |")
    w("| BX     | 729         | 10     | 72.9     | 2-8              |")
    w("| CXC    | 6,561       | 50     | 131.2    | 1-8              |")
    w()
    w("All orbit x stabilizer = 216 verified.")
    w("Stabilizer element sets (not just sizes) are now explicitly exported.")
    w()
    w("### Repaired Signature Summary")
    w()
    w("[MEASURED_FROM_CODE] / [REPAIRED]")
    w()
    w("| Schema | Orbit Count | Distinct Signatures | Orbit Complete |")
    w("|--------|-------------|---------------------|----------------|")
    for sch in ['XX','CX','XC','CC','AX','BX','CXC']:
        oc = ORBIT_COUNTS[sch]
        sc = SIG_COUNTS[sch]
        comp = "yes" if sch in ORBIT_COMPLETE else "no"
        w(f"| {sch:<6} | {oc:>11} | {sc:>19} | {comp:>14} |")

    # ── 10 ── (PATCHED: composition facts)
    w()
    w("## 10. EXACT DERIVED RESULTS CURRENTLY KNOWN")
    w()
    w("[EXACT_DERIVED]")
    w()
    w("### Dead-Dead Pair Classification")
    w()
    w("- 24 orbits classified exactly")
    w("- Classification by: row equality, column equality, one of 6 shared-index patterns")
    w("- Full resolution with refined signature tuple (row_eq, col_eq, shared_pattern)")
    w()
    w("### CX x XC -> CC Composition Result (Corrected Orbit-Based)")
    w()
    w("[EXACT_DERIVED] / [REPAIRED]")
    w()
    w("- CX orbits = 8, XC orbits = 8, CC orbits = 4")
    w("- Possible orbit-pairs = 64")
    w("- Realized keys = 32")
    w("- Deterministic = 18")
    w("- Mixed = 14")
    w()
    w("An older type-based layer reported 256/64/36/28. Those counts used a different")
    w("type system and are now superseded by the corrected orbit-based counts above.")
    w()
    w("### Refinement Engine Result")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)")
    w("- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target")
    w("- Best 2-tuple: [r1_eq_r, u1_eq_u]")
    w()
    w("### Orbit-Complete Schemas (Repaired)")
    w()
    w("[EXACT_DERIVED] / [REPAIRED]")
    w()
    w("- CX: 8 orbits, 8 distinct signatures")
    w("- XC: 8 orbits, 8 distinct signatures")
    w("- CC: 4 orbits, 4 distinct signatures")
    w("- CXC: 50 orbits, 50 distinct signatures")
    w()
    w("### Schemas with Coarse Signatures (Repaired)")
    w()
    w("[EXACT_DERIVED] / [REPAIRED]")
    w()
    w("- XX: 56 orbits, 48 distinct signatures")
    w("- AX: 10 orbits, 8 distinct signatures")
    w("- BX: 10 orbits, 8 distinct signatures")
    w()
    w("### Cross-Schema Raw Arity-3 Alignment")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("- Schemas CX, XC, AX, BX all bridge to raw arity 3")
    w("- Each image size = 729")
    w("- Distinct raw tuples occupied = 729")
    w("- Aggregate image count = 2,916")
    w("- Occupancy ratio = 4.0 (all four images coincide exactly)")

    # ── 11 ──
    w()
    w("## 11. OBJECT VS LENS DISTINCTION")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### The Object (Ground Truth)")
    w()
    w("- Full raw base-9 warehouse (all 9^k tuples)")
    w("- Typed species A, B, C, X")
    w("- Typed schema definitions including first arity-4 schema CXXC")
    w("- Primitive exact rules (live/dead, fibers)")
    w("- Typed/raw bridge embeddings for all schemas")
    w()
    w("### Lens/Infrastructure (Not the Object)")
    w()
    w("- Orbit metadata caches")
    w("- Signature caches")
    w("- Stabilizer element lists")
    w("- Canonicalization tables")
    w("- Projection metadata")
    w("- Composition caches")
    w("- Cross-schema alignment tables")
    w()
    w("**Critical:** Downstream agents must distinguish object facts from lens facts.")
    w("Lens updates do not change the underlying object.")

    # ── 12 ──
    w()
    w("## 12. PROVENANCE / EVIDENCE LABELS")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("Evidence taxonomy used in this document:")
    w()
    w("- [GROUND_TRUTH] - Core definition, object structure")
    w("- [EXACT_DERIVED] - Results derived from ground truth by exact computation")
    w("- [MEASURED_FROM_CODE] - Measured numeric outputs from executed code")
    w("- [REPAIRED] - Corrected after bug fix in orbit metadata cache")
    w("- [INTERPRETATION] - Analysis or interpretation, not ground truth")
    w("- [OPEN_FRONT] - Known gaps or incomplete areas")

    # ── 13 ── (UPDATED: CXXC gap revised)
    w()
    w("## 13. CURRENT GAPS / OPEN FRONTS")
    w()
    w("[OPEN_FRONT]")
    w()
    w("- First arity-4 typed schema exists (CXXC); broader higher-arity typed schema")
    w("  family expansion remains open")
    w("- Full orbit-complete signatures not yet known for XX, AX, BX schemas")
    w("- Unified raw-backed composition caches not yet built")
    w("- Closure/refinement not yet globally rerun inside unified raw-backed warehouse")
    w()
    w("These are genuine incompletions, not promises.")

    # ── 14 ──
    w()
    w("## 14. INSTRUCTIONS FOR DOWNSTREAM AGENTS")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Usage Guidelines")
    w()
    w("1. **Do NOT replace the full object with a reduced surrogate**")
    w("2. **Distinguish object facts from lens facts** in any derivation")
    w("3. **Cite exact sections/facts** used in any derivation")
    w("4. **If proposing compression** or quotients, state explicitly what raw distinctions are being discarded")
    w("5. **Prefer additions** that preserve fidelity unless reduction is explicitly the task")
    w()
    w("### Basis for Reasoning")
    w()
    w("This dossier provides the canonical basis. Any downstream derivation should")
    w("explicitly reference which sections are being used and which are being assumed.")

    # ── 15 ── (NEW: cross-schema alignment section)
    w()
    w("## 15. CROSS-SCHEMA RAW ALIGNMENT")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("### Raw Arity-3 Alignment (CX, XC, AX, BX)")
    w()
    w("All four schemas bridge to raw arity 3 under different role overlays:")
    w()
    w("| Schema | Role Overlay |")
    w("|--------|--------------|")
    w("| CX | (C, A_X, B_X) |")
    w("| XC | (A_X, B_X, C) |")
    w("| AX | (A, A_X, B_X) |")
    w("| BX | (B, A_X, B_X) |")
    w()
    w("Alignment facts:")
    w()
    w("- Each schema image size = 729")
    w("- Distinct raw tuples occupied = 729")
    w("- Aggregate image count = 2,916")
    w("- Occupancy ratio = 4.0")
    w("- All four schema images coincide exactly on raw tuple support")
    w()
    w("Supporting artifacts: exports/raw3_alignment.csv, exports/raw3_alignment.md")

    # ── 16 ── (from 13b repair note, preserved)
    w()
    w("## 16. CORRECTION NOTE: ORBIT METADATA REPAIR")
    w()
    w("[REPAIRED]")
    w()
    w("### Bug")
    w()
    w("The original step10 orbit metadata cache keyed records by orbit_id and")
    w("implicitly used orbit_id as a config_id index into the configs list.")
    w("Since orbit_id != canonical representative config_id for most orbits,")
    w("every signature was computed from the wrong representative config.")
    w()
    w("### Impact")
    w()
    w("Old cached signature counts were partly wrong:")
    w()
    w("| Schema | Old (buggy) | Repaired | Changed |")
    w("|--------|-------------|----------|---------|")
    w("| XX     | 20          | 48       | yes     |")
    w("| CX     | 4           | 8        | yes     |")
    w("| XC     | 4           | 8        | yes     |")
    w("| CC     | 3           | 4        | yes     |")
    w("| AX     | 3           | 8        | yes     |")
    w("| BX     | 8           | 8        | no      |")
    w("| CXC    | 50          | 50       | no      |")
    w()
    w("Additionally, the earlier BX orbit count of 18 was wrong due to a")
    w("separate bug (identity B-action). The corrected BX orbit count is 10.")
    w()
    w("### Resolution")
    w()
    w("- Repaired orbit metadata now uses actual canonical orbit representatives")
    w("- Signatures are computed from the correct representative config")
    w("- See exports/orbit_metadata_repair_report.md for full details")
    w("- Old cached signature counts are superseded")

    # ── 17 ── (UPDATED: full artifact inventory)
    w()
    w("## 17. SUPPORTING EXPORT ARTIFACTS")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("The following raw evidence tables have been exported and are available")
    w("as supporting artifacts alongside this dossier:")
    w()
    for group, files in EXPORT_ARTIFACTS.items():
        w(f"**{group}:**")
        for f in files:
            w(f"- `{f}`")
        w()

    # ── 18 ── (NEW: first arity-4 schema)
    w()
    w("## 18. FIRST ARITY-4 TYPED SCHEMA: CXXC")
    w()
    w("[MEASURED_FROM_CODE]")
    w()
    w("Typed schema growth beyond the arity-3 core has begun.")
    w()
    w("| Property | Value |")
    w("|----------|-------|")
    w("| Schema | C × X × X × C |")
    w("| Typed arity | 4 |")
    w("| Raw arity | 6 |")
    w("| Typed config count | 531,441 |")
    w("| Bridge export | full population |")
    w("| Injective | yes |")
    w("| Role overlay | (C, A_X1, B_X1, A_X2, B_X2, C) |")
    w()
    w("Supporting artifacts: exports/schema_CXXC_summary.md, exports/bridge_CXXC.csv")

    # ── End ──
    w()
    w("-" * 70)
    w("END OF DOSSIER")
    w("-" * 70)

    return "\n".join(L)


def main():
    output_path = Path("ADE3x3_CANONICAL_OBJECT.md")
    content = generate()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    n_sections = content.count("\n## ")
    print("DOSSIER UPDATE COMPLETE")
    print(f"Output: {output_path.absolute()}")
    print(f"Sections: {n_sections}")
    print(f"File size: {len(content)} bytes")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Updates applied:")
    print("  - XX bridge added (no longer 'not bridged')")
    print("  - Stabilizer element exports referenced")
    print("  - Raw3 alignment section added (Section 15)")
    print("  - Corrected orbit-based composition facts (Section 10)")
    print("  - CXXC first arity-4 schema added (Section 18)")
    print("  - Open fronts updated for CXXC")
    print("  - Full artifact inventory expanded (Section 17)")
    print("  - Corrected orbit/signature facts preserved")


if __name__ == "__main__":
    main()
