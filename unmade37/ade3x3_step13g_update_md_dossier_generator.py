from pathlib import Path
from datetime import datetime

ROOT = Path('/mnt/data/adeproj/ADE3X3')
DOC = ROOT / 'ADE3x3_CANONICAL_OBJECT.md'


def replace_once(text, old, new):
    if old not in text:
        return text
    return text.replace(old, new, 1)


def insert_after(text, marker, insert):
    if insert.strip() in text:
        return text
    if marker not in text:
        return text
    return text.replace(marker, marker + insert, 1)


def main():
    text = DOC.read_text(encoding='utf-8')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # metadata
    text = replace_once(text, 'Generator: ade3x3_step13f_update_md_dossier_generator.py',
                        'Generator: ade3x3_step13g_update_md_dossier_generator.py')
    text = replace_once(text, '**Generator Script:** ade3x3_step13f_update_md_dossier_generator.py',
                        '**Generator Script:** ade3x3_step13g_update_md_dossier_generator.py')
    # update top generated lines conservatively
    lines = text.splitlines()
    for i, line in enumerate(lines[:20]):
        if line.startswith('Generated: '):
            lines[i] = f'Generated: {now}'
        if line.startswith('**Generated:** '):
            lines[i] = f'**Generated:** {now}'
    text = '\n'.join(lines)

    # Infrastructure bullets
    text = replace_once(
        text,
        '- First two arity-4 typed schemas (CXXC, AXXC) registered and bridged\n- CXXC face inventory, face-pattern summary, and joint interior co-occurrence export\n- CXXC marginals conditioned on XX and both CXC faces\n- AXXC face inventory export\n',
        '- First two arity-4 typed schemas (CXXC, AXXC) registered and bridged\n- CXXC face inventory, face-pattern summary, middle-XX marginals, left/right CXC marginals, and joint interior co-occurrence export\n- AXXC face inventory, face-pattern summary, middle-XX marginals, and joint interior co-occurrence export\n- Raw arity-6 alignment export for CXXC and AXXC\n'
    )

    # Refinement engine wording as tested subset
    text = replace_once(
        text,
        '### Refinement Engine Result\n\n[MEASURED_FROM_CODE]\n\n- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)\n- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target\n- Best 2-tuple: [r1_eq_r, u1_eq_u]\n',
        '### Refinement Engine Result\n\n[MEASURED_FROM_CODE]\n\nThe following separator results come from a tested subset of simple coordinate-alignment predicates on the older composition layer; they are not claimed to exhaust all possible separator families.\n\n- Single separator r1_eq_r resolves ALL 28 mixed composition keys (older layer)\n- Also resolved by u1_eq_u, x_live, c1_equals_target, c2_equals_target\n- Best 2-tuple within the tested subset: [r1_eq_r, u1_eq_u]\n'
    )

    # Arity-4 inventories section replacement/expansion
    old_block = '''### Arity-4 Marginal Inventories\n\n[MEASURED_FROM_CODE]\n\n- CXXC conditioned on middle XX orbit: 56 rows, counts 2,187–17,496, distinct face-patterns 25–81\n- CXXC conditioned on left CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81\n- CXXC conditioned on right CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81\n- AXXC face inventory exported: full population 531,441 rows with faces AX, XX, XC, AC, left AXC, right AXC\n'''
    new_block = '''### Arity-4 Interaction Inventories\n\n[MEASURED_FROM_CODE]\n\n- CXXC face inventory exported: full population 531,441 rows with faces CX, XX, XC, CC, left CXC, right CXC\n- CXXC distinct face-patterns: 2,744, with multiplicities 27–216\n- CXXC conditioned on middle XX orbit: 56 rows, counts 2,187–17,496, distinct face-patterns 25–81\n- CXXC conditioned on left CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81\n- CXXC conditioned on right CXC orbit: 50 rows, counts 2,187–17,496, distinct face-patterns 20–81\n- CXXC joint interior table (left CXC, middle XX, right CXC): 2,744 distinct joint keys, exactly matching the 2,744 full face-patterns\n- AXXC face inventory exported: full population 531,441 rows with faces AX, XX, XC, AC, left AXC, right AXC\n- AXXC distinct face-patterns: 2,870, with multiplicities 27–216\n- AXXC conditioned on middle XX orbit: 56 rows, counts 2,187–17,496, distinct face-patterns 20–81\n- AXXC joint interior table (left AXC, middle XX, right AXC): 2,870 distinct joint keys, exactly matching the 2,870 full face-patterns\n'''
    text = replace_once(text, old_block, new_block)

    # object vs lens clarification schemas treated as built object structure once bridged
    text = replace_once(
        text,
        '- Typed schema definitions including first arity-4 schema CXXC\n',
        '- Typed schema definitions currently instantiated and bridged in the warehouse (including CXXC and AXXC)\n'
    )
    text = insert_after(
        text,
        '**Critical:** Object facts and lens facts must be kept distinct.\nA lens update does not change the underlying object.\n',
        '\nIn this dossier, once a typed schema has been explicitly registered and injectively bridged into the warehouse, it is treated as part of the built object structure. Orbit tables, signature caches, marginal summaries, and alignment summaries remain organizational lenses over that built structure.\n'
    )

    # Open fronts explicit arity-4 parity note
    text = replace_once(
        text,
        '- First two arity-4 typed schemas exist (CXXC, AXXC); broader higher-arity typed schema\n  family expansion remains open\n- Full orbit-complete signatures not yet known for XX, AX, BX schemas\n',
        '- First two arity-4 typed schemas exist (CXXC, AXXC); broader higher-arity typed schema\n  family expansion remains open\n- Arity-4 orbit/signature/stabilizer parity has not yet been established for CXXC or AXXC\n- Full orbit-complete signatures not yet known for XX, AX, BX schemas\n'
    )

    # How to read - soften reduced view wording maybe already okay; add note about historical superseded counts?
    text = replace_once(
        text,
        '4. **Any reduced view must state what it omits.** If a future document\n   presents a simplified picture of the object, it must say explicitly\n   which raw distinctions have been dropped.\n',
        '4. **Any partial or reorganized view must state what it omits or re-indexes.** If a future document\n   presents a simplified or reorganized picture of the object, it must say explicitly\n   which raw distinctions have been dropped, merged, or re-keyed.\n'
    )

    # Add raw arity-6 alignment section under section 15
    marker15 = 'Supporting artifacts: exports/raw3_alignment.csv, exports/raw3_alignment.md\n'
    add15 = '''\n\n### Raw Arity-6 Alignment (CXXC, AXXC)\n\n[MEASURED_FROM_CODE]\n\nBoth arity-4 schemas bridge to raw arity 6:\n\n| Schema | Role Overlay | Image Size |\n|--------|--------------|-----------:|\n| CXXC | (C, A_X1, B_X1, A_X2, B_X2, C) | 531,441 |\n| AXXC | (A, A_X1, B_X1, A_X2, B_X2, C) | 531,441 |\n\nAlignment facts:\n\n- Distinct raw arity-6 tuples occupied by CXXC = 531,441\n- Distinct raw arity-6 tuples occupied by AXXC = 531,441\n- Overlap size = 531,441\n- The two schema images coincide exactly on raw arity-6 support\n\nSupporting artifacts: exports/raw6_alignment_CXXC_AXXC.csv, exports/raw6_alignment_CXXC_AXXC.md\n'''
    text = insert_after(text, marker15, add15)

    # update artifacts list with raw6 and AXXC pattern files
    marker_art = "**Cross-Schema Alignment:**\n- `exports/raw3_alignment.csv`\n- `exports/raw3_alignment.md`\n"
    repl_art = "**Cross-Schema Alignment:**\n- `exports/raw3_alignment.csv`\n- `exports/raw3_alignment.md`\n- `exports/raw6_alignment_CXXC_AXXC.csv`\n- `exports/raw6_alignment_CXXC_AXXC.md`\n"
    text = replace_once(text, marker_art, repl_art)
    marker_higher = "**Higher-Arity Schema Exports:**\n- `exports/schema_CXXC_summary.md`\n- `exports/bridge_CXXC.csv`\n- `exports/schema_AXXC_summary.md`\n- `exports/bridge_AXXC.csv`\n\n**Arity-4 Interaction Inventories:**\n- `exports/CXXC_face_inventory.csv`\n"
    repl_higher = "**Higher-Arity Schema Exports:**\n- `exports/schema_CXXC_summary.md`\n- `exports/bridge_CXXC.csv`\n- `exports/schema_AXXC_summary.md`\n- `exports/bridge_AXXC.csv`\n\n**Arity-4 Interaction Inventories:**\n- `exports/CXXC_face_inventory.csv`\n"
    text = replace_once(text, marker_higher, repl_higher)
    # append AXXC artifacts in existing block
    text = replace_once(
        text,
        '- `exports/AXXC_face_inventory.csv`\n- `exports/AXXC_face_inventory.md`\n',
        '- `exports/AXXC_face_inventory.csv`\n- `exports/AXXC_face_inventory.md`\n- `exports/AXXC_face_patterns.csv`\n- `exports/AXXC_face_patterns.md`\n- `exports/AXXC_xx_marginals.csv`\n- `exports/AXXC_xx_marginals.md`\n- `exports/AXXC_joint_AXC_XX_AXC.csv`\n- `exports/AXXC_joint_AXC_XX_AXC.md`\n'
    )

    # replace section 18 with combined arity-4 section
    import re
    pattern = r"## 18\. FIRST ARITY-4 TYPED SCHEMA: CXXC.*?Supporting artifacts: exports/schema_CXXC_summary\.md, exports/bridge_CXXC\.csv\n"
    replacement = '''## 18. ARITY-4 TYPED SCHEMAS: CXXC AND AXXC\n\n[MEASURED_FROM_CODE]\n\nThe current warehouse contains two fully bridged arity-4 schemas.\n\n| Property | CXXC | AXXC |\n|----------|------|------|\n| Schema | C × X × X × C | A × X × X × C |\n| Typed arity | 4 | 4 |\n| Raw arity | 6 | 6 |\n| Typed config count | 531,441 | 531,441 |\n| Bridge export | full population | full population |\n| Injective | yes | yes |\n| Role overlay | (C, A_X1, B_X1, A_X2, B_X2, C) | (A, A_X1, B_X1, A_X2, B_X2, C) |\n| Distinct face-patterns | 2,744 | 2,870 |\n| Joint interior keys | 2,744 (CXC–XX–CXC) | 2,870 (AXC–XX–AXC) |\n\nSupporting artifacts:\n- CXXC: exports/schema_CXXC_summary.md, exports/bridge_CXXC.csv, exports/CXXC_face_inventory.csv, exports/CXXC_face_patterns.csv, exports/CXXC_xx_marginals.csv, exports/CXXC_joint_CXC_XX_CXC.csv\n- AXXC: exports/schema_AXXC_summary.md, exports/bridge_AXXC.csv, exports/AXXC_face_inventory.csv, exports/AXXC_face_patterns.csv, exports/AXXC_xx_marginals.csv, exports/AXXC_joint_AXC_XX_AXC.csv\n'''
    text = re.sub(pattern, replacement, text, flags=re.S)

    DOC.write_text(text, encoding='utf-8')
    print(f'Wrote {DOC}')
    print(f'Generated: {now}')
    print('Added raw arity-6 alignment, AXXC parity facts, and provenance/lens cleanup.')
    print(f'File size: {DOC.stat().st_size} bytes')

if __name__ == '__main__':
    main()
