from pathlib import Path
from datetime import datetime
import re

ROOT = Path('/mnt/data/adeproj/ADE3X3')
DOC = ROOT / 'ADE3x3_CANONICAL_OBJECT.md'


def replace_once(text: str, old: str, new: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    return text


def insert_after(text: str, marker: str, insert: str) -> str:
    if insert.strip() in text:
        return text
    if marker not in text:
        return text
    return text.replace(marker, marker + insert, 1)


def main() -> None:
    text = DOC.read_text(encoding='utf-8')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    text = replace_once(text, 'Generator: ade3x3_step13g_update_md_dossier_generator.py',
                        'Generator: ade3x3_step13h_update_md_dossier_generator.py')
    text = replace_once(text, '**Generator Script:** ade3x3_step13g_update_md_dossier_generator.py',
                        '**Generator Script:** ade3x3_step13h_update_md_dossier_generator.py')
    lines = text.splitlines()
    for i, line in enumerate(lines[:20]):
        if line.startswith('Generated: '):
            lines[i] = f'Generated: {now}'
        if line.startswith('**Generated:** '):
            lines[i] = f'**Generated:** {now}'
    text = '\n'.join(lines)

    # Update schema table orbits for arity-4 schemas
    text = text.replace('| CXXC   | 4           | 6         | 531,441 | -      | Yes     | first arity-4 schema |',
                        '| CXXC   | 4           | 6         | 531,441 | 2,744  | Yes     | first arity-4 schema |')
    text = text.replace('| AXXC   | 4           | 6         | 531,441 | -      | Yes     | second arity-4 schema |',
                        '| AXXC   | 4           | 6         | 531,441 | 2,870  | Yes     | second arity-4 schema |')

    # Infrastructure bullets
    text = replace_once(
        text,
        '- Raw arity-6 alignment export for CXXC and AXXC\n',
        '- Raw arity-6 alignment export for CXXC and AXXC\n- Arity-4 orbit/signature/stabilizer parity export for CXXC and AXXC\n'
    )

    # Orbit summaries add arity-4 rows
    old_orbit_table = '''| XX     | 6,561       | 56     | 117.2    | 1-8              |
| CX     | 729         | 8      | 91.1     | 1-8              |
| XC     | 729         | 8      | 91.1     | 1-8              |
| CC     | 81          | 4      | 20.2     | 6-24             |
| AX     | 729         | 10     | 72.9     | 2-8              |
| BX     | 729         | 10     | 72.9     | 2-8              |
| CXC    | 6,561       | 50     | 131.2    | 1-8              |'''
    new_orbit_table = '''| XX     | 6,561       | 56     | 117.2    | 1-8              |
| CX     | 729         | 8      | 91.1     | 1-8              |
| XC     | 729         | 8      | 91.1     | 1-8              |
| CC     | 81          | 4      | 20.2     | 6-24             |
| AX     | 729         | 10     | 72.9     | 2-8              |
| BX     | 729         | 10     | 72.9     | 2-8              |
| CXC    | 6,561       | 50     | 131.2    | 1-8              |
| CXXC   | 531,441     | 2,744  | 193.7    | 1-8              |
| AXXC   | 531,441     | 2,870  | 185.2    | 1-8              |'''
    text = replace_once(text, old_orbit_table, new_orbit_table)

    # Repaired signature summary add arity-4 rows
    old_sig_table = '''| XX     |          56 |                  48 |             no |
| CX     |           8 |                   8 |            yes |
| XC     |           8 |                   8 |            yes |
| CC     |           4 |                   4 |            yes |
| AX     |          10 |                   8 |             no |
| BX     |          10 |                   8 |             no |
| CXC    |          50 |                  50 |            yes |'''
    new_sig_table = '''| XX     |          56 |                  48 |             no |
| CX     |           8 |                   8 |            yes |
| XC     |           8 |                   8 |            yes |
| CC     |           4 |                   4 |            yes |
| AX     |          10 |                   8 |             no |
| BX     |          10 |                   8 |             no |
| CXC    |          50 |                  50 |            yes |
| CXXC   |       2,744 |               2,744 |            yes |
| AXXC   |       2,870 |               2,870 |            yes |'''
    text = replace_once(text, old_sig_table, new_sig_table)

    # Add explicit arity-4 parity subsection after arity-4 interaction inventories
    marker = '- AXXC joint interior table (left AXC, middle XX, right AXC): 2,870 distinct joint keys, exactly matching the 2,870 full face-patterns\n'
    insert = '\n### Arity-4 Orbit / Signature / Stabilizer Parity\n\n[MEASURED_FROM_CODE]\n\n- CXXC: 2,744 orbits, 2,744 distinct signatures, orbit-complete at the current arity-4 signature layer\n- AXXC: 2,870 orbits, 2,870 distinct signatures, orbit-complete at the current arity-4 signature layer\n- Orbit size range for both schemas: 27–216\n- Stabilizer size range for both schemas: 1–8\n'
    text = insert_after(text, marker, insert)

    # Update object section mention
    text = replace_once(text,
        '- Typed schema definitions currently instantiated and bridged in the warehouse (including CXXC and AXXC)\n',
        '- Typed schema definitions currently instantiated and bridged in the warehouse (including CXXC and AXXC, with arity-4 orbit/signature/stabilizer parity now recorded)\n')

    # Open fronts: remove arity-4 parity open item
    text = replace_once(text,
        '- Arity-4 orbit/signature/stabilizer parity has not yet been established for CXXC or AXXC\n',
        '')

    # Supporting artifacts add arity-4 parity files
    text = replace_once(text,
        '- `exports/raw6_alignment_CXXC_AXXC.csv`\n- `exports/raw6_alignment_CXXC_AXXC.md`\n',
        '- `exports/raw6_alignment_CXXC_AXXC.csv`\n- `exports/raw6_alignment_CXXC_AXXC.md`\n- `exports/orbits_CXXC.csv`\n- `exports/orbits_AXXC.csv`\n- `exports/arity4_orbit_signature_stabilizer_summary.md`\n')

    # Section 18 table add orbit counts and parity
    old18 = '''| Property | CXXC | AXXC |
|----------|------|------|
| Schema | C × X × X × C | A × X × X × C |
| Typed arity | 4 | 4 |
| Raw arity | 6 | 6 |
| Typed config count | 531,441 | 531,441 |
| Bridge export | full population | full population |
| Injective | yes | yes |
| Role overlay | (C, A_X1, B_X1, A_X2, B_X2, C) | (A, A_X1, B_X1, A_X2, B_X2, C) |
| Distinct face-patterns | 2,744 | 2,870 |
| Joint interior keys | 2,744 (CXC–XX–CXC) | 2,870 (AXC–XX–AXC) |'''
    new18 = '''| Property | CXXC | AXXC |
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
| Joint interior keys | 2,744 (CXC–XX–CXC) | 2,870 (AXC–XX–AXC) |'''
    text = replace_once(text, old18, new18)

    DOC.write_text(text, encoding='utf-8')
    print(f'Wrote {DOC}')
    print(f'Generated: {now}')
    print('Added arity-4 orbit/signature/stabilizer parity to canon.')
    print(f'File size: {DOC.stat().st_size} bytes')


if __name__ == '__main__':
    main()
