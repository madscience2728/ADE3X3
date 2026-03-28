from pathlib import Path
from datetime import datetime
import csv

ROOT = Path('/mnt/data/adeproj/ADE3X3')
DOC = ROOT / 'ADE3x3_CANONICAL_OBJECT.md'
EXPORTS = ROOT / 'exports'


def abc_table(label):
    hdr = [f"| {label}_local_id | {label}_name | row | col | raw_idx |", "|---:|---|---:|---:|---:|"]
    rows = []
    for i in range(9):
        r, c = divmod(i, 3)
        rows.append(f"| {i} | {label}[{r},{c}] | {r} | {c} | {i} |")
    return "\n".join(hdr + rows)


def x_rows():
    rows = []
    x_id = 0
    for r in range(3):
        for s in range(3):
            for t in range(3):
                for u in range(3):
                    live = int(s == t)
                    target = f"C[{r},{u}]" if live else "-"
                    rows.append(f"| {x_id} | X[{r},{s}|{t},{u}] | {r} | {s} | {t} | {u} | {live} | {3*r+s} | {3*t+u} | {target} |")
                    x_id += 1
    return rows


def build_x_table():
    hdr = ["| x_local_id | x_name | r | s | t | u | live | a_idx | b_idx | target_c |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    return "\n".join(hdr + x_rows())


def build_c_fiber_table():
    path = EXPORTS / 'C_fibers.csv'
    hdr = ["| c_local_id | c_name | x0_name | x1_name | x2_name |", "|---:|---|---|---|---|"]
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(f"| {row['c_local_id']} | {row['c_name']} | {row['x0_name']} | {row['x1_name']} | {row['x2_name']} |")
    return "\n".join(hdr + rows)


def build_comp_grid():
    comp = {}
    with (EXPORTS / 'comp_CX_XC_to_CC.csv').open() as f:
        for row in csv.DictReader(f):
            comp[(int(row['cx_orbit_id']), int(row['xc_orbit_id']))] = row
    hdr = ["| CX_orbit | XC_orbit | realized | cc_orbit_ids | deterministic | witness_count |", "|---:|---:|---:|---|---:|---:|"]
    rows = []
    for cx in range(8):
        for xc in range(8):
            row = comp.get((cx, xc))
            if row is None:
                rows.append(f"| {cx} | {xc} | 0 | [] | 0 | 0 |")
            else:
                rows.append(f"| {cx} | {xc} | 1 | {row['cc_orbit_ids']} | {row['deterministic']} | {row['witness_count']} |")
    return "\n".join(hdr + rows)


def insert_once(text, marker, insert):
    if insert.strip() in text:
        return text
    return text.replace(marker, marker + insert, 1)


def main():
    text = DOC.read_text()
    gen = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    primitive_insert = """

### Atomic Index Tables

[GROUND_TRUTH]

#### A index table

%s

#### B index table

%s

#### C index table

%s

### C Fiber Table

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

%s

### X Atom Table (Inline)

[GROUND_TRUTH] / [MEASURED_FROM_CODE]

%s
""" % (abc_table('A'), abc_table('B'), abc_table('C'), build_c_fiber_table(), build_x_table())

    comp_insert = """

### Full Corrected 8×8 Orbit-Based Composition Grid

[MEASURED_FROM_CODE] / [REPAIRED]

%s
""" % build_comp_grid()

    marker1 = "4. **A/B Participation**: Each X atom has left A index and right B index\n"
    text = insert_once(text, marker1, primitive_insert)

    marker2 = "An older type-based layer reported 256/64/36/28. Those counts used a different\ntype system and are now superseded by the corrected orbit-based counts above.\n"
    text = insert_once(text, marker2, comp_insert)

    # refresh top metadata lines if present
    text = text.replace("Generator: ade3x3_step13d_debias_md_dossier_generator.py", "Generator: ade3x3_step13f_update_md_dossier_generator.py")
    text = text.replace("Generator Script:** ade3x3_step13d_debias_md_dossier_generator.py", "Generator Script:** ade3x3_step13f_update_md_dossier_generator.py")
    text = text.replace("Generated: 2026-03-27 18:29:56", f"Generated: {gen}")
    text = text.replace("**Generated:** 2026-03-27 18:29:56", f"**Generated:** {gen}")

    DOC.write_text(text)
    print(f'Wrote {DOC}')
    print(f'Generated: {gen}')
    print('Added inline primitive tables and full corrected 8x8 composition grid.')
    print(f'File size: {DOC.stat().st_size} bytes')


if __name__ == '__main__':
    main()
