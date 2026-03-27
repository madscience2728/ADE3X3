import csv
import os
from pathlib import Path
from datetime import datetime

def generate_x_table():
    atoms = []
    
    r_idx = 0
    for r in range(3):
        for s in range(3):
            for t in range(3):
                for u in range(3):
                    a_idx = 3 * r + s
                    b_idx = 3 * t + u
                    c_idx = 3 * r + u
                    
                    live = (s == t)
                    
                    a_name = f"A[{r},{s}]"
                    b_name = f"B[{t},{u}]"
                    x_name = f"X[{r},{s}|{t},{u}]"
                    
                    if live:
                        targ_c_idx = c_idx
                        targ_c_name = f"C[{r},{u}]"
                    else:
                        targ_c_idx = -1
                        targ_c_name = ""
                    
                    atoms.append({
                        'x_local_id': r_idx,
                        'x_name': x_name,
                        'r': r,
                        's': s,
                        't': t,
                        'u': u,
                        'a_local_id': a_idx,
                        'a_name': a_name,
                        'b_local_id': b_idx,
                        'b_name': b_name,
                        'live': 1 if live else 0,
                        'target_c_local_id': targ_c_idx,
                        'target_c_name': targ_c_name,
                    })
                    r_idx += 1
    
    return atoms

def write_csv(atoms, path):
    fieldnames = [
        'x_local_id', 'x_name', 'r', 's', 't', 'u',
        'a_local_id', 'a_name', 'b_local_id', 'b_name',
        'live', 'target_c_local_id', 'target_c_name'
    ]
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(atoms)

def write_markdown(atoms, path):
    live_count = sum(1 for a in atoms if a['live'] == 1)
    dead_count = len(atoms) - live_count
    
    lines = []
    lines.append("# X Atoms Table")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This is the explicit object-level table for all 81 X atoms.")
    lines.append("")
    lines.append("## Summary Counts")
    lines.append("")
    lines.append(f"- Total X atoms: 81")
    lines.append(f"- Live atoms: 27")
    lines.append(f"- Dead atoms: 54")
    lines.append("")
    lines.append("## First 10 Rows (Preview)")
    lines.append("")
    lines.append("| x_local_id | x_name | r | s | t | u | a_name | b_name | live | target_c |")
    lines.append("|------------|--------|---|---|---|---|--------|--------|------|----------|")
    
    for a in atoms[:10]:
        target = a['target_c_name'] if a['live'] else "-"
        lines.append(f"| {a['x_local_id']} | {a['x_name']} | {a['r']} | {a['s']} | {a['t']} | {a['u']} | {a['a_name']} | {a['b_name']} | {a['live']} | {target} |")
    
    lines.append("")
    lines.append("See `X_atoms.csv` for the complete machine-readable table.")
    lines.append("")
    lines.append("-" * 50)
    
    content = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def sanity_checks(atoms):
    print("\n--- SANITY CHECKS ---")
    
    total = len(atoms)
    print(f"Total atoms: {total}")
    assert total == 81, f"Expected 81, got {total}"
    
    live = sum(1 for a in atoms if a['live'] == 1)
    dead = sum(1 for a in atoms if a['live'] == 0)
    print(f"Live: {live}, Dead: {dead}")
    assert live == 27, f"Expected 27 live, got {live}"
    assert dead == 54, f"Expected 54 dead, got {dead}"
    
    for a in atoms:
        s, t = a['s'], a['t']
        if a['live'] == 1:
            assert s == t, f"Live atom {a['x_name']} has s={s}, t={t}"
        else:
            assert s != t, f"Dead atom {a['x_name']} has s={s}, t={t}"
    
    print("All live atoms satisfy s == t: OK")
    print("All dead atoms satisfy s != t: OK")
    
    for a in atoms:
        if a['live'] == 1:
            r, u = a['r'], a['u']
            expected = 3 * r + u
            assert a['target_c_local_id'] == expected, f"Target mismatch for {a['x_name']}"
    
    print("All live targets match formula target_c = 3*r + u: OK")
    print("\nAll sanity checks passed!")

def main():
    print("Generating X atoms table...")
    
    os.makedirs('exports', exist_ok=True)
    
    atoms = generate_x_table()
    
    csv_path = Path('exports/X_atoms.csv')
    md_path = Path('exports/X_atoms.md')
    
    write_csv(atoms, csv_path)
    print(f"CSV written: {csv_path}")
    
    write_markdown(atoms, md_path)
    print(f"Markdown written: {md_path}")
    
    sanity_checks(atoms)
    
    live_count = sum(1 for a in atoms if a['live'] == 1)
    dead_count = len(atoms) - live_count
    
    print(f"\n--- SUMMARY ---")
    print(f"Output paths:")
    print(f"  - {csv_path.absolute()}")
    print(f"  - {md_path.absolute()}")
    print(f"Row count: 81")
    print(f"Live: {live_count}, Dead: {dead_count}")

if __name__ == "__main__":
    main()