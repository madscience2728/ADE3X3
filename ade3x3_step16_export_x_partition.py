import csv
import os
from pathlib import Path
from datetime import datetime

def generate_x_atoms():
    atoms = []
    for r in range(3):
        for s in range(3):
            for t in range(3):
                for u in range(3):
                    x_idx = len(atoms)
                    a_idx = 3 * r + s
                    b_idx = 3 * t + u
                    live = (s == t)
                    target_c = -1
                    target_c_name = ""
                    if live:
                        target_c = 3 * r + u
                        target_c_name = f"C[{r},{u}]"
                    atoms.append({
                        'x_local_id': x_idx,
                        'x_name': f"X[{r},{s}|{t},{u}]",
                        'r': r, 's': s, 't': t, 'u': u,
                        'a_local_id': a_idx,
                        'a_name': f"A[{r},{s}]",
                        'b_local_id': b_idx,
                        'b_name': f"B[{t},{u}]",
                        'live': 1 if live else 0,
                        'target_c_local_id': target_c,
                        'target_c_name': target_c_name,
                    })
    return atoms

def split_atoms(atoms):
    live = [a for a in atoms if a['live'] == 1]
    dead = [a for a in atoms if a['live'] == 0]
    return live, dead

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

def write_markdown(live, dead, path):
    lines = []
    lines.append("# X Live/Dead Partition")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This is the explicit live/dead partition of all 81 X atoms.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total X atoms: {len(live) + len(dead)}")
    lines.append(f"- Live atoms: {len(live)}")
    lines.append(f"- Dead atoms: {len(dead)}")
    lines.append("")
    lines.append("## First 5 Live Rows")
    lines.append("")
    lines.append("| x_local_id | x_name | r | s | t | u | live | target_c |")
    lines.append("|------------|--------|---|---|---|---|------|----------|")
    for a in live[:5]:
        lines.append(f"| {a['x_local_id']} | {a['x_name']} | {a['r']} | {a['s']} | {a['t']} | {a['u']} | {a['live']} | {a['target_c_name']} |")
    lines.append("")
    lines.append("## First 5 Dead Rows")
    lines.append("")
    lines.append("| x_local_id | x_name | r | s | t | u | live |")
    lines.append("|------------|--------|---|---|---|---|------|")
    for a in dead[:5]:
        lines.append(f"| {a['x_local_id']} | {a['x_name']} | {a['r']} | {a['s']} | {a['t']} | {a['u']} | {a['live']} |")
    lines.append("")
    lines.append("See `X_live.csv` and `X_dead.csv` for complete machine-readable tables.")
    lines.append("")
    lines.append("-" * 50)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def sanity_checks(live, dead):
    print("\n--- SANITY CHECKS ---")
    
    print(f"Live count: {len(live)}, Dead count: {len(dead)}")
    assert len(live) == 27, f"Expected 27 live, got {len(live)}"
    assert len(dead) == 54, f"Expected 54 dead, got {len(dead)}"
    
    for a in live:
        assert a['s'] == a['t'], f"Live atom {a['x_name']} has s={a['s']}, t={a['t']}"
    print("All live atoms satisfy s == t: OK")
    
    for a in dead:
        assert a['s'] != a['t'], f"Dead atom {a['x_name']} has s={a['s']}, t={a['t']}"
    print("All dead atoms satisfy s != t: OK")
    
    for a in live:
        expected = 3 * a['r'] + a['u']
        assert a['target_c_local_id'] == expected, f"Target mismatch for {a['x_name']}"
    print("All live targets match formula: OK")
    
    for a in dead:
        assert a['target_c_local_id'] == -1, f"Dead atom {a['x_name']} should have no target"
    print("All dead atoms have no target: OK")
    
    live_ids = set(a['x_local_id'] for a in live)
    dead_ids = set(a['x_local_id'] for a in dead)
    assert len(live_ids) == 27, "Duplicate live IDs"
    assert len(dead_ids) == 54, "Duplicate dead IDs"
    assert len(live_ids & dead_ids) == 0, "Overlap between live and dead"
    assert len(live_ids | dead_ids) == 81, "Missing IDs in union"
    print("No overlap, complete coverage: OK")
    
    print("\nAll sanity checks passed!")

def main():
    print("Generating X partition tables...")
    
    os.makedirs('exports', exist_ok=True)
    
    atoms = generate_x_atoms()
    live, dead = split_atoms(atoms)
    
    live_csv = Path('exports/X_live.csv')
    dead_csv = Path('exports/X_dead.csv')
    md_path = Path('exports/X_partition.md')
    
    write_csv(live, live_csv)
    write_csv(dead, dead_csv)
    write_markdown(live, dead, md_path)
    
    print(f"Live CSV: {live_csv}")
    print(f"Dead CSV: {dead_csv}")
    print(f"Markdown: {md_path}")
    
    sanity_checks(live, dead)
    
    print(f"\n--- SUMMARY ---")
    print(f"Live rows: {len(live)}")
    print(f"Dead rows: {len(dead)}")

if __name__ == "__main__":
    main()