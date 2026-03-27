import csv
import os
from pathlib import Path
from datetime import datetime

def generate_c_fibers():
    fibers = []
    
    for r in range(3):
        for u in range(3):
            c_idx = 3 * r + u
            c_name = f"C[{r},{u}]"
            
            x0_idx = 9 * (3 * r + 0) + (3 * 0 + u)
            x0_name = f"X[{r},0|0,{u}]"
            
            x1_idx = 9 * (3 * r + 1) + (3 * 1 + u)
            x1_name = f"X[{r},1|1,{u}]"
            
            x2_idx = 9 * (3 * r + 2) + (3 * 2 + u)
            x2_name = f"X[{r},2|2,{u}]"
            
            fibers.append({
                'c_local_id': c_idx,
                'c_name': c_name,
                'r': r,
                'u': u,
                'fiber_size': 3,
                'x0_local_id': x0_idx,
                'x0_name': x0_name,
                'x1_local_id': x1_idx,
                'x1_name': x1_name,
                'x2_local_id': x2_idx,
                'x2_name': x2_name,
            })
    
    return fibers

def write_csv(fibers, path):
    fieldnames = [
        'c_local_id', 'c_name', 'r', 'u', 'fiber_size',
        'x0_local_id', 'x0_name', 'x1_local_id', 'x1_name',
        'x2_local_id', 'x2_name'
    ]
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fibers)

def write_markdown(fibers, path):
    live_total = len(fibers) * 3
    
    lines = []
    lines.append("# C Fibers Table")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This is the explicit object-level fiber table for all 9 C atoms.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Total C atoms: 9")
    lines.append("- Fiber size per C: 3")
    lines.append(f"- Total live fiber memberships: {live_total}")
    lines.append("")
    lines.append("## Fibers Table")
    lines.append("")
    lines.append("| c_local_id | c_name | r | u | x0 | x1 | x2 |")
    lines.append("|------------|--------|---|---|----|----|----|")
    
    for f in fibers:
        lines.append(f"| {f['c_local_id']} | {f['c_name']} | {f['r']} | {f['u']} | {f['x0_name']} | {f['x1_name']} | {f['x2_name']} |")
    
    lines.append("")
    lines.append("See `C_fibers.csv` for the complete machine-readable table.")
    lines.append("")
    lines.append("-" * 50)
    
    content = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def sanity_checks(fibers):
    print("\n--- SANITY CHECKS ---")
    
    assert len(fibers) == 9, f"Expected 9 C atoms, got {len(fibers)}"
    print(f"Row count: {len(fibers)}")
    
    for f in fibers:
        assert f['fiber_size'] == 3, f"Fiber size for {f['c_name']} is {f['fiber_size']}, expected 3"
    print("All fiber_size == 3: OK")
    
    all_x_ids = set()
    for f in fibers:
        for xkey in ['x0_local_id', 'x1_local_id', 'x2_local_id']:
            xid = f[xkey]
            all_x_ids.add(xid)
    
    assert len(all_x_ids) == 27, f"Expected 27 unique X atoms in fibers, got {len(all_x_ids)}"
    print(f"Unique X atoms in fibers: {len(all_x_ids)}")
    
    for f in fibers:
        for xkey in ['x0_local_id', 'x1_local_id', 'x2_local_id']:
            xid = f[xkey]
            r, s = (xid // 9) // 3, (xid // 9) % 3
            t, u_ = xid % 9 // 3, xid % 9 % 3
            assert s == t, f"Fiber X atom {xid} has s={s}, t={t}, not live"
    print("All X atoms in fibers are live (s == t): OK")
    
    for f in fibers:
        c_r, c_u = f['r'], f['u']
        for xkey in ['x0_local_id', 'x1_local_id', 'x2_local_id']:
            xid = f[xkey]
            x_r = (xid // 9) // 3
            x_u = xid % 9 % 3
            expected_c_idx = 3 * x_r + x_u
            actual_c_idx = f['c_local_id']
            assert expected_c_idx == actual_c_idx, f"X {xid} targets C idx {expected_c_idx}, expected {actual_c_idx}"
    print("All X atoms target the correct C atom: OK")
    
    print("\nAll sanity checks passed!")

def main():
    print("Generating C fibers table...")
    
    os.makedirs('exports', exist_ok=True)
    
    fibers = generate_c_fibers()
    
    csv_path = Path('exports/C_fibers.csv')
    md_path = Path('exports/C_fibers.md')
    
    write_csv(fibers, csv_path)
    print(f"CSV written: {csv_path}")
    
    write_markdown(fibers, md_path)
    print(f"Markdown written: {md_path}")
    
    sanity_checks(fibers)
    
    total_x_in_fibers = sum(3 for _ in fibers)
    
    print(f"\n--- SUMMARY ---")
    print(f"Output paths:")
    print(f"  - {csv_path.absolute()}")
    print(f"  - {md_path.absolute()}")
    print(f"Row count: 9")
    print(f"Total fiber memberships: {total_x_in_fibers}")
    print(f"Distinct live X atoms covered: 27")

if __name__ == "__main__":
    main()