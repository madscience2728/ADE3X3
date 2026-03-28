import csv
import os
from pathlib import Path
from datetime import datetime
from itertools import permutations

def invert_perm(p):
    inv = [0] * 3
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)

def generate_s3():
    return [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

def generate_actions():
    s3 = generate_s3()
    actions = []
    for pi_rA in s3:
        for pi_shared in s3:
            for pi_cB in s3:
                actions.append({
                    'pi_rA': pi_rA,
                    'pi_shared': pi_shared,
                    'pi_cB': pi_cB,
                    'inv_pi_rA': invert_perm(pi_rA),
                    'inv_pi_shared': invert_perm(pi_shared),
                    'inv_pi_cB': invert_perm(pi_cB),
                })
    return actions

def action_on_a(a_idx, act):
    r, s = divmod(a_idx, 3)
    r_new = act['pi_rA'][r]
    s_new = act['pi_shared'][s]
    return 3 * r_new + s_new

def action_on_b(b_idx, act):
    t, u = divmod(b_idx, 3)
    t_new = act['pi_shared'][t]
    u_new = act['pi_cB'][u]
    return 3 * t_new + u_new

def action_on_c(c_idx, act):
    r, u = divmod(c_idx, 3)
    r_new = act['pi_rA'][r]
    u_new = act['pi_cB'][u]
    return 3 * r_new + u_new

def action_on_x(x_idx, act):
    r, s = divmod(x_idx // 9, 3)
    t, u = divmod(x_idx % 9, 3)
    r_new = act['pi_rA'][r]
    s_new = act['pi_shared'][s]
    t_new = act['pi_shared'][t]
    u_new = act['pi_cB'][u]
    return 9 * (3 * r_new + s_new) + (3 * t_new + u_new)

def find_action_by_perms(actions, pi_rA, pi_shared, pi_cB):
    for i, a in enumerate(actions):
        if a['pi_rA'] == pi_rA and a['pi_shared'] == pi_shared and a['pi_cB'] == pi_cB:
            return i
    return -1

def write_summary_csv(actions, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['action_id', 'pi_rA', 'pi_shared', 'pi_cB', 'inverse_action_id'])
        for i, a in enumerate(actions):
            inv_idx = find_action_by_perms(actions, a['inv_pi_rA'], a['inv_pi_shared'], a['inv_pi_cB'])
            writer.writerow([i, a['pi_rA'], a['pi_shared'], a['pi_cB'], inv_idx])

def write_species_csv(actions, species_fn, path, species_size):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        header = ['action_id'] + [f'img_{i}' for i in range(species_size)]
        writer = csv.writer(f)
        writer.writerow(header)
        for i, a in enumerate(actions):
            row = [i] + [species_fn(idx, a) for idx in range(species_size)]
            writer.writerow(row)

def write_markdown(actions, path):
    lines = []
    lines.append("# Compatible Actions Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This is the explicit compatible symmetry action table export.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total actions: {len(actions)}")
    lines.append("- Species covered: A, B, C, X")
    lines.append("")
    lines.append("## First 10 Actions")
    lines.append("")
    lines.append("| action_id | pi_rA | pi_shared | pi_cB | inverse |")
    lines.append("|-----------|-------|-----------|-------|---------|")
    for a in actions[:10]:
        inv = find_action_by_perms(actions, a['inv_pi_rA'], a['inv_pi_shared'], a['inv_pi_cB'])
        lines.append(f"| {actions.index(a)} | {a['pi_rA']} | {a['pi_shared']} | {a['pi_cB']} | {inv} |")
    lines.append("")
    lines.append("See `actions_A.csv`, `actions_B.csv`, `actions_C.csv`, `actions_X.csv` for complete action tables.")
    lines.append("")
    lines.append("-" * 50)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def sanity_checks(actions):
    print("\n--- SANITY CHECKS ---")
    
    assert len(actions) == 216, f"Expected 216 actions, got {len(actions)}"
    print(f"Total actions: {len(actions)}")
    
    for i, a in enumerate(actions):
        for idx in range(9):
            img = action_on_a(idx, a)
            assert 0 <= img < 9, f"A action {i} maps {idx} to invalid {img}"
        for idx in range(9):
            img = action_on_b(idx, a)
            assert 0 <= img < 9, f"B action {i} maps {idx} to invalid {img}"
        for idx in range(9):
            img = action_on_c(idx, a)
            assert 0 <= img < 9, f"C action {i} maps {idx} to invalid {img}"
        for idx in range(81):
            img = action_on_x(idx, a)
            assert 0 <= img < 81, f"X action {i} maps {idx} to invalid {img}"
    print("All species action outputs are valid permutations: OK")
    
    for i, a in enumerate(actions):
        inv_a = actions[i]['inv_pi_rA'], actions[i]['inv_pi_shared'], actions[i]['inv_pi_cB']
        for idx in range(9):
            twice = action_on_a(action_on_a(idx, a), actions[find_action_by_perms(actions, *inv_a)])
            assert twice == idx, f"A double action failed at {idx}"
        for idx in range(9):
            twice = action_on_b(action_on_b(idx, a), actions[find_action_by_perms(actions, *inv_a)])
            assert twice == idx, f"B double action failed at {idx}"
        for idx in range(9):
            twice = action_on_c(action_on_c(idx, a), actions[find_action_by_perms(actions, *inv_a)])
            assert twice == idx, f"C double action failed at {idx}"
        for idx in range(81):
            twice = action_on_x(action_on_x(idx, a), actions[find_action_by_perms(actions, *inv_a)])
            assert twice == idx, f"X double action failed at {idx}"
    print("All inverse actions work correctly: OK")
    
    identity_idx = find_action_by_perms(actions, (0,1,2), (0,1,2), (0,1,2))
    assert identity_idx >= 0, "Identity action not found"
    print(f"Identity action ID: {identity_idx}")
    
    for idx in range(9):
        assert action_on_a(idx, actions[identity_idx]) == idx, "A identity check failed"
        assert action_on_b(idx, actions[identity_idx]) == idx, "B identity check failed"
        assert action_on_c(idx, actions[identity_idx]) == idx, "C identity check failed"
    for idx in range(81):
        assert action_on_x(idx, actions[identity_idx]) == idx, "X identity check failed"
    print("Identity action verified on all species: OK")
    
    print("\nAll sanity checks passed!")

def main():
    print("Generating action tables...")
    
    os.makedirs('exports', exist_ok=True)
    
    actions = generate_actions()
    
    summary_csv = Path('exports/actions_summary.csv')
    a_csv = Path('exports/actions_A.csv')
    b_csv = Path('exports/actions_B.csv')
    c_csv = Path('exports/actions_C.csv')
    x_csv = Path('exports/actions_X.csv')
    md = Path('exports/actions_summary.md')
    
    write_summary_csv(actions, summary_csv)
    write_species_csv(actions, action_on_a, a_csv, 9)
    write_species_csv(actions, action_on_b, b_csv, 9)
    write_species_csv(actions, action_on_c, c_csv, 9)
    write_species_csv(actions, action_on_x, x_csv, 81)
    write_markdown(actions, md)
    
    print(f"Summary: {summary_csv}")
    print(f"A actions: {a_csv}")
    print(f"B actions: {b_csv}")
    print(f"C actions: {c_csv}")
    print(f"X actions: {x_csv}")
    print(f"Markdown: {md}")
    
    sanity_checks(actions)
    
    identity_idx = find_action_by_perms(actions, (0,1,2), (0,1,2), (0,1,2))
    print(f"\n--- SUMMARY ---")
    print(f"Actions export complete: {len(actions)} actions")
    print(f"Identity action ID: {identity_idx}")

if __name__ == "__main__":
    main()