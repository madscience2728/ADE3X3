from dataclasses import dataclass
from typing import Tuple
import ade3x3_step1 as s1

def generate_s3():
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    return perms

def compose_perm(p1, p2):
    return tuple(p1[p2[i]] for i in range(3))

def invert_perm(p):
    inv = [0] * 3
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)

def act_on_u(u_idx, perm, which):
    u = s1.u_decode(u_idx)
    if which == 'rows':
        r = perm[u.r]
        return s1.u_encode(r, u.s)
    elif which == 'cols':
        s = perm[u.s]
        return s1.u_encode(u.r, s)

def act_on_v(v_idx, perm, which):
    v = s1.v_decode(v_idx)
    if which == 'rows':
        t = perm[v.t]
        return s1.v_encode(t, v.u)
    elif which == 'cols':
        u = perm[v.u]
        return s1.v_encode(v.t, u)

def act_on_w(w_idx, perm_rowA, perm_colB):
    w = s1.w_decode(w_idx)
    r = perm_rowA[w.r]
    u = perm_colB[w.u]
    return s1.w_encode(r, u)

def act_on_p(p_idx, pi_rA, pi_cA, pi_rB, pi_cB):
    p = s1.p_decode(p_idx)
    r, s = p.left
    t, u = p.right
    r_new = pi_rA[r]
    s_new = pi_cA[s]
    t_new = pi_rB[t]
    u_new = pi_cB[u]
    return s1.p_encode(r_new, s_new, t_new, u_new)

def is_compatible_action(pi_rA, pi_cA, pi_rB, pi_cB):
    return pi_cA == pi_rB

def induced_w_action(pi_rA, pi_cA, pi_rB, pi_cB):
    return pi_rA, pi_cB

def apply_action_to_live(selection, pi_rA, pi_cA, pi_rB, pi_cB):
    live_pairs = []
    w_action = induced_w_action(pi_rA, pi_cA, pi_rB, pi_cB)
    for p_idx, w_idx in enumerate(selection):
        if w_idx >= 0:
            p_new_idx = act_on_p(p_idx, pi_rA, pi_cA, pi_rB, pi_cB)
            w_new_idx = act_on_w(w_idx, pi_rA, pi_cB)
            live_pairs.append((p_idx, w_idx, p_new_idx, w_new_idx))
    return live_pairs

def compute_orbits(selection, compatible_actions):
    p_orbits = {}
    live_orbits = {}
    dead_orbits = {}
    w_orbits = {}
    
    all_p_sets = {}
    
    for p_idx in range(81):
        orbit = []
        for pi_rA, pi_cA, pi_rB, pi_cB in compatible_actions:
            new_idx = act_on_p(p_idx, pi_rA, pi_cA, pi_rB, pi_cB)
            orbit.append(new_idx)
        orbit_set = frozenset(orbit)
        all_p_sets[p_idx] = orbit_set
    
    unique_p_orbits = {}
    for p_idx, orbit_set in all_p_sets.items():
        key = tuple(sorted(orbit_set))
        if key not in unique_p_orbits:
            unique_p_orbits[key] = []
        unique_p_orbits[key].append(p_idx)
    
    for key, members in unique_p_orbits.items():
        rep = min(members)
        p_orbits[rep] = members
        
        live_members = [m for m in members if selection[m] >= 0]
        dead_members = [m for m in members if selection[m] < 0]
        
        if live_members:
            live_orbits[rep] = live_members
        if dead_members:
            dead_orbits[rep] = dead_members
    
    all_w_sets = {}
    for w_idx in range(9):
        orbit = []
        for pi_rA, pi_cA, pi_rB, pi_cB in compatible_actions:
            new_idx = act_on_w(w_idx, pi_rA, pi_cB)
            orbit.append(new_idx)
        orbit_set = frozenset(orbit)
        all_w_sets[w_idx] = orbit_set
    
    unique_w_orbits = {}
    for w_idx, orbit_set in all_w_sets.items():
        key = tuple(sorted(orbit_set))
        if key not in unique_w_orbits:
            unique_w_orbits[key] = []
        unique_w_orbits[key].append(w_idx)
    
    for key, members in unique_w_orbits.items():
        rep = min(members)
        w_orbits[rep] = members
    
    return p_orbits, live_orbits, dead_orbits, w_orbits

def canonical_name_p(p_idx):
    p = s1.p_decode(p_idx)
    return f"P[{p.left[0]},{p.left[1]}|{p.right[0]},{p.right[1]}]"

def canonical_name_w(w_idx):
    w = s1.w_decode(w_idx)
    return f"W[{w.r},{w.u}]"

def print_report(selection, compatible_actions):
    s3 = generate_s3()
    total_raw = len(s3) ** 4
    total_compatible = len(compatible_actions)
    
    p_orbits, live_orbits, dead_orbits, w_orbits = compute_orbits(selection, compatible_actions)
    
    print("=" * 70)
    print("ADE3x3 Step 2: Permutation Actions Report")
    print("=" * 70)
    
    print("\n1. PERMUTATION ACTIONS COUNT")
    print(f"   Total raw combined actions: {total_raw} (6^4)")
    print(f"   Matmul-compatible actions: {total_compatible}")
    
    print("\n2. COMPATIBILITY VERIFICATION")
    compatible_count_by_cA_rB = 0
    for pi_rA in s3:
        for pi_cA in s3:
            for pi_rB in s3:
                for pi_cB in s3:
                    if pi_cA == pi_rB:
                        compatible_count_by_cA_rB += 1
    print(f"   Verification: pi_cA == pi_rB gives {compatible_count_by_cA_rB} compatible actions")
    print(f"   Matches above: {compatible_count_by_cA_rB == total_compatible}")
    
    print("\n3. ORBIT COUNTS (under compatible actions)")
    print(f"   Orbits on all P atoms: {len(p_orbits)}")
    print(f"   Orbits on live P atoms: {len(live_orbits)}")
    print(f"   Orbits on dead P atoms: {len(dead_orbits)}")
    print(f"   Orbits on W atoms: {len(w_orbits)}")
    
    print("\n4. ORBIT REPRESENTATIVES")
    print("   All P orbits:")
    for rep in sorted(p_orbits.keys()):
        print(f"      {canonical_name_p(rep)} : {len(p_orbits[rep])} atoms")
    
    print("   Live P orbits:")
    for rep in sorted(live_orbits.keys()):
        print(f"      {canonical_name_p(rep)} : {len(live_orbits[rep])} atoms")
    
    print("   Dead P orbits:")
    for rep in sorted(dead_orbits.keys()):
        print(f"      {canonical_name_p(rep)} : {len(dead_orbits[rep])} atoms")
    
    print("   W orbits:")
    for rep in sorted(w_orbits.keys()):
        print(f"      {canonical_name_w(rep)} : {len(w_orbits[rep])} atoms")
    
    print("\n5. EXAMPLE ACTIONS")
    s3 = generate_s3()
    ex_pi = (0, 2, 1)
    actions_to_show = [
        (ex_pi, s3[0], ex_pi, s3[0]),
        (s3[0], ex_pi, ex_pi, s3[0]),
        (ex_pi, s3[1], s3[1], ex_pi),
    ]
    for pi_rA, pi_cA, pi_rB, pi_cB in actions_to_show:
        compat = is_compatible_action(pi_rA, pi_cA, pi_rB, pi_cB)
        compat_str = "compatible" if compat else "NOT compatible"
        if compat:
            w_action = induced_w_action(pi_rA, pi_cA, pi_rB, pi_cB)
            print(f"   Action {pi_rA},{pi_cA},{pi_rB},{pi_cB}: {compat_str}")
            print(f"      W[r,u] -> W[{w_action[0]}(r),{w_action[1]}(u)]")
        else:
            print(f"   Action {pi_rA},{pi_cA},{pi_rB},{pi_cB}: {compat_str}")
        
        p_ex = 0
        p_new = act_on_p(p_ex, pi_rA, pi_cA, pi_rB, pi_cB)
        print(f"      P[{p_ex}] -> {canonical_name_p(p_new)}")
    
    print("\n" + "=" * 70)

def main():
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    s3 = generate_s3()
    
    compatible_actions = []
    for pi_rA in s3:
        for pi_cA in s3:
            for pi_rB in s3:
                for pi_cB in s3:
                    if is_compatible_action(pi_rA, pi_cA, pi_rB, pi_cB):
                        compatible_actions.append((pi_rA, pi_cA, pi_rB, pi_cB))
    
    print_report(selection, compatible_actions)

if __name__ == "__main__":
    main()