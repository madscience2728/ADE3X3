import ade3x3_step1 as s1
from ade3x3_step2_permutations import (
    generate_s3, is_compatible_action, act_on_p, act_on_w
)

def list_compatible_actions():
    s3 = generate_s3()
    actions = []
    for pi_rA in s3:
        for pi_shared in s3:
            for pi_cB in s3:
                actions.append((pi_rA, pi_shared, pi_cB))
    return actions

def stabilizer_size(element_idx, actions, which):
    count = 0
    for pi_rA, pi_shared, pi_cB in actions:
        if which == 'w':
            new_idx = act_on_w(element_idx, pi_rA, pi_cB)
        else:
            new_idx = act_on_p(element_idx, pi_rA, pi_shared, pi_shared, pi_cB)
        if new_idx == element_idx:
            count += 1
    return count

def pair_signature(p1, p2, selection):
    s1p1 = s1.p_decode(p1)
    s1p2 = s1.p_decode(p2)
    r1, s1_ = s1p1.left
    t1, u1 = s1p1.right
    r2, s2_ = s1p2.left
    t2, u2 = s1p2.right
    
    live1 = selection[p1] >= 0
    live2 = selection[p2] >= 0
    
    s_self = "live" if live1 else "dead"
    s_other = "live" if live2 else "dead"
    
    w1 = selection[p1]
    w2 = selection[p2]
    same_w = (w1 == w2) and (w1 >= 0)
    
    return {
        'type': f"{s_self}-{s_other}",
        'r_eq': r1 == r2,
        's_eq': s1_ == s2_,
        't_eq': t1 == t2,
        'u_eq': u1 == u2,
        'same_w': same_w,
        'same_output_row': (w1 >= 0 and w2 >= 0 and (s1.w_decode(w1).r == s1.w_decode(w2).r)),
        'same_output_col': (w1 >= 0 and w2 >= 0 and (s1.w_decode(w1).u == s1.w_decode(w2).u)),
        'same_left_U': (r1 == r2 and s1_ == s2_),
        'same_right_V': (t1 == t2 and u1 == u2),
    }

def compute_pair_orbits(elements1, elements2, actions, selection):
    pair_sets = {}
    
    for p1 in elements1:
        for p2 in elements2:
            pair = (p1, p2)
            orbit = []
            for pi_rA, pi_shared, pi_cB in actions:
                p1_new = act_on_p(p1, pi_rA, pi_shared, pi_shared, pi_cB)
                p2_new = act_on_p(p2, pi_rA, pi_shared, pi_shared, pi_cB)
                orbit.append((p1_new, p2_new))
            
            orbit_key = tuple(sorted([(a, b) for a, b in orbit]))
            if orbit_key not in pair_sets:
                pair_sets[orbit_key] = {
                    'members': [],
                    'repr': None,
                    'sig': None
                }
            pair_sets[orbit_key]['members'].append((p1, p2))
    
    reps = []
    for key, data in pair_sets.items():
        members = data['members']
        data['repr'] = members[0]
        data['sig'] = pair_signature(members[0][0], members[0][1], selection)
        reps.append(data)
    
    return reps

def fiber_signature(p_idx, selection):
    w_idx = selection[p_idx]
    live = w_idx >= 0
    
    if not live:
        p = s1.p_decode(p_idx)
        s, t = p.left[1], p.right[0]
        return {'type': 'dead', 's_t_mismatch': (s, t)}
    else:
        p = s1.p_decode(p_idx)
        r, s = p.left
        t, u = p.right
        return {
            'type': 'live',
            'target_w': w_idx,
            'fiber_index': s,
            'contracted': f"s={s}=t"
        }

def print_report():
    actions = list_compatible_actions()
    group_size = len(actions)
    
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    live_ps = [i for i in range(81) if selection[i] >= 0]
    dead_ps = [i for i in range(81) if selection[i] < 0]
    
    live_rep = 0
    dead_rep = 0
    while selection[dead_rep] >= 0:
        dead_rep += 1
    w_rep = 0
    
    print("=" * 70)
    print("ADE3x3 Step 3: Stabilizers and Pair Orbits Report")
    print("=" * 70)
    
    print(f"\n1. GROUP SIZE")
    print(f"   |G| = {group_size}")
    
    print("\n2. STABILIZERS AND ORBIT-STABILIZER CHECK")
    
    stab_live = stabilizer_size(live_rep, actions, 'p')
    orb_live = group_size // stab_live
    print(f"   Live P atom P[0,0|0,0]:")
    print(f"      |Stab| = {stab_live}, |Orbit| = {orb_live}")
    print(f"      |Orbit| * |Stab| = {orb_live * stab_live} (verify={orb_live * stab_live == group_size})")
    
    stab_dead = stabilizer_size(dead_rep, actions, 'p')
    orb_dead = group_size // stab_dead
    p_dead = s1.p_decode(dead_rep)
    print(f"   Dead P atom P[{p_dead.left[0]},{p_dead.left[1]}|{p_dead.right[0]},{p_dead.right[1]}]:")
    print(f"      |Stab| = {stab_dead}, |Orbit| = {orb_dead}")
    print(f"      |Orbit| * |Stab| = {orb_dead * stab_dead} (verify={orb_dead * stab_dead == group_size})")
    
    stab_w = stabilizer_size(w_rep, actions, 'w')
    orb_w = group_size // stab_w
    print(f"   W atom W[0,0]:")
    print(f"      |Stab| = {stab_w}, |Orbit| = {orb_w}")
    print(f"      |Orbit| * |Stab| = {orb_w * stab_w} (verify={orb_w * stab_w == group_size})")
    
    print("\n3. PAIR ORBIT COUNTS")
    
    ll_orbits = compute_pair_orbits(live_ps, live_ps, actions, selection)
    print(f"   Live-live pairs: {len(ll_orbits)} orbits ({len(live_ps)**2} ordered pairs)")
    
    dd_orbits = compute_pair_orbits(dead_ps, dead_ps, actions, selection)
    print(f"   Dead-dead pairs: {len(dd_orbits)} orbits ({len(dead_ps)**2} ordered pairs)")
    
    ld_orbits = compute_pair_orbits(live_ps, dead_ps, actions, selection)
    print(f"   Live-dead pairs: {len(ld_orbits)} orbits ({len(live_ps)*len(dead_ps)} ordered pairs)")
    
    print("\n4. PAIR ORBIT REPRESENTATIVES AND SIGNATURES")
    
    print("   Live-live selected orbits:")
    if len(ll_orbits) <= 3:
        for orb in ll_orbits[:3]:
            p1, p2 = orb['repr']
            sig = orb['sig']
            print(f"      {s1.p_decode(p1)} -- {s1.p_decode(p2)}")
            print(f"         r_eq={sig['r_eq']}, s_eq={sig['s_eq']}, t_eq={sig['t_eq']}, u_eq={sig['u_eq']}, same_w={sig['same_w']}")
    else:
        for orb in ll_orbits[:2]:
            p1, p2 = orb['repr']
            sig = orb['sig']
            print(f"      {s1.p_decode(p1)} -- {s1.p_decode(p2)}")
            print(f"         r_eq={sig['r_eq']}, s_eq={sig['s_eq']}, t_eq={sig['t_eq']}, u_eq={sig['u_eq']}, same_w={sig['same_w']}")
        print(f"      ... and {len(ll_orbits)-2} more live-live orbits")
    
    print("\n5. FIBER SIGNATURES FOR REPRESENTATIVES")
    live_fs = fiber_signature(live_rep, selection)
    print(f"   Live atom P[0,0|0,0]: {live_fs}")
    dead_fs = fiber_signature(dead_rep, selection)
    p_d = s1.p_decode(dead_rep)
    print(f"   Dead atom P[{p_d.left[0]},{p_d.left[1]}|{p_d.right[0]},{p_d.right[1]}]: {dead_fs}")
    print(f"   W atom W[0,0] fiber size = 3")
    
    print("\n6. CLASSIFICATION COMPLETENESS CHECK")
    
    ll_sig_groups = {}
    for orb in ll_orbits:
        p1, p2 = orb['repr']
        key = (orb['sig']['r_eq'], orb['sig']['s_eq'], orb['sig']['t_eq'], orb['sig']['u_eq'], orb['sig']['same_w'])
        if key not in ll_sig_groups:
            ll_sig_groups[key] = []
        ll_sig_groups[key].append(orb['repr'])
    
    print(f"   Live-live: {len(ll_sig_groups)} unique signatures for {len(ll_orbits)} orbits")
    if len(ll_sig_groups) != len(ll_orbits):
        print(f"   WARNING: signatures do NOT fully separate live-live orbits")
        for key, pairs in ll_sig_groups.items():
            if len(pairs) > 1:
                print(f"      Shared sig {key}: {len(pairs)} orbits")
    
    dd_sig_groups = {}
    for orb in dd_orbits:
        p1, p2 = orb['repr']
        key = (orb['sig']['r_eq'], orb['sig']['s_eq'], orb['sig']['t_eq'], orb['sig']['u_eq'])
        if key not in dd_sig_groups:
            dd_sig_groups[key] = []
        dd_sig_groups[key].append(orb['repr'])
    
    print(f"   Dead-dead: {len(dd_sig_groups)} unique signatures for {len(dd_orbits)} orbits")
    if len(dd_sig_groups) != len(dd_orbits):
        print(f"   WARNING: signatures do NOT fully separate dead-dead orbits")
    
    ld_sig_groups = {}
    for orb in ld_orbits:
        p1, p2 = orb['repr']
        key = (orb['sig']['r_eq'], orb['sig']['s_eq'], orb['sig']['t_eq'], orb['sig']['u_eq'])
        if key not in ld_sig_groups:
            ld_sig_groups[key] = []
        ld_sig_groups[key].append(orb['repr'])
    
    print(f"   Live-dead: {len(ld_sig_groups)} unique signatures for {len(ld_orbits)} orbits")
    if len(ld_sig_groups) != len(ld_orbits):
        print(f"   WARNING: signatures do NOT fully separate live-dead orbits")
    
    print("\n7. DETAILED ORBIT-SIGNATURE COMPARISON")
    print("   Live-live orbit details:")
    for i, orb in enumerate(ll_orbits):
        p1, p2 = orb['repr']
        sig = orb['sig']
        print(f"      Orbit {i}: {s1.p_decode(p1)} -- {s1.p_decode(p2)}")
        print(f"         sig: r_eq={sig['r_eq']}, s_eq={sig['s_eq']}, t_eq={sig['t_eq']}, u_eq={sig['u_eq']}, same_w={sig['same_w']}")
        print(f"         size: {len(orb['members'])}")
    
    print("\n" + "=" * 70)

def main():
    print_report()

if __name__ == "__main__":
    main()