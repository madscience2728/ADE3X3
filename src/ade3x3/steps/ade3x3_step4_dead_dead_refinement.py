import ade3x3_step1 as s1
from ade3x3_step2_permutations import (
    generate_s3, act_on_p
)
from ade3x3_step3_stabilizers import list_compatible_actions, pair_signature

def dead_atoms():
    selection = s1.build_selection_map([s1.p_decode(i) for i in range(81)])
    return [i for i in range(81) if selection[i] < 0]

def canonical_equality_pattern(vals):
    mapping = {}
    next_label = 0
    pattern = []
    for v in vals:
        if v not in mapping:
            mapping[v] = next_label
            next_label += 1
        pattern.append(mapping[v])
    return tuple(pattern)

def refined_signature(p1, p2):
    p1d = s1.p_decode(p1)
    p2d = s1.p_decode(p2)
    
    r1, s1_ = p1d.left
    t1, u1 = p1d.right
    r2, s2_ = p2d.left
    t2, u2 = p2d.right
    
    same = (p1 == p2)
    row_eq = (r1 == r2)
    col_eq = (u1 == u2)
    
    shared_pattern = canonical_equality_pattern([s1_, t1, s2_, t2])
    
    match_orient = (s1_ == s2_ and t1 == t2)
    reversed_orient = (s1_ == t2 and t1 == s2_)
    same_mismatch = match_orient
    reversed_mismatch = reversed_orient
    
    all_distinct = len(set([s1_, t1, s2_, t2])) == 4
    
    distinct = canonical_equality_pattern([s1_, t1, s2_, t2])
    num_distinct = len(set([s1_, t1, s2_, t2]))
    
    return {
        'same': same,
        'row_eq': row_eq,
        'col_eq': col_eq,
        'shared_pattern': shared_pattern,
        'num_distinct': num_distinct,
        'same_mismatch': same_mismatch,
        'reversed_mismatch': reversed_mismatch,
    }

def orbit_decomposition(elements, actions, selection):
    pair_sets = {}
    
    for p1 in elements:
        for p2 in elements:
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
                    'old_sig': None,
                    'refined_sig': None
                }
            pair_sets[orbit_key]['members'].append((p1, p2))
    
    reps = []
    for key, data in pair_sets.items():
        members = data['members']
        data['repr'] = members[0]
        data['old_sig'] = pair_signature(members[0][0], members[0][1], selection)
        data['refined_sig'] = refined_signature(members[0][0], members[0][1])
        reps.append(data)
    
    return reps

def sig_to_key(sig):
    return (sig['same'], sig['row_eq'], sig['col_eq'], sig['shared_pattern'])

def test_completeness(orbits):
    sig_groups = {}
    for orb in orbits:
        key = sig_to_key(orb['refined_sig'])
        if key not in sig_groups:
            sig_groups[key] = []
        sig_groups[key].append(orb['repr'])
    
    return len(sig_groups), sig_groups

def drop_component_test(orbits):
    results = {}
    
    full_key = lambda s: (s['same'], s['row_eq'], s['col_eq'], s['shared_pattern'])
    
    no_same = {}
    for orb in orbits:
        key = (orb['refined_sig']['row_eq'], orb['refined_sig']['col_eq'], orb['refined_sig']['shared_pattern'])
        if key not in no_same:
            no_same[key] = []
        no_same[key].append(orb['repr'])
    results['without_same'] = len(no_same)
    
    no_row_eq = {}
    for orb in orbits:
        key = (orb['refined_sig']['same'], orb['refined_sig']['col_eq'], orb['refined_sig']['shared_pattern'])
        if key not in no_row_eq:
            no_row_eq[key] = []
        no_row_eq[key].append(orb['repr'])
    results['without_row_eq'] = len(no_row_eq)
    
    no_col_eq = {}
    for orb in orbits:
        key = (orb['refined_sig']['same'], orb['refined_sig']['row_eq'], orb['refined_sig']['shared_pattern'])
        if key not in no_col_eq:
            no_col_eq[key] = []
        no_col_eq[key].append(orb['repr'])
    results['without_col_eq'] = len(no_col_eq)
    
    no_shared = {}
    for orb in orbits:
        key = (orb['refined_sig']['same'], orb['refined_sig']['row_eq'], orb['refined_sig']['col_eq'])
        if key not in no_shared:
            no_shared[key] = []
        no_shared[key].append(orb['repr'])
    results['without_shared_pattern'] = len(no_shared)
    
    return results

def print_report():
    actions = list_compatible_actions()
    
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    dead_ps = dead_atoms()
    
    print("=" * 70)
    print("ADE3x3 Step 4: Dead-Dead Pair Refinement Report")
    print("=" * 70)
    
    print("\n1. DEAD-DEAD PAIR ORBIT ENUMERATION")
    dd_orbits = orbit_decomposition(dead_ps, actions, selection)
    print(f"   Total dead-dead ordered pairs: {len(dead_ps)**2}")
    print(f"   Number of orbits: {len(dd_orbits)}")
    
    print("\n2. REFINED SIGNATURE COMPONENTS")
    print("   - same: whether p1 == p2")
    print("   - row_eq: (r1 == r2)")
    print("   - col_eq: (u1 == u2)")
    print("   - shared_pattern: canonical equality pattern of (s1,t1,s2,t2)")
    
    print("\n3. REFINED SIGNATURE COMPLETENESS TEST")
    num_sigs, sig_groups = test_completeness(dd_orbits)
    print(f"   Dead-dead orbits: {len(dd_orbits)}")
    print(f"   Distinct refined signatures: {num_sigs}")
    
    if num_sigs == len(dd_orbits):
        print("   RESULT: Refined signatures FULLY SEPARATE all orbits")
    else:
        print("   RESULT: Still incomplete")
        collisions = [(k, v) for k, v in sig_groups.items() if len(v) > 1]
        print(f"   Collisions: {len(collisions)}")
        for key, pairs in collisions[:3]:
            p1, p2 = pairs[0]
            print(f"      Sig {key}: {len(pairs)} orbits collapsed")
    
    print("\n4. MINIMALITY ANALYSIS (drop-one-component test)")
    minimal = drop_component_test(dd_orbits)
    print(f"   Using all 4 components: {len(dd_orbits)} unique (complete)")
    print(f"   Without 'same': {minimal['without_same']} unique")
    print(f"   Without 'row_eq': {minimal['without_row_eq']} unique")
    print(f"   Without 'col_eq': {minimal['without_col_eq']} unique")
    print(f"   Without 'shared_pattern': {minimal['without_shared_pattern']} unique")
    
    necessary = []
    if minimal['without_same'] < len(dd_orbits):
        necessary.append('same')
    if minimal['without_row_eq'] < len(dd_orbits):
        necessary.append('row_eq')
    if minimal['without_col_eq'] < len(dd_orbits):
        necessary.append('col_eq')
    if minimal['without_shared_pattern'] < len(dd_orbits):
        necessary.append('shared_pattern')
    
    print(f"   Necessary components: {necessary}")
    
    print("\n5. STRUCTURAL INTERPRETATION")
    p1 = dd_orbits[0]['repr'][0]
    p2 = dd_orbits[0]['repr'][1]
    sig0 = dd_orbits[0]['refined_sig']
    print(f"   Example orbit 0: {s1.p_decode(p1)} -- {s1.p_decode(p2)}")
    print(f"      same={sig0['same']}, row_eq={sig0['row_eq']}, col_eq={sig0['col_eq']}")
    print(f"      shared_pattern={sig0['shared_pattern']}, num_distinct={sig0['num_distinct']}")
    
    print("\n6. ORBIT SIZES AND SAMPLE REPRESENTATIVES")
    for i, orb in enumerate(dd_orbits):
        p1, p2 = orb['repr']
        sig = orb['refined_sig']
        print(f"   Orbit {i}: size={len(orb['members'])}")
        print(f"      {s1.p_decode(p1)} -- {s1.p_decode(p2)}")
        print(f"      sig: same={sig['same']}, row_eq={sig['row_eq']}, col_eq={sig['col_eq']}, pattern={sig['shared_pattern']}")
    
    print("\n7. EQUALITY PATTERN TYPES DISCOVERED")
    pattern_types = set()
    for orb in dd_orbits:
        pattern_types.add(orb['refined_sig']['shared_pattern'])
    print(f"   Distinct shared_pattern values: {len(pattern_types)}")
    print(f"   Patterns: {sorted(pattern_types)}")
    
    print("\n" + "=" * 70)

def main():
    print_report()

if __name__ == "__main__":
    main()