import ade3x3_step1 as s1
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from collections import defaultdict

@dataclass
class DepthTracker:
    depth_1_atomic_species_complete: bool
    depth_2_ambient_product_universe_complete: bool
    depth_2_symmetry_layer_complete: bool
    depth_2_pair_relation_catalogs_complete: bool
    current_explicit_depth: str
    next_intended_depth: str
    longterm_target: str

def canonical_pattern(vals):
    mapping = {}
    next_label = 0
    pattern = []
    for v in vals:
        if v not in mapping:
            mapping[v] = next_label
            next_label += 1
        pattern.append(mapping[v])
    return tuple(pattern)

def signature_XX(x1, x2, selection):
    p1 = s1.p_decode(x1)
    p2 = s1.p_decode(x2)
    r1, s1_ = p1.left
    t1, u1 = p1.right
    r2, s2_ = p2.left
    t2, u2 = p2.right
    
    live1 = selection[x1] >= 0
    live2 = selection[x2] >= 0
    
    return {
        'x1_live': live1,
        'x2_live': live2,
        'same_atom': x1 == x2,
        'r_eq': r1 == r2,
        's_eq': s1_ == s2_,
        't_eq': t1 == t2,
        'u_eq': u1 == u2,
        'same_A': (r1 == r2 and s1_ == s2_),
        'same_B': (t1 == t2 and u1 == u2),
        'same_target_C': (selection[x1] == selection[x2] and selection[x1] >= 0),
        'same_fiber': (live1 and live2 and s1_ == s2_ and t1 == t2),
        'shared_s_t_pattern': canonical_pattern([s1_, t1, s2_, t2]),
        'r_pattern': canonical_pattern([r1, r2]),
        'u_pattern': canonical_pattern([u1, u2]),
    }

def signature_CX(c, x, selection):
    c_w = s1.w_decode(c)
    p = s1.p_decode(x)
    r2, s2_ = p.left
    t2, u2 = p.right
    
    live = selection[x] >= 0
    target_c = selection[x]
    target_match = (target_c == c) if live else False
    
    fiber_idx = s2_ if live else None
    
    return {
        'x_live': live,
        'x_maps_to_c': target_match,
        'c_row_eq_x_target_row': (live and c_w.r == r2),
        'c_col_eq_x_target_col': (live and c_w.u == u2),
        'fiber_index': fiber_idx,
        'x_target_equals_c': target_match,
        'r_pattern': canonical_pattern([c_w.r, r2]),
        'u_pattern': canonical_pattern([c_w.u, u2]),
    }

def signature_CC(c1, c2):
    w1 = s1.w_decode(c1)
    w2 = s1.w_decode(c2)
    
    return {
        'same': c1 == c2,
        'same_row': w1.r == w2.r,
        'same_col': w1.u == w2.u,
        'r_pattern': canonical_pattern([w1.r, w2.r]),
        'u_pattern': canonical_pattern([w1.u, w2.u]),
    }

def signature_AX(a, x, selection):
    a_u = s1.u_decode(a)
    p = s1.p_decode(x)
    r2, s2_ = p.left
    
    use_exact_A = (a_u.r == r2 and a_u.s == s2_)
    same_row = a_u.r == r2
    same_col_in_A = a_u.s == s2_
    live = selection[x] >= 0
    
    return {
        'x_uses_exact_A': use_exact_A,
        'same_row_A': same_row,
        'same_col_A': same_col_in_A,
        'x_live': live,
        'in_contributing_fiber': (live and a_u.r == r2),
    }

def signature_BX(b, x, selection):
    b_v = s1.v_decode(b)
    p = s1.p_decode(x)
    t2, u2 = p.right
    
    use_exact_B = (b_v.t == t2 and b_v.u == u2)
    same_row_B = b_v.t == t2
    same_col_B = b_v.u == u2
    live = selection[x] >= 0
    
    return {
        'x_uses_exact_B': use_exact_B,
        'same_row_B': same_row_B,
        'same_col_B': same_col_B,
        'x_live': live,
        'in_contributing_fiber': (live and b_v.u == u2),
    }

def build_catalog(pairs, selection, sig_func, extra_args=()):
    sig_to_type = {}
    type_id_counter = 0
    type_info = {}
    
    for p1, p2 in pairs:
        if extra_args:
            sig = sig_func(p1, p2, *extra_args)
        else:
            sig = sig_func(p1, p2)
        
        sig_key = tuple(sorted(str(k) + str(v) for k, v in sig.items()))
        
        if sig_key not in sig_to_type:
            type_id = type_id_counter
            type_id_counter += 1
            sig_to_type[sig_key] = {
                'type_id': type_id,
                'sig': sig,
                'representative': (p1, p2),
                'count': 0,
            }
        sig_to_type[sig_key]['count'] += 1
    
    return list(sig_to_type.values())

def print_report():
    tracker = DepthTracker(
        depth_1_atomic_species_complete=True,
        depth_2_ambient_product_universe_complete=True,
        depth_2_symmetry_layer_complete=True,
        depth_2_pair_relation_catalogs_complete=True,
        current_explicit_depth="9^2",
        next_intended_depth="composition / relation-of-relations",
        longterm_target="9^9 closure (long-term, not yet explicit)"
    )
    
    print("=" * 70)
    print("ADE3x3 Step 4: Depth Tracker & Pair Relation Catalogs")
    print("=" * 70)
    
    print("\n--- DEPTH TRACKER ---")
    print(f"Current explicit depth: {tracker.current_explicit_depth}")
    print(f"Next intended depth: {tracker.next_intended_depth}")
    print(f"Long-term target: {tracker.longterm_target}")
    print("\nCompleted sublayers:")
    print(f"  - depth_1_atomic_species_complete: {tracker.depth_1_atomic_species_complete}")
    print(f"  - depth_2_ambient_product_universe_complete: {tracker.depth_2_ambient_product_universe_complete}")
    print(f"  - depth_2_symmetry_layer_complete: {tracker.depth_2_symmetry_layer_complete}")
    print(f"  - depth_2_pair_relation_catalogs_complete: {tracker.depth_2_pair_relation_catalogs_complete}")
    
    print("\n--- SPECIES SIZES ---")
    print("|A| = 9 (left input basis)")
    print("|B| = 9 (right input basis)")  
    print("|C| = 9 (output basis)")
    print("|X| = 81 (ambient product)")
    
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    all_X = list(range(81))
    all_C = list(range(9))
    all_A = list(range(9))
    all_B = list(range(9))
    
    XX_pairs = [(x1, x2) for x1 in all_X for x2 in all_X]
    CX_pairs = [(c, x) for c in all_C for x in all_X]
    CC_pairs = [(c1, c2) for c1 in all_C for c2 in all_C]
    AX_pairs = [(a, x) for a in all_A for x in all_X]
    BX_pairs = [(b, x) for b in all_B for x in all_X]
    
    print("\n--- CATALOG SUMMARIES ---")
    
    print("\n[1] XX Catalog (X x X)")
    xx_types = build_catalog(XX_pairs, selection, signature_XX, (selection,))
    print(f"    Total ordered pairs: {len(XX_pairs)}")
    print(f"    Distinct relation types: {len(xx_types)}")
    print("    Sample types:")
    for t in xx_types[:3]:
        p1, p2 = t['representative']
        print(f"      Type {t['type_id']}: count={t['count']}, repr: X[...] -- X[...]")
    
    print("\n[2] CX Catalog (C x X)")
    cx_types = build_catalog(CX_pairs, selection, signature_CX, (selection,))
    print(f"    Total ordered pairs: {len(CX_pairs)}")
    print(f"    Distinct relation types: {len(cx_types)}")
    print("    Sample types:")
    for t in cx_types[:3]:
        c, x = t['representative']
        c_w = s1.w_decode(c)
        p = s1.p_decode(x)
        print(f"      Type {t['type_id']}: count={t['count']}, repr: C[{c_w.r},{c_w.u}] -- X[{p.left[0]},{p.left[1]}|{p.right[0]},{p.right[1]}]")
    
    print("\n[3] CC Catalog (C x C)")
    cc_types = build_catalog(CC_pairs, selection, signature_CC)
    print(f"    Total ordered pairs: {len(CC_pairs)}")
    print(f"    Distinct relation types: {len(cc_types)}")
    print("    Types:")
    for t in cc_types:
        c1, c2 = t['representative']
        w1 = s1.w_decode(c1)
        w2 = s1.w_decode(c2)
        print(f"      Type {t['type_id']}: count={t['count']}, repr: C[{w1.r},{w1.u}] -- C[{w2.r},{w2.u}]")
    
    print("\n[4] AX Catalog (A x X)")
    ax_types = build_catalog(AX_pairs, selection, signature_AX, (selection,))
    print(f"    Total ordered pairs: {len(AX_pairs)}")
    print(f"    Distinct relation types: {len(ax_types)}")
    print("    Sample types:")
    for t in ax_types[:3]:
        a, x = t['representative']
        a_u = s1.u_decode(a)
        p = s1.p_decode(x)
        print(f"      Type {t['type_id']}: count={t['count']}, repr: A[{a_u.r},{a_u.s}] -- X[{p.left[0]},{p.left[1]}|{p.right[0]},{p.right[1]}]")
    
    print("\n[5] BX Catalog (B x X)")
    bx_types = build_catalog(BX_pairs, selection, signature_BX, (selection,))
    print(f"    Total ordered pairs: {len(BX_pairs)}")
    print(f"    Distinct relation types: {len(bx_types)}")
    print("    Sample types:")
    for t in bx_types[:3]:
        b, x = t['representative']
        b_v = s1.v_decode(b)
        p = s1.p_decode(x)
        print(f"      Type {t['type_id']}: count={t['count']}, repr: B[{b_v.t},{b_v.u}] -- X[{p.left[0]},{p.left[1]}|{p.right[0]},{p.right[1]}]")
    
    print("\n--- INTERPRETATION ---")
    print("This step builds canonical typed pair-relation catalogs at depth 9^2.")
    print("Relation types are exact, discrete, and carry stable IDs.")
    print("Future depths will compose relations of relations, building upward.")
    print("9^9 long-term closure target is not yet explicit.")
    print("\n" + "=" * 70)

def main():
    print_report()

if __name__ == "__main__":
    main()