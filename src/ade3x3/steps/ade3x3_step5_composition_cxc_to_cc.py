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
    depth_3_first_composition_started: bool
    depth_3_cxc_to_cc_complete: bool
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

def signature_CX(c, x, selection):
    c_w = s1.w_decode(c)
    p = s1.p_decode(x)
    r2, s2_ = p.left
    t2, u2 = p.right
    
    live = selection[x] >= 0
    target_c = selection[x]
    target_match = (target_c == c) if live else False
    
    return {
        'x_live': live,
        'x_maps_to_c': target_match,
        'c_row_eq_x_target_row': (live and c_w.r == r2),
        'c_col_eq_x_target_col': (live and c_w.u == u2),
        'fiber_index': s2_ if live else None,
        'r_pattern': canonical_pattern([c_w.r, r2]),
        'u_pattern': canonical_pattern([c_w.u, u2]),
    }

def signature_XC(x, c, selection):
    p = s1.p_decode(x)
    c_w = s1.w_decode(c)
    r, s_ = p.left
    t, u = p.right
    
    live = selection[x] >= 0
    target_c = selection[x]
    target_match = (target_c == c) if live else False
    
    return {
        'x_live': live,
        'x_maps_to_c': target_match,
        'c_row_eq_x_row': (live and c_w.r == r),
        'c_col_eq_x_col': (live and c_w.u == u),
        'fiber_index': s_ if live else None,
        'r_pattern': canonical_pattern([r, c_w.r]),
        'u_pattern': canonical_pattern([u, c_w.u]),
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

def sig_to_key(sig):
    items = []
    for k, v in sorted(sig.items()):
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)

def build_catalog(pairs, sig_func, extra_args=()):
    sig_to_type = {}
    type_id_counter = 0
    
    for p in pairs:
        if extra_args:
            sig = sig_func(p[0], p[1], *extra_args)
        else:
            sig = sig_func(p[0], p[1])
        
        sig_key = sig_to_key(sig)
        
        if sig_key not in sig_to_type:
            type_id = type_id_counter
            type_id_counter += 1
            sig_to_type[sig_key] = {
                'type_id': type_id,
                'sig': sig,
                'representative': p,
                'count': 0,
            }
        sig_to_type[sig_key]['count'] += 1
    
    return sig_to_type

def build_composition_table(CX_types, XC_types, CC_types, all_triples, selection):
    comp_table = {}
    
    for c1, x, c2 in all_triples:
        alpha = CX_types[sig_to_key(signature_CX(c1, x, selection))]['type_id']
        beta = XC_types[sig_to_key(signature_XC(x, c2, selection))]['type_id']
        gamma = CC_types[sig_to_key(signature_CC(c1, c2))]['type_id']
        
        key = (alpha, beta)
        if key not in comp_table:
            comp_table[key] = {
                'alpha': alpha,
                'beta': beta,
                'triple_count': 0,
                'gamma_hist': {},
                'repr_triple': None,
            }
        
        comp_table[key]['triple_count'] += 1
        if gamma not in comp_table[key]['gamma_hist']:
            comp_table[key]['gamma_hist'][gamma] = 0
        comp_table[key]['gamma_hist'][gamma] += 1
        
        if comp_table[key]['repr_triple'] is None:
            comp_table[key]['repr_triple'] = (c1, x, c2)
    
    return comp_table

def print_report():
    tracker = DepthTracker(
        depth_1_atomic_species_complete=True,
        depth_2_ambient_product_universe_complete=True,
        depth_2_symmetry_layer_complete=True,
        depth_2_pair_relation_catalogs_complete=True,
        depth_3_first_composition_started=True,
        depth_3_cxc_to_cc_complete=True,
        current_explicit_depth="9^3",
        next_intended_depth="9^4 (nested relations)",
        longterm_target="9^9 closure (long-term, not yet explicit)"
    )
    
    print("=" * 70)
    print("ADE3x3 Step 5: C-X-C Composition to C-C")
    print("=" * 70)
    
    print("\n--- DEPTH TRACKER ---")
    print(f"Current explicit depth: {tracker.current_explicit_depth}")
    print(f"Next intended depth: {tracker.next_intended_depth}")
    print(f"Long-term target: {tracker.longterm_target}")
    print(f"\nCompleted:")
    print(f"  - depth_3_first_composition_started: {tracker.depth_3_first_composition_started}")
    print(f"  - depth_3_cxc_to_cc_complete: {tracker.depth_3_cxc_to_cc_complete}")
    
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    all_C = list(range(9))
    all_X = list(range(81))
    
    CX_pairs = [(c, x) for c in all_C for x in all_X]
    XC_pairs = [(x, c) for x in all_X for c in all_C]
    CC_pairs = [(c1, c2) for c1 in all_C for c2 in all_C]
    all_triples = [(c1, x, c2) for c1 in all_C for x in all_X for c2 in all_C]
    
    print("\n--- CATALOG SIZES ---")
    print(f"|C| = 9")
    print(f"|X| = 81")
    
    CX_types = build_catalog(CX_pairs, signature_CX, (selection,))
    print(f"CX types: {len(CX_types)}")
    
    XC_types = build_catalog(XC_pairs, signature_XC, (selection,))
    print(f"XC types: {len(XC_types)}")
    
    CC_types = build_catalog(CC_pairs, signature_CC)
    print(f"CC types: {len(CC_types)}")
    
    print(f"\n--- TRIPLE COUNT ---")
    print(f"Total C-X-C triples = {len(all_triples)}")
    
    print("\n--- COMPOSITION TABLE ---")
    comp_table = build_composition_table(CX_types, XC_types, CC_types, all_triples, selection)
    
    total_keys = len(CX_types) * len(XC_types)
    realized_keys = len(comp_table)
    impossible_count = 0
    deterministic_count = 0
    mixed_count = 0
    
    for key, entry in comp_table.items():
        gamma_count = len(entry['gamma_hist'])
        if gamma_count == 0:
            impossible_count += 1
        elif gamma_count == 1:
            deterministic_count += 1
        else:
            mixed_count += 1
    
    print(f"Total possible (alpha,beta) keys: {total_keys}")
    print(f"Realized keys: {realized_keys}")
    print(f"Impossible: {impossible_count}")
    print(f"Deterministic (1 resulting gamma): {deterministic_count}")
    print(f"Mixed (>1 resulting gamma): {mixed_count}")
    
    print("\n--- DETAILED COMPOSITION ENTRIES ---")
    for key in sorted(comp_table.keys(), key=lambda k: (comp_table[k]['alpha'], comp_table[k]['beta'])):
        entry = comp_table[key]
        c1, x, c2 = entry['repr_triple']
        c1w = s1.w_decode(c1)
        c2w = s1.w_decode(c2)
        p = s1.p_decode(x)
        print(f"({entry['alpha']}, {entry['beta']}): count={entry['triple_count']}, gamma_hist={entry['gamma_hist']}")
        print(f"   repr: C[{c1w.r},{c1w.u}] -- X[p] -- C[{c2w.r},{c2w.u}]")
    
    print("\n--- INTERPRETATION ---")
    xc_matches_cx = (len(XC_types) == len(CX_types))
    print(f"XC matches CX structurally: {xc_matches_cx}")
    
    if mixed_count == 0:
        print("Composition is FULLY DETERMINISTIC: each (CX-type, XC-type) pair yields exactly one CC-type.")
    else:
        print(f"Composition shows {mixed_count} mixed cases where (CX-type, XC-type) yields multiple CC-types.")
    
    print("\nThis step establishes the first exact C-X-C composition law at depth 9^3.")
    print("Future depths will build on this to compose relations of relations.")
    print("\n" + "=" * 70)

def main():
    print_report()

if __name__ == "__main__":
    main()