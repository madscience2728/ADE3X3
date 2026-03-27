import ade3x3_step1 as s1
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any
from collections import defaultdict

@dataclass
class DepthTracker:
    depth_1_atomic_species_complete: bool = True
    depth_2_ambient_product_universe_complete: bool = True
    depth_2_symmetry_layer_complete: bool = True
    depth_2_pair_relation_catalogs_complete: bool = True
    depth_3_first_composition_started: bool = True
    depth_3_first_composition_complete: bool = True
    depth_3_refinement_engine_started: bool = True
    current_explicit_depth: str = "9^3"
    next_intended_depth: str = "9^4 (apply refinements)"
    longterm_target: str = "9^9 closure"

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
    r, s_ = p.left
    t, u = p.right
    live = selection[x] >= 0
    target_c = selection[x]
    return {
        'x_live': live,
        'x_maps_to_c': (target_c == c) if live else False,
        'c_row_eq': c_w.r == r,
        'c_col_eq': c_w.u == u,
        'fiber_idx': s_ if live else None,
    }

def signature_XC(x, c, selection):
    p = s1.p_decode(x)
    c_w = s1.w_decode(c)
    r, s_ = p.left
    t, u = p.right
    live = selection[x] >= 0
    target_c = selection[x]
    return {
        'x_live': live,
        'x_maps_to_c': (target_c == c) if live else False,
        'c_row_eq': c_w.r == r,
        'c_col_eq': c_w.u == u,
        'fiber_idx': s_ if live else None,
    }

def signature_CC(c1, c2):
    w1 = s1.w_decode(c1)
    w2 = s1.w_decode(c2)
    return {
        'same': c1 == c2,
        'row_eq': w1.r == w2.r,
        'col_eq': w1.u == w2.u,
    }

def sig_to_key(sig):
    items = []
    for k, v in sorted(sig.items()):
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)

def build_catalog_simple(pairs, sig_func, selection):
    sig_to_type = {}
    type_id_counter = 0
    for p in pairs:
        sig = sig_func(p[0], p[1], selection)
        sig_key = sig_to_key(sig)
        if sig_key not in sig_to_type:
            sig_to_type[sig_key] = type_id_counter
            type_id_counter += 1
    return sig_to_type

def build_catalog_simple_no_sel(pairs, sig_func):
    sig_to_type = {}
    type_id_counter = 0
    for p in pairs:
        sig = sig_func(p[0], p[1])
        sig_key = sig_to_key(sig)
        if sig_key not in sig_to_type:
            sig_to_type[sig_key] = type_id_counter
            type_id_counter += 1
    return sig_to_type

def print_report():
    tracker = DepthTracker()
    
    print("=" * 70)
    print("ADE3x3 Step 6: Refinement Engine")
    print("=" * 70)
    
    print("\n--- DEPTH TRACKER ---")
    print(f"Current explicit depth: {tracker.current_explicit_depth}")
    print(f"Refinement engine started: {tracker.depth_3_refinement_engine_started}")
    
    _, _, _, p_table = s1.build_basis_tables()
    selection = s1.build_selection_map(p_table)
    
    all_C = list(range(9))
    all_X = list(range(81))
    
    CX_pairs = [(c, x) for c in all_C for x in all_X]
    XC_pairs = [(x, c) for x in all_X for c in all_C]
    CC_pairs = [(c1, c2) for c1 in all_C for c2 in all_C]
    
    CX_sig_to_type = build_catalog_simple(CX_pairs, signature_CX, selection)
    XC_sig_to_type = build_catalog_simple(XC_pairs, signature_XC, selection)
    CC_sig_to_type = build_catalog_simple_no_sel(CC_pairs, signature_CC)
    
    print("\n--- COMPOSITION RECAP ---")
    print("Testing: CX x XC -> CC")
    
    realized = {}
    for c1 in all_C:
        for x in all_X:
            for c2 in all_C:
                key_CX = sig_to_key(signature_CX(c1, x, selection))
                key_XC = sig_to_key(signature_XC(x, c2, selection))
                alpha = CX_sig_to_type[key_CX]
                beta = XC_sig_to_type[key_XC]
                
                key_CC = sig_to_key(signature_CC(c1, c2))
                gamma = CC_sig_to_type[key_CC]
                
                key = (alpha, beta)
                if key not in realized:
                    realized[key] = {'alpha': alpha, 'beta': beta, 'triple_count': 0, 'gamma_hist': {}, 'repr': (c1, x, c2)}
                realized[key]['triple_count'] += 1
                if gamma not in realized[key]['gamma_hist']:
                    realized[key]['gamma_hist'][gamma] = 0
                realized[key]['gamma_hist'][gamma] += 1
    
    deterministic = 0
    mixed = 0
    for key, entry in realized.items():
        if len(entry['gamma_hist']) == 1:
            deterministic += 1
        else:
            mixed += 1
    
    total_triples = len(all_C) * len(all_X) * len(all_C)
    print(f"Total triples: {total_triples}")
    print(f"Realized keys: {len(realized)}")
    print(f"Deterministic: {deterministic}")
    print(f"Mixed: {mixed}")
    
    print(f"\n--- MIXED KEY ANALYSIS ---")
    print(f"Number of mixed keys: {mixed}")
    
    mixed_keys_data = [(k, v) for k, v in realized.items() if len(v['gamma_hist']) > 1]
    
    witnesses = {}
    for key, entry in mixed_keys_data:
        alpha, beta = key
        witnesses[key] = []
        for c1 in all_C:
            for x in all_X:
                for c2 in all_C:
                    key_CX = sig_to_key(signature_CX(c1, x, selection))
                    key_XC = sig_to_key(signature_XC(x, c2, selection))
                    if CX_sig_to_type[key_CX] == alpha and XC_sig_to_type[key_XC] == beta:
                        key_CC = sig_to_key(signature_CC(c1, c2))
                        gamma = CC_sig_to_type[key_CC]
                        witnesses[key].append((c1, x, c2, gamma))
    
    def sep_c1_eq_c2(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        c2_w = s1.w_decode(u)
        return c1_w.r == c2_w.r and c1_w.u == c2_w.u
    
    def sep_r1_eq_r(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        p = s1.p_decode(t)
        return c1_w.r == p.left[0]
    
    def sep_r1_eq_r2(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        c2_w = s1.w_decode(u)
        return c1_w.r == c2_w.r
    
    def sep_u1_eq_u(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        p = s1.p_decode(t)
        return c1_w.u == p.right[1]
    
    def sep_u1_eq_u2(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        c2_w = s1.w_decode(u)
        return c1_w.u == c2_w.u
    
    def sep_row_pat(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        p = s1.p_decode(t)
        c2_w = s1.w_decode(u)
        return canonical_pattern([c1_w.r, p.left[0], c2_w.r])
    
    def sep_col_pat(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        p = s1.p_decode(t)
        c2_w = s1.w_decode(u)
        return canonical_pattern([c1_w.u, p.right[1], c2_w.u])
    
    def sep_x_live(d, ctx):
        s, t, u = d
        sel = ctx['selection']
        return sel[t] >= 0
    
    def sep_c1_equals_target(d, ctx):
        s, t, u = d
        c1_w = s1.w_decode(s)
        sel = ctx['selection']
        if sel[t] >= 0:
            target = sel[t]
            target_w = s1.w_decode(target)
            return c1_w.r == target_w.r and c1_w.u == target_w.u
        return False
    
    def sep_c2_equals_target(d, ctx):
        s, t, u = d
        c2_w = s1.w_decode(u)
        sel = ctx['selection']
        if sel[t] >= 0:
            target = sel[t]
            target_w = s1.w_decode(target)
            return c2_w.r == target_w.r and c2_w.u == target_w.u
        return False
    
    def sep_contracted(d, ctx):
        s, t, u = d
        p = s1.p_decode(t)
        return p.left[1] == p.right[0]
    
    separators = [
        ("c1_eq_c2", sep_c1_eq_c2),
        ("r1_eq_r", sep_r1_eq_r),
        ("r1_eq_r2", sep_r1_eq_r2),
        ("u1_eq_u", sep_u1_eq_u),
        ("u1_eq_u2", sep_u1_eq_u2),
        ("row_pat", sep_row_pat),
        ("col_pat", sep_col_pat),
        ("x_live", sep_x_live),
        ("c1_equals_target", sep_c1_equals_target),
        ("c2_equals_target", sep_c2_equals_target),
        ("contracted", sep_contracted),
    ]
    
    print("\n--- SINGLE SEPARATOR PERFORMANCE ---")
    ctx = {'selection': selection}
    
    for name, sep in separators:
        resolved = 0
        for key, wlist in witnesses.items():
            values = set()
            for s, t, u, gamma in wlist:
                val = sep((s, t, u), ctx)
                values.add(val)
            if len(values) == 1:
                resolved += 1
        print(f"  {name}: resolves {resolved} mixed keys")
    
    print("\n--- GREEDY CUMULATIVE SEARCH ---")
    resolved = set()
    current_combo = []
    remaining = len(witnesses)
    
    while remaining > 0:
        best_idx = None
        best_gain = 0
        
        for idx, (name, sep) in enumerate(separators):
            if idx in [i for i, _ in current_combo]:
                continue
            
            test_idx = current_combo + [(idx, name)]
            test_resolved = set()
            
            for key, wlist in witnesses.items():
                values = set()
                for s, t, u, gamma in wlist:
                    tuple_val = tuple(separators[i][1]((s, t, u), ctx) for i, _ in test_idx)
                    values.add(tuple_val)
                if len(values) == 1:
                    test_resolved.add(key)
            
            gain = len(test_resolved) - len(resolved)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx is None or best_gain == 0:
            break
        
        current_combo.append((best_idx, separators[best_idx][0]))
        
        test_resolved = set()
        for key, wlist in witnesses.items():
            values = set()
            for s, t, u, gamma in wlist:
                tuple_val = tuple(separators[i][1]((s, t, u), ctx) for i, _ in current_combo)
                values.add(tuple_val)
            if len(values) == 1:
                test_resolved.add(key)
        
        resolved = test_resolved
        remaining = len(witnesses) - len(resolved)
    
    greedy_resolved = len(resolved)
    print(f"Greedy resolved: {greedy_resolved}/{len(witnesses)}")
    print(f"Greedy combo: {[name for _, name in current_combo]}")
    
    print("\n--- TESTING SMALL TUPLES ---")
    full_resolved = 0
    best_tuple = None
    
    for i1 in range(len(separators)):
        for i2 in range(i1 + 1, len(separators)):
            for i3 in range(i2 + 1, len(separators)):
                combo = [separators[i1], separators[i2], separators[i3]]
                res = 0
                for key, wlist in witnesses.items():
                    values = set()
                    for s, t, u, gamma in wlist:
                        tuple_val = tuple(sep[1]((s, t, u), ctx) for sep in combo)
                        values.add(tuple_val)
                    if len(values) == 1:
                        res += 1
                if res > full_resolved:
                    full_resolved = res
                    best_tuple = [separators[i1][0], separators[i2][0], separators[i3][0]]
    
    print(f"Best 3-tuple resolves: {full_resolved}/{len(witnesses)}")
    if best_tuple:
        print(f"Best 3-tuple: {best_tuple}")
    
    two_full = 0
    best_two = None
    for i1 in range(len(separators)):
        for i2 in range(i1 + 1, len(separators)):
            combo = [separators[i1], separators[i2]]
            res = 0
            for key, wlist in witnesses.items():
                values = set()
                for s, t, u, gamma in wlist:
                    tuple_val = tuple(sep[1]((s, t, u), ctx) for sep in combo)
                    values.add(tuple_val)
                if len(values) == 1:
                    res += 1
            if res > two_full:
                two_full = res
                best_two = [separators[i1][0], separators[i2][0]]
    
    print(f"Best 2-tuple resolves: {two_full}/{len(witnesses)}")
    if best_two:
        print(f"Best 2-tuple: {best_two}")
    
    print("\n--- WITNESS EXAMPLES ---")
    for key, entry in mixed_keys_data[:3]:
        alpha, beta = key
        wlist = witnesses[key]
        gammas = list(set(g for _, _, _, g in wlist))
        print(f"Mixed key ({alpha}, {beta}):")
        print(f"  Witness gammas: {gammas}")
    
    print("\n--- INTERPRETATION ---")
    if full_resolved == len(witnesses):
        print("FULL RESOLUTION: All mixed keys CAN BE SEPARATED by small separator tuple")
        print(f"Minimal approach: Use tuple {best_tuple} or similar")
    elif greedy_resolved == len(witnesses):
        print("Greedy finds full separation")
    else:
        print(f"PARTIAL: Greedy resolves {greedy_resolved}/{len(witnesses)}, best 3-resolves {full_resolved}")
        print("The current separator family does NOT fully resolve all mixed keys")
    
    print("\n--- NEXT STEPS ---")
    print("Options:")
    print("1. Extend separator family with more coordinates")
    print("2. Note that current separators capture structural constraints")
    print("3. Investigate whether C-X-C truly requires more granularity")
    print("\n" + "=" * 70)

def main():
    print_report()

if __name__ == "__main__":
    main()