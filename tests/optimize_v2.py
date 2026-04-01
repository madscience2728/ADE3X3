"""Aggressive optimizer v2: 27-enriched lookup + support mutations + pair swaps.
Goal: break below 0.1 max_abs.
"""
import json, copy, time, sys, os
import numpy as np
from itertools import combinations

# ══════════════════════════════════════════════════════════════════════════════
# Target tensor
# ══════════════════════════════════════════════════════════════════════════════
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# 27-enriched algebraic lookup table
# ══════════════════════════════════════════════════════════════════════════════
def build_lookup():
    table = {}

    # ── Original families ──
    for x in range(0, 21):
        for y in range(1, 21):
            v = x / y
            if v > 3.5: continue
            table[f"{x}/{y}"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = np.sqrt(x / y)
            if v > 3.5: continue
            table[f"sqrt({x}/{y})"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** (1/3)
            if v > 3.5: continue
            table[f"cbrt({x}/{y})"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** 0.25
            if v > 3.5: continue
            table[f"4rt({x}/{y})"] = v

    for x in range(1, 11):
        for y in range(1, 11):
            v = (x / y) ** (1/6)
            if v > 3.5: continue
            table[f"6rt({x}/{y})"] = v

    # ── 27-family: the tensor norm is 27, so these are natural ──
    for y in range(1, 28):
        v = np.sqrt(27.0 / y)
        if v < 3.5:
            table[f"sqrt(27/{y})"] = v
        v = (27.0 / y) ** (1/3)
        if v < 3.5:
            table[f"cbrt(27/{y})"] = v
        v = (27.0 / y) ** 0.25
        if v < 3.5:
            table[f"4rt(27/{y})"] = v
        v = (27.0 / y) ** (1/6)
        if v < 3.5:
            table[f"6rt(27/{y})"] = v

    # 27 = 3^3, so also 3^(k/n) forms
    for k in range(-6, 7):
        for n in [2, 3, 4, 6]:
            v = 3.0 ** (k / n)
            if 0.01 < v < 3.5:
                table[f"3^({k}/{n})"] = v

    # √(27/m) * simple_fraction
    for m in [1, 2, 3, 4, 6, 9, 12, 18, 27]:
        base = np.sqrt(27.0 / m)
        for p in range(1, 4):
            for q in range(1, 4):
                v = base * p / q
                if 0.01 < v < 3.5:
                    table[f"{p}/{q}*sqrt(27/{m})"] = v

    # ── Products: sqrt * cbrt ──
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                for d in range(1, 6):
                    v = np.sqrt(a/b) * ((c/d)**(1/3))
                    if 0.01 < v < 3.0:
                        table[f"sqrt({a}/{b})*cbrt({c}/{d})"] = v

    # ── Products: rational * sqrt ──
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = np.sqrt(a/b) * (c/d)
                    if 0.01 < v < 3.0:
                        table[f"{c}/{d}*sqrt({a}/{b})"] = v

    # ── Products: rational * cbrt ──
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = ((a/b)**(1/3)) * (c/d)
                    if 0.01 < v < 3.0:
                        table[f"{c}/{d}*cbrt({a}/{b})"] = v

    # ── 2^a * 3^b powers ──
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in [2, 3, 4, 6]:
                v = abs((2**a) * (3**b)) ** (1/c)
                if 0.01 < v < 3.0:
                    table[f"(2^{a}*3^{b})^(1/{c})"] = v

    # ── 27-specific: cbrt(27) * rational, sqrt(3) * rational ──
    for p in range(1, 10):
        for q in range(1, 10):
            v = 3.0 * p / q  # cbrt(27) = 3
            if 0.01 < v < 3.5:
                table[f"3*{p}/{q}"] = v
            v = np.sqrt(3.0) * p / q
            if 0.01 < v < 3.5:
                table[f"sqrt3*{p}/{q}"] = v
            v = 3.0**(2/3) * p / q  # 27^(2/9)... no, 3^(2/3)
            if 0.01 < v < 3.5:
                table[f"3^(2/3)*{p}/{q}"] = v

    table["0"] = 0.0

    # Deduplicate
    deduped = {}
    for name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
        rounded = round(val, 8)
        if rounded not in deduped:
            deduped[rounded] = (name, val)
        elif len(name) < len(deduped[rounded][0]):
            deduped[rounded] = (name, val)

    lookup = sorted(deduped.values(), key=lambda x: x[1])
    return np.array([v for _, v in lookup]), [n for n, _ in lookup]

lookup_vals, lookup_names = build_lookup()
print(f"Lookup table: {len(lookup_vals)} algebraic values")

# ══════════════════════════════════════════════════════════════════════════════
# I/O + Eval
# ══════════════════════════════════════════════════════════════════════════════
def load_candidate(path):
    with open(path) as f:
        return json.load(f)

def save_candidate(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def reconstruct(data):
    out = np.zeros((9, 9, 9))
    for term in data['terms']:
        a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
        for i, v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
        for i, v in zip(term['beta_support'], term['beta_values']): b[i] = v
        for i, v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
        out += np.einsum('c,a,b->cab', g, a, b)
    return out

def evaluate(data):
    R = reconstruct(data) - T
    return np.max(np.abs(R)), np.linalg.norm(R)

def build_term_tensor(term):
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i, v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i, v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i, v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    return np.einsum('c,a,b->cab', g, a, b)

def nearest_algebraic(absval, k=8):
    idx = np.searchsorted(lookup_vals, absval)
    candidates = []
    for i in range(max(0, idx - k), min(len(lookup_vals), idx + k + 1)):
        candidates.append((abs(absval - lookup_vals[i]), lookup_vals[i], lookup_names[i]))
    candidates.sort()
    return candidates[:k]

# ══════════════════════════════════════════════════════════════════════════════
# Move generators
# ══════════════════════════════════════════════════════════════════════════════

# ── Move 1: Single-coefficient tweak (same as v1) ──
def gen_single_coeff_trials(val, n_nearby=8, fine_steps=5, fine_range=0.003):
    trials = set()
    sign = 1.0 if val >= 0 else -1.0
    absv = abs(val)
    neighbors = nearest_algebraic(absv, k=n_nearby)
    for _, aval, _ in neighbors:
        trials.add(sign * aval)
        if absv < 0.3:
            trials.add(-sign * aval)
    for delta in np.linspace(-fine_range, fine_range, fine_steps * 2 + 1):
        if delta != 0:
            trials.add(val + delta)
    for _, aval, _ in neighbors[:3]:
        for delta in np.linspace(-fine_range/2, fine_range/2, fine_steps):
            trials.add(sign * aval + delta)
    trials.discard(val)
    return sorted(trials)

# ── Move 2: Support mutation (add or remove an index) ──
AXIS_MAP = {
    'alpha_values': 'alpha_support',
    'beta_values': 'beta_support', 
    'gamma_values': 'gamma_support',
}

def try_support_mutations(data, R, best_mx, verbose=False):
    """Try adding/removing support indices. Returns (improved, new_R, new_mx, desc)."""
    accepts = 0
    for ti, term in enumerate(data['terms']):
        for vkey, skey in AXIS_MAP.items():
            current_support = list(term[skey])
            current_values = list(term[vkey])
            
            # Try ADDING each missing index with small algebraic values
            all_idx = set(range(9))
            used = set(current_support)
            missing = all_idx - used
            
            old_tt = build_term_tensor(term)
            
            for idx in missing:
                # Try a few seed values for the new index
                for seed_val in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 
                                 1/np.sqrt(2), -1/np.sqrt(2),
                                 np.sqrt(3)/3, -np.sqrt(3)/3,
                                 1/3, -1/3]:
                    # Temporarily add
                    term[skey] = current_support + [idx]
                    term[vkey] = current_values + [seed_val]
                    new_tt = build_term_tensor(term)
                    R_new = R - old_tt + new_tt
                    new_mx = np.max(np.abs(R_new))
                    
                    if new_mx < best_mx - 1e-10:
                        R = R_new
                        best_mx = new_mx
                        current_support = list(term[skey])
                        current_values = list(term[vkey])
                        old_tt = new_tt
                        accepts += 1
                        if verbose:
                            print(f"  +SUPPORT t{ti+1} {vkey[0]}[{idx}]={seed_val:+.4f}  "
                                  f"max_abs={best_mx:.10f}")
                        break  # accept first improvement for this idx
                    else:
                        # revert
                        term[skey] = current_support
                        term[vkey] = current_values
            
            # Try REMOVING each existing index (if value is small)
            for pi in range(len(current_support) - 1, -1, -1):
                if abs(current_values[pi]) > 0.5:
                    continue  # don't remove large coefficients
                old_tt = build_term_tensor(term)
                removed_idx = current_support[pi]
                removed_val = current_values[pi]
                
                term[skey] = current_support[:pi] + current_support[pi+1:]
                term[vkey] = current_values[:pi] + current_values[pi+1:]
                new_tt = build_term_tensor(term)
                R_new = R - old_tt + new_tt
                new_mx = np.max(np.abs(R_new))
                
                if new_mx < best_mx - 1e-10:
                    R = R_new
                    best_mx = new_mx
                    current_support = list(term[skey])
                    current_values = list(term[vkey])
                    accepts += 1
                    if verbose:
                        print(f"  -SUPPORT t{ti+1} {vkey[0]}[{removed_idx}] (was {removed_val:+.4f})  "
                              f"max_abs={best_mx:.10f}")
                else:
                    term[skey] = current_support
                    term[vkey] = current_values
    
    return accepts, R, best_mx

# ── Move 3: Pair-coefficient swap within a term (capped for speed) ──
def try_pair_swaps(data, R, best_mx, fine_range=0.003, max_pairs_per_axis=6, verbose=False):
    """Try moving two coefficients in the same term simultaneously.
    Only tries top-K largest coefficients to limit combinatorics."""
    accepts = 0
    deltas = np.array([-fine_range*3, -fine_range, fine_range, fine_range*3])
    
    for ti, term in enumerate(data['terms']):
        for vkey in ['alpha_values', 'beta_values', 'gamma_values']:
            vals = term[vkey]
            if len(vals) < 2:
                continue
            
            old_tt = build_term_tensor(term)
            
            # Only try pairs among top-K largest coefficients
            ranked = sorted(range(len(vals)), key=lambda k: abs(vals[k]), reverse=True)
            top = ranked[:max_pairs_per_axis]
            
            for i, j in combinations(top, 2):
                v_i, v_j = vals[i], vals[j]
                best_pair = None
                pair_best_mx = best_mx
                
                for di in deltas:
                    for dj in deltas:
                        vals[i] = v_i + di
                        vals[j] = v_j + dj
                        new_tt = build_term_tensor(term)
                        R_new = R - old_tt + new_tt
                        new_mx = np.max(np.abs(R_new))
                        if new_mx < pair_best_mx - 1e-12:
                            best_pair = (v_i + di, v_j + dj)
                            pair_best_mx = new_mx
                
                if best_pair is not None:
                    vals[i], vals[j] = best_pair
                    new_tt = build_term_tensor(term)
                    R = R - old_tt + new_tt
                    best_mx = pair_best_mx
                    old_tt = new_tt
                    accepts += 1
                    if verbose:
                        print(f"  PAIR t{ti+1} {vkey[0]}[{i},{j}]  max_abs={best_mx:.10f}")
                else:
                    vals[i], vals[j] = v_i, v_j
    
    return accepts, R, best_mx

# ── Move 4: Cross-axis pair (top-K only for speed) ──
def try_cross_axis_pairs(data, R, best_mx, fine_range=0.003, max_per_axis=4, verbose=False):
    """Try moving one coeff from one axis + one from another in same term.
    Only top-K largest per axis to limit combinatorics."""
    accepts = 0
    axes = ['alpha_values', 'beta_values', 'gamma_values']
    deltas = np.array([-fine_range*3, -fine_range, fine_range, fine_range*3])
    
    for ti, term in enumerate(data['terms']):
        old_tt = build_term_tensor(term)
        
        for ax1, ax2 in combinations(range(3), 2):
            vkey1, vkey2 = axes[ax1], axes[ax2]
            vals1, vals2 = term[vkey1], term[vkey2]
            
            top1 = sorted(range(len(vals1)), key=lambda k: abs(vals1[k]), reverse=True)[:max_per_axis]
            top2 = sorted(range(len(vals2)), key=lambda k: abs(vals2[k]), reverse=True)[:max_per_axis]
            
            for i in top1:
                for j in top2:
                    v1, v2 = vals1[i], vals2[j]
                    best_cross = None
                    cross_best_mx = best_mx
                    
                    for d1 in deltas:
                        for d2 in deltas:
                            vals1[i] = v1 + d1
                            vals2[j] = v2 + d2
                            new_tt = build_term_tensor(term)
                            R_new = R - old_tt + new_tt
                            new_mx = np.max(np.abs(R_new))
                            if new_mx < cross_best_mx - 1e-12:
                                best_cross = (v1 + d1, v2 + d2)
                                cross_best_mx = new_mx
                    
                    if best_cross is not None:
                        vals1[i], vals2[j] = best_cross
                        new_tt = build_term_tensor(term)
                        R = R - old_tt + new_tt
                        best_mx = cross_best_mx
                        old_tt = new_tt
                        accepts += 1
                        if verbose:
                            print(f"  CROSS t{ti+1} {vkey1[0]}[{i}]×{vkey2[0]}[{j}]  "
                                  f"max_abs={best_mx:.10f}")
                    else:
                        vals1[i], vals2[j] = v1, v2
    
    return accepts, R, best_mx

# ══════════════════════════════════════════════════════════════════════════════
# Main optimizer: interleave all move types
# ══════════════════════════════════════════════════════════════════════════════
def optimize_v2(input_path, output_path, max_sweeps=200, patience=5, fine_range=0.003):
    data = load_candidate(input_path)
    best_mx, best_fro = evaluate(data)
    print(f"\nStarting: max_abs = {best_mx:.10f}  fro = {best_fro:.8f}")
    print(f"Moves: single-coeff + support-mut + pair-swap + cross-axis")
    print(f"{'='*90}")

    R = reconstruct(data) - T
    stale = 0
    total_accepts = 0
    t0 = time.time()
    local_range = fine_range

    for sweep in range(1, max_sweeps + 1):
        sweep_accepts = 0

        # ─── Phase A: single-coefficient descent (shuffled) ───
        coeff_idx = []
        for ti, term in enumerate(data['terms']):
            for axis in ['alpha_values', 'beta_values', 'gamma_values']:
                for pi, v in enumerate(term[axis]):
                    coeff_idx.append((ti, axis, pi, v))
        
        order = np.random.permutation(len(coeff_idx))
        for ci in order:
            ti, axis, pi, old_val = coeff_idx[ci]
            term = data['terms'][ti]
            old_tt = build_term_tensor(term)
            trials = gen_single_coeff_trials(old_val, fine_range=local_range)
            
            bt = None
            bt_mx = best_mx
            for tv in trials:
                term[axis][pi] = tv
                new_tt = build_term_tensor(term)
                R_new = R - old_tt + new_tt
                new_mx = np.max(np.abs(R_new))
                if new_mx < bt_mx - 1e-12:
                    bt = tv
                    bt_mx = new_mx
            
            if bt is not None:
                term[axis][pi] = bt
                new_tt = build_term_tensor(term)
                R = R - old_tt + new_tt
                best_mx = bt_mx
                sweep_accepts += 1
            else:
                term[axis][pi] = old_val

        # ─── Phase B: support mutations ───
        s_acc, R, best_mx = try_support_mutations(data, R, best_mx, verbose=True)
        sweep_accepts += s_acc

        # ─── Phase C: pair swaps (within-axis) ───
        p_acc, R, best_mx = try_pair_swaps(data, R, best_mx, fine_range=local_range, verbose=True)
        sweep_accepts += p_acc

        # ─── Phase D: cross-axis pairs (every 3rd sweep to save time) ───
        if sweep % 3 == 0:
            x_acc, R, best_mx = try_cross_axis_pairs(data, R, best_mx, fine_range=local_range, verbose=True)
            sweep_accepts += x_acc
        else:
            x_acc = 0

        elapsed = time.time() - t0
        total_accepts += sweep_accepts
        print(f"  Sweep {sweep}: single={sweep_accepts - s_acc - p_acc - x_acc} "
              f"support={s_acc} pair={p_acc} cross={x_acc} "
              f"| max_abs={best_mx:.10f} | {elapsed:.1f}s")

        save_candidate(data, output_path)

        if sweep_accepts == 0:
            stale += 1
            if stale >= patience:
                print(f"\n  Converged after {sweep} sweeps.")
                break
            local_range *= 1.5
            print(f"  Widening: range={local_range:.4f}")
        else:
            stale = 0

    final_mx, final_fro = evaluate(data)
    elapsed = time.time() - t0
    print(f"\n{'='*90}")
    print(f"FINAL: max_abs={final_mx:.10f}  fro={final_fro:.8f}  "
          f"accepts={total_accepts}  time={elapsed:.1f}s")
    return data, final_mx

# ══════════════════════════════════════════════════════════════════════════════
# Parallel wrapper
# ══════════════════════════════════════════════════════════════════════════════
def _worker_v2(args_tuple):
    (input_path, output_path, seed, max_sweeps, patience, fine_range, wid) = args_tuple
    np.random.seed(seed)
    
    data = load_candidate(input_path)
    best_mx, _ = evaluate(data)
    R = reconstruct(data) - T
    stale = 0
    total_acc = 0
    t0 = time.time()
    local_range = fine_range

    for sweep in range(1, max_sweeps + 1):
        sweep_acc = 0

        # Phase A: single-coeff
        coeff_idx = []
        for ti, term in enumerate(data['terms']):
            for axis in ['alpha_values', 'beta_values', 'gamma_values']:
                for pi, v in enumerate(term[axis]):
                    coeff_idx.append((ti, axis, pi, v))
        
        order = np.random.permutation(len(coeff_idx))
        for ci in order:
            ti, axis, pi, old_val = coeff_idx[ci]
            term = data['terms'][ti]
            old_tt = build_term_tensor(term)
            trials = gen_single_coeff_trials(old_val, fine_range=local_range)
            bt = None; bt_mx = best_mx
            for tv in trials:
                term[axis][pi] = tv
                new_tt = build_term_tensor(term)
                R_new = R - old_tt + new_tt
                new_mx = np.max(np.abs(R_new))
                if new_mx < bt_mx - 1e-12:
                    bt = tv; bt_mx = new_mx
            if bt is not None:
                term[axis][pi] = bt
                new_tt = build_term_tensor(term)
                R = R - old_tt + new_tt
                best_mx = bt_mx; sweep_acc += 1
            else:
                term[axis][pi] = old_val

        # Phase B: support mutations
        s_acc, R, best_mx = try_support_mutations(data, R, best_mx)
        sweep_acc += s_acc

        # Phase C: pair swaps
        p_acc, R, best_mx = try_pair_swaps(data, R, best_mx, fine_range=local_range)
        sweep_acc += p_acc

        # Phase D: cross-axis (every 5th)
        if sweep % 5 == 0:
            x_acc, R, best_mx = try_cross_axis_pairs(data, R, best_mx, fine_range=local_range)
            sweep_acc += x_acc

        total_acc += sweep_acc
        if sweep_acc == 0:
            stale += 1
            if stale >= patience:
                break
            local_range *= 1.5
        else:
            stale = 0

    elapsed = time.time() - t0
    save_candidate(data, output_path)
    final_mx, final_fro = evaluate(data)
    return (wid, seed, final_mx, final_fro, total_acc, sweep, elapsed, output_path)

def run_parallel_v2(input_path, final_output, n_workers=24, max_sweeps=200,
                    patience=5, fine_range=0.003):
    import multiprocessing as mp
    os.makedirs('optim_workers', exist_ok=True)

    tasks = []
    for i in range(n_workers):
        seed = 2000 + i * 171
        out_path = f'optim_workers/v2_worker_{i:02d}.json'
        tasks.append((input_path, out_path, seed, max_sweeps, patience, fine_range, i))

    data0 = load_candidate(input_path)
    mx0, _ = evaluate(data0)
    print(f"Baseline max_abs = {mx0:.10f}")
    print(f"Launching {n_workers} v2 workers (sweeps={max_sweeps}, patience={patience}, range={fine_range})")
    print(f"{'='*90}")

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_v2, tasks)
    total_elapsed = time.time() - t0

    results.sort(key=lambda x: x[2])
    print(f"\n{'='*90}")
    print(f"ALL WORKERS COMPLETE  ({total_elapsed:.1f}s wall-clock)")
    print(f"{'='*90}")
    print(f"{'Worker':>7} {'Seed':>6} {'max_abs':>14} {'fro':>12} {'Accepts':>8} {'Sweeps':>7} {'Time':>7}")
    print(f"{'─'*70}")
    for wid, seed, mx, fro, acc, sw, el, path in results:
        flag = " ★" if mx == results[0][2] else ""
        print(f"  W{wid:02d}   {seed:6d}  {mx:14.10f}  {fro:12.8f}  {acc:6d}  {sw:6d}  {el:6.1f}s{flag}")

    best = results[0]
    import shutil
    shutil.copy2(best[7], final_output)
    print(f"\nBEST: W{best[0]} (seed={best[1]})")
    print(f"  max_abs = {best[2]:.10f}  (baseline: {mx0:.10f}, Δ = {best[2] - mx0:+.10f})")
    print(f"  saved to {final_output}")
    return best[2]

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='optimized_candidate_r7.json')
    parser.add_argument('--output', default='optimized_v2.json')
    parser.add_argument('--sweeps', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--fine-range', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--parallel', type=int, default=0)
    args = parser.parse_args()

    if args.parallel > 0:
        run_parallel_v2(args.input, args.output,
                        n_workers=args.parallel,
                        max_sweeps=args.sweeps, patience=args.patience,
                        fine_range=args.fine_range)
    else:
        np.random.seed(args.seed)
        optimize_v2(args.input, args.output,
                    max_sweeps=args.sweeps, patience=args.patience,
                    fine_range=args.fine_range)
