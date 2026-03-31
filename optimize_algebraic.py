"""Algebraic coordinate descent optimizer.
Start from best candidate. For each coefficient:
  - Try nearby algebraic grid values (± small perturbations)
  - Accept if max_abs improves
  - Repeat sweeps until converged.
"""
import json, copy, time, sys
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Target tensor
# ══════════════════════════════════════════════════════════════════════════════
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# Algebraic lookup table
# ══════════════════════════════════════════════════════════════════════════════
def build_lookup():
    table = {}
    for x in range(0, 21):
        for y in range(1, 21):
            v = x / y
            if v > 3.0: continue
            table[f"{x}/{y}"] = v
    for x in range(1, 21):
        for y in range(1, 21):
            v = np.sqrt(x / y)
            if v > 3.0: continue
            table[f"sqrt({x}/{y})"] = v
    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** (1/3)
            if v > 3.0: continue
            table[f"cbrt({x}/{y})"] = v
    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** 0.25
            if v > 3.0: continue
            table[f"4rt({x}/{y})"] = v
    for x in range(1, 11):
        for y in range(1, 11):
            v = (x / y) ** (1/6)
            if v > 3.0: continue
            table[f"6rt({x}/{y})"] = v
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                for d in range(1, 6):
                    v = np.sqrt(a/b) * ((c/d)**(1/3))
                    if 0.01 < v < 2.5:
                        table[f"sqrt({a}/{b})*cbrt({c}/{d})"] = v
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = np.sqrt(a/b) * (c/d)
                    if 0.01 < v < 2.5:
                        table[f"{c}/{d}*sqrt({a}/{b})"] = v
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = ((a/b)**(1/3)) * (c/d)
                    if 0.01 < v < 2.5:
                        table[f"{c}/{d}*cbrt({a}/{b})"] = v
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in [2, 3, 6]:
                v = abs((2**a) * (3**b)) ** (1/c)
                if 0.01 < v < 2.5:
                    table[f"(2^{a}*3^{b})^(1/{c})"] = v
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
# Candidate I/O
# ══════════════════════════════════════════════════════════════════════════════
def load_candidate(path):
    with open(path) as f:
        return json.load(f)

def save_candidate(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# Fast evaluation
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct(data):
    """Build 9x9x9 tensor from CP terms."""
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

# ══════════════════════════════════════════════════════════════════════════════
# Incremental evaluation (fast path)
# ══════════════════════════════════════════════════════════════════════════════
def build_term_tensor(term):
    """Build the rank-1 tensor for a single term."""
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i, v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i, v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i, v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    return np.einsum('c,a,b->cab', g, a, b)

def eval_with_residual(R_current):
    """Evaluate from precomputed residual."""
    return np.max(np.abs(R_current)), np.linalg.norm(R_current)

# ══════════════════════════════════════════════════════════════════════════════
# Find K nearest algebraic values to a given magnitude
# ══════════════════════════════════════════════════════════════════════════════
def nearest_algebraic(absval, k=5):
    """Return up to k nearest algebraic lookup values to |absval|."""
    idx = np.searchsorted(lookup_vals, absval)
    candidates = []
    for i in range(max(0, idx - k), min(len(lookup_vals), idx + k + 1)):
        candidates.append((abs(absval - lookup_vals[i]), lookup_vals[i], lookup_names[i]))
    candidates.sort()
    return candidates[:k]

# ══════════════════════════════════════════════════════════════════════════════
# Generate trial values for a coefficient
# ══════════════════════════════════════════════════════════════════════════════
def trial_values(current_val, n_nearby=6, fine_steps=5, fine_range=0.005):
    """Generate candidate replacement values:
    1) Nearby algebraic grid values (both signs)
    2) Fine grid around current value
    3) Fine grid around each algebraic neighbor
    """
    trials = set()
    sign = 1.0 if current_val >= 0 else -1.0
    absv = abs(current_val)

    # Nearby algebraic values
    neighbors = nearest_algebraic(absv, k=n_nearby)
    for _, aval, _ in neighbors:
        trials.add(sign * aval)
        # Also try the opposite sign for small values
        if absv < 0.2:
            trials.add(-sign * aval)

    # Fine grid around current
    for delta in np.linspace(-fine_range, fine_range, fine_steps * 2 + 1):
        if delta == 0:
            continue
        trials.add(current_val + delta)

    # Fine grid around each algebraic neighbor
    for _, aval, _ in neighbors[:3]:
        for delta in np.linspace(-fine_range/2, fine_range/2, fine_steps):
            trials.add(sign * aval + delta)

    # Remove current value
    trials.discard(current_val)
    return sorted(trials)

# ══════════════════════════════════════════════════════════════════════════════
# Build flat index of all mutable coefficients
# ══════════════════════════════════════════════════════════════════════════════
def build_coeff_index(data):
    """Return list of (term_idx, axis_key, position, current_value)."""
    index = []
    axes = ['alpha_values', 'beta_values', 'gamma_values']
    for ti, term in enumerate(data['terms']):
        for axis in axes:
            for pi, v in enumerate(term[axis]):
                index.append((ti, axis, pi, v))
    return index

# ══════════════════════════════════════════════════════════════════════════════
# Main optimizer
# ══════════════════════════════════════════════════════════════════════════════
def optimize(input_path, output_path, max_sweeps=50, patience=3,
             n_nearby=6, fine_steps=5, fine_range=0.005):
    data = load_candidate(input_path)
    best_mx, best_fro = evaluate(data)
    print(f"\nStarting: max_abs = {best_mx:.10f}  fro = {best_fro:.8f}")
    print(f"Config: max_sweeps={max_sweeps}, patience={patience}, "
          f"n_nearby={n_nearby}, fine_steps={fine_steps}, fine_range={fine_range}")
    print(f"{'='*90}")

    # Precompute residual for incremental updates
    R = reconstruct(data) - T

    stale_sweeps = 0
    total_accepts = 0
    total_evals = 0
    t0 = time.time()

    for sweep in range(1, max_sweeps + 1):
        sweep_accepts = 0
        sweep_best_delta = 0.0
        coeff_idx = build_coeff_index(data)

        # Shuffle to avoid systematic bias
        order = np.random.permutation(len(coeff_idx))

        for ci in order:
            ti, axis, pi, old_val = coeff_idx[ci]
            term = data['terms'][ti]

            # Build current rank-1 contribution
            old_term_tensor = build_term_tensor(term)

            # Generate trial values
            trials = trial_values(old_val, n_nearby, fine_steps, fine_range)
            if not trials:
                continue

            best_trial = None
            best_trial_mx = best_mx

            for tv in trials:
                # Temporarily set new value
                term[axis][pi] = tv
                new_term_tensor = build_term_tensor(term)

                # Incremental residual update
                R_new = R - old_term_tensor + new_term_tensor
                new_mx = np.max(np.abs(R_new))
                total_evals += 1

                if new_mx < best_trial_mx - 1e-12:
                    best_trial = tv
                    best_trial_mx = new_mx

            if best_trial is not None:
                # Accept the best trial
                term[axis][pi] = best_trial
                new_term_tensor = build_term_tensor(term)
                R = R - old_term_tensor + new_term_tensor
                delta = best_trial_mx - best_mx
                best_mx = best_trial_mx
                sweep_accepts += 1
                total_accepts += 1
                sweep_best_delta = min(sweep_best_delta, delta)

                axis_ch = axis[0]  # a, b, or g
                print(f"  S{sweep:02d} t{ti+1:2d} {axis_ch}[{pi}]: "
                      f"{old_val:+.8f} → {best_trial:+.8f}  "
                      f"max_abs={best_mx:.10f} (Δ={delta:+.2e})")
            else:
                # Revert
                term[axis][pi] = old_val

        # Sweep summary
        elapsed = time.time() - t0
        print(f"\n  Sweep {sweep}: {sweep_accepts} accepts, "
              f"best_delta={sweep_best_delta:+.2e}, "
              f"max_abs={best_mx:.10f}, "
              f"evals={total_evals}, time={elapsed:.1f}s")
        print(f"  {'─'*80}")

        # Save after each sweep
        save_candidate(data, output_path)

        if sweep_accepts == 0:
            stale_sweeps += 1
            if stale_sweeps >= patience:
                print(f"\n  Converged after {sweep} sweeps ({stale_sweeps} stale).")
                break
            # Widen search for next sweep
            fine_range *= 1.5
            n_nearby += 2
            print(f"  Widening: fine_range={fine_range:.4f}, n_nearby={n_nearby}")
        else:
            stale_sweeps = 0

    # Final report
    final_mx, final_fro = evaluate(data)
    elapsed = time.time() - t0
    print(f"\n{'='*90}")
    print(f"FINAL RESULT")
    print(f"  max_abs = {final_mx:.10f}")
    print(f"  fro     = {final_fro:.8f}")
    print(f"  total accepts = {total_accepts}")
    print(f"  total evals   = {total_evals}")
    print(f"  elapsed       = {elapsed:.1f}s")
    print(f"  saved to      = {output_path}")
    return data, final_mx

# ══════════════════════════════════════════════════════════════════════════════
# Worker for parallel execution
# ══════════════════════════════════════════════════════════════════════════════
def _worker(args_tuple):
    """Run one optimizer instance. Imports are module-level so this works with mp."""
    (input_path, output_path, seed, max_sweeps, patience,
     n_nearby, fine_steps, fine_range, worker_id) = args_tuple

    np.random.seed(seed)

    data = load_candidate(input_path)
    best_mx, _ = evaluate(data)
    R = reconstruct(data) - T

    stale_sweeps = 0
    total_accepts = 0
    t0 = time.time()
    local_nearby = n_nearby
    local_fine = fine_range

    for sweep in range(1, max_sweeps + 1):
        sweep_accepts = 0
        coeff_idx = build_coeff_index(data)
        order = np.random.permutation(len(coeff_idx))

        for ci in order:
            ti, axis, pi, old_val = coeff_idx[ci]
            term = data['terms'][ti]
            old_term_tensor = build_term_tensor(term)
            trials = trial_values(old_val, local_nearby, fine_steps, local_fine)
            if not trials:
                continue

            best_trial = None
            best_trial_mx = best_mx

            for tv in trials:
                term[axis][pi] = tv
                new_term_tensor = build_term_tensor(term)
                R_new = R - old_term_tensor + new_term_tensor
                new_mx = np.max(np.abs(R_new))

                if new_mx < best_trial_mx - 1e-12:
                    best_trial = tv
                    best_trial_mx = new_mx

            if best_trial is not None:
                term[axis][pi] = best_trial
                new_term_tensor = build_term_tensor(term)
                R = R - old_term_tensor + new_term_tensor
                best_mx = best_trial_mx
                sweep_accepts += 1
                total_accepts += 1
            else:
                term[axis][pi] = old_val

        if sweep_accepts == 0:
            stale_sweeps += 1
            if stale_sweeps >= patience:
                break
            local_fine *= 1.5
            local_nearby += 2
        else:
            stale_sweeps = 0

    elapsed = time.time() - t0
    # Save this worker's result
    save_candidate(data, output_path)
    final_mx, final_fro = evaluate(data)
    return (worker_id, seed, final_mx, final_fro, total_accepts, sweep, elapsed, output_path)

# ══════════════════════════════════════════════════════════════════════════════
def run_parallel(input_path, final_output, n_workers=24, max_sweeps=100,
                 patience=5, n_nearby=8, fine_steps=7, fine_range=0.01):
    """Launch n_workers independent coordinate-descent runs with different seeds."""
    import multiprocessing as mp
    import os

    os.makedirs('optim_workers', exist_ok=True)

    tasks = []
    for i in range(n_workers):
        seed = 1000 + i * 137  # spread seeds
        out_path = f'optim_workers/worker_{i:02d}.json'
        tasks.append((input_path, out_path, seed, max_sweeps, patience,
                       n_nearby, fine_steps, fine_range, i))

    data0 = load_candidate(input_path)
    mx0, _ = evaluate(data0)
    print(f"Baseline max_abs = {mx0:.10f}")
    print(f"Launching {n_workers} parallel workers "
          f"(sweeps={max_sweeps}, patience={patience}, "
          f"nearby={n_nearby}, fine_steps={fine_steps}, range={fine_range})")
    print(f"{'='*90}")

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, tasks)
    total_elapsed = time.time() - t0

    # Sort by max_abs
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
    best_wid, best_seed, best_mx, best_fro, best_acc, best_sw, best_el, best_path = best
    print(f"\n{'='*90}")
    print(f"BEST: Worker {best_wid} (seed={best_seed})")
    print(f"  max_abs = {best_mx:.10f}  (baseline: {mx0:.10f}, Δ = {best_mx - mx0:+.10f})")
    print(f"  fro     = {best_fro:.8f}")
    print(f"  accepts = {best_acc}, sweeps = {best_sw}")

    # Copy best to final output
    import shutil
    shutil.copy2(best_path, final_output)
    print(f"  saved to {final_output}")
    return best_mx

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Algebraic coordinate descent optimizer')
    parser.add_argument('--input', default='outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json',
                        help='Input candidate JSON')
    parser.add_argument('--output', default='optimized_candidate.json',
                        help='Output candidate JSON')
    parser.add_argument('--sweeps', type=int, default=100, help='Max sweeps')
    parser.add_argument('--patience', type=int, default=5, help='Stale sweeps before stop')
    parser.add_argument('--nearby', type=int, default=8, help='Algebraic neighbors to try')
    parser.add_argument('--fine-steps', type=int, default=7, help='Fine grid steps per side')
    parser.add_argument('--fine-range', type=float, default=0.01, help='Fine grid half-width')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Number of parallel workers (0 = single-threaded)')
    args = parser.parse_args()

    if args.parallel > 0:
        run_parallel(args.input, args.output,
                     n_workers=args.parallel,
                     max_sweeps=args.sweeps, patience=args.patience,
                     n_nearby=args.nearby, fine_steps=args.fine_steps,
                     fine_range=args.fine_range)
    else:
        np.random.seed(args.seed)
        optimize(args.input, args.output,
                 max_sweeps=args.sweeps, patience=args.patience,
                 n_nearby=args.nearby, fine_steps=args.fine_steps,
                 fine_range=args.fine_range)
