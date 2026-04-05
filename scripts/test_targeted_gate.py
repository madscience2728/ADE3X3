"""Test support pattern enumeration and gate checking — 24 workers."""
import numpy as np
import time
import sys
from multiprocessing import freeze_support, Pool
from CANON_ADE.axiom_engine import operators as ops, tensor_core as tc
from CANON_ADE.axiom_engine.axiom_graph import bfs_for_rank
from CANON_ADE.axiom_engine.axiom_evaluators import matrix_rank

WORKERS = 24


def _check_one(args):
    """Worker function — must be picklable top-level."""
    pat_name, pattern, kept, R, fiber_gammas, seed = args
    # Re-import inside worker to avoid Windows spawn issues
    from CANON_ADE.axiom_engine import operators as _ops
    a, b, g = _ops._build_factors_support_pattern(kept, R, fiber_gammas, pattern, seed=seed)
    passed, det = _ops._check_gates(a, b, R)
    return pat_name, seed, passed, det['aug_gap'], det['delta_leak']


def main():
    for R in [13, 27]:
        print(f"\n=== R = {R} ===", flush=True)
        t0 = time.time()
        results = bfs_for_rank(R, max_states=500)
        leaves = [r for r in results if 'A3' in r.satisfied]
        print(f"  BFS done in {time.time()-t0:.1f}s, {len(leaves)} A3 leaves", flush=True)
        if not leaves:
            print("  No A3 leaf found")
            continue
        st = leaves[0].state
        kept = st['kept']
        fiber_gammas = st.get('fiber_gammas')

        # Orbit classes
        orbit_classes = {}
        for idx, (r, s, u) in enumerate(kept):
            key = tuple(sorted([r, s, u]))
            orbit_classes.setdefault(key, []).append(idx)
        print(f"  {len(kept)} terms, {len(orbit_classes)} orbit classes")

        patterns = ops._enumerate_support_patterns(kept, R)
        n_pat = len(patterns)

        # Build work
        work = []
        for pat_name, pattern in patterns:
            for seed in range(3):
                work.append((pat_name, pattern, kept, R, fiber_gammas, seed))
        print(f"  {len(work)} jobs dispatching to {WORKERS} workers...", flush=True)

        gate_pass = 0
        best_ag = 0
        best_name = None
        best_leak = -1
        t1 = time.time()

        with Pool(processes=WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(_check_one, work, chunksize=50)):
                pat_name, seed, passed, ag, leak = result
                if passed:
                    gate_pass += 1
                if ag > best_ag or (ag == best_ag and leak < best_leak):
                    best_ag = ag
                    best_leak = leak
                    best_name = f"{pat_name}_s{seed}"
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1}/{len(work)} done, best aug_gap={best_ag}, "
                          f"elapsed={time.time()-t1:.1f}s", flush=True)

        elapsed = time.time() - t1
        print(f"  Finished in {elapsed:.1f}s")
        print(f"  Gate pass: {gate_pass}/{len(work)}")
        print(f"  Best aug_gap={best_ag} (leak={best_leak}) from {best_name}")


if __name__ == '__main__':
    freeze_support()
    main()



