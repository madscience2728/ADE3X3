import sys, time
from collections import defaultdict
sys.path.insert(0, '.')
from CANON_ADE.axiom_engine.operators import OPERATORS, _make_initial_state, _propagate_constraints
from collections import deque

for R in [27]:
    initial = _make_initial_state(R)
    queue = deque([initial])
    explored = 0
    op_times = defaultdict(float)
    op_calls = defaultdict(int)
    best_seen = set()
    
    while queue and explored < 100:
        state = queue.popleft()
        explored += 1
        for src, tgt, op_fn, prereqs in OPERATORS:
            if not prereqs.issubset(state['satisfied']):
                continue
            if tgt in state['satisfied'] and tgt != 'free':
                continue
            t0 = time.time()
            try:
                children = op_fn(state)
            except:
                children = []
            dt = time.time() - t0
            op_times[f"{src}->{tgt}"] += dt
            op_calls[f"{src}->{tgt}"] += 1
            
            for child in children:
                if not _propagate_constraints(child):
                    continue
                sat = frozenset(child['satisfied'])
                shape = child.get('kernel_shape')
                fg = child.get('fiber_gammas')
                fg_key = tuple(sorted(fg.items())) if fg else ()
                dedup_key = (sat, shape, fg_key)
                if dedup_key in best_seen:
                    continue
                best_seen.add(dedup_key)
                queue.append(child)
    
    print(f"\nR={R}: {explored} states explored, queue={len(queue)}")
    print(f"{'Op':<20} {'Calls':>6} {'Total(s)':>10} {'Avg(ms)':>10}")
    for op in sorted(op_times, key=lambda k: -op_times[k]):
        total = op_times[op]
        calls = op_calls[op]
        avg_ms = 1000 * total / calls if calls else 0
        print(f"{op:<20} {calls:>6} {total:>10.2f} {avg_ms:>10.1f}")
