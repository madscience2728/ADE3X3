import sys, time
sys.path.insert(0, '.')
from CANON_ADE.axiom_engine.operators import _make_initial_state, OPERATORS

for R in [13, 19, 27]:
    t0 = time.time()
    s = _make_initial_state(R)
    t1 = time.time()
    kept = s.get('kept', [])
    n = len(kept) if kept else 0
    print(f"R={R}: _make_initial_state took {t1-t0:.1f}s, kept={n}")
    for src, tgt, op_fn, prereqs in OPERATORS:
        if src == 'root':
            t2 = time.time()
            try:
                children = op_fn(s)
                t3 = time.time()
                print(f"  {src}->{tgt}: {t3-t2:.1f}s, {len(children)} children")
            except Exception as e:
                t3 = time.time()
                print(f"  {src}->{tgt}: {t3-t2:.1f}s, EXCEPTION: {e}")
