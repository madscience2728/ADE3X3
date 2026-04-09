import sys, time
sys.path.insert(0, '.')
from CANON_ADE.axiom_engine.axiom_graph import bfs_for_rank

for R in [19, 27]:
    t0 = time.time()
    print(f"R={R}: starting BFS (max 200 states)...")
    def prog(explored, queued, depth, ax):
        elapsed = time.time() - t0
        print(f"  R={R} explored={explored} queued={queued} depth={depth} ax={ax} elapsed={elapsed:.1f}s")
    leaves = bfs_for_rank(R, max_states=200, on_progress=prog)
    t1 = time.time()
    print(f"R={R}: done in {t1-t0:.1f}s, {len(leaves)} leaves\n")
