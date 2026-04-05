"""
axiom_graph.py — BFS engine over the axiom graph.

Nodes = partial algebra states (which axioms are satisfied + accumulated structure).
Edges = constructive operators from operators.py.

BFS discovers all reachable axiom subsets for a given rank R.
Each terminal state (no more operators can fire) is a "leaf" — 
the maximal set of simultaneously satisfiable axioms via that path.
"""
import time
from collections import deque
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor

from .operators import (
    OPERATORS, _make_initial_state, State,
)


class BFSResult:
    """One completed BFS path."""
    __slots__ = ('rank', 'satisfied', 'path', 'state', 'depth')

    def __init__(self, rank, satisfied, path, state, depth):
        self.rank = rank
        self.satisfied = frozenset(satisfied)
        self.path = list(path)
        self.state = state
        self.depth = depth

    def __repr__(self):
        axioms = ','.join(sorted(self.satisfied))
        return f"BFSResult(R={self.rank}, depth={self.depth}, axioms={{{axioms}}})"


def bfs_for_rank(R: int, max_states: int = 100_000,
                 on_progress: Optional[Callable] = None) -> List[BFSResult]:
    """
    BFS over the axiom graph for rank R.
    
    Returns all terminal states (leaves where no further operator can fire).
    Also returns intermediate states that reached high axiom counts.
    
    Args:
        R: target decomposition rank
        max_states: safety cap on total states explored
        on_progress: callback(explored, queued, best_depth, best_axioms)
    """
    initial = _make_initial_state(R)
    queue = deque([initial])
    
    leaves = []          # terminal states
    best_seen = set()    # track unique (satisfied_set) to prune duplicates
    explored = 0
    best_depth = 0
    best_axiom_count = 0

    while queue and explored < max_states:
        state = queue.popleft()
        explored += 1

        # Try all operators whose prereqs are met
        any_fired = False
        for src, tgt, op_fn, prereqs in OPERATORS:
            if not prereqs.issubset(state['satisfied']):
                continue
            # Don't re-apply an operator that already produced this target
            if tgt in state['satisfied'] and tgt != 'free':
                continue

            try:
                children = op_fn(state)
            except Exception:
                continue

            for child in children:
                # Dedup by (satisfied_set, kernel_shape, fiber_gammas_hash)
                sat = frozenset(child['satisfied'])
                shape = child.get('kernel_shape')
                fg = child.get('fiber_gammas')
                fg_key = tuple(sorted(fg.items())) if fg else ()
                dedup_key = (sat, shape, fg_key)
                if dedup_key in best_seen:
                    continue
                best_seen.add(dedup_key)
                any_fired = True

                depth = len(child['path'])
                if depth > best_depth:
                    best_depth = depth
                if len(child['satisfied']) > best_axiom_count:
                    best_axiom_count = len(child['satisfied'])

                queue.append(child)

        # If no operator fired, this is a leaf
        if not any_fired:
            depth = len(state['path'])
            leaves.append(BFSResult(
                rank=R,
                satisfied=state['satisfied'],
                path=state['path'],
                state=state,
                depth=depth,
            ))

        if on_progress and explored % 100 == 0:
            on_progress(explored, len(queue), best_depth, best_axiom_count)
        elif explored % 500 == 0:
            import sys
            print(f"  [progress] explored={explored}, queue={len(queue)}, "
                  f"leaves={len(leaves)}, best_axioms={best_axiom_count}",
                  file=sys.stderr, flush=True)

    # Also capture any remaining queued states as partial results
    # (in case we hit max_states)
    if explored >= max_states:
        for state in queue:
            if len(state['satisfied']) >= best_axiom_count - 1:
                leaves.append(BFSResult(
                    rank=R,
                    satisfied=state['satisfied'],
                    path=state['path'],
                    state=state,
                    depth=len(state['path']),
                ))

    # Sort: most axioms satisfied first, then shortest path
    leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))

    return leaves


def bfs_multi_rank(ranks: List[int], max_states_per_rank: int = 100_000,
                   on_progress: Optional[Callable] = None) -> Dict[int, List[BFSResult]]:
    """Run BFS for multiple ranks sequentially (each rank is already parallel internally)."""
    results = {}
    for R in ranks:
        t0 = time.time()
        leaves = bfs_for_rank(R, max_states=max_states_per_rank,
                              on_progress=on_progress)
        elapsed = time.time() - t0
        results[R] = leaves
        if on_progress:
            on_progress(-1, 0, 0, 0)  # signal rank complete
    return results
