"""
axiom_graph.py — BFS engine over the axiom graph.

Nodes = partial algebra states (which axioms are satisfied + accumulated structure).
Edges = constructive operators from operators.py.

BFS discovers all reachable axiom subsets for a given rank R.
Each terminal state (no more operators can fire) is a "leaf" — 
the maximal set of simultaneously satisfiable axioms via that path.

Supports:
  - Multiple entry points (A2, A4, A6 can all be roots)
  - Constraint propagation (exact, rule-based filtering)
  - No fixed axiom ordering
"""
import time
import random
from collections import deque
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor

from .operators import (
    OPERATORS, _make_initial_state, _propagate_constraints, State,
)


class BFSResult:
    """One completed BFS path."""
    __slots__ = ('rank', 'satisfied', 'path', 'state', 'depth', 'constraints')

    def __init__(self, rank, satisfied, path, state, depth):
        self.rank = rank
        self.satisfied = frozenset(satisfied)
        self.path = list(path)
        self.state = state
        self.depth = depth
        self.constraints = state.get('constraints', {}) if isinstance(state, dict) else {}

    def __repr__(self):
        axioms = ','.join(sorted(self.satisfied))
        return f"BFSResult(R={self.rank}, depth={self.depth}, axioms={{{axioms}}})"


def expand_roots(R: int) -> list:
    """
    Expand the initial state through all root operators, returning
    a flat list of depth-1 states. Each state is deterministically
    reproducible from (R, index) so workers can reconstruct locally.
    """
    initial = _make_initial_state(R)
    children = []
    for src, tgt, op_fn, prereqs in OPERATORS:
        if src != 'root':
            continue
        try:
            results = op_fn(initial)
        except Exception:
            continue
        for child in results:
            if _propagate_constraints(child):
                children.append(child)
    return children


def bfs_from_states(R: int, starting_states: list, max_states: int = 100_000,
                    on_progress=None) -> List[BFSResult]:
    """
    BFS from a list of pre-expanded starting states (subtree sharding).
    Same logic as bfs_for_rank but starts from given states instead of root.
    on_progress: callback(explored, queued, best_depth, best_axioms, axiom_frontier)
        axiom_frontier is a dict {frozenset_of_axioms: count}
    """
    queue = list(starting_states)

    leaves = []
    best_seen = set()
    explored = 0
    best_depth = 0
    best_axiom_count = 0
    axiom_frontier = {}  # {frozenset: count} — which axiom sets are being seen

    def _dedup_key(st):
        sat = frozenset(st['satisfied'])
        shape = st.get('kernel_shape')
        fg = st.get('fiber_gammas')
        fg_key = tuple(sorted(fg.items())) if fg else ()
        kept = tuple(st['kept']) if st.get('kept') else ()
        cd = tuple(sorted(st['cd_block_map'].items())) if st.get('cd_block_map') else ()
        rs = st.get('relation_spectrum') or ()
        fs = st.get('factor_strategy') or ()
        stab = st.get('stability_tag') or ''
        return (sat, kept, shape, fg_key, cd, rs, fs, stab)

    # Seed dedup with starting states
    for state in starting_states:
        best_seen.add(_dedup_key(state))

    while queue and explored < max_states:
        # Random pop: O(1) via swap-with-last
        idx = random.randrange(len(queue))
        queue[idx], queue[-1] = queue[-1], queue[idx]
        state = queue.pop()
        explored += 1

        any_fired = False
        for src, tgt, op_fn, prereqs in OPERATORS:
            if not prereqs.issubset(state['satisfied']):
                continue
            if tgt in state['satisfied'] and tgt != 'free':
                continue

            try:
                children = op_fn(state)
            except Exception:
                continue

            for child in children:
                if not _propagate_constraints(child):
                    continue

                dedup_key = _dedup_key(child)
                if dedup_key in best_seen:
                    continue
                best_seen.add(dedup_key)
                any_fired = True

                sat = frozenset(child['satisfied'])
                depth = len(child['path'])
                if depth > best_depth:
                    best_depth = depth
                if len(child['satisfied']) > best_axiom_count:
                    best_axiom_count = len(child['satisfied'])

                # Track axiom frontier
                axiom_frontier[sat] = axiom_frontier.get(sat, 0) + 1

                queue.append(child)

        if not any_fired:
            depth = len(state['path'])
            sat = frozenset(state['satisfied'])
            axiom_frontier[sat] = axiom_frontier.get(sat, 0) + 1
            leaves.append(BFSResult(
                rank=R,
                satisfied=state['satisfied'],
                path=state['path'],
                state=state,
                depth=depth,
            ))

        if on_progress and explored % 50 == 0:
            # Send top-5 axiom sets as frontier summary
            top_frontier = sorted(axiom_frontier.items(), key=lambda x: -x[1])[:5]
            frontier_summary = [(sorted(k), v) for k, v in top_frontier]
            on_progress(explored, len(queue), best_depth, best_axiom_count, frontier_summary)

    # Capture ALL remaining queued states with good axiom counts
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

    leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))
    return leaves


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
    queue = [initial]
    
    leaves = []          # terminal states
    best_seen = set()    # track unique (satisfied_set) to prune duplicates
    explored = 0
    best_depth = 0
    best_axiom_count = 0

    def _dk(st):
        sat = frozenset(st['satisfied'])
        shape = st.get('kernel_shape')
        fg = st.get('fiber_gammas')
        fg_key = tuple(sorted(fg.items())) if fg else ()
        kept = tuple(st['kept']) if st.get('kept') else ()
        cd = tuple(sorted(st['cd_block_map'].items())) if st.get('cd_block_map') else ()
        rs = st.get('relation_spectrum') or ()
        fs = st.get('factor_strategy') or ()
        stab = st.get('stability_tag') or ''
        return (sat, kept, shape, fg_key, cd, rs, fs, stab)

    while queue and explored < max_states:
        # Random pop: O(1) via swap-with-last
        idx = random.randrange(len(queue))
        queue[idx], queue[-1] = queue[-1], queue[idx]
        state = queue.pop()
        explored += 1

        # Try all operators whose prereqs are met
        any_fired = False
        for src, tgt, op_fn, prereqs in OPERATORS:
            if not prereqs.issubset(state['satisfied']):
                continue
            # Don't re-apply an operator that already produced this target
            if tgt in state['satisfied'] and tgt != 'free':
                continue

            t_op = time.time()
            try:
                children = op_fn(state)
            except Exception:
                continue


            for child in children:
                # Constraint propagation: reject inconsistent states
                if not _propagate_constraints(child):
                    continue

                # Dedup by full state signature
                dedup_key = _dk(child)
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

        if on_progress and explored % 50 == 0:
            on_progress(explored, len(queue), best_depth, best_axiom_count)


    # Capture ALL remaining queued states with good axiom counts
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
