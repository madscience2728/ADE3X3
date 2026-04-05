# coding: utf-8
"""
axiom_bfs.py -- BFS over algebra axiom lattice, orbit-reduced basis (439 vars).

Key optimisation: C4 G-equivariance collapses 6859 structure constants to 439
orbit representatives.  Every axiom constraint is expressed in orbit coords,
so each SVD is at most 439x439 instead of 6859x6859 (~250x faster).

Linear axioms explored:
  AX_COMM        f[i,j,k] = f[j,i,k]
  AX_ANTI_COMM   f[i,j,k] = -f[j,i,k], f[i,i,k]=0
  AX_R_BLIND     r-variants within each fiber equated
  AX_FIBER_ID    Forced-fiber elements act as partial identity
  AX_HULL_ZD     Hull-adjacent cross-products zeroed

Quadratic axioms (Phase 2, not yet linearised):
  AX_ASSOC, AX_FIBER_ASSOC
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import deque

from canon_constraints import INTERIOR, IDX, C3_FiberConstraint, C4_Symmetry
from algebra_template import AlgebraTemplate, N

# ---------------------------------------------------------------------------
# One-time orbit map  (module-level, cached)
# ---------------------------------------------------------------------------
print("Building G-orbit map for INTERIOR^3 (~30s)...", flush=True)
_t0 = time.time()
_tmpl = AlgebraTemplate()
_triple_orbits: List[List[Tuple]] = _tmpl.compute_triple_orbits()
N_ORBITS: int = len(_triple_orbits)          # 439

_triple_to_orbit: Dict[Tuple[int, int, int], int] = {}
for _oid, _orb in enumerate(_triple_orbits):
    for _trip in _orb:
        _triple_to_orbit[_trip] = _oid
print(f"  Done: {N_ORBITS} orbits in {time.time()-_t0:.1f}s", flush=True)


def orbit_of(i: int, j: int, k: int) -> int:
    return _triple_to_orbit[(i, j, k)]


# ---------------------------------------------------------------------------
# Orbit-reduced constraint system
# ---------------------------------------------------------------------------
class OCS:
    """Linear system on N_ORBITS (439) orbit-representative variables."""

    def __init__(self) -> None:
        self.n = N_ORBITS
        self.rows: List[Tuple[np.ndarray, float]] = []
        self._dirty = True
        self._A: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    def copy(self) -> "OCS":
        cs = OCS()
        cs.rows = list(self.rows)
        cs._dirty = True
        return cs

    def add_eq(self, row: np.ndarray, rhs: float) -> None:
        self.rows.append((row.copy(), rhs))
        self._dirty = True

    def force_zero(self, i: int, j: int, k: int) -> None:
        oid = orbit_of(i, j, k)
        row = np.zeros(self.n); row[oid] = 1.0
        self.add_eq(row, 0.0)

    def force_equal(self, i1: int, j1: int, k1: int,
                    i2: int, j2: int, k2: int, sign: float = 1.0) -> None:
        oid1 = orbit_of(i1, j1, k1)
        oid2 = orbit_of(i2, j2, k2)
        if oid1 == oid2:
            if abs(sign - 1.0) < 1e-12:
                return          # trivially satisfied
            row = np.zeros(self.n); row[oid1] = 1.0
            self.add_eq(row, 0.0)
            return
        row = np.zeros(self.n)
        row[oid1] = 1.0; row[oid2] = -sign
        self.add_eq(row, 0.0)

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._dirty:
            if not self.rows:
                self._A = np.zeros((0, self.n))
                self._b = np.zeros(0)
            else:
                self._A = np.vstack([r for r, _ in self.rows])
                self._b = np.array([b for _, b in self.rows])
            self._dirty = False
        return self._A, self._b

    def dof(self) -> int:
        """Affine subspace dimension; -1 if inconsistent."""
        A, b = self.build()
        if A.shape[0] == 0:
            return self.n
        sv = np.linalg.svd(A, compute_uv=False)
        tol = max(A.shape) * (sv[0] if sv[0] > 0 else 1.0) * 1e-10
        rk_A = int(np.sum(sv > tol))
        Ab = np.column_stack([A, b.reshape(-1, 1)])
        sv_ab = np.linalg.svd(Ab, compute_uv=False)
        rk_Ab = int(np.sum(sv_ab > tol))
        if rk_Ab > rk_A:
            return -1
        return self.n - rk_A

    def is_consistent(self) -> bool:
        return self.dof() >= 0


# ---------------------------------------------------------------------------
# Axiom encoders
# ---------------------------------------------------------------------------
ALL_AXIOMS = [
    "AX_ASSOC", "AX_COMM", "AX_ANTI_COMM",
    "AX_FIBER_ASSOC", "AX_R_BLIND", "AX_FIBER_ID", "AX_HULL_ZD",
]
INCOMPATIBLE = [("AX_COMM", "AX_ANTI_COMM")]
QUADRATIC_AXIOMS = {"AX_ASSOC", "AX_FIBER_ASSOC"}


def enc_comm(cs: OCS) -> None:
    for i in range(N):
        for j in range(i+1, N):
            for k in range(N):
                cs.force_equal(i, j, k, j, i, k)


def enc_anti(cs: OCS) -> None:
    for i in range(N):
        for k in range(N):
            cs.force_zero(i, i, k)
    for i in range(N):
        for j in range(i+1, N):
            for k in range(N):
                cs.force_equal(i, j, k, j, i, k, sign=-1.0)


def enc_rblind(cs: OCS) -> None:
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for j in range(N):
                for k in range(N):
                    cs.force_equal(rep, j, k, other, j, k)
                    cs.force_equal(j, rep, k, j, other, k)
                    cs.force_equal(j, k, rep, j, k, other)


def enc_fibid(cs: OCS) -> None:
    for (s, u), indices in C3_FiberConstraint.FORCED_FIBERS.items():
        id_idx = indices[0]
        for i in range(N):
            for k in range(N):
                rhs = 1.0 if i == k else 0.0
                r1 = np.zeros(N_ORBITS); r1[orbit_of(i, id_idx, k)] = 1.0
                cs.add_eq(r1, rhs)
                r2 = np.zeros(N_ORBITS); r2[orbit_of(id_idx, i, k)] = 1.0
                cs.add_eq(r2, rhs)


def enc_hullzd(cs: OCS) -> None:
    hull = {idx for lst in C3_FiberConstraint.FORCED_FIBERS.values() for idx in lst}
    nonhull = set(range(N)) - hull
    for i in hull:
        for j in hull:
            for k in nonhull:
                cs.force_zero(i, j, k)


ENCODERS = {
    "AX_ASSOC": None, "AX_COMM": enc_comm, "AX_ANTI_COMM": enc_anti,
    "AX_FIBER_ASSOC": None, "AX_R_BLIND": enc_rblind,
    "AX_FIBER_ID": enc_fibid, "AX_HULL_ZD": enc_hullzd,
}


def enc_rblind_first(cs: OCS) -> None:
    """R-blindness on first index only: f[(r,s),j,k] independent of r.
    This is the correct C2 encoding: the 'input mode' r-variant doesn't
    distinguish algebra products.  All-3-index r-blindness over-constrains
    to the trivial constant algebra (d.o.f.=1).
    """
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for j in range(N):
                for k in range(N):
                    cs.force_equal(rep, j, k, other, j, k)


def build_canon() -> OCS:
    """C4 baked into orbit basis.  Add C2 (first-index r-blindness).
    Result: d.o.f.=27 = |INTERIOR^2 / G|, the correct ADE root.
    (All-3-index r-blindness collapses to d.o.f.=1 trivial constant algebra.)
    """
    cs = OCS()
    enc_rblind_first(cs)
    return cs


# ---------------------------------------------------------------------------
# BFS node
# ---------------------------------------------------------------------------
@dataclass
class AxiomState:
    imposed: FrozenSet[str] = field(default_factory=frozenset)
    cs: OCS = field(default_factory=OCS)
    _dof_cache: Optional[int] = field(default=None, init=False, repr=False, compare=False)

    def impose(self, ax: str) -> Optional["AxiomState"]:
        if ax in self.imposed:
            return self
        for a, b in INCOMPATIBLE:
            if (ax == a and b in self.imposed) or (ax == b and a in self.imposed):
                return None
        ncs = self.cs.copy()
        enc = ENCODERS.get(ax)
        if enc:
            enc(ncs)
        d = ncs.dof()
        if d < 0:
            return None
        ns = AxiomState(self.imposed | {ax}, ncs)
        ns._dof_cache = d
        return ns

    def dof(self) -> int:
        if self._dof_cache is None:
            self._dof_cache = self.cs.dof()
        return self._dof_cache

    def summary(self) -> str:
        label = ", ".join(sorted(self.imposed)) if self.imposed else "C4+C2 only"
        return f"[{label}]  d.o.f.={self.dof()}"


# ---------------------------------------------------------------------------
# BFS driver
# ---------------------------------------------------------------------------
def run_bfs(max_depth: int = 3, verbose: bool = True) -> List[AxiomState]:
    linear = [a for a in ALL_AXIOMS if a not in QUADRATIC_AXIOMS]
    root = AxiomState(frozenset(), build_canon())
    root._dof_cache = root.cs.dof()
    if verbose:
        print(f"  Root: d.o.f.={root.dof()} / {N_ORBITS} orbit vars", flush=True)
    visited: Set[FrozenSet] = {root.imposed}
    q: deque = deque([root])
    results: List[AxiomState] = [root]
    n_tested = 0

    while q:
        state = q.popleft()
        d = len(state.imposed)
        if d >= max_depth:
            continue
        for ax in linear:
            if ax in state.imposed:
                continue
            n_tested += 1
            ts = time.time()
            ns = state.impose(ax)
            dt = time.time() - ts
            if ns is None:
                if verbose:
                    print(
                        f"  [d{d+1}] INCONS: {sorted(state.imposed)}+{ax}"
                        f"  ({dt:.2f}s #{n_tested} q={len(q)})",
                        flush=True,
                    )
                continue
            if ns.imposed in visited:
                continue
            visited.add(ns.imposed)
            results.append(ns)
            q.append(ns)
            if verbose:
                print(
                    f"  [d{d+1}] {ns.summary()}"
                    f"  ({dt:.2f}s #{n_tested} q={len(q)})",
                    flush=True,
                )

    return sorted(results, key=lambda s: s.dof())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("AXIOM BFS -- orbit-reduced basis")
    la = [a for a in ALL_AXIOMS if a not in QUADRATIC_AXIOMS]
    qa = [a for a in ALL_AXIOMS if a in QUADRATIC_AXIOMS]
    print(f"Linear:    {la}")
    print(f"Quadratic: {qa}  (Phase 2)")
    print(f"Variables: {N_ORBITS} orbit-reps vs {N**3} full-space (~{N**3//N_ORBITS}x speedup)")

    prev = None
    for depth, label in [(1, "single axioms"), (2, "pairs"), (3, "triples")]:
        print(f"\n--- BFS depth={depth} ({label}) ---", flush=True)
        t0 = time.time()
        res = run_bfs(max_depth=depth, verbose=True)
        print(f"  Done: {len(res)} states  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nFull summary ({len(res)} consistent combos):", flush=True)
    for s in res[:20]:
        line = s.summary()
        print(f"  {line}", flush=True)
    print("\nNear-exact (d.o.f. <= 10):", flush=True)
    for s in res:
        d = s.dof()
        if 0 <= d <= 10:
            print(f"  d={d}: {sorted(s.imposed)}", flush=True)
    print("\nDone.", flush=True)
