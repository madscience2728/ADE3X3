"""
canon_constraints.py — Formal encoding of the ADE constraint canon C1–C5.

Each constraint is a pure Python object with:
  .name        — identifier
  .check(obj)  — returns True/False given a candidate algebra or decomposition
  .describe()  — human-readable statement

Nothing in this file uses scipy or any optimizer. This is the algebra layer.
"""

from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Set


# ---------------------------------------------------------------------------
# Fundamental objects
# ---------------------------------------------------------------------------

# The index cube Z_3^3
CUBE = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]

# C1 — the partition
HULL     = [(r, s, u) for r, s, u in CUBE if r % 2 == 0 and s % 2 == 0 and u % 2 == 0]
INTERIOR = [(r, s, u) for r, s, u in CUBE if not (r % 2 == 0 and s % 2 == 0 and u % 2 == 0)]

assert len(HULL)     == 8
assert len(INTERIOR) == 19

# Canonical index maps
IDX  = {xi: i for i, xi in enumerate(INTERIOR)}   # interior point -> 0..18
CIDX = {xi: i for i, xi in enumerate(CUBE)}        # full cube -> 0..26

# The matmul tensor T in R^{9x9x9}
def build_T() -> np.ndarray:
    T = np.zeros((9, 9, 9))
    for r, s, u in CUBE:
        T[3*r+s, 3*s+u, 3*r+u] = 1.0
    return T

T_MATMUL = build_T()


# ---------------------------------------------------------------------------
# C1 — Support Partition
# ---------------------------------------------------------------------------

class C1_SupportPartition:
    """
    The 27 nonzero entries of T split into:
      HULL     (8 pts):  all-even coordinates — the L-sector / deleted set
      INTERIOR (19 pts): at least one odd coordinate — the violation locus

    This is a Z_2 x S_3 wreath-product orbit partition, not a choice.
    """
    name = "C1_SupportPartition"

    @staticmethod
    def is_hull(rsu: Tuple[int,int,int]) -> bool:
        r, s, u = rsu
        return r % 2 == 0 and s % 2 == 0 and u % 2 == 0

    @staticmethod
    def is_interior(rsu: Tuple[int,int,int]) -> bool:
        return not C1_SupportPartition.is_hull(rsu)

    @staticmethod
    def cd_block_type(rsu: Tuple[int,int,int]) -> Tuple[int,int,int]:
        """Parity signature: (r%2, s%2, u%2)."""
        r, s, u = rsu
        return (r % 2, s % 2, u % 2)

    @staticmethod
    def hull() -> List[Tuple]:
        return HULL

    @staticmethod
    def interior() -> List[Tuple]:
        return INTERIOR

    @staticmethod
    def describe() -> str:
        return (
            "C1: The 27 nonzero entries of T partition into HULL (8, all-even coords) "
            "and INTERIOR (19, at least one odd coord). HULL = L-sector = deleted. "
            "INTERIOR = CD parity-violation locus."
        )


# ---------------------------------------------------------------------------
# C2 — r-Drops-Out Theorem
# ---------------------------------------------------------------------------

class C2_RDropsOut:
    """
    The Fourier character chi_{p,q,w}(r,s,u) = exp(2pi*i*(ps+qu+wu)/3)
    is independent of r. Consequently:

      T_hat(p,q,w) = 27 * delta_{p,0} * delta_{q+w=0 mod 3}

    Active modes: (0,0,0), (0,1,2), (0,2,1)  — only 3 of 27
    Dead modes:   the other 24

    For any valid CP decomposition sum_t a_t x b_t x c_t:
      The a-factor vectors are 'r-transparent': a_t[3r+s] depends on s only,
      not on r independently (up to the fiber structure).
    """
    name = "C2_RDropsOut"

    OMEGA = np.exp(2j * np.pi / 3)
    ALL_MODES    = [(p, q, w) for p in range(3) for q in range(3) for w in range(3)]
    ACTIVE_MODES = [(0, 0, 0), (0, 1, 2), (0, 2, 1)]
    DEAD_MODES   = [m for m in [(p,q,w) for p in range(3) for q in range(3) for w in range(3)]
                    if m not in [(0,0,0),(0,1,2),(0,2,1)]]

    @classmethod
    def fourier_coeff(cls, T: np.ndarray, p: int, q: int, w: int) -> complex:
        """Compute T_hat(p,q,w) via direct sum."""
        omega = cls.OMEGA
        I, J, K = np.mgrid[0:9, 0:9, 0:9]
        phase = omega ** (p * I + q * J + w * K)
        return np.sum(T * phase)

    @classmethod
    def verify(cls, T: np.ndarray = T_MATMUL) -> Dict:
        results = {}
        for mode in cls.ALL_MODES:
            val = cls.fourier_coeff(T, *mode)
            results[mode] = val
        active_nonzero = all(abs(results[m]) > 1.0 for m in cls.ACTIVE_MODES)
        dead_zero      = all(abs(results[m]) < 1e-9 for m in cls.DEAD_MODES)
        return {
            "coeffs": results,
            "active_nonzero": active_nonzero,
            "dead_zero": dead_zero,
            "verified": active_nonzero and dead_zero,
        }

    @staticmethod
    def describe() -> str:
        return (
            "C2: r drops out of the Fourier character. Only 3 of 27 modes are active: "
            "(0,0,0), (0,1,2), (0,2,1). The 24 dead modes are exactly zero. "
            "Any CP decomposition must cancel all contributions to dead modes."
        )


# ---------------------------------------------------------------------------
# C3 — Fiber Constraint
# ---------------------------------------------------------------------------

class C3_FiberConstraint:
    """
    Group the 19 interior points by (s,u)-fiber:
      F_{s,u} = {r in Z_3 : (r,s,u) in INTERIOR}

    For each fiber, the sum of gamma values (norms of rank-1 terms) must equal 3:
      Gamma(s,u) = sum_{t in F_{s,u}} gamma_t = 3

    Fiber types:
      Forced (size 1): (s,u) in {0,2}^2  ->  gamma is pinned = 3, no d.o.f.
      Free   (size 3): all others (5 fibers) -> sum=3, 2 d.o.f. each
    Total gamma d.o.f. = 10
    """
    name = "C3_FiberConstraint"

    # Build fiber map
    FIBERS: Dict[Tuple, List[int]] = {}
    for _r, _s, _u in INTERIOR:
        key = (_s, _u)
        if key not in FIBERS:
            FIBERS[key] = []
        FIBERS[key].append(IDX[(_r, _s, _u)])

    FORCED_FIBERS = {k: v for k, v in FIBERS.items() if len(v) == 1}
    FREE_FIBERS   = {k: v for k, v in FIBERS.items() if len(v) == 3}

    assert len(FORCED_FIBERS) == 4
    assert len(FREE_FIBERS)   == 5
    assert sum(len(v) for v in FIBERS.values()) == 19

    @classmethod
    def check_gamma(cls, gammas: np.ndarray, tol: float = 1e-8) -> Dict:
        """
        gammas: length-19 array of gamma values (one per interior term).
        Returns per-fiber sums and whether all equal 3.
        """
        fiber_sums = {}
        for (s, u), indices in cls.FIBERS.items():
            fiber_sums[(s, u)] = sum(gammas[i] for i in indices)
        all_ok = all(abs(v - 3.0) < tol for v in fiber_sums.values())
        return {"fiber_sums": fiber_sums, "all_equal_3": all_ok}

    @classmethod
    def fiber_constraint_matrix(cls) -> np.ndarray:
        """
        Returns a 9x19 matrix M such that M @ gamma = 3*ones(9)
        encodes the fiber constraint.
        """
        M = np.zeros((9, 19))
        for row, (key, indices) in enumerate(sorted(cls.FIBERS.items())):
            for i in indices:
                M[row, i] = 1.0
        return M

    @staticmethod
    def describe() -> str:
        return (
            "C3: For each (s,u)-fiber, the sum of gammas over the fiber must equal 3. "
            "4 forced fibers (size 1, gamma=3 pinned) + 5 free fibers (size 3, sum=3, 2 d.o.f.) "
            "= 10 total gamma d.o.f."
        )


# ---------------------------------------------------------------------------
# C4 — Symmetry Group Z_2 wreath S_3
# ---------------------------------------------------------------------------

class C4_Symmetry:
    """
    The tensor T is invariant under G = Z_2 wr S_3 (order 48).

    Action on (r,s,u) in Z_3^3:
      Inner Z_2^3: swap 1<->2 in each coordinate independently
      Outer S_3:   permute the coordinates (r,s,u)

    Any algebra A whose structure encodes T must be G-equivariant:
    the structure constants f^zeta_{xi,eta} satisfy
      f^{g(zeta)}_{g(xi), g(eta)} = f^zeta_{xi,eta}   for all g in G.

    G-orbits on INTERIOR x INTERIOR reduce the 19x19 = 361 pairs
    to a much smaller number of orbit representatives.
    """
    name = "C4_Symmetry"

    @staticmethod
    def swap12(x: int) -> int:
        """Swap 1<->2, fix 0: used for Z_2 inner action."""
        return {0: 0, 1: 2, 2: 1}[x]

    @classmethod
    def inner_action(cls, rsu: Tuple, bits: Tuple[int,int,int]) -> Tuple:
        """Apply Z_2^3 inner action: bit_i=1 means swap 1<->2 in coordinate i."""
        r, s, u = rsu
        br, bs, bu = bits
        return (
            cls.swap12(r) if br else r,
            cls.swap12(s) if bs else s,
            cls.swap12(u) if bu else u,
        )

    @staticmethod
    def outer_action(rsu: Tuple, perm: Tuple[int,int,int]) -> Tuple:
        """Apply S_3 outer permutation: perm is a permutation of (0,1,2)."""
        coords = list(rsu)
        return tuple(coords[perm[i]] for i in range(3))

    @classmethod
    def group_elements(cls) -> List[Tuple]:
        """All 48 group elements as (perm, bits) pairs."""
        from itertools import permutations, product
        perms = list(permutations([0, 1, 2]))
        bits  = list(product([0, 1], repeat=3))
        return [(p, b) for p in perms for b in bits]

    @classmethod
    def apply(cls, rsu: Tuple, perm: Tuple, bits: Tuple) -> Tuple:
        """Apply group element (perm, bits) to a point."""
        return cls.outer_action(cls.inner_action(rsu, bits), perm)

    @classmethod
    def orbit(cls, rsu: Tuple) -> Set[Tuple]:
        """Compute the G-orbit of a point."""
        return {cls.apply(rsu, p, b) for p, b in cls.group_elements()}

    @classmethod
    def interior_orbits(cls) -> List[Set]:
        """Partition INTERIOR into G-orbits."""
        remaining = set(INTERIOR)
        orbits = []
        while remaining:
            pt = next(iter(remaining))
            orb = cls.orbit(pt) & set(INTERIOR)
            orbits.append(orb)
            remaining -= orb
        return orbits

    @classmethod
    def pair_orbits(cls) -> List[Set]:
        """
        Partition INTERIOR x INTERIOR into G-orbits.
        Each orbit representative (xi, eta) gives one free structure constant
        (up to equivariance).
        """
        all_pairs = set(itertools.product(INTERIOR, INTERIOR))
        remaining = set(all_pairs)
        orbits = []
        G = cls.group_elements()
        while remaining:
            xi, eta = next(iter(remaining))
            orb = set()
            for p, b in G:
                orb.add((cls.apply(xi, p, b), cls.apply(eta, p, b)))
            orb &= all_pairs
            orbits.append(orb)
            remaining -= orb
        return orbits

    @staticmethod
    def describe() -> str:
        return (
            "C4: T is invariant under G = Z_2 wr S_3 (order 48). "
            "Structure constants of A must be G-equivariant: "
            "f^{g(zeta)}_{g(xi),g(eta)} = f^zeta_{xi,eta} for all g in G. "
            "G-orbits on INTERIOR^2 give the true d.o.f. count."
        )


# ---------------------------------------------------------------------------
# C5 — Sedenion Zero-Divisor Graph
# ---------------------------------------------------------------------------

class C5_SedenionZDGraph:
    """
    In the sedenion algebra S (CD level 4, dim 16), there are:
      42 ZD vertices: (e_i + e_j), i in {1..7} (L-imaginary), j in {9..15} (U-sector), j != i+8
      84 undirected ZD pairs (sign-quotiented: identifying (i,j) across ± signs)
      Degree-4 regular (each vertex has 4 ZD partners)

    PARITY-BLOCK BRIDGE (sedenion → INTERIOR):
      Sedenion L-imaginary index i in {1..7} has binary (b2,b1,b0) = (i>>2, (i>>1)&1, i&1).
      This matches the parity type (r%2, s%2, u%2) of INTERIOR points.
      The 7 nonzero parity types biject with octonion imaginary indices 1..7.
      Each parity block P_i = {ξ ∈ INTERIOR : parity(ξ) = bits(i)}.
      Block sizes: 4,4,2,4,2,2,1 for i=1..7 (total 19).

      A ZD vertex (i, j) with j in {9..15} maps to the pair of
      parity blocks (P_i, P_{j-8}).  A ZD pair ((i1,j1),(i2,j2)) = 0
      means: the composite algebra element
        Σ_{ξ∈P_{i1}} e_ξ + Σ_{η∈P_{j1-8}} e_η
      annihilates
        Σ_{ξ'∈P_{i2}} e_{ξ'} + Σ_{η'∈P_{j2-8}} e_{η'}
      in the algebra A.
    """
    name = "C5_SedenionZDGraph"

    # --- Parity-block bridge ---

    @staticmethod
    def _idx_to_parity(i: int) -> Tuple[int, int, int]:
        """Sedenion imaginary index (1-7) → parity triple (r%2, s%2, u%2)."""
        return ((i >> 2) & 1, (i >> 1) & 1, i & 1)

    @staticmethod
    def _parity_to_idx(p: Tuple[int, int, int]) -> int:
        """Parity triple → sedenion imaginary index (1-7)."""
        return p[0] * 4 + p[1] * 2 + p[2]

    _parity_blocks: Dict[int, List[int]] = {}  # L-imag index → list of IDX indices

    @classmethod
    def _build_parity_blocks(cls) -> Dict[int, List[int]]:
        if cls._parity_blocks:
            return cls._parity_blocks
        for xi in INTERIOR:
            r, s, u = xi
            p = (r % 2, s % 2, u % 2)
            idx_val = cls._parity_to_idx(p)
            cls._parity_blocks.setdefault(idx_val, []).append(IDX[xi])
        return cls._parity_blocks

    @classmethod
    def parity_blocks(cls) -> Dict[int, List[int]]:
        """Return {imag_index: [IDX indices of INTERIOR points in that parity block]}."""
        return cls._build_parity_blocks()

    # --- Cayley-Dickson arithmetic ---

    @staticmethod
    def _conj(x: List[float], lv: int) -> List[float]:
        if lv == 0:
            return x[:]
        h = len(x) // 2
        return C5_SedenionZDGraph._conj(x[:h], lv-1) + [-v for v in x[h:]]

    @staticmethod
    def _cd_mul(a: List[float], b: List[float], lv: int) -> List[float]:
        if lv == 0:
            return [a[0] * b[0]]
        h = 1 << (lv - 1)
        a1, a2 = a[:h], a[h:]
        b1, b2 = b[:h], b[h:]
        conj = C5_SedenionZDGraph._conj
        mul  = C5_SedenionZDGraph._cd_mul
        add  = lambda u, v: [x + y for x, y in zip(u, v)]
        sub  = lambda u, v: [x - y for x, y in zip(u, v)]
        c1 = sub(mul(a1, b1, lv-1), mul(conj(b2, lv-1), a2, lv-1))
        c2 = add(mul(b2, a1, lv-1), mul(a2, conj(b1, lv-1), lv-1))
        return c1 + c2

    @classmethod
    def _e(cls, i: int, lv: int = 4) -> List[float]:
        v = [0.0] * (1 << lv)
        v[i] = 1.0
        return v

    # --- ZD pair computation ---

    @classmethod
    def build_zd_pairs(cls) -> List[Tuple[Tuple, Tuple]]:
        """
        Find all weight-2 zero-divisor pairs in the sedenions, including
        sign variants.  Returns directed pairs ((i,j), (k,l)) where
        (e_i ± e_j)*(e_k ± e_l) = 0 for some sign choice.

        After sign-quotienting (identifying vertex (i,j) across ± choices),
        there are 42 vertices and 84 undirected edges, degree-4 regular.
        """
        LV = 4
        N  = 16

        def basis_elem(i):
            return cls._e(i, LV)

        def is_zero(v, tol=1e-9):
            return all(abs(x) < tol for x in v)

        # Collect sign-quotiented undirected edges
        edges: Set[Tuple[Tuple, Tuple]] = set()
        for i in range(1, 8):
            for j in range(9, 16):
                if j == i + 8:
                    continue
                for k in range(1, 8):
                    for l in range(9, 16):
                        if l == k + 8:
                            continue
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                a = [s1 * basis_elem(i)[m] + basis_elem(j)[m] for m in range(16)]
                                b = [s2 * basis_elem(k)[m] + basis_elem(l)[m] for m in range(16)]
                                prod = cls._cd_mul(a, b, LV)
                                if is_zero(prod):
                                    v1, v2 = (i, j), (k, l)
                                    edge = (min(v1, v2), max(v1, v2))
                                    edges.add(edge)

        return sorted(edges)

    @classmethod
    def zd_graph_stats(cls) -> Dict:
        """Compute the ZD graph vertex/edge statistics (sign-quotiented)."""
        edges = cls.build_zd_pairs()
        a_vertices: Set = set()
        for (v1, v2) in edges:
            a_vertices.add(v1)
            a_vertices.add(v2)

        degree: Dict = {}
        for v in a_vertices:
            nbrs = set()
            for (v1, v2) in edges:
                if v1 == v:
                    nbrs.add(v2)
                if v2 == v:
                    nbrs.add(v1)
            degree[v] = len(nbrs)

        return {
            "n_pairs": len(edges),
            "n_vertices": len(a_vertices),
            "degree_min": min(degree.values()) if degree else 0,
            "degree_max": max(degree.values()) if degree else 0,
            "degree_uniform": len(set(degree.values())) <= 1,
            "pairs": edges,
            "vertices": a_vertices,
        }

    @classmethod
    def zd_pairs_as_parity_blocks(cls) -> List[Tuple[Tuple, Tuple]]:
        """
        Return ZD pairs translated to parity-block pairs.
        Each entry: ((p_a, p_b), (p_c, p_d)) where p_x is a parity triple
        (r%2, s%2, u%2).

        Meaning: the composite element over blocks p_a, p_b annihilates
        the composite element over blocks p_c, p_d in the algebra.
        """
        edges = cls.build_zd_pairs()
        result = []
        for (i, j), (k, l) in edges:
            pa = cls._idx_to_parity(i)
            pb = cls._idx_to_parity(j - 8)
            pc = cls._idx_to_parity(k)
            pd = cls._idx_to_parity(l - 8)
            result.append(((pa, pb), (pc, pd)))
        return result

    _zd_indicator_cache = None  # (n_edges, 2, N) indicator matrices

    @classmethod
    def _build_zd_indicators(cls):
        """Precompute indicator vectors for fast ZD constraint evaluation."""
        if cls._zd_indicator_cache is not None:
            return cls._zd_indicator_cache
        blocks = cls.parity_blocks()
        edges = cls.build_zd_pairs()
        N = 19
        # For each edge, build indicator vectors for blocks A and B
        ind_a = np.zeros((len(edges), N))
        ind_b = np.zeros((len(edges), N))
        for e, ((i, j), (k, l)) in enumerate(edges):
            for idx in blocks[i] + blocks[j - 8]:
                ind_a[e, idx] = 1.0
            for idx in blocks[k] + blocks[l - 8]:
                ind_b[e, idx] = 1.0
        cls._zd_indicator_cache = (ind_a, ind_b)
        return ind_a, ind_b

    @classmethod
    def algebra_zd_constraints(cls, f: 'np.ndarray') -> List[float]:
        """
        Given a 19×19×19 structure constant tensor f, evaluate the 84 ZD
        constraints.  For each ZD edge ((i,j),(k,l)):

          Composite element A = Σ_{ξ∈P_i} e_ξ + Σ_{η∈P_{j-8}} e_η
          Composite element B = Σ_{ξ'∈P_k} e_{ξ'} + Σ_{η'∈P_{l-8}} e_{η'}
          Constraint: A * B = 0, i.e. for every output index ζ:
            Σ_{a∈P_i∪P_{j-8}} Σ_{b∈P_k∪P_{l-8}} f[a,b,ζ] = 0

        Returns a list of 84 * 19 scalar constraint values (should all be 0).
        """
        blocks = cls.parity_blocks()
        edges = cls.build_zd_pairs()
        violations = []
        N = f.shape[0]
        for (i, j), (k, l) in edges:
            block_a = blocks[i] + blocks[j - 8]
            block_b = blocks[k] + blocks[l - 8]
            for zeta in range(N):
                val = sum(f[a, b, zeta] for a in block_a for b in block_b)
                violations.append(val)
        return violations

    @classmethod
    def algebra_zd_loss_fast(cls, f: 'np.ndarray') -> float:
        """Fast sum-of-squares C5 ZD loss using precomputed indicators."""
        ind_a, ind_b = cls._build_zd_indicators()
        # For each edge e: constraint[e, zeta] = ind_a[e] @ f[:,:,zeta] @ ind_b[e]
        #   = sum_{a,b} ind_a[e,a] * f[a,b,zeta] * ind_b[e,b]
        # Vectorized: C[e, zeta] = ind_a[e] @ f_reshaped @ ind_b[e]
        # f is (19,19,19). Reshape to do batch matmul.
        # C = einsum('ea,abz,eb->ez', ind_a, f, ind_b)
        C = np.einsum('ea,abz,eb->ez', ind_a, f, ind_b)
        return float(np.sum(C ** 2))

    @classmethod
    def algebra_zd_max_violation_fast(cls, f: 'np.ndarray') -> float:
        """Fast max |violation| using precomputed indicators."""
        ind_a, ind_b = cls._build_zd_indicators()
        C = np.einsum('ea,abz,eb->ez', ind_a, f, ind_b)
        return float(np.max(np.abs(C)))

    @staticmethod
    def describe() -> str:
        return (
            "C5: The sedenion ZD graph has 42 vertices and 84 undirected pairs, "
            "degree-4 regular. Vertices are L-imaginary × U-sector pairs (e_i+e_j). "
            "The parity-block bridge maps sedenion index i to INTERIOR points with "
            "matching parity (r%2,s%2,u%2) = bits(i). ZD pairs constrain the algebra: "
            "composite block-sum elements must annihilate each other."
        )


# ---------------------------------------------------------------------------
# Canon registry
# ---------------------------------------------------------------------------

CANON = [
    C1_SupportPartition,
    C2_RDropsOut,
    C3_FiberConstraint,
    C4_Symmetry,
    C5_SedenionZDGraph,
]


def print_canon_summary():
    print("=" * 70)
    print("ADE CONSTRAINT CANON")
    print("=" * 70)
    for C in CANON:
        print(f"\n[{C.name}]")
        print(C.describe())
    print()


if __name__ == "__main__":
    print_canon_summary()

    print("\n--- Verifying C2 (r-drops-out) ---")
    result = C2_RDropsOut.verify()
    print(f"  Active modes nonzero: {result['active_nonzero']}")
    print(f"  Dead modes zero:      {result['dead_zero']}")
    print(f"  VERIFIED: {result['verified']}")

    print("\n--- C3 Fiber structure ---")
    print(f"  Forced fibers (size 1): {sorted(C3_FiberConstraint.FORCED_FIBERS.keys())}")
    print(f"  Free   fibers (size 3): {sorted(C3_FiberConstraint.FREE_FIBERS.keys())}")
    M = C3_FiberConstraint.fiber_constraint_matrix()
    print(f"  Constraint matrix shape: {M.shape}  (9 fibers x 19 terms)")

    print("\n--- C4 Symmetry group orbits ---")
    orbits = C4_Symmetry.interior_orbits()
    print(f"  G-orbits on INTERIOR: {len(orbits)}")
    for o in sorted(orbits, key=lambda x: sorted(x)[0]):
        print(f"    size {len(o)}: {sorted(o)[0]} ...")
    print("\n  Computing pair orbits (19x19=361 pairs)...")
    pair_orbs = C4_Symmetry.pair_orbits()
    print(f"  G-orbits on INTERIOR^2: {len(pair_orbs)}  (= free structure constants before other constraints)")

    print("\n--- C5 Sedenion ZD graph ---")
    stats = C5_SedenionZDGraph.zd_graph_stats()
    print(f"  ZD pairs:    {stats['n_pairs']}")
    print(f"  ZD vertices: {stats['n_vertices']}")
    print(f"  Degree range: [{stats['degree_min']}, {stats['degree_max']}]")
    print(f"  Degree-uniform: {stats['degree_uniform']}")
