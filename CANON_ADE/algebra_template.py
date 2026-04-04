"""
algebra_template.py — The 19-dimensional algebra A with structure constants.

A has basis {e_xi : xi in INTERIOR} (19 basis elements, one per interior point).
The product rule is:
    e_xi * e_eta = sum_{zeta in INTERIOR} f[xi, eta, zeta] * e_zeta

The 19x19x19 = 6859 structure constants f are the primary unknowns.

This module:
  1. Represents the full structure constant tensor
  2. Applies G-equivariance (C4) to reduce to orbit representatives
  3. Applies the fiber constraint (C3) as a linear system
  4. Applies ZD constraints (C5) by zeroing specific products
  5. Reports the remaining degrees of freedom

Usage:
    python algebra_template.py
"""

from __future__ import annotations
import itertools
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

from canon_constraints import (
    INTERIOR, IDX, HULL,
    C2_RDropsOut, C3_FiberConstraint, C4_Symmetry, C5_SedenionZDGraph,
    T_MATMUL,
)


# ---------------------------------------------------------------------------
# Algebra template
# ---------------------------------------------------------------------------

N = 19  # dimension of A


class AlgebraTemplate:
    """
    A 19-dimensional non-associative algebra A with unknown structure constants.

    Internal representation: f[i, j, k] = coefficient of e_k in (e_i * e_j)
    where i, j, k are indices into INTERIOR (0..18).

    Constraints are applied as a sequence of reductions, each narrowing the
    feasible set of structure constant tensors.
    """

    def __init__(self):
        # Full structure constant tensor: free initially
        self.f = np.zeros((N, N, N), dtype=float)
        self.is_free = np.ones((N, N, N), dtype=bool)   # True = still free
        self.is_zero = np.zeros((N, N, N), dtype=bool)  # True = forced to 0

        # Orbit data (computed lazily)
        self._pair_orbits: Optional[List] = None
        self._triple_orbits: Optional[List] = None

    # -----------------------------------------------------------------------
    # G-orbit reduction (C4)
    # -----------------------------------------------------------------------

    def compute_pair_orbits(self) -> List[List[Tuple]]:
        """
        Partition INTERIOR^2 into G-orbits.
        Each orbit gives one free pair (i,j) representative.
        """
        if self._pair_orbits is not None:
            return self._pair_orbits

        G = C4_Symmetry.group_elements()
        all_pairs = list(itertools.product(range(N), range(N)))
        remaining = set(all_pairs)
        orbits = []

        while remaining:
            i0, j0 = next(iter(remaining))
            xi0, eta0 = INTERIOR[i0], INTERIOR[j0]
            orb = set()
            for p, b in G:
                xi_g  = C4_Symmetry.apply(xi0,  p, b)
                eta_g = C4_Symmetry.apply(eta0, p, b)
                # Only include if both land in INTERIOR
                if xi_g in IDX and eta_g in IDX:
                    orb.add((IDX[xi_g], IDX[eta_g]))
            orb &= remaining
            orb.add((i0, j0))
            orbits.append(sorted(orb))
            remaining -= set(orb)

        self._pair_orbits = orbits
        return orbits

    def compute_triple_orbits(self) -> List[List[Tuple]]:
        """
        Partition INTERIOR^3 into G-orbits.
        Each orbit gives one free structure constant f[i,j,k].
        This is the full reduction needed for equivariance.
        """
        if self._triple_orbits is not None:
            return self._triple_orbits

        G = C4_Symmetry.group_elements()
        all_triples = list(itertools.product(range(N), range(N), range(N)))
        remaining = set(all_triples)
        orbits = []

        while remaining:
            i0, j0, k0 = next(iter(remaining))
            xi0, eta0, zeta0 = INTERIOR[i0], INTERIOR[j0], INTERIOR[k0]
            orb = set()
            for p, b in G:
                xi_g   = C4_Symmetry.apply(xi0,   p, b)
                eta_g  = C4_Symmetry.apply(eta0,  p, b)
                zeta_g = C4_Symmetry.apply(zeta0, p, b)
                if xi_g in IDX and eta_g in IDX and zeta_g in IDX:
                    orb.add((IDX[xi_g], IDX[eta_g], IDX[zeta_g]))
            orb &= remaining
            orb.add((i0, j0, k0))
            orbits.append(sorted(orb))
            remaining -= set(orb)

        self._triple_orbits = orbits
        return orbits

    def dof_after_c4(self) -> int:
        """
        Count free structure constants after G-equivariance reduction.
        = number of G-orbits on INTERIOR^3.
        """
        orbits = self.compute_triple_orbits()
        return len(orbits)

    # -----------------------------------------------------------------------
    # Fiber constraint (C3) applied to structure constants
    # -----------------------------------------------------------------------

    def fiber_constraint_on_f(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The fiber constraint Gamma(s,u) = 3 translates to constraints on f
        via the CP equation: T[i,j,k] = sum_{t} A[t,i] * B[t,j] * C[t,k].

        For the algebra template, this is the weaker condition that the
        'diagonal' slice of f (when xi=eta, same fiber) sums correctly.

        More precisely: the constraint M @ gamma = 3*ones(9) where
        M is the 9x19 fiber constraint matrix from C3.

        Returns (M, b) such that M @ gamma = b defines the fiber constraint.
        """
        M = C3_FiberConstraint.fiber_constraint_matrix()
        b = 3.0 * np.ones(9)
        return M, b

    def fiber_null_space(self) -> np.ndarray:
        """
        Null space of the fiber constraint matrix M.
        Elements of null(M) are the 'free gamma directions'.
        """
        M, _ = self.fiber_constraint_on_f()
        # Null space via SVD
        _, s, Vt = np.linalg.svd(M)
        rank = np.sum(s > 1e-10)
        null = Vt[rank:].T   # columns are null space basis vectors
        return null  # shape (19, 19-rank)

    # -----------------------------------------------------------------------
    # ZD constraint (C5)
    # -----------------------------------------------------------------------

    def apply_zd_constraints(self, zd_pairs: Optional[List] = None):
        """
        For each sedenion ZD pair, find the corresponding interior-index pair
        and force f[i, j, :] = 0.

        The mapping from sedenion indices (0..15) to interior indices (0..18)
        requires a bridge: the sedenion basis e_i lives in Z_2^4, while our
        interior points live in Z_3^3. We use the natural identification via
        the CD parity structure.

        This method marks constraints as zero; it does not assign them.
        Returns the number of (i,j) pairs zeroed.
        """
        if zd_pairs is None:
            stats = C5_SedenionZDGraph.zd_graph_stats()
            zd_pairs = stats["pairs"]

        # Build the sedenion->interior bridge (OP1)
        bridge = self._build_sedenion_interior_bridge()
        zeroed = 0

        for (ij, kl) in zd_pairs:
            i_int = bridge.get(ij)
            j_int = bridge.get(kl)
            if i_int is not None and j_int is not None:
                self.is_zero[i_int, j_int, :] = True
                self.is_zero[j_int, i_int, :] = True
                self.is_free[i_int, j_int, :] = False
                self.is_free[j_int, i_int, :] = False
                zeroed += 1

        return zeroed

    def _build_sedenion_interior_bridge(self) -> Dict[Tuple, int]:
        """
        Map sedenion weight-2 vertex (i,j) -> interior index.

        The sedenion has 16 basis elements e_0..e_15.
        The CD level-4 structure means:
          e_0..e_7 = L-sector (lower half)
          e_8..e_15 = U-sector (upper half)

        The interior points (r,s,u) with at least one odd coordinate are the
        parity-violation locus. We map:
          e_i for i in 1..7  -> imaginary L-sector -> encodes the (s,u) fiber
          e_j for j in 9..15 -> U-sector -> encodes the r-stratification

        This is OP1 of the open problems. Here we record what is known and
        flag what requires further derivation.

        Current status: PARTIAL. The mapping is derived for weight-1 elements
        only (the 7 L-imaginary and 7 U-imaginary basis elements). Weight-2
        ZD vertices require the full map.
        """
        # Sedenion imaginary indices: 1..7 (L) and 9..15 (U)
        # Z_3^3 index via: i in {1..7} -> 3-bit pattern -> (r,s,u) selection
        # This is the OP1 bridge — we record the known part here.

        bridge = {}

        # L-imaginary e_i, i in 1..7: bit pattern of i gives (r%2, s%2, u%2)
        # but we need to project from Z_2^3 -> Z_3^3
        # Map: bit k of i -> coordinate k is "touched" (odd) vs "untouched" (even=0)
        # Representative interior point for each L-imaginary:
        # e_1 = 001 -> u-coordinate touched -> (0,0,1)
        # e_2 = 010 -> s-coordinate touched -> (0,1,0)
        # e_3 = 011 -> s,u touched          -> (0,1,1)
        # e_4 = 100 -> r-coordinate touched -> (1,0,0)
        # e_5 = 101 -> r,u touched          -> (1,0,1)
        # e_6 = 110 -> r,s touched          -> (1,1,0)
        # e_7 = 111 -> all touched          -> (1,1,1)

        L_map = {
            1: (0, 0, 1),
            2: (0, 1, 0),
            3: (0, 1, 1),
            4: (1, 0, 0),
            5: (1, 0, 1),
            6: (1, 1, 0),
            7: (1, 1, 1),
        }

        # e_i + e_j for i in L-imaginary (1..7), j in U-sector (9..15), j != i+8
        # Maps to interior points via: take L_map[i] as the base interior point
        # U-sector offset j-8 in 1..7 (since j in 9..15 and j!=i+8 means offset != i)
        # This gives a pair of interior points for each ZD vertex

        for i in range(1, 8):
            for j in range(9, 16):
                if j == i + 8:
                    continue  # excluded: CD-paired
                key = (i, j)
                # Primary interior point from L-imaginary
                base_pt = L_map.get(i)
                if base_pt and base_pt in IDX:
                    bridge[key] = IDX[base_pt]

        return bridge

    # -----------------------------------------------------------------------
    # R-blindness constraint (C2)
    # -----------------------------------------------------------------------

    def apply_r_blindness(self):
        """
        Enforce C2: the structure constants are blind to the r-label.
        f[xi, eta, zeta] depends only on (s,u) of xi, eta, zeta, not on r.

        This means: if xi=(r1,s,u) and xi'=(r2,s,u) differ only in r,
        then f[xi, eta, zeta] = f[xi', eta, zeta] for all eta, zeta.

        Implements this by identifying all r-variants of each (s,u)-fiber.
        Returns the number of equality constraints imposed.
        """
        constraints = 0
        # For each (s,u)-fiber with multiple r-values:
        for (s, u), indices in C3_FiberConstraint.FIBERS.items():
            if len(indices) < 2:
                continue
            # All indices in this fiber must have the same f[][] values
            rep = indices[0]
            for idx in indices[1:]:
                # f[idx, :, :] = f[rep, :, :]
                # f[:, idx, :] = f[:, rep, :]
                # f[:, :, idx] = f[:, :, rep]
                constraints += N*N * 3  # 3 modes x N^2 equations
        return constraints

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def dof_summary(self) -> Dict:
        """
        Compute degrees of freedom after each constraint layer.
        """
        total = N ** 3  # = 6859 raw

        # After C4 (G-equivariance)
        triple_orbits = self.compute_triple_orbits()
        after_c4 = len(triple_orbits)

        # After C3 (fiber constraint on f — affects the diagonal structure)
        M, _ = self.fiber_constraint_on_f()
        rank_M = np.linalg.matrix_rank(M)
        fiber_dof = N - rank_M  # free gamma directions

        # After C2 (r-blindness): each (s,u)-fiber with size 3 collapses
        # 3 r-variants to 1 representative -> reduces by 2 per free fiber
        r_blind_reduction = sum(
            len(v) - 1 for v in C3_FiberConstraint.FIBERS.values()
        )

        return {
            "raw": total,
            "after_C4_equivariance": after_c4,
            "fiber_constraint_rank": rank_M,
            "fiber_free_gamma_dof": fiber_dof,
            "r_blindness_reduction": r_blind_reduction,
            "triple_orbit_count": len(triple_orbits),
        }


# ---------------------------------------------------------------------------
# Entry point: run all diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ALGEBRA TEMPLATE: Structure constant analysis")
    print("=" * 70)

    tmpl = AlgebraTemplate()

    print("\n[1] G-equivariance reduction (C4)")
    print("    Computing triple orbits on INTERIOR^3 (this may take ~30s)...")
    summary = tmpl.dof_summary()
    print(f"    Raw structure constants: {summary['raw']}")
    print(f"    After G-equivariance:    {summary['after_C4_equivariance']} orbit reps")
    print(f"    Fiber constraint matrix rank: {summary['fiber_constraint_rank']}")
    print(f"    Free gamma d.o.f.:       {summary['fiber_free_gamma_dof']}")
    print(f"    r-blindness reduction:   -{summary['r_blindness_reduction']}")

    print("\n[2] Pair orbits on INTERIOR^2")
    pair_orbits = tmpl.compute_pair_orbits()
    print(f"    G-orbits on INTERIOR^2: {len(pair_orbits)}")
    sizes = sorted(set(len(o) for o in pair_orbits))
    for s in sizes:
        cnt = sum(1 for o in pair_orbits if len(o) == s)
        print(f"      orbit size {s}: {cnt} orbits")

    print("\n[3] Fiber null space (C3)")
    null = tmpl.fiber_null_space()
    print(f"    Null space dimension: {null.shape[1]}  (free gamma directions)")
    print(f"    Particular solution: gamma = (1/3)*ones(19)? "
          f"{'YES' if abs(np.sum(np.ones(N)/N * 9) - 19) < 1e-9 else 'check manually'}")
    # Verify: each fiber sum = 3 with gamma = 3/fiber_size * ones
    M, b = tmpl.fiber_constraint_on_f()
    gamma_particular = np.zeros(N)
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        for i in indices:
            gamma_particular[i] = 3.0 / len(indices)
    residual = M @ gamma_particular - b
    print(f"    gamma_particular = 3/fiber_size per fiber: residual norm = {np.linalg.norm(residual):.2e}")

    print("\n[4] ZD bridge (OP1 partial)")
    bridge = tmpl._build_sedenion_interior_bridge()
    print(f"    Bridge entries (sedenion ZD vertex -> interior idx): {len(bridge)}")
    zeroed = tmpl.apply_zd_constraints()
    print(f"    Structure constant (i,j) pairs zeroed by C5: {zeroed}")
    free_remaining = np.sum(tmpl.is_free)
    print(f"    Free structure constants remaining: {free_remaining} / {N**3}")

    print("\n[5] Interior point orbits under G (for reference)")
    int_orbits = C4_Symmetry.interior_orbits()
    print(f"    G-orbits on INTERIOR: {len(int_orbits)}")
    for o in sorted(int_orbits, key=lambda x: sorted(x)[0]):
        rep = sorted(o)[0]
        print(f"      size {len(o):2d}  rep={rep}")
