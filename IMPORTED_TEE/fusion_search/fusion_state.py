"""FusionState — one point in the scalar curvature search space.

Search domain: λ ∈ ℝ  (a single real parameter)
Fusion algebra: S = ℝ²

Fixed observable geometry (never searched over):
    u₀ = e₁ = [1,0]     v₀ = e₁ = [1,0]
    u₁ = e₂ = [0,1]     v₁ = e₂ = [0,1]
    u₂ = e₁+e₂ = [1,1]  v₂ = e₁+e₂ = [1,1]
    M₀ = I,  M₁ = [[-1,1],[1,0]],  M₂ = I

Scalar curvature stratum (parameterised by λ):
    N₀ = 0,  N₁ = λ·M₁,  N₂ = 0

Local fusion laws  K_t : S⊗S → S:
    K₀(x,y) = K₂(x,y) = [x·y,  0]
    K₁(x,y) = (x^T M₁ y) · [1, λ]

Global decode  (fixed for all λ):
    d_t = [1, 0]  (Γ projects onto first coordinate)
    α₀ = α₂ = 1,  α₁ + λβ₁ = 1  solved by d₁ = [1,0]

Observable correctness constraint:
    u_a^T M_{a+b} v_b = 1   for all  a, b ∈ ℤ₃
This is automatically satisfied by the geometry above (verified in
verify_observables()).

Z₃ grading of M₃(ℝ) basis elements  e_{ij}  (flat index k = 3i+j):
    grade(k) = (k%3 - k//3) % 3
    grade-0: flat indices [0, 4, 8]   →  (0,0),(1,1),(2,2)
    grade-1: flat indices [1, 5, 6]   →  (0,1),(1,2),(2,0)
    grade-2: flat indices [2, 3, 7]   →  (0,2),(1,0),(2,1)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed observable geometry
# ---------------------------------------------------------------------------

_U: list[np.ndarray] = [
    np.array([1.0, 0.0]),   # u₀ = e₁
    np.array([0.0, 1.0]),   # u₁ = e₂
    np.array([1.0, 1.0]),   # u₂ = e₁+e₂
]

_V: list[np.ndarray] = [
    np.array([1.0, 0.0]),   # v₀ = e₁
    np.array([0.0, 1.0]),   # v₁ = e₂
    np.array([1.0, 1.0]),   # v₂ = e₁+e₂
]

_M0 = np.eye(2)
_M1 = np.array([[-1.0, 1.0], [1.0, 0.0]])
_M2 = np.eye(2)

_M = [_M0, _M1, _M2]  # indexed by Z₃

# Flat-index → Z₃ grade for k = 3i+j:  grade = (j - i) % 3
_GRADE: np.ndarray = np.array(
    [(k % 3 - k // 3) % 3 for k in range(9)], dtype=np.int32
)
# Position of flat index k within its grade (0, 1, or 2) = row index = k // 3
_POS: np.ndarray = np.array([k // 3 for k in range(9)], dtype=np.int32)


# ---------------------------------------------------------------------------
# FusionState
# ---------------------------------------------------------------------------

class FusionState:
    """Encapsulates the fusion algebra at curvature parameter λ.

    Parameters
    ----------
    lam : float
        The single real search parameter.  λ=0 is the flat/associative case.
    """

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)

        # Fixed matrices
        self.M = [m.copy() for m in _M]

        # Derived N matrices
        self.N = [
            np.zeros((2, 2)),           # N₀ = 0
            self.lam * _M1.copy(),      # N₁ = λ·M₁
            np.zeros((2, 2)),           # N₂ = 0
        ]

        # Decode vectors  d_t = [1,0] for all t
        self.d = [np.array([1.0, 0.0]) for _ in range(3)]

        self._verify()

    # ------------------------------------------------------------------
    # Core fusion law
    # ------------------------------------------------------------------

    def K(self, t: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute K_t(x, y) = [x^T M_t y,  x^T N_t y] ∈ ℝ²."""
        t = int(t) % 3
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return np.array([
            float(x @ self.M[t] @ y),
            float(x @ self.N[t] @ y),
        ])

    def K_decoded(self, t: int, x: np.ndarray, y: np.ndarray) -> float:
        """Apply Γ·K_t(x,y) — returns the single scalar seen by the output."""
        return float(self.d[t] @ self.K(t, x, y))

    # ------------------------------------------------------------------
    # Observable correctness
    # ------------------------------------------------------------------

    def verify_observables(self) -> dict[tuple[int, int], float]:
        """Check u_a^T M_{a+b} v_b = 1 for all (a,b) ∈ ℤ₃².

        Returns a dict mapping (a,b) → value.  All values must be 1.0.
        """
        results = {}
        for a in range(3):
            for b in range(3):
                t = (a + b) % 3
                val = float(_U[a] @ _M[t] @ _V[b])
                results[(a, b)] = val
        return results

    def _verify(self) -> None:
        obs = self.verify_observables()
        bad = {k: v for k, v in obs.items() if abs(v - 1.0) > 1e-12}
        if bad:
            raise RuntimeError(f"Observable correctness violated: {bad}")

    # ------------------------------------------------------------------
    # Fusion matrix (λ-dependent ALS seed)
    # ------------------------------------------------------------------

    def fusion_matrix(self) -> np.ndarray:
        """Build the 9×9 λ-dependent seed matrix for ALS on T₃₃₃.

        Entry F[α, β] uses the full K_t output (both components):
            F[α,β] = K_t(u_{row(α)}, v_{row(β)})[0]
                   + K_t(u_{row(α)}, v_{row(β)})[1]
                   = (1 + λ·δ_{t,1}) × u_{row(α)}^T M_t v_{row(β)}

        where t = (grade(α) + grade(β)) % 3 and row(k) = k // 3.

        For λ=0 the second component is zero (flat case).
        For λ≠0 the grade-1 interaction entries are scaled by (1+λ),
        making the matrix, and hence the ALS initialisation, λ-sensitive.
        """
        F = np.zeros((9, 9))
        for alpha in range(9):
            for beta in range(9):
                t = int((_GRADE[alpha] + _GRADE[beta]) % 3)
                p = int(_POS[alpha])   # row of A = position within grade
                q = int(_POS[beta])    # row of B
                kval = self.K(t, _U[p], _V[q])
                F[alpha, beta] = kval[0] + kval[1]
        return F

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"FusionState(λ={self.lam:.6g})"


# ---------------------------------------------------------------------------
# MultiSectorFusionState — per-sector curvature (symmetry breaking)
# ---------------------------------------------------------------------------

class MultiSectorFusionState:
    """Fusion algebra with per-sector curvature matrices — breaks Z₃ symmetry.

    Search parameters: 5 total
        λ   — scalar (A₀ = λI, A₂ = λI, both fixed to this value)
        A₁  — free 2×2 matrix (4 independent entries)

    Fusion laws:
        K₀(x,y) = [x^T M₀ y,  x^T (λI·M₀) y]  =  [x·y,  λ(x·y)]
        K₁(x,y) = [x^T M₁ y,  x^T (A₁·M₁) y]
        K₂(x,y) = [x^T M₂ y,  x^T (λI·M₂) y]  =  [x·y,  λ(x·y)]

    The scalar (Z₃-symmetric) ansatz is recovered when A₁ = λI.  A free A₁
    allows ALS to escape the structural floor at residual ≈ 1.76.

    The fixed observable geometry (M_t, u_a, v_b) is never modified.  The
    observable correctness constraint u_a^T M_{a+b} v_b = 1 depends only on
    M matrices and _U/_V vectors, which are unchanged.
    """

    def __init__(self, lam: float, A1: np.ndarray) -> None:
        self.lam = float(lam)
        self.A1  = np.asarray(A1, dtype=np.float64).reshape(2, 2)

        lam_I = self.lam * np.eye(2)
        # N_t = A_t · M_t
        self.N = [
            lam_I @ _M0,    # N₀ = λI·I  = λI
            self.A1 @ _M1,  # N₁ = A₁·M₁  (free 2×2)
            lam_I @ _M2,    # N₂ = λI·I  = λI
        ]
        self.d = [np.array([1.0, 0.0]) for _ in range(3)]

    # ------------------------------------------------------------------
    # Core fusion law
    # ------------------------------------------------------------------

    def K(self, t: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """K_t(x,y) = [x^T M_t y,  x^T N_t y] ∈ ℝ²."""
        t = int(t) % 3
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return np.array([
            float(x @ _M[t] @ y),
            float(x @ self.N[t] @ y),
        ])

    def K_decoded(self, t: int, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.d[t] @ self.K(t, x, y))

    # ------------------------------------------------------------------
    # Fusion matrix (A₁-dependent ALS seed)
    # ------------------------------------------------------------------

    def fusion_matrix(self) -> np.ndarray:
        """Build the 9×9 multi-sector seed matrix for ALS on T₃₃₃.

        Same index structure as FusionState.fusion_matrix():
            F[α,β] = K_t(u_p, v_q)[0] + K_t(u_p, v_q)[1]
        """
        F = np.zeros((9, 9))
        for alpha in range(9):
            for beta in range(9):
                t = int((_GRADE[alpha] + _GRADE[beta]) % 3)
                p = int(_POS[alpha])
                q = int(_POS[beta])
                kval = self.K(t, _U[p], _V[q])
                F[alpha, beta] = kval[0] + kval[1]
        return F

    def __repr__(self) -> str:  # pragma: no cover
        return f"MultiSectorFusionState(λ={self.lam:.4g}, A1={self.A1.tolist()})"
