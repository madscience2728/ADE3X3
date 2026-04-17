"""rank_probe.py — CP-rank estimation of T₃₃₃ seeded from FusionState(λ).

For each λ the probe:
  1. Builds the exact 9×9×9 matrix-multiplication tensor T₃₃₃.
  2. Constructs a λ-dependent 9×9 seed matrix F_λ from the fusion law.
  3. Runs a structured ALS restart (seed = top-r SVD of F_λ) plus several
     random ALS+L-BFGS-B restarts against T₃₃₃.
  4. Returns (best_residual, best_rank_estimate, best_factors).

ALS pattern copied verbatim from bini_slime/slime_search.py — no import,
no dependency on any other module in the repo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from fusion_search.fusion_state import FusionState

# ---------------------------------------------------------------------------
# Hyper-parameters (same defaults as bini_slime)
# ---------------------------------------------------------------------------
RANK_RANGE      = list(range(9, 20))    # ranks to probe (9..19 inclusive)
ALS_ITERS       = 80
LBFGS_MAXITER   = 300
RANK_TOL        = 1e-7


# ---------------------------------------------------------------------------
# Build T<3,3,3>  (copied from bini_slime/slime_search.py)
# ---------------------------------------------------------------------------

def _build_T333() -> np.ndarray:
    n = 3
    dim = n * n
    T = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                alpha = i * n + j
                beta  = j * n + k
                gamma = i * n + k
                T[alpha, beta, gamma] = 1.0
    return T


T333: np.ndarray = _build_T333()


# ---------------------------------------------------------------------------
# CP machinery  (ALS + L-BFGS-B)  — same as bini_slime/slime_search.py
# ---------------------------------------------------------------------------

def _reconstruct(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.einsum("ra,rb,rc->abc", U, V, W)


def _loss_and_grad(params: np.ndarray, T: np.ndarray, r: int):
    d1, d2, d3 = T.shape
    U = params[: r * d1].reshape(r, d1)
    V = params[r * d1 : r * (d1 + d2)].reshape(r, d2)
    W = params[r * (d1 + d2) :].reshape(r, d3)
    R = _reconstruct(U, V, W) - T
    loss = 0.5 * float(np.dot(R.ravel(), R.ravel()))
    dU = np.einsum("abc,rb,rc->ra", R, V, W)
    dV = np.einsum("abc,ra,rc->rb", R, U, W)
    dW = np.einsum("abc,ra,rb->rc", R, U, V)
    return loss, np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])


def _als_step(T: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray):
    r = U.shape[0]
    d1, d2, d3 = T.shape
    reg = 1e-10 * np.eye(r)

    def _update(T_mode, F1, F2):
        KR = np.einsum("ra,rb->rab", F1, F2).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T_mode @ KR.T
        return np.linalg.lstsq(gram, rhs.T, rcond=None)[0]

    T1 = T.reshape(d1, d2 * d3)
    U = _update(T1, V, W)
    T2 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
    V = _update(T2, U, W)
    T3 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)
    W = _update(T3, U, V)
    return U, V, W


def _one_restart(
    T_init: np.ndarray,
    rank: int,
    seed: int,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    T_target: Optional[np.ndarray] = None,
    U0: Optional[np.ndarray] = None,
    V0: Optional[np.ndarray] = None,
    W0: Optional[np.ndarray] = None,
):
    """Single ALS warm-start + L-BFGS-B restart.

    Returns (residual, U, V, W).
    """
    if T_target is None:
        T_target = T_init
    d1, d2, d3 = T_init.shape
    rng = np.random.default_rng(seed)
    scale = max((float(np.linalg.norm(T_init)) / rank) ** (1 / 3), 0.1)

    U = U0 if U0 is not None else rng.standard_normal((rank, d1)) * scale
    V = V0 if V0 is not None else rng.standard_normal((rank, d2)) * scale
    W = W0 if W0 is not None else rng.standard_normal((rank, d3)) * scale

    for _ in range(als_iters):
        U, V, W = _als_step(T_init, U, V, W)

    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    try:
        opt = minimize(
            _loss_and_grad,
            x0,
            args=(T_target, rank),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": lbfgs_maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        res = float(np.sqrt(max(2.0 * opt.fun, 0.0)))
        params = opt.x
    except Exception:
        res = float(np.linalg.norm(T_target - _reconstruct(U, V, W)))
        params = x0

    d1t, d2t, d3t = T_target.shape
    U_ = params[: rank * d1t].reshape(rank, d1t)
    V_ = params[rank * d1t : rank * (d1t + d2t)].reshape(rank, d2t)
    W_ = params[rank * (d1t + d2t) :].reshape(rank, d3t)
    return res, U_, V_, W_


# ---------------------------------------------------------------------------
# Fusion-law seed
# ---------------------------------------------------------------------------

def _fusion_seed(state: FusionState, rank: int, rng: np.random.Generator):
    """Build structured (U₀, V₀, W₀) from SVD of F_λ."""
    F = state.fusion_matrix()                      # 9×9
    U_svd, s, Vt_svd = np.linalg.svd(F, full_matrices=False)
    r_eff = min(rank, len(s))
    scale = float(s[0]) ** (1 / 3) if s[0] > 0 else 1.0

    U0 = np.zeros((rank, 9))
    V0 = np.zeros((rank, 9))
    W0 = rng.standard_normal((rank, 9)) * scale

    U0[:r_eff] = (U_svd[:, :r_eff] * (s[:r_eff] ** (1 / 3))).T
    V0[:r_eff] = (Vt_svd[:r_eff, :] * (s[:r_eff] ** (1 / 3)).reshape(-1, 1))

    # Fill remaining rows randomly
    if r_eff < rank:
        fill = rng.standard_normal((rank - r_eff, 9)) * scale * 0.01
        U0[r_eff:] = fill
        V0[r_eff:] = fill

    return U0, V0, W0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    lam: float
    residual: float
    rank_estimate: int   # rank of best residual; confirmed only when solved=True
    solved: bool
    timestamp: float
    cond_num: float = 1.0  # condition number of the fusion matrix F_λ
    per_rank: dict = None  # {rank: best_residual} for every rank tried
    factors: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def __post_init__(self):
        if self.per_rank is None:
            self.per_rank = {}


def probe(
    lam: float,
    ranks: list[int] = RANK_RANGE,
    n_restarts: int = 6,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    tol: float = RANK_TOL,
    keep_factors: bool = False,
    verbose: bool = False,
) -> ProbeResult:
    """Probe the CP rank of T₃₃₃ for a given fusion parameter λ.

    Parameters
    ----------
    lam : float
        Curvature parameter.
    ranks : list[int]
        Candidate ranks to try (ascending).  Stops early when residual < tol.
    n_restarts : int
        Random restarts per rank (one additional structured restart is always
        run from the fusion-law seed).
    als_iters : int
        ALS iterations per warm-start.
    lbfgs_maxiter : int
        L-BFGS-B iterations per restart.
    tol : float
        Residual threshold for declaring a rank achieved.
    keep_factors : bool
        If True, store the best (U, V, W) in ProbeResult.factors.
    verbose : bool
        Print per-rank progress.
    """
    state  = FusionState(lam)
    try:
        cond_num = float(np.linalg.cond(state.fusion_matrix()))
    except Exception:
        cond_num = float("inf")
    T_tgt  = T333
    t0     = time.time()
    rng_seed_base = int(abs(lam * 1e6)) % (2 ** 31)
    rng    = np.random.default_rng(rng_seed_base)

    best_res  = float("inf")
    best_rank = -1         # rank of best residual; confirmed only when solved=True
    best_UVW: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    solved    = False
    per_rank: dict[int, float] = {}

    for rank in ranks:
        rank_best = float("inf")
        rank_UVW  = None

        # -------- structured restart (fusion-law seed) --------
        U0, V0, W0 = _fusion_seed(state, rank, rng)
        res, U, V, W = _one_restart(
            T_tgt, rank,
            seed=rng_seed_base,
            als_iters=als_iters,
            lbfgs_maxiter=lbfgs_maxiter,
            U0=U0, V0=V0, W0=W0,
        )
        if res < rank_best:
            rank_best = res
            rank_UVW  = (U, V, W)

        # -------- random restarts --------
        for i in range(n_restarts):
            seed = rng_seed_base + 1 + i
            res, U, V, W = _one_restart(
                T_tgt, rank,
                seed=seed,
                als_iters=als_iters,
                lbfgs_maxiter=lbfgs_maxiter,
            )
            if res < rank_best:
                rank_best = res
                rank_UVW  = (U, V, W)

        per_rank[rank] = rank_best

        # Track which rank achieved the best residual (even if not yet solved).
        if rank_best < best_res:
            best_res  = rank_best
            best_UVW  = rank_UVW
            best_rank = rank   # rank that got closest, confirmed or not

        if rank_best < tol:
            # First (lowest) rank to achieve tolerance wins.
            best_rank = rank
            solved    = True
            if verbose:
                print(
                    f"  λ={lam:+.4f}  rank={rank}  res={rank_best:.4e}  ***SOLVED***",
                    flush=True,
                )
            break

        if verbose:
            print(
                f"  λ={lam:+.4f}  rank={rank}  res={rank_best:.4e}",
                flush=True,
            )

    return ProbeResult(
        lam=lam,
        residual=best_res,
        rank_estimate=best_rank,
        solved=solved,
        timestamp=time.time() - t0,
        cond_num=cond_num,
        per_rank=per_rank,
        factors=best_UVW if keep_factors else None,
    )


# ---------------------------------------------------------------------------
# Multi-sector probe  (accepts any state with a fusion_matrix() method)
# ---------------------------------------------------------------------------

FLOOR_RESIDUAL: float = 1.76  # Z₃-symmetric scalar curvature structural floor


def probe_state(
    state,
    ranks: list[int] = RANK_RANGE,
    n_restarts: int = 6,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    tol: float = RANK_TOL,
    keep_factors: bool = False,
    verbose: bool = False,
) -> ProbeResult:
    """Generic rank probe accepting any state object with a fusion_matrix() method.

    Identical ALS+L-BFGS-B loop to probe(), but takes a pre-built state so
    that MultiSectorFusionState (and future state types) work without modifying
    the original probe().

    The RNG seed is diversified per A₁ value so parallel workers with the same
    λ but different A₁ matrices explore distinct random restarts.
    """
    import hashlib

    lam = float(getattr(state, "lam", 0.0))
    try:
        cond_num = float(np.linalg.cond(state.fusion_matrix()))
    except Exception:
        cond_num = float("inf")

    T_tgt = T333
    t0    = time.time()

    # Seed diversification: fold the A₁ bytes into the base seed so that each
    # distinct A₁ gets a distinct random-restart sequence.
    A1_attr = getattr(state, "A1", None)
    if A1_attr is not None:
        a1_bytes = np.asarray(A1_attr, dtype=np.float64).ravel().tobytes()
        a1_seed  = int(hashlib.md5(a1_bytes).hexdigest()[:8], 16) % (2 ** 30)
    else:
        a1_seed = 0
    rng_seed_base = (int(abs(lam * 1e6)) + a1_seed) % (2 ** 31)
    rng = np.random.default_rng(rng_seed_base)

    best_res  = float("inf")
    best_rank = -1
    best_UVW: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    solved    = False
    per_rank: dict[int, float] = {}

    for rank in ranks:
        rank_best = float("inf")
        rank_UVW  = None

        # Structured restart from fusion seed
        U0, V0, W0 = _fusion_seed(state, rank, rng)
        res, U, V, W = _one_restart(
            T_tgt, rank,
            seed=rng_seed_base,
            als_iters=als_iters,
            lbfgs_maxiter=lbfgs_maxiter,
            U0=U0, V0=V0, W0=W0,
        )
        if res < rank_best:
            rank_best = res
            rank_UVW  = (U, V, W)

        # Random restarts
        for i in range(n_restarts):
            seed = rng_seed_base + 1 + i
            res, U, V, W = _one_restart(
                T_tgt, rank,
                seed=seed,
                als_iters=als_iters,
                lbfgs_maxiter=lbfgs_maxiter,
            )
            if res < rank_best:
                rank_best = res
                rank_UVW  = (U, V, W)

        per_rank[rank] = rank_best

        if rank_best < best_res:
            best_res  = rank_best
            best_UVW  = rank_UVW
            best_rank = rank

        if rank_best < tol:
            best_rank = rank
            solved    = True
            if verbose:
                print(f"  rank={rank}  res={rank_best:.4e}  ***SOLVED***", flush=True)
            break

        if verbose:
            print(f"  rank={rank}  res={rank_best:.4e}", flush=True)

    return ProbeResult(
        lam=lam,
        residual=best_res,
        rank_estimate=best_rank,
        solved=solved,
        timestamp=time.time() - t0,
        cond_num=cond_num,
        per_rank=per_rank,
        factors=best_UVW if keep_factors else None,
    )
