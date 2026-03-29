"""
ade3x3_step77_border_rank_singular_pair_witness.py

Step 77: Border Rank Singular-Pair Witness Construction.

For each of the 59 constant-multiple AlphaTensor pairs identified in Step 70,
we test whether a singular-pair deformation can produce a border-rank reduction.

Background
----------
A pair (t_i, t_j) is a "constant-multiple" pair when their combined alpha*beta
index support intersects significantly (Step 70 found 59 such pairs with ratio 1
or -1 on common entries).  The singular construction is:

  Consider the one-parameter family
      f_i(eps) = t_i  (alpha unchanged)
      f_j(eps) = t_j + eps * delta
  where delta is chosen to make the *combined* contribution alpha*beta-singular
  as eps -> 0.  If the eps->0 limit of (f_i + f_j) equals a *single* rank-1
  term, the pair can be "fused" and rank would drop.

The necessary condition for this fusion to work is that the gamma output
vectors of t_i and t_j are *compatible*: specifically

    Condition A (Disjoint):  support(gamma_i) ∩ support(gamma_j) = ∅
    Condition B (Proportional): gamma_j = k * gamma_i for some scalar k

Under Condition A: the fused term needs gamma_i + gamma_j, which has rank-1
contribution to all outputs in the union — possible if alpha*beta supports align.

Under Condition B: gamma_i and gamma_j contribute to exactly the same outputs
with a fixed ratio.  The fused term has gamma = gamma_i (rescaled).

Neither condition alone guarantees border rank reduction; each condition is a
necessary check that filters out pairs where fusion is algebraically impossible.

We also compute the rank-1 "target" tensor for each pair:
    T_pair = T_i + T_j   (rank <= 2 submatrix of the full decomposition)
and measure:
    - flattening lower bound for T_pair (gives rank >= flattening_lb)
    - whether T_pair is exactly rank 1 (would confirm fusion is possible without
      any deformation)
    - singular value structure of T_pair's mode-1 unfolding

Finally, for ALL 59 pairs we try a direct numerical search: find a rank-1 tensor
R such that (T_full - T_pair + R) has rank <= 22 exactly.  This is a bounded
randomized 22-restart search per pair.

Outputs
-------
outputs/exports/step77_pair_analysis.csv
    One row per constant-multiple pair: gamma conditions, mode-1 rank,
    flattening lb, exact rank-1 status, search result.

outputs/exports/step77_summary.csv
    Scalar key-value summary for the canon doc generator.

outputs/exports/step77_border_rank_witnesses.csv
    Any pair for which the numerical search found an exact rank-22 decomposition.
"""

from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
)

EXPORTS = Path("outputs/exports")
EXACT_TOL = 1e-6
SEARCH_RESTARTS = int(os.getenv("STEP77_SEARCH_RESTARTS", "50"))
SEARCH_MAXITER = int(os.getenv("STEP77_SEARCH_MAXITER", "2000"))
TARGET_RANK = 22  # we aim to beat rank 23
WORKERS = max(1, os.cpu_count() or 1)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  Wrote -> {path}")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Load Step-70 pair table
# ---------------------------------------------------------------------------

def load_constant_multiple_pairs() -> list[dict]:
    """Return the 59 constant-multiple pairs from the Step 70 export."""
    rows = read_csv_rows(EXPORTS / "step70_alphatensor_pair_ratios.csv")
    return [r for r in rows if r["classification"] == "constant_multiple"]


# ---------------------------------------------------------------------------
# Gamma-support conditions
# ---------------------------------------------------------------------------

def gamma_support(g: np.ndarray) -> frozenset:
    return frozenset(int(k) for k in range(9) if g.flatten()[k] != 0)


def check_gamma_conditions(
    gi: np.ndarray, gj: np.ndarray
) -> tuple[bool, bool, Optional[float]]:
    """
    Returns (disjoint, proportional, ratio).
    disjoint:     support(gi) ∩ support(gj) = ∅
    proportional: gj = k*gi for some scalar k (or gi = k*gj)
    ratio:        scalar ratio if proportional, else None
    """
    si = gamma_support(gi)
    sj = gamma_support(gj)
    disjoint = len(si & sj) == 0

    proportional = False
    ratio: Optional[float] = None
    gi_f = gi.flatten().astype(float)
    gj_f = gj.flatten().astype(float)
    # Check proportionality: gj = k * gi
    nz_i = np.where(gi_f != 0)[0]
    nz_j = np.where(gj_f != 0)[0]
    if len(nz_i) > 0 and len(nz_j) > 0 and len(nz_i) == len(nz_j) and set(nz_i) == set(nz_j):
        ratios = gj_f[nz_i] / gi_f[nz_i]
        if np.allclose(ratios, ratios[0], atol=1e-9):
            proportional = True
            ratio = float(ratios[0])

    return disjoint, proportional, ratio


# ---------------------------------------------------------------------------
# Tensor construction and flattening lower bound
# ---------------------------------------------------------------------------

def term_tensor(t: Term) -> np.ndarray:
    a = t.alpha.reshape(-1).astype(float)
    b = t.beta.reshape(-1).astype(float)
    g = t.gamma.reshape(-1).astype(float)
    return np.einsum("i,j,k->ijk", a, b, g)


def flattening_rank(T: np.ndarray, mode: int) -> int:
    """Rank of the mode-n unfolding of a 9x9x9 tensor."""
    if mode == 0:
        M = T.reshape(9, 81)
    elif mode == 1:
        M = T.transpose(1, 0, 2).reshape(9, 81)
    else:
        M = T.transpose(2, 0, 1).reshape(9, 81)
    return int(np.linalg.matrix_rank(M, tol=1e-9))


def pair_tensor(terms: list[Term], ti: str, tj: str) -> np.ndarray:
    """Rank-at-most-2 sub-tensor contributed by the pair."""
    term_map = {t.term_id: t for t in terms}
    return term_tensor(term_map[ti]) + term_tensor(term_map[tj])


# ---------------------------------------------------------------------------
# Rank-22 search: replace pair with a single rank-1 term
# ---------------------------------------------------------------------------

def _cp_pack22(factors: list[np.ndarray]) -> np.ndarray:
    """Pack 22 (alpha, beta, gamma) triples into a flat parameter vector."""
    return np.concatenate([f.flatten() for f in factors])


def _cp_unpack22(x: np.ndarray, rank: int = TARGET_RANK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = x[:rank * 9].reshape(rank, 9)
    beta = x[rank * 9:rank * 18].reshape(rank, 9)
    gamma = x[rank * 18:].reshape(rank, 9)
    return alpha, beta, gamma


def _reconstruct(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.einsum("ri,rj,rk->ijk", alpha, beta, gamma)


def _loss_pair_replacement(
    x: np.ndarray,
    T_reduced: np.ndarray,
    rank: int,
) -> float:
    """Loss for fitting T_reduced with rank-`rank` CP decomposition."""
    alpha, beta, gamma = _cp_unpack22(x, rank)
    T_hat = _reconstruct(alpha, beta, gamma)
    diff = T_hat - T_reduced
    return float(np.sum(diff ** 2))


def search_rank22_replacement(
    T_full: np.ndarray,
    T_pair: np.ndarray,
    restarts: int = SEARCH_RESTARTS,
    maxiter: int = SEARCH_MAXITER,
    seed_offset: int = 0,
    rank: int = TARGET_RANK,
) -> dict:
    """
    Try to express (T_full - T_pair) as a rank-`rank` tensor.

    If this succeeds, then T_full = (T_full - T_pair) + T_pair can be expressed
    as rank + 1 terms total (the rank many from T_full - T_pair plus exactly
    the pair contribution which itself has rank <= 2, but we're replacing the pair
    with ONE rank-1 term by the border rank construction).

    Actually what we want:  T_full = R22 + R1_new, where R22 is rank <= 22
    and R1_new is a single rank-1 tensor that approximates T_pair.
    Equivalently: T_full - T_pair + R1_fit fits as rank-22,
    which is equivalent to (T_full - T_pair) being rank <= 22 minus 1 = 21.

    Simpler: just check if (T_full - T_pair) has rank < 22.
    If rank(T_full - T_pair) <= 21, then rank(T_full) <= 21 + 1 = 22.

    We approximate by trying to fit (T_full - T_pair) with rank-21 CP.
    """
    T_target = T_full - T_pair  # we want this to be rank <= 21
    search_rank = rank - 1  # 21

    best_loss = float("inf")
    best_residual = float("inf")
    exact_hit = False
    best_x = None

    n_params = search_rank * 27  # 3 matrices of shape (search_rank, 9)

    rng = np.random.default_rng(77000 + seed_offset)

    for restart in range(restarts):
        x0 = rng.standard_normal(n_params)
        result = minimize(
            _loss_pair_replacement,
            x0,
            args=(T_target, search_rank),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-30, "gtol": 1e-12},
        )
        loss = result.fun
        if loss < best_loss:
            best_loss = loss
            best_x = result.x

    if best_x is not None:
        alpha, beta, gamma = _cp_unpack22(best_x, search_rank)
        T_hat = _reconstruct(alpha, beta, gamma)
        diff = T_hat - T_target
        best_residual = float(np.max(np.abs(diff)))
        exact_hit = best_residual < EXACT_TOL

    return {
        "search_rank": search_rank,
        "restarts": restarts,
        "best_loss": best_loss,
        "best_max_abs_residual": best_residual,
        "exact_hit": exact_hit,
    }


# ---------------------------------------------------------------------------
# Parallel pair worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def pair_worker(payload: dict) -> dict:
    """Process one pair: compute gamma conditions + run numerical rank search."""
    import numpy as np
    from scipy.optimize import minimize

    ti_id: str = payload["term_i"]
    tj_id: str = payload["term_j"]
    ab_common: int = payload["ab_common_support"]
    pair_index: int = payload["pair_index"]
    restarts: int = payload["restarts"]
    maxiter: int = payload["maxiter"]
    exact_tol: float = payload["exact_tol"]
    target_rank: int = payload["target_rank"]

    ai = np.array(payload["alpha_i"], dtype=np.float64).reshape(-1)
    bi = np.array(payload["beta_i"],  dtype=np.float64).reshape(-1)
    gi = np.array(payload["gamma_i"], dtype=np.float64).reshape(-1)
    aj = np.array(payload["alpha_j"], dtype=np.float64).reshape(-1)
    bj = np.array(payload["beta_j"],  dtype=np.float64).reshape(-1)
    gj = np.array(payload["gamma_j"], dtype=np.float64).reshape(-1)
    T_full = np.array(payload["T_full"], dtype=np.float64)

    # Gamma conditions
    si = frozenset(int(k) for k in range(9) if gi[k] != 0)
    sj = frozenset(int(k) for k in range(9) if gj[k] != 0)
    inter = si & sj
    disjoint = len(inter) == 0

    proportional = False
    ratio: Optional[float] = None
    nz_i = [k for k in range(9) if gi[k] != 0]
    nz_j = [k for k in range(9) if gj[k] != 0]
    if nz_i and nz_j and len(nz_i) == len(nz_j) and set(nz_i) == set(nz_j):
        ratios = gj[nz_i] / gi[nz_i]
        if np.allclose(ratios, ratios[0], atol=1e-9):
            proportional = True
            ratio = float(ratios[0])

    # Flattening ranks of T_pair
    T_i = np.einsum("i,j,k->ijk", ai, bi, gi)
    T_j = np.einsum("i,j,k->ijk", aj, bj, gj)
    T_pair = T_i + T_j

    def flat_rank(T, mode):
        if mode == 0:
            M = T.reshape(9, 81)
        elif mode == 1:
            M = T.transpose(1, 0, 2).reshape(9, 81)
        else:
            M = T.transpose(2, 0, 1).reshape(9, 81)
        return int(np.linalg.matrix_rank(M, tol=1e-9))

    m0 = flat_rank(T_pair, 0)
    m1 = flat_rank(T_pair, 1)
    m2 = flat_rank(T_pair, 2)
    flat_lb = max(m0, m1, m2)
    is_rank1 = (flat_lb <= 1)

    # Numeric search: fit T_full - T_pair at rank (target_rank - 1)
    T_target = T_full - T_pair
    search_rank = target_rank - 1  # 21
    n_params = search_rank * 27

    def _loss(x):
        alpha = x[:search_rank * 9].reshape(search_rank, 9)
        beta  = x[search_rank * 9:search_rank * 18].reshape(search_rank, 9)
        gamma = x[search_rank * 18:].reshape(search_rank, 9)
        diff = np.einsum("ri,rj,rk->ijk", alpha, beta, gamma) - T_target
        return float(np.sum(diff ** 2))

    rng = np.random.default_rng(77000 + pair_index)
    best_loss = math.inf
    best_x = None
    for _ in range(restarts):
        x0 = rng.standard_normal(n_params)
        res = minimize(_loss, x0, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-30, "gtol": 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x

    best_residual = math.inf
    exact_hit = False
    if best_x is not None:
        alpha = best_x[:search_rank * 9].reshape(search_rank, 9)
        beta  = best_x[search_rank * 9:search_rank * 18].reshape(search_rank, 9)
        gamma = best_x[search_rank * 18:].reshape(search_rank, 9)
        diff = np.einsum("ri,rj,rk->ijk", alpha, beta, gamma) - T_target
        best_residual = float(np.max(np.abs(diff)))
        exact_hit = best_residual < exact_tol

    return {
        "term_i": ti_id,
        "term_j": tj_id,
        "pair_index": pair_index,
        "ab_common_support": ab_common,
        "gamma_i_support_size": len(si),
        "gamma_j_support_size": len(sj),
        "gamma_intersection_size": len(inter),
        "gamma_disjoint": disjoint,
        "gamma_proportional": proportional,
        "gamma_ratio": ratio,
        "T_pair_mode0_rank": m0,
        "T_pair_mode1_rank": m1,
        "T_pair_mode2_rank": m2,
        "T_pair_flattening_lb": flat_lb,
        "T_pair_is_rank1": is_rank1,
        "search_rank": search_rank,
        "search_restarts": restarts,
        "search_best_loss": best_loss,
        "search_best_max_abs_residual": best_residual,
        "search_exact_hit": exact_hit,
        "provenance": "MEASURED_FROM_CODE",
    }


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    term_i: str
    term_j: str
    ab_common_support: int
    gamma_i_support_size: int
    gamma_j_support_size: int
    gamma_intersection_size: int
    gamma_disjoint: bool
    gamma_proportional: bool
    gamma_ratio: Optional[float]
    T_pair_mode0_rank: int
    T_pair_mode1_rank: int
    T_pair_mode2_rank: int
    T_pair_flattening_lb: int
    T_pair_is_rank1: bool
    search_rank: int
    search_restarts: int
    search_best_loss: float
    search_best_residual: float
    search_exact_hit: bool
    provenance: str = "MEASURED_FROM_CODE"


def analyse_pair(
    terms: list[Term],
    pair_row: dict,
    T_full: np.ndarray,
    pair_index: int,
) -> PairResult:
    ti_id = pair_row["term_i"]
    tj_id = pair_row["term_j"]
    term_map = {t.term_id: t for t in terms}
    ti = term_map[ti_id]
    tj = term_map[tj_id]

    # Gamma support conditions
    gi = ti.gamma.flatten().astype(float)
    gj = tj.gamma.flatten().astype(float)
    si = gamma_support(ti.gamma)
    sj = gamma_support(tj.gamma)
    inter = si & sj
    disjoint, proportional, ratio = check_gamma_conditions(ti.gamma, tj.gamma)

    # Pair sub-tensor
    T_pair = pair_tensor(terms, ti_id, tj_id)
    m0 = flattening_rank(T_pair, 0)
    m1 = flattening_rank(T_pair, 1)
    m2 = flattening_rank(T_pair, 2)
    flat_lb = max(m0, m1, m2)
    is_rank1 = (flat_lb <= 1)

    # Alpha*beta common support size (from Step 70 export)
    ab_common = int(pair_row["common_support_size"])

    # Numerical rank-22 search
    search_result = search_rank22_replacement(T_full, T_pair, seed_offset=pair_index)

    return PairResult(
        term_i=ti_id,
        term_j=tj_id,
        ab_common_support=ab_common,
        gamma_i_support_size=len(si),
        gamma_j_support_size=len(sj),
        gamma_intersection_size=len(inter),
        gamma_disjoint=disjoint,
        gamma_proportional=proportional,
        gamma_ratio=ratio,
        T_pair_mode0_rank=m0,
        T_pair_mode1_rank=m1,
        T_pair_mode2_rank=m2,
        T_pair_flattening_lb=flat_lb,
        T_pair_is_rank1=is_rank1,
        search_rank=search_result["search_rank"],
        search_restarts=search_result["restarts"],
        search_best_loss=search_result["best_loss"],
        search_best_residual=search_result["best_max_abs_residual"],
        search_exact_hit=search_result["exact_hit"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading AlphaTensor rank-23 terms...")
    terms, gamma_orientation, recon_max_abs = load_public_rank23_terms()
    print(f"  Orientation: {gamma_orientation}, reconstruction max-abs: {recon_max_abs}")

    T_true = matrix_multiplication_tensor(3).astype(float)
    T_full = sum(term_tensor(t) for t in terms)
    full_residual = float(np.max(np.abs(T_full - T_true)))
    print(f"  Full rank-23 reconstruction max-abs residual: {full_residual:.2e}")

    print("Loading Step 70 constant-multiple pairs...")
    pairs = load_constant_multiple_pairs()
    print(f"  Found {len(pairs)} constant-multiple pairs")

    # Build payloads for parallel dispatch
    term_map = {t.term_id: t for t in terms}
    T_full_list = T_full.tolist()
    payloads = []
    for idx, pair_row in enumerate(pairs):
        ti = term_map[pair_row["term_i"]]
        tj = term_map[pair_row["term_j"]]
        payloads.append({
            "term_i": pair_row["term_i"],
            "term_j": pair_row["term_j"],
            "pair_index": idx,
            "ab_common_support": int(pair_row["common_support_size"]),
            "alpha_i": ti.alpha.tolist(),
            "beta_i":  ti.beta.tolist(),
            "gamma_i": ti.gamma.tolist(),
            "alpha_j": tj.alpha.tolist(),
            "beta_j":  tj.beta.tolist(),
            "gamma_j": tj.gamma.tolist(),
            "T_full": T_full_list,
            "restarts": SEARCH_RESTARTS,
            "maxiter": SEARCH_MAXITER,
            "exact_tol": EXACT_TOL,
            "target_rank": TARGET_RANK,
        })

    print(f"  Dispatching {len(payloads)} pairs across {WORKERS} workers...", flush=True)

    # Run per-pair analysis in parallel
    raw_results: dict[int, dict] = {}
    disjoint_count = 0
    proportional_count = 0
    rank1_pair_count = 0
    exact_hit_count = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(pair_worker, p): p["pair_index"] for p in payloads}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            raw_results[res["pair_index"]] = res
            completed += 1
            flags = []
            if res["gamma_disjoint"]:     flags.append("DISJOINT_GAMMA")
            if res["gamma_proportional"]: flags.append("PROPORTIONAL_GAMMA")
            if res["T_pair_is_rank1"]:    flags.append("PAIR_RANK1")
            if res["search_exact_hit"]:   flags.append("EXACT_HIT_R22")
            flags_str = ", ".join(flags) if flags else "no condition passes"
            print(f"  [{completed:02d}/{len(payloads)}] {res['term_i']},{res['term_j']} ... "
                  f"{flags_str} "
                  f"resid={res['search_best_max_abs_residual']:.2e}", flush=True)

    # Reassemble in original order
    results_dicts = [raw_results[i] for i in range(len(payloads))]

    for res in results_dicts:
        if res["gamma_disjoint"]:     disjoint_count += 1
        if res["gamma_proportional"]: proportional_count += 1
        if res["T_pair_is_rank1"]:    rank1_pair_count += 1
        if res["search_exact_hit"]:   exact_hit_count += 1

    # --- Export pair analysis ---
    pair_rows: list[dict] = []
    for r in results_dicts:
        pair_rows.append({
            "term_i": r["term_i"],
            "term_j": r["term_j"],
            "ab_common_support": r["ab_common_support"],
            "gamma_i_support_size": r["gamma_i_support_size"],
            "gamma_j_support_size": r["gamma_j_support_size"],
            "gamma_intersection_size": r["gamma_intersection_size"],
            "gamma_disjoint": str(r["gamma_disjoint"]),
            "gamma_proportional": str(r["gamma_proportional"]),
            "gamma_ratio": "" if r["gamma_ratio"] is None else str(r["gamma_ratio"]),
            "T_pair_mode0_rank": r["T_pair_mode0_rank"],
            "T_pair_mode1_rank": r["T_pair_mode1_rank"],
            "T_pair_mode2_rank": r["T_pair_mode2_rank"],
            "T_pair_flattening_lb": r["T_pair_flattening_lb"],
            "T_pair_is_rank1": str(r["T_pair_is_rank1"]),
            "search_rank": r["search_rank"],
            "search_restarts": r["search_restarts"],
            "search_best_loss": r["search_best_loss"],
            "search_best_max_abs_residual": r["search_best_max_abs_residual"],
            "search_exact_hit": str(r["search_exact_hit"]),
            "provenance": r["provenance"],
        })

    write_csv(
        EXPORTS / "step77_pair_analysis.csv",
        pair_rows,
        list(pair_rows[0].keys()) if pair_rows else [],
    )

    # --- Export witnesses ---
    witness_rows = [row for row in pair_rows if row["search_exact_hit"] == "True"]
    if witness_rows:
        write_csv(
            EXPORTS / "step77_border_rank_witnesses.csv",
            witness_rows,
            list(witness_rows[0].keys()),
        )
    else:
        write_csv(
            EXPORTS / "step77_border_rank_witnesses.csv",
            [],
            list(pair_rows[0].keys()) if pair_rows else ["term_i", "term_j"],
        )

    # --- Flattening lower bounds summary ---
    flat_lbs = [r["T_pair_flattening_lb"] for r in results_dicts]
    gamma_inter_sizes = [r["gamma_intersection_size"] for r in results_dicts]

    # --- Summary CSV ---
    any_exact_hit = exact_hit_count > 0
    summary: list[dict] = [
        {
            "summary_name": "step77_total_constant_multiple_pairs",
            "summary_value": str(len(results_dicts)),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_gamma_disjoint_pairs",
            "summary_value": str(disjoint_count),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_gamma_proportional_pairs",
            "summary_value": str(proportional_count),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_pair_tensor_rank1_count",
            "summary_value": str(rank1_pair_count),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_pair_flattening_lb_max",
            "summary_value": str(max(flat_lbs) if flat_lbs else "n/a"),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_pair_flattening_lb_min",
            "summary_value": str(min(flat_lbs) if flat_lbs else "n/a"),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_gamma_intersection_zero_count",
            "summary_value": str(sum(1 for x in gamma_inter_sizes if x == 0)),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_search_restarts_per_pair",
            "summary_value": str(SEARCH_RESTARTS),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_search_target_rank",
            "summary_value": str(TARGET_RANK),
            "provenance": "EXACT_DERIVED",
        },
        {
            "summary_name": "step77_any_exact_hit_rank22",
            "summary_value": str(any_exact_hit),
            "provenance": "MEASURED_FROM_CODE",
        },
        {
            "summary_name": "step77_exact_hit_count",
            "summary_value": str(exact_hit_count),
            "provenance": "MEASURED_FROM_CODE",
        },
        {
            "summary_name": "step77_best_max_abs_residual_across_pairs",
            "summary_value": str(min(r["search_best_max_abs_residual"] for r in results_dicts)),
            "provenance": "MEASURED_FROM_CODE",
        },
    ]

    write_csv(
        EXPORTS / "step77_summary.csv",
        summary,
        ["summary_name", "summary_value", "provenance"],
    )

    # --- Human-readable text report ---
    lines: list[str] = []
    lines.append("# Step 77: Border Rank Singular-Pair Witness Construction")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("[MEASURED_FROM_CODE]")
    lines.append("")
    lines.append(f"Pairs analysed: {len(results_dicts)} (all constant-multiple pairs from Step 70)")
    lines.append(f"Gamma disjoint pairs:       {disjoint_count}")
    lines.append(f"Gamma proportional pairs:   {proportional_count}")
    lines.append(f"Pair tensor rank-1 cases:   {rank1_pair_count}")
    lines.append(f"Search target rank:         {TARGET_RANK} (trying T_full - T_pair at rank {TARGET_RANK-1})")
    lines.append(f"Restarts per pair:          {SEARCH_RESTARTS}")
    lines.append(f"Exact hits (rank-22 found): {exact_hit_count}")
    lines.append("")
    lines.append("## Gamma-condition breakdown")
    lines.append("")
    lines.append("| pair | gamma_i | gamma_j | intersect | disjoint | proportional | ratio | T_pair_rank_lb | exact_hit |")
    lines.append("|------|---------|---------|-----------|----------|--------------|-------|----------------|-----------|")
    for r in results_dicts:
        ratio_str = f"{r['gamma_ratio']:.4g}" if r["gamma_ratio"] is not None else ""
        lines.append(
            f"| {r['term_i']},{r['term_j']} | {r['gamma_i_support_size']} | {r['gamma_j_support_size']} "
            f"| {r['gamma_intersection_size']} | {r['gamma_disjoint']} | {r['gamma_proportional']} "
            f"| {ratio_str} | {r['T_pair_flattening_lb']} | {r['search_exact_hit']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if exact_hit_count > 0:
        lines.append(f"POSITIVE: {exact_hit_count} pair(s) admitted an exact rank-22 replacement.")
        lines.append("See step77_border_rank_witnesses.csv for the corresponding pairs.")
    else:
        lines.append(
            f"NEGATIVE: No pair produced an exact rank-22 replacement at {SEARCH_RESTARTS} restarts per pair."
        )
        best_r = min(results_dicts, key=lambda r: r["search_best_max_abs_residual"])
        lines.append(
            f"Best residual across all pairs: {best_r['search_best_max_abs_residual']:.4e} "
            f"(pair {best_r['term_i']},{best_r['term_j']})."
        )
        lines.append("")
        lines.append(
            "The gamma-support conditions provide the algebraic filter:"
        )
        lines.append(
            f"  {disjoint_count} / {len(results_dicts)} pairs have disjoint gamma support (cleanest fusion condition)."
        )
        lines.append(
            f"  {proportional_count} / {len(results_dicts)} pairs have proportional gamma vectors."
        )
        lines.append(
            f"  {rank1_pair_count} / {len(results_dicts)} pair sub-tensors are already rank-1 "
            "(these are trivially fused without deformation, but the residual T_full - T_pair "
            "still needs rank <= 21 for a net gain)."
        )
        lines.append("")
        lines.append(
            "A true border-rank argument would need to exhibit an explicit one-parameter family "
            "of rank-22 tensors converging to T_full, which requires the residual (T_full - T_pair) "
            "to itself be expressible at rank <= 21.  The bounded LP-BFGS search here is a "
            "necessary (not sufficient) test at the chosen restart budget."
        )

    write_text(EXPORTS / "step77_singular_pair_report.md", "\n".join(lines) + "\n")
    print("Step 77 complete.")
    print(f"  Pairs with gamma disjoint: {disjoint_count}/{len(results_dicts)}")
    print(f"  Pairs with proportional gamma: {proportional_count}/{len(results_dicts)}")
    print(f"  Pair tensors already rank-1: {rank1_pair_count}/{len(results_dicts)}")
    print(f"  Exact rank-22 hits: {exact_hit_count}/{len(results_dicts)}")


if __name__ == "__main__":
    run()
