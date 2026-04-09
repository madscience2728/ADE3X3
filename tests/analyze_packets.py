"""
Phase 3 Packet Analysis — Mining the Near-Miss Landscape
=========================================================

Reads Stage-1 packets from CANON_DATABASE/packets/{R13,R19,R20}/,
computes all canon diagnostics, and produces:
  - Rich console summary tables
  - CSV of per-packet diagnostics      → CANON_OPTIMIZER/analysis/
  - Correlation matrix CSV             → CANON_OPTIMIZER/analysis/
  - Markdown report                    → CANON_OPTIMIZER/analysis/

Packet format (Stage 1 output):
  rank                 : (1,) int32  — target R
  basis_record_indices : (R-9,)      — H-basis terms in TermDB
  hit_record_indices   : (N_hits,)   — pool for dependent terms
  dependent_estimate   : (1,)        — number of dependents needed (usually 9)
  V_basis / V_perp / sigma_basis     — precomputed stage-1 data

For each packet:
  1. Diagnose the basis subspace (R-9 terms).
  2. Sample --completions random 9-term draws from hit pool → full R-term configs.
  3. Compute all Block A-G diagnostics on each full config.
  4. Aggregate and report.

Usage:
    python analyze_packets.py                          # all ranks, 50 completions/packet
    python analyze_packets.py --rank 19                # R=19 only
    python analyze_packets.py --completions 200        # more random samples
    python analyze_packets.py --db ./CANON_DATABASE/data
    python analyze_packets.py --output ./CANON_OPTIMIZER/analysis
    python analyze_packets.py --verbose
"""

import argparse
import glob
import os
import sys
import time
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Rich imports ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("WARNING: rich not installed — plain text output only")

console = Console() if HAS_RICH else None

# ── TermDB import ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "CANON_DATABASE"))
from term_db import TermDB

# ── SVD rank helper ───────────────────────────────────────────────────────────
SVD_TOL = 1e-10

def matrix_rank(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    A = A.astype(np.float64)
    sv = np.linalg.svd(A, compute_uv=False)
    tol = SVD_TOL * sv[0] if sv[0] > 0 else SVD_TOL
    return int(np.sum(sv > tol))

def safe_svd_values(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    if A.size == 0:
        return np.array([])
    return np.linalg.svd(A, compute_uv=False)

# ── Tensor construction ───────────────────────────────────────────────────────
def build_T() -> np.ndarray:
    """Build the 3×3 matrix multiplication tensor T of shape (9,9,9)."""
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[r * 3 + s, s * 3 + u, r * 3 + u] = 1.0
    return T

TARGET_T = build_T()

# ── Diagnostics ───────────────────────────────────────────────────────────────

def compute_diagnostics(db: TermDB, indices: np.ndarray, rank_target: int) -> dict:
    """
    Compute all Block A-G diagnostics for a complete R-term config.

    Parameters
    ----------
    db           : loaded TermDB
    indices      : (R,) int64 — record indices for all R terms
    rank_target  : the target R (13, 19, or 20)

    Returns
    -------
    dict of scalar/array diagnostics
    """
    R = len(indices)
    diag = {"R": R, "rank_target": rank_target}

    # ── Load matrices ──────────────────────────────────────────────────────
    try:
        H       = db.H[indices].astype(np.float64)          # (R, 18)
        sigma   = db.sigma[indices].astype(np.float64)       # (R, 9)
        delta   = db.compute_delta_batch(indices).astype(np.float64)  # (R, 54)
        alphas, betas = db.get_factors_batch(indices)        # (R,3,3) int8
        alphas  = alphas.astype(np.float64)
        betas   = betas.astype(np.float64)
    except Exception as e:
        diag["_load_error"] = str(e)
        return diag

    nuisance = np.hstack([H, delta])                         # (R, 72)
    SN       = np.hstack([sigma, nuisance])                  # (R, 81)

    # ── Block A: Core gate diagnostics ────────────────────────────────────
    target_H_rank = rank_target - 9

    rank_H   = matrix_rank(H)
    rank_N   = matrix_rank(nuisance)
    delta_leak = rank_N - rank_H

    rank_sigma = matrix_rank(sigma)
    rank_SN_val = matrix_rank(SN)
    augmented_gap = rank_target - rank_SN_val

    eta_nullity = 18 - rank_H
    conservation = rank_target + eta_nullity
    conservation_violation = abs(conservation - 27)

    diag.update({
        "rank_H": rank_H,
        "rank_H_target": target_H_rank,
        "gate1_pass": int(rank_H == target_H_rank),
        "rank_N": rank_N,
        "delta_leak": delta_leak,
        "gate2_pass": int(delta_leak == 0),
        "rank_sigma": rank_sigma,
        "rank_SN": rank_SN_val,
        "augmented_gap": augmented_gap,
        "gate3_pass": int(augmented_gap == 0),
        "eta_nullity": eta_nullity,
        "conservation": conservation,
        "conservation_violation": conservation_violation,
    })

    # ── Block B: Sigma innovation (THE key diagnostic) ────────────────────
    rank_nuisance_alone = rank_N   # already computed
    sigma_innovation = rank_SN_val - rank_nuisance_alone

    sv_sigma = safe_svd_values(sigma)
    sigma_condition = float(sv_sigma[0] / sv_sigma[-1]) if len(sv_sigma) > 0 and sv_sigma[-1] > SVD_TOL else np.inf
    sigma_rank_gap = 9 - rank_sigma

    diag.update({
        "sigma_innovation": sigma_innovation,
        "sigma_condition_number": sigma_condition,
        "sigma_rank_gap": sigma_rank_gap,
        "sigma_sv_min": float(sv_sigma[-1]) if len(sv_sigma) > 0 else 0.0,
        "sigma_sv_max": float(sv_sigma[0]) if len(sv_sigma) > 0 else 0.0,
    })

    # ── Block C: Delta containment ────────────────────────────────────────
    delta_rank = matrix_rank(delta)

    # Residual after projecting delta columns onto span(H)
    try:
        H_pinv = np.linalg.pinv(H)
        D_proj = H @ (H_pinv @ delta)
        delta_resid = delta - D_proj
        delta_resid_frob = float(np.linalg.norm(delta_resid, 'fro'))
        delta_resid_max  = float(np.max(np.abs(delta_resid)))
    except Exception:
        delta_resid_frob = np.nan
        delta_resid_max  = np.nan

    delta_in_H_span = int(delta_leak == 0)
    delta_leak_dims = delta_leak

    # Per-channel leak: 6 (s,t) pairs with s≠t
    pairs_st = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
    per_channel_leak = []
    for pair_idx, (s, t) in enumerate(pairs_st):
        # Delta column group for this (s,t): 9 entries for all (r,u)
        col_start = pair_idx * 9
        d_block = delta[:, col_start:col_start + 9]  # (R, 9)
        combined = np.hstack([H, d_block])
        leak_dim = matrix_rank(combined) - rank_H
        per_channel_leak.append(int(leak_dim > 0))

    diag.update({
        "delta_rank": delta_rank,
        "delta_in_H_span": delta_in_H_span,
        "delta_residual_frobenius": delta_resid_frob,
        "delta_residual_max": delta_resid_max,
        "delta_leak_dims": delta_leak_dims,
        "n_channels_leaking": sum(per_channel_leak),
    })

    # ── Block D: Spectral analysis ────────────────────────────────────────
    sv_H = safe_svd_values(H)
    sv_SN = safe_svd_values(SN)
    sv_nuisance = safe_svd_values(nuisance)

    # Spectral gap at rank cutoff
    if len(sv_H) > target_H_rank:
        H_spectral_gap = float(sv_H[target_H_rank - 1] - sv_H[target_H_rank]) if target_H_rank > 0 else 0.0
    else:
        H_spectral_gap = 0.0

    SN_spectral_gap = float(sv_SN[rank_target - 1] - sv_SN[rank_target]) if len(sv_SN) > rank_target else 0.0

    diag.update({
        "H_sv_1": float(sv_H[0]) if len(sv_H) > 0 else 0.0,
        "H_sv_last": float(sv_H[target_H_rank - 1]) if len(sv_H) >= target_H_rank else 0.0,
        "H_spectral_gap": H_spectral_gap,
        "SN_spectral_gap": SN_spectral_gap,
        "SN_sv_min": float(sv_SN[rank_target - 1]) if len(sv_SN) >= rank_target else 0.0,
    })

    # ── Block E: Omega operator ───────────────────────────────────────────
    omega_diag = {"Z_dim": -1, "omega_exists": 0, "omega_pd": 0,
                  "omega_condition": np.nan, "omega_eig_min": np.nan, "omega_eig_max": np.nan}

    if rank_H == target_H_rank and delta_leak == 0:
        try:
            N_mat = nuisance.T  # (72, R)
            _, S_N, Vt_N = np.linalg.svd(N_mat, full_matrices=True)
            null_start = int(np.sum(S_N > SVD_TOL * S_N[0] if S_N[0] > 0 else SVD_TOL))
            Z_basis = Vt_N[null_start:]  # (Z_dim, R)
            Z_dim = Z_basis.shape[0]
            omega_diag["Z_dim"] = Z_dim
            omega_diag["omega_exists"] = int(Z_dim == 9)

            if Z_dim == 9:
                Sigma_Z = Z_basis @ sigma   # (9, 9)
                G = Sigma_Z @ Sigma_Z.T
                eigs = np.linalg.eigvalsh(G)
                omega_pd = bool(np.all(eigs > SVD_TOL))
                omega_diag["omega_pd"] = int(omega_pd)
                if omega_pd:
                    Omega = 3.0 * np.linalg.inv(G)
                    omega_eigs = np.linalg.eigvalsh(Omega)
                    omega_diag["omega_condition"] = float(omega_eigs[-1] / omega_eigs[0])
                    omega_diag["omega_eig_min"] = float(omega_eigs[0])
                    omega_diag["omega_eig_max"] = float(omega_eigs[-1])
        except Exception:
            pass

    diag.update(omega_diag)

    # ── Block F: Gamma solve & reconstruction ─────────────────────────────
    try:
        # lstsq: min ‖SN @ Gamma_col - target_col‖ for each of 9 target columns
        # target: [Sigma | N]^T @ Gamma^T = [3I_9 | 0]^T
        # i.e. SN.T @ gamma_k = e_k * 3  for k=0..8
        rhs = np.zeros((R, 9))
        rhs[:, :] = 0.0
        # We need SN^T (81, R) @ Gamma^T (R, 9) = target (81, 9)
        # Equivalently SN (R, 81) rows → solve SN @ x = b for each gamma row? No.
        # The condition is: for each tensor output k, sum_i gamma_{k,i} * sigma_i = 3 * delta_{kk'}
        # i.e. sigma^T @ gamma_k = 3 e_k, nuisance^T @ gamma_k = 0
        # Together: SN^T (81, R) @ gamma_k (R,) = rhs_k (81,)  where rhs_k[k] = 3, rest 0
        target_mat = np.zeros((81, 9))
        target_mat[:9, :] = 3.0 * np.eye(9)   # sigma part
        Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target_mat, rcond=None)
        Gamma = Gamma_T.T  # (9, R)

        # Reconstruction
        alphas_flat = alphas.reshape(R, 9)   # (R, 9)
        betas_flat  = betas.reshape(R, 9)    # (R, 9)
        # T_hat[i,j,k] = sum_r Gamma[k,r] * alpha[r,i] * beta[r,j]
        T_hat = np.einsum('kr,ri,rj->ijk', Gamma, alphas_flat, betas_flat)
        residual = TARGET_T - T_hat
        fitness_inf = float(np.max(np.abs(residual)))
        fitness_rmse = float(np.linalg.norm(residual) / np.sqrt(729))

        gamma_resid = target_mat - SN.T @ Gamma_T
        gamma_lstsq_residual = float(np.linalg.norm(gamma_resid, 'fro'))
        gamma_max_resid = float(np.max(np.abs(gamma_resid)))
    except Exception as e:
        fitness_inf = np.nan
        fitness_rmse = np.nan
        gamma_lstsq_residual = np.nan
        gamma_max_resid = np.nan

    diag.update({
        "gamma_lstsq_residual": gamma_lstsq_residual,
        "gamma_max_entry_residual": gamma_max_resid,
        "tensor_reconstruction_error": fitness_inf,
        "tensor_reconstruction_rmse": fitness_rmse,
    })

    # ── Block G: Structural classification ───────────────────────────────
    alphas_int = alphas.astype(np.int8)
    betas_int  = betas.astype(np.int8)

    n_zero_alpha = int(np.sum(np.all(alphas_int.reshape(R, 9) == 0, axis=1)))
    n_zero_beta  = int(np.sum(np.all(betas_int.reshape(R, 9) == 0, axis=1)))
    alpha_density = float(np.mean(alphas_int != 0))
    beta_density  = float(np.mean(betas_int != 0))

    # Fiber partition: dominant sigma channel per term
    abs_sigma = np.abs(sigma)
    fiber_assignments = np.argmax(abs_sigma, axis=1)  # (R,)
    fiber_counts = np.bincount(fiber_assignments, minlength=9)
    fiber_tuple = tuple(sorted(fiber_counts.tolist(), reverse=True))

    diag.update({
        "n_zero_alpha_rows": n_zero_alpha,
        "n_zero_beta_cols": n_zero_beta,
        "alpha_density": alpha_density,
        "beta_density": beta_density,
        "fiber_partition_str": str(fiber_tuple),
    })

    return diag


# ── Packet loading ────────────────────────────────────────────────────────────

def load_packets(packet_dir: Path, target_rank: int | None = None) -> list[dict]:
    """Load all packet files and return list of dicts with metadata + indices."""
    patterns = ["*.npz"]
    all_files = []
    for pat in patterns:
        all_files.extend(sorted(packet_dir.glob(pat)))

    packets = []
    for fpath in all_files:
        try:
            d = np.load(str(fpath), allow_pickle=True)
            rank = int(d["rank"][0])
            if target_rank is not None and rank != target_rank:
                continue
            basis_len = len(d["basis_record_indices"])
            packets.append({
                "file": str(fpath),
                "rank": rank,
                "basis_indices": d["basis_record_indices"].astype(np.int64),
                "hit_indices": d["hit_record_indices"].astype(np.int64)
                    if "hit_record_indices" in d else np.array([], dtype=np.int64),
                "dependent_estimate": rank - basis_len,  # number of dep terms needed
            })
        except Exception as e:
            print(f"WARNING: skipping {fpath}: {e}")
    return packets


# ── Per-packet analysis ───────────────────────────────────────────────────────

RNG = np.random.default_rng(0xADE3)

def analyze_packet(db: TermDB, pkt: dict, n_completions: int) -> list[dict]:
    """
    Return a list of diagnostic dicts — one per sampled completion.
    Each dict augments compute_diagnostics output with packet metadata.
    """
    rank = pkt["rank"]
    basis = pkt["basis_indices"]
    hits  = pkt["hit_indices"]
    n_dep = pkt["dependent_estimate"]
    fname = Path(pkt["file"]).name

    results = []

    # Minimum hit pool size check
    if len(hits) < n_dep:
        # Not enough hits — diagnose basis alone if it's a full R-term set
        if len(basis) == rank:
            d = compute_diagnostics(db, basis, rank)
            d.update({"packet_file": fname, "completion_id": 0,
                      "n_hits_in_pool": len(hits), "basis_size": len(basis)})
            results.append(d)
        return results

    for trial in range(n_completions):
        try:
            dep_indices = RNG.choice(hits, size=n_dep, replace=False)
        except ValueError:
            dep_indices = RNG.choice(hits, size=min(n_dep, len(hits)), replace=False)

        full_indices = np.concatenate([basis, dep_indices])
        d = compute_diagnostics(db, full_indices, rank)
        d.update({
            "packet_file": fname,
            "completion_id": trial,
            "n_hits_in_pool": len(hits),
            "basis_size": len(basis),
        })
        results.append(d)

    return results


# ── Aggregate stats ───────────────────────────────────────────────────────────

def agg(vals: list, key: str) -> dict:
    v = [x[key] for x in vals if key in x and x[key] is not None and not (isinstance(x[key], float) and np.isnan(x[key]))]
    if not v:
        return {"min": None, "max": None, "mean": None, "median": None, "std": None}
    v = np.array(v, dtype=float)
    return {"min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())}


# ── Rich output ───────────────────────────────────────────────────────────────

def print_summary_table(rank: int, rows: list[dict], console: "Console"):
    total = len(rows)
    if total == 0:
        console.print(f"[yellow]R={rank}: no data[/yellow]")
        return

    g1 = sum(r.get("gate1_pass", 0) for r in rows)
    g2 = sum(r.get("gate2_pass", 0) for r in rows if r.get("gate1_pass", 0))
    g3 = sum(r.get("gate3_pass", 0) for r in rows if r.get("gate2_pass", 0))

    g1_tested = total
    g2_tested = g1
    g3_tested = g2

    # Gate table
    gt = Table(title=f"R={rank} — Gate Pass Rates (N={total} sampled configs)", box=box.SIMPLE_HEAVY)
    gt.add_column("Gate", style="bold")
    gt.add_column("Tested", justify="right")
    gt.add_column("Passed", justify="right")
    gt.add_column("Rate", justify="right")
    gt.add_row("Gate 1  rank(H)=R-9", str(g1_tested), str(g1),
               f"[green]{100*g1/g1_tested:.1f}%[/green]" if g1_tested else "—")
    gt.add_row("Gate 2  Δ⊂span(H)", str(g2_tested),  str(g2),
               f"{100*g2/g2_tested:.1f}%" if g2_tested else "—")
    gt.add_row("Gate 3  rank([Σ|N])=R", str(g3_tested), str(g3),
               f"[red]{100*g3/g3_tested:.1f}%[/red]" if g3_tested else "—")
    console.print(gt)

    # Sigma innovation table — THE key table
    si_counts = defaultdict(int)
    for r in rows:
        v = r.get("sigma_innovation")
        if v is not None:
            si_counts[v] += 1

    sit = Table(title="Sigma Innovation (σ_innov = rank([Σ|N]) − rank([H|Δ]))", box=box.SIMPLE)
    sit.add_column("σ_innov", justify="right")
    sit.add_column("Count", justify="right")
    sit.add_column("Fraction", justify="right")
    for k in sorted(si_counts):
        pct = 100 * si_counts[k] / total
        style = "[green]" if k == 9 else ("[yellow]" if k == 8 else "")
        end_style = "[/green]" if k == 9 else ("[/yellow]" if k == 8 else "")
        sit.add_row(f"{style}{k}{end_style}", f"{style}{si_counts[k]}{end_style}",
                    f"{style}{pct:.1f}%{end_style}")
    console.print(sit)

    # Augmented gap + reconstruction
    ag_s   = agg(rows, "augmented_gap")
    rc_s   = agg(rows, "tensor_reconstruction_error")
    hg_s   = agg(rows, "H_spectral_gap")
    cons_s = agg(rows, "conservation_violation")

    misc = Table(title="Key Metrics", box=box.SIMPLE)
    misc.add_column("Metric")
    misc.add_column("min", justify="right")
    misc.add_column("median", justify="right")
    misc.add_column("max", justify="right")
    for label, d in [
        ("augmented_gap", ag_s),
        ("reconstruction_error (∞)", rc_s),
        ("H_spectral_gap", hg_s),
        ("conservation_violation", cons_s),
    ]:
        misc.add_row(label,
                     f"{d['min']:.4f}" if d["min"] is not None else "—",
                     f"{d['median']:.4f}" if d["median"] is not None else "—",
                     f"{d['max']:.4f}" if d["max"] is not None else "—")
    console.print(misc)

    # Conservation law
    cons_ok = sum(1 for r in rows if r.get("conservation_violation", 999) == 0)
    console.print(f"  Conservation R + η_null = 27: "
                  f"[bold]{cons_ok}[/bold]/{total} ({100*cons_ok/total:.1f}%)")


# ── Correlation matrix ────────────────────────────────────────────────────────

CORR_KEYS = [
    "rank_H", "delta_leak", "rank_sigma", "rank_SN", "augmented_gap",
    "sigma_innovation", "delta_residual_frobenius",
    "gamma_lstsq_residual", "tensor_reconstruction_error",
    "H_spectral_gap", "alpha_density", "beta_density",
]

def compute_correlation_matrix(rows: list[dict]) -> np.ndarray:
    mat = []
    for k in CORR_KEYS:
        col = [r.get(k, np.nan) for r in rows]
        mat.append([float(x) if x is not None else np.nan for x in col])
    mat = np.array(mat, dtype=np.float64)  # (n_keys, n_rows)

    n = len(CORR_KEYS)
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi, xj = mat[i], mat[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() > 2:
                try:
                    corr[i, j] = float(np.corrcoef(xi[mask], xj[mask])[0, 1])
                except Exception:
                    pass
    return corr


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown_report(all_rows: dict[int, list[dict]], output_dir: Path,
                           n_completions: int):
    lines = ["# Packet Analysis Report — Phase 3\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"Completions per packet: {n_completions}\n\n"]

    for rank in sorted(all_rows.keys()):
        rows = all_rows[rank]
        n = len(rows)
        if n == 0:
            continue
        lines.append(f"## R={rank} ({n} sampled configurations)\n\n")

        g1 = sum(r.get("gate1_pass", 0) for r in rows)
        g2 = sum(r.get("gate2_pass", 0) for r in rows if r.get("gate1_pass", 0))
        g3 = sum(r.get("gate3_pass", 0) for r in rows if r.get("gate2_pass", 0))

        lines.append("### Gate Pass Rates\n\n")
        lines.append("| Gate | Tested | Passed | Rate |\n")
        lines.append("|------|--------|--------|------|\n")
        lines.append(f"| Gate 1 rank(H)=R-9 | {n} | {g1} | {100*g1/n:.1f}% |\n")
        lines.append(f"| Gate 2 Δ⊂span(H) | {g1} | {g2} | {100*g2/g1:.1f}% |\n" if g1 else "| Gate 2 | — | — | — |\n")
        lines.append(f"| Gate 3 rank([Σ|N])=R | {g2} | {g3} | {100*g3/g2:.1f}% |\n" if g2 else "| Gate 3 | — | — | — |\n")
        lines.append("\n")

        # Sigma innovation distribution
        si_counts = defaultdict(int)
        for r in rows:
            v = r.get("sigma_innovation")
            if v is not None:
                si_counts[v] += 1
        lines.append("### Sigma Innovation Distribution\n\n")
        lines.append("| σ_innov | Count | Fraction |\n")
        lines.append("|---------|-------|----------|\n")
        for k in sorted(si_counts):
            lines.append(f"| {k} | {si_counts[k]} | {100*si_counts[k]/n:.1f}% |\n")
        max_si = max(si_counts.keys()) if si_counts else 0
        lines.append(f"\n**Maximum σ_innov observed: {max_si}**\n\n")

        # Conservation
        cons_ok = sum(1 for r in rows if r.get("conservation_violation", 999) == 0)
        lines.append(f"### Conservation Law\n\n")
        lines.append(f"R + η_null = 27 satisfied: {cons_ok}/{n} ({100*cons_ok/n:.1f}%)\n\n")

        # Key metrics
        lines.append("### Key Metric Distributions\n\n")
        lines.append("| Metric | min | mean | median | max |\n")
        lines.append("|--------|-----|------|--------|-----|\n")
        for metric in ["augmented_gap", "sigma_innovation", "tensor_reconstruction_error",
                       "H_spectral_gap", "delta_residual_frobenius", "delta_leak_dims",
                       "gamma_lstsq_residual"]:
            s = agg(rows, metric)
            if s["min"] is not None:
                lines.append(f"| {metric} | {s['min']:.4f} | {s['mean']:.4f} | {s['median']:.4f} | {s['max']:.4f} |\n")
        lines.append("\n")

        # Interpretation
        lines.append("### Interpretation\n\n")
        g2_tested_n = g1
        if g2 == 0 and g2_tested_n > 0:
            lines.append(
                "**Gate 2 is the active bottleneck.** "
                f"Zero of {g2_tested_n} Gate-1-passing configs satisfy Δ⊂span(H). "
                "When Gate 2 fails, Nuisance=[H|Δ] achieves full row rank R, "
                "making σ_innov=0 a trivial algebraic consequence — Σ is automatically "
                "contained in a full-row-rank matrix. "
                "This is NOT an independent sigma obstruction; it is a Gate 2 cascade. "
                "The meaningful sigma analysis requires Gate-2-passing configurations.\n\n"
                "**Strategy:** Understand structurally what forces Δ⊂span(H). "
                "The Stage-1 hit pool enforces H-row containment by construction. "
                "The delta containment (Gate 2) requires (α[r,s]·β[t,u]) for s≠t to lie "
                "in the span of [Eta1|Eta2] — a far stronger constraint than H-row membership.\n\n")
        elif max_si == 0:
            lines.append(f"**Scenario 1 (Universal Obstruction candidate):** σ_innov=0 universally. "
                         "If Gate 2 also failed universally, see Gate 2 cascade note above. "
                         "If Gate 2 passed and σ_innov=0, this is a true structural obstruction.\n\n")
        elif max_si < 8:
            lines.append(f"**Scenario 1 (Universal Obstruction candidate):** σ_innov never exceeds {max_si}. "
                         "The sigma image appears algebraically constrained. "
                         "Possible structural ceiling at codimension ≥ 2 for {{-1,0,1}} configurations.\n\n")
        elif max_si == 8:
            lines.append("**Scenario 2 (Soft ceiling):** σ_innov reaches 8 but not 9. "
                         "One sigma dimension is consistently blocked. "
                         "Investigate whether the missing direction is fixed across packets.\n\n")
        elif max_si == 9 and g3 == 0:
            lines.append("**Scenario 3 (Heterogeneous landscape):** σ_innov reaches 9 but Gate 3 still fails. "
                         "The augmented rank condition is passable, but the Gamma solve fails. "
                         "Investigate Gamma residual patterns for σ_innov=9 packets.\n\n")
        elif g3 > 0:
            lines.append(f"**Gate 3 PASSED** in {g3} configurations! "
                         "These are candidate solutions — verify tensor reconstruction error approaches 0.\n\n")

    path = output_dir / "analysis_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ── CSV export ─────────────────────────────────────────────────────────────────

NUMERIC_DIAG_KEYS = [
    "rank_H", "rank_H_target", "gate1_pass", "rank_N", "delta_leak", "gate2_pass",
    "rank_sigma", "rank_SN", "augmented_gap", "gate3_pass",
    "eta_nullity", "conservation", "conservation_violation",
    "sigma_innovation", "sigma_condition_number", "sigma_rank_gap",
    "sigma_sv_min", "sigma_sv_max",
    "delta_rank", "delta_in_H_span", "delta_residual_frobenius", "delta_residual_max",
    "delta_leak_dims", "n_channels_leaking",
    "H_sv_1", "H_sv_last", "H_spectral_gap", "SN_spectral_gap", "SN_sv_min",
    "Z_dim", "omega_exists", "omega_pd", "omega_condition", "omega_eig_min", "omega_eig_max",
    "gamma_lstsq_residual", "gamma_max_entry_residual",
    "tensor_reconstruction_error", "tensor_reconstruction_rmse",
    "n_zero_alpha_rows", "n_zero_beta_cols", "alpha_density", "beta_density",
    "n_hits_in_pool", "basis_size",
]

def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = ["rank_target", "packet_file", "completion_id", "fiber_partition_str"] + NUMERIC_DIAG_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_correlation_csv(corr: np.ndarray, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + CORR_KEYS)
        for i, key in enumerate(CORR_KEYS):
            row = [key] + [f"{corr[i,j]:.4f}" if np.isfinite(corr[i,j]) else "" for j in range(len(CORR_KEYS))]
            w.writerow(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 packet analysis")
    parser.add_argument("--rank", type=int, action="append", dest="ranks",
                        help="Target rank(s) to analyze (default: all found)")
    parser.add_argument("--completions", type=int, default=50,
                        help="Number of random completions to sample per packet (default: 50)")
    parser.add_argument("--db", default="./CANON_DATABASE/data",
                        help="Path to TermDB data directory")
    parser.add_argument("--packets", default="./CANON_DATABASE/packets",
                        help="Path to packets root directory")
    parser.add_argument("--output", default="./CANON_DATABASE/analysis",
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    packets_root = Path(args.packets)

    if console:
        console.print(Panel("[bold cyan]Phase 3 — Packet Analysis[/bold cyan]", expand=False))

    # ── Load TermDB ──────────────────────────────────────────────────────────
    if console:
        console.print(f"[dim]Loading TermDB from {args.db}...[/dim]")
    t0 = time.time()
    db = TermDB(args.db)
    if console:
        console.print(f"  TermDB loaded in {time.time()-t0:.1f}s")

    # ── Discover rank subdirs ────────────────────────────────────────────────
    rank_dirs = {}
    for subdir in sorted(packets_root.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("R"):
            try:
                r = int(subdir.name[1:])
                rank_dirs[r] = subdir
            except ValueError:
                pass

    if args.ranks:
        rank_dirs = {r: d for r, d in rank_dirs.items() if r in args.ranks}

    if not rank_dirs:
        print(f"ERROR: no packet directories found under {packets_root}")
        sys.exit(1)

    if console:
        console.print(f"Ranks to analyze: {sorted(rank_dirs)}")

    # ── Process each rank ────────────────────────────────────────────────────
    all_rows: dict[int, list[dict]] = {}

    for rank, pdir in sorted(rank_dirs.items()):
        packets = load_packets(pdir, target_rank=rank)
        if console:
            console.print(f"\n[bold]R={rank}[/bold]: {len(packets)} packets, "
                          f"{args.completions} completions each "
                          f"→ up to {len(packets)*args.completions} configs")

        rows: list[dict] = []

        if HAS_RICH:
            prog = Progress(SpinnerColumn(), TextColumn("{task.description}"),
                            BarColumn(), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), console=console)
            task = prog.add_task(f"R={rank}", total=len(packets))
            prog.start()
        else:
            prog = None

        for i, pkt in enumerate(packets):
            pkt_rows = analyze_packet(db, pkt, args.completions)
            rows.extend(pkt_rows)
            if args.verbose and console:
                for r in pkt_rows[:1]:
                    console.print(f"  [{i}] g1={r.get('gate1_pass')} g2={r.get('gate2_pass')} "
                                  f"g3={r.get('gate3_pass')} si={r.get('sigma_innovation')} "
                                  f"err={r.get('tensor_reconstruction_error', 0):.3f}")
            if prog:
                prog.advance(task)

        if prog:
            prog.stop()

        all_rows[rank] = rows

        # Print Rich summary
        if console:
            print_summary_table(rank, rows, console)

        # Write CSV
        csv_path = output_dir / f"packets_R{rank}_diagnostics.csv"
        write_csv(rows, csv_path)
        if console:
            console.print(f"  [dim]→ {csv_path}[/dim]")

        # Correlation matrix
        if len(rows) > 5:
            corr = compute_correlation_matrix(rows)
            corr_path = output_dir / f"packets_R{rank}_correlations.csv"
            write_correlation_csv(corr, corr_path)
            if console:
                console.print(f"  [dim]→ {corr_path}[/dim]")

    # ── Markdown report ──────────────────────────────────────────────────────
    md_path = write_markdown_report(all_rows, output_dir, args.completions)
    if console:
        console.print(f"\n[green]Report written:[/green] {md_path}")
    else:
        print(f"Report written: {md_path}")

    # ── Cross-rank summary ────────────────────────────────────────────────────
    if len(all_rows) > 1 and console:
        console.print("\n[bold]Cross-rank comparison[/bold]")
        ct = Table(box=box.SIMPLE)
        ct.add_column("R")
        ct.add_column("N configs")
        ct.add_column("G1 rate")
        ct.add_column("G2|G1 rate")
        ct.add_column("max σ_innov")
        ct.add_column("min recon err")
        for rank, rows in sorted(all_rows.items()):
            n = len(rows)
            if n == 0:
                continue
            g1 = sum(r.get("gate1_pass", 0) for r in rows)
            g2 = sum(r.get("gate2_pass", 0) for r in rows if r.get("gate1_pass", 0))
            max_si = max((r.get("sigma_innovation", 0) or 0) for r in rows)
            min_err = min((r.get("tensor_reconstruction_error") or np.inf)
                          for r in rows if r.get("tensor_reconstruction_error") is not None)
            ct.add_row(str(rank), str(n), f"{100*g1/n:.1f}%",
                       f"{100*g2/g1:.1f}%" if g1 else "—",
                       str(max_si),
                       f"{min_err:.4f}" if np.isfinite(min_err) else "—")
        console.print(ct)

    console.print("\n[bold green]Analysis complete.[/bold green]") if console else print("Done.")


if __name__ == "__main__":
    main()
