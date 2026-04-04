"""
Gate 2 Hit-Pool Scan
====================

For each Stage-1 packet, scans the hit pool to count how many candidate
records are "delta-neutral" — i.e., appending them to the basis does NOT
increase delta_leak.

Math
----
delta_leak = rank([H|Delta]) - rank(H)

When we append a record j:
  - rank(H_aug) = rank(H_basis)  [since h_j ∈ row-span(H_basis) by Stage-1 construction]
  - rank(N_aug) = rank(N_basis) + 1  if [h_j|d_j] ∉ row-span(N_basis)
               = rank(N_basis)       if [h_j|d_j] ∈ row-span(N_basis)

So:
  delta-neutral: [h_j|d_j] ∈ row-span(N_basis = [H_basis|D_basis])
  delta-leaking: [h_j|d_j] ∉ row-span(N_basis)

Fast batch test: precompute the row-space projection matrix of N_basis,
then compute residual norms for all hits at once.

Output
------
  - Console table: per-packet n_neutral / n_hits
  - CANON_DATABASE/analysis/gate2_scan_R{rank}.csv
  - Per-rank summary printed at end

Usage
-----
    python CANON_DATABASE/gate2_scan.py
    python CANON_DATABASE/gate2_scan.py --rank 19
    python CANON_DATABASE/gate2_scan.py --rank 19 --max-packets 10
    python CANON_DATABASE/gate2_scan.py --tol 1e-8
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── TermDB import ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from term_db import TermDB

# ── Rich ─────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

SVD_TOL = 1e-10


def matrix_rank(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    sv = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    tol = SVD_TOL * sv[0] if sv[0] > 0 else SVD_TOL
    return int(np.sum(sv > tol))


def row_space_projector(A: np.ndarray, tol: float = SVD_TOL):
    """
    Return (Q, rank) where Q is (rank × n_cols) orthonormal row-space basis of A.
    P = Q^T @ Q is the (n_cols × n_cols) row-space projection matrix.
    """
    A = A.astype(np.float64)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = tol * sv[0] if sv[0] > 0 else tol
    r = int(np.sum(sv > cutoff))
    return Vt[:r], r  # (r, n_cols)


def scan_packet(db: TermDB, basis_indices: np.ndarray, hit_indices: np.ndarray,
                tol: float, chunk_size: int = 50_000) -> dict:
    """
    Scan the hit pool for delta-neutral records.

    Returns a dict with:
        delta_leak_basis : int
        n_hits           : int
        n_neutral        : int  — records that do NOT increase delta_leak
        n_leaking        : int  — records that DO increase delta_leak
        resid_min        : float  — min residual norm (neutrals have resid ≈ 0)
        resid_max        : float
        resid_median     : float
        resid_p10        : float  — 10th percentile (probe of cluster structure)
        resid_p90        : float
    """
    n_basis = len(basis_indices)
    rank_target = n_basis + 9   # Stage-1 packets have basis_size = R - 9

    H_basis = db.H[basis_indices].astype(np.float64)       # (n_basis, 18)
    D_basis = db.compute_delta_batch(basis_indices).astype(np.float64)  # (n_basis, 54)
    N_basis = np.hstack([H_basis, D_basis])                # (n_basis, 72)

    rank_H = matrix_rank(H_basis)
    rank_N = matrix_rank(N_basis)
    delta_leak_basis = rank_N - rank_H

    # Row-space basis of N_basis
    Q, row_rank = row_space_projector(N_basis, tol=tol)  # (row_rank, 72)
    # Projection: for vector v, proj = v @ Q.T @ Q; resid = v - proj

    n_hits = len(hit_indices)
    if n_hits == 0:
        return {"delta_leak_basis": delta_leak_basis, "n_hits": 0,
                "n_neutral": 0, "n_leaking": 0,
                "resid_min": np.nan, "resid_max": np.nan,
                "resid_median": np.nan, "resid_p10": np.nan, "resid_p90": np.nan,
                "rank_H_basis": rank_H, "rank_N_basis": rank_N, "row_rank_N": row_rank}

    # Process in chunks to avoid OOM (150K × 72 × 8 bytes = ~86 MB — fine, but chunk anyway)
    all_resid_norms = np.empty(n_hits, dtype=np.float64)

    QQt = Q.T @ Q  # (72, 72) projection matrix — precompute once

    for start in range(0, n_hits, chunk_size):
        end = min(start + chunk_size, n_hits)
        idx_chunk = hit_indices[start:end]

        H_chunk = db.H[idx_chunk].astype(np.float64)           # (k, 18)
        D_chunk = db.compute_delta_batch(idx_chunk).astype(np.float64)  # (k, 54)
        ND_chunk = np.hstack([H_chunk, D_chunk])                # (k, 72)

        proj = ND_chunk @ QQt                       # (k, 72)
        resid = ND_chunk - proj                     # (k, 72)
        norms = np.linalg.norm(resid, axis=1)       # (k,)
        all_resid_norms[start:end] = norms

    n_neutral = int(np.sum(all_resid_norms < tol * 100))  # generous threshold
    n_leaking  = n_hits - n_neutral

    return {
        "delta_leak_basis": delta_leak_basis,
        "n_hits": n_hits,
        "n_neutral": n_neutral,
        "n_leaking": n_leaking,
        "resid_min":    float(np.min(all_resid_norms)),
        "resid_max":    float(np.max(all_resid_norms)),
        "resid_median": float(np.median(all_resid_norms)),
        "resid_p10":    float(np.percentile(all_resid_norms, 10)),
        "resid_p90":    float(np.percentile(all_resid_norms, 90)),
        "rank_H_basis": rank_H,
        "rank_N_basis": rank_N,
        "row_rank_N":   row_rank,
    }


def load_packets(packets_root: Path, target_rank: int | None, max_packets: int | None):
    """Yield (rank, packet_file_path, basis_indices, hit_indices) for each packet."""
    rank_dirs = {}
    for subdir in sorted(packets_root.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("R"):
            try:
                r = int(subdir.name[1:])
                rank_dirs[r] = subdir
            except ValueError:
                pass
    if target_rank is not None:
        rank_dirs = {r: d for r, d in rank_dirs.items() if r == target_rank}

    for rank, pdir in sorted(rank_dirs.items()):
        files = sorted(pdir.glob("*.npz"))
        if max_packets:
            files = files[:max_packets]
        for fpath in files:
            try:
                d = np.load(str(fpath), allow_pickle=True)
                yield (rank, fpath,
                       d["basis_record_indices"].astype(np.int64),
                       d["hit_record_indices"].astype(np.int64)
                           if "hit_record_indices" in d else np.array([], np.int64))
            except Exception as e:
                print(f"WARNING: skipping {fpath}: {e}")


CSV_FIELDS = ["rank", "packet_file", "delta_leak_basis", "rank_H_basis",
              "rank_N_basis", "row_rank_N", "n_hits", "n_neutral", "n_leaking",
              "neutral_fraction", "resid_min", "resid_p10", "resid_median",
              "resid_p90", "resid_max"]


def main():
    parser = argparse.ArgumentParser(description="Gate 2 hit-pool scan")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--db", default="CANON_DATABASE/data")
    parser.add_argument("--packets", default="CANON_DATABASE/packets")
    parser.add_argument("--output", default="CANON_DATABASE/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if console:
        console.print("[bold cyan]Gate 2 Hit-Pool Scan[/bold cyan]")

    t0 = time.time()
    db = TermDB(args.db)
    if console:
        console.print(f"  TermDB loaded in {time.time()-t0:.1f}s")

    packets_list = list(load_packets(Path(args.packets), args.rank, args.max_packets))
    if console:
        console.print(f"  Packets to scan: {len(packets_list)}")

    # ── Per-rank results accumulator ─────────────────────────────────────────
    by_rank: dict[int, list[dict]] = defaultdict(list)

    # ── Open CSV writers per rank (lazy) ─────────────────────────────────────
    csv_files: dict[int, tuple] = {}

    def get_csv(rank):
        if rank not in csv_files:
            path = output_dir / f"gate2_scan_R{rank}.csv"
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            csv_files[rank] = (f, w)
        return csv_files[rank][1]

    # ── Main scan loop ────────────────────────────────────────────────────────
    if HAS_RICH:
        prog = Progress(SpinnerColumn(), TextColumn("{task.description}"),
                        BarColumn(), TextColumn("{task.completed}/{task.total}"),
                        TimeElapsedColumn(), console=console)
        task = prog.add_task("scanning", total=len(packets_list))
        prog.start()
    else:
        prog = None

    try:
        for rank, fpath, basis_idx, hit_idx in packets_list:
            result = scan_packet(db, basis_idx, hit_idx, tol=args.tol)
            result["rank"] = rank
            result["packet_file"] = fpath.name
            result["neutral_fraction"] = (result["n_neutral"] / result["n_hits"]
                                           if result["n_hits"] > 0 else 0.0)
            by_rank[rank].append(result)

            row = {k: result.get(k, "") for k in CSV_FIELDS}
            get_csv(rank).writerow(row)

            if prog:
                prog.advance(task)

    finally:
        if prog:
            prog.stop()
        for f, w in csv_files.values():
            f.close()

    # ── Per-rank summary ─────────────────────────────────────────────────────
    for rank in sorted(by_rank.keys()):
        rows = by_rank[rank]
        n_pkts = len(rows)
        total_hits    = sum(r["n_hits"] for r in rows)
        total_neutral = sum(r["n_neutral"] for r in rows)
        max_neutral   = max(r["n_neutral"] for r in rows)
        any_neutral   = sum(1 for r in rows if r["n_neutral"] > 0)
        all_zero      = sum(1 for r in rows if r["n_neutral"] == 0)
        median_leak   = float(np.median([r["delta_leak_basis"] for r in rows]))
        median_resid_p10 = float(np.median([r["resid_p10"] for r in rows]))

        if console:
            t = Table(title=f"R={rank} — Gate 2 Scan Summary ({n_pkts} packets)",
                      box=box.SIMPLE_HEAVY)
            t.add_column("Metric", style="bold")
            t.add_column("Value", justify="right")
            t.add_row("Total hit records scanned", f"{total_hits:,}")
            t.add_row("Total delta-neutral hits", f"[green]{total_neutral:,}[/green]"
                      if total_neutral > 0 else "[red]0[/red]")
            t.add_row("Packets with ≥1 neutral hit", f"{any_neutral} / {n_pkts}")
            t.add_row("Packets with 0 neutral hits", f"[red]{all_zero}[/red]")
            t.add_row("Max neutral hits (single packet)", str(max_neutral))
            t.add_row("Median delta_leak_basis", f"{median_leak:.1f}")
            t.add_row("Median resid_p10 (10th‐pctile resid)", f"{median_resid_p10:.4f}")
            console.print(t)

            # Interpretation
            if total_neutral == 0:
                console.print(
                    f"[bold red]R={rank}: ZERO delta-neutral hits across all {n_pkts} packets.[/bold red]\n"
                    "  → The hit pools contain NO records that preserve delta_leak.\n"
                    "  → Gate 2 is not navigable from any of these basis subspaces.\n"
                    "  → This is strong evidence of a structural obstruction:\n"
                    "    the {-1,0,1} integer lattice may structurally prevent\n"
                    "    col(Delta) ⊆ col(H) for R=19.\n"
                )
            elif total_neutral < 10 * n_pkts:
                console.print(
                    f"[yellow]R={rank}: very few neutral hits (~{total_neutral/n_pkts:.1f}/packet avg).[/yellow]\n"
                    "  → Gate 2 is extremely sparse but not measure-zero.\n"
                    "  → Targeted search focusing on these neutral records may be viable.\n"
                )
            else:
                console.print(
                    f"[green]R={rank}: {total_neutral:,} neutral hits found "
                    f"({100*total_neutral/total_hits:.2f}% of pool).[/green]\n"
                    "  → Gate 2-aware assembly is feasible. Focus Stage 2 on neutral-hit pool.\n"
                )
        else:
            print(f"\nR={rank}: {n_pkts} packets | {total_hits:,} hits | "
                  f"{total_neutral:,} neutral ({100*total_neutral/max(total_hits,1):.2f}%) | "
                  f"packets with neutral: {any_neutral}/{n_pkts}")
            if total_neutral == 0:
                print(f"  *** ZERO neutral hits — structural obstruction likely ***")

    if console:
        console.print(f"\n[bold green]Scan complete.[/bold green]")
        for rank, rows in by_rank.items():
            path = output_dir / f"gate2_scan_R{rank}.csv"
            console.print(f"  [dim]→ {path}[/dim]")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
