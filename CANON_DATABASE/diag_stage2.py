"""
Single-packet Stage 2 diagnostic — no multiprocessing, no dashboard.
Loads the DB once, picks the first packet for each rank, and runs the
assembler in-process with full verbose output.

Usage:
    python CANON_DATABASE/diag_stage2.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

# Make local imports work when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from term_db import TermDB
from search_config import SearchConfig
from assembler import DependentAssembler
from gate_check import full_gate_check

WORKSPACE = Path(__file__).resolve().parents[1]
DB_PATH   = WORKSPACE / "CANON_DATABASE" / "data"
PKT_ROOT  = WORKSPACE / "CANON_DATABASE" / "packets"

def run_packet(db, config, pkt_path: Path):
    print(f"\n{'='*60}")
    print(f"Packet: {pkt_path.name}  ({pkt_path.stat().st_size/1024:.1f} KB)")
    data = np.load(str(pkt_path), allow_pickle=False)
    rank               = int(data["rank"][0])
    basis_rec_idx      = data["basis_record_indices"].tolist()
    hit_rec_idx        = data["hit_record_indices"]
    dependent_estimate = int(data["dependent_estimate"][0])
    V_basis            = data["V_basis"]

    print(f"  rank={rank}  basis_records={len(basis_rec_idx)}  hits={len(hit_rec_idx):,}  dep_est={dependent_estimate:,}")
    print(f"  V_basis shape: {V_basis.shape}")
    print(f"  basis_record_indices[:5]: {basis_rec_idx[:5]}")

    # Sanity check: do basis H-rows span V_basis?
    H_basis = db.H[basis_rec_idx].astype(np.float64)
    rank_H = int(np.linalg.matrix_rank(H_basis))
    print(f"  rank(H_basis) = {rank_H}  (target = {rank - 9})")

    # Count how many hits are actually in the subspace
    if len(hit_rec_idx) == 0:
        print("  WARNING: 0 hits in packet!")
        return

    # Run assembler
    config2 = SearchConfig(target_ranks=[rank])
    assembler = DependentAssembler(db, config2)

    t0 = time.perf_counter()
    solutions = assembler.assemble(rank, basis_rec_idx, hit_rec_idx)
    elapsed = time.perf_counter() - t0

    s = assembler.stats
    target_dep = rank - len(basis_rec_idx)
    print(f"  target_dependents={target_dep}")
    print(f"  configs_tested={s.configs_tested:,}  gate2_passes={s.gate2_passes}  gate3_passes={s.gate3_passes}")
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  solutions: {len(solutions)}")
    if solutions:
        print("  *** SOLUTION FOUND ***", solutions[0].get("gate"), solutions[0].keys())


def main():
    print(f"Loading DB from {DB_PATH} ...")
    t0 = time.perf_counter()
    db = TermDB.__new__(TermDB)
    db._load(str(DB_PATH), progress_callback=lambda m: print(f"  [{m}]"), readonly=True)
    print(f"  DB loaded in {time.perf_counter()-t0:.1f}s  ({db.H.shape[0]:,} records)")

    config = SearchConfig()

    for rank_dir in sorted(PKT_ROOT.iterdir()):
        if not rank_dir.is_dir():
            continue
        pkts = sorted(rank_dir.glob("basis_*.npz"))
        if not pkts:
            print(f"\nNo packets in {rank_dir.name}")
            continue
        print(f"\n{'#'*60}")
        print(f"# {rank_dir.name}: {len(pkts)} packets — testing first 3")
        for pkt in pkts[:3]:
            run_packet(db, config, pkt)


if __name__ == "__main__":
    main()
