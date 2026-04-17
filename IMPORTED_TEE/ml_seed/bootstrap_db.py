from __future__ import annotations

import argparse
import io
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.encode import seed_to_image
from ml_seed.schema import K_CHECKPOINT, TrainingSample


def _score_tensor_against_t333(
    C: np.ndarray,
    ranks: list[int],
    restarts_per_rank: int,
    als_iters: int,
    lbfgs_maxiter: int,
    k_checkpoint: int,
    seed_base: int | None = None,
) -> tuple[float, int, float | None]:
    from bini_slime.slime_search import T333, _one_restart

    best_res = float("inf")
    best_rank = -1
    best_res_k = None
    rng_base = seed_base if seed_base is not None else int(time.time() * 1000) % (2**31)

    for rank in ranks:
        for restart_idx in range(restarts_per_rank):
            res, _U, _V, _W, res_k = _one_restart(
                C,
                rank,
                seed=rng_base + rank * 100003 + restart_idx,
                als_iters=als_iters,
                lbfgs_maxiter=lbfgs_maxiter,
                T_target=T333,
                k_checkpoint=k_checkpoint,
            )
            if res < best_res:
                best_res = float(res)
                best_rank = int(rank)
                best_res_k = None if res_k is None else float(res_k)
    return best_res, best_rank, best_res_k


def run(
    db_path: str,
    out: str,
    novel_only: bool = True,
    limit: int | None = None,
    ranks: list[int] | None = None,
    restarts_per_rank: int = 2,
    als_iters: int = 40,
    lbfgs_maxiter: int = 150,
    k_checkpoint: int = K_CHECKPOINT,
) -> None:
    if ranks is None:
        ranks = [19, 20, 21, 22]

    writer = DatasetWriter(out)
    con = sqlite3.connect(db_path)
    try:
        query = (
            "SELECT fp_hash, name, C_blob FROM algebras "
            "WHERE n = 9 AND C_blob IS NOT NULL"
        )
        if novel_only:
            query += " AND is_known = 0"
        query += " ORDER BY created_at ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    print(f"Found {len(rows)} DB tensors to relabel.")
    converted = 0

    for fp_hash, name, c_blob in rows:
        try:
            C = np.load(io.BytesIO(c_blob)).astype(np.float64)
            if C.shape != (9, 9, 9):
                continue
            res_final, rank, res_k = _score_tensor_against_t333(
                C=C,
                ranks=ranks,
                restarts_per_rank=restarts_per_rank,
                als_iters=als_iters,
                lbfgs_maxiter=lbfgs_maxiter,
                k_checkpoint=k_checkpoint,
            )
            sample = TrainingSample(
                image=seed_to_image(C),
                res_k=res_k,
                res_final=res_final,
                rank=rank,
                node_name=name or fp_hash,
                ts=time.time(),
            )
            writer.append(sample)
            converted += 1
            print(
                f"  [{converted}/{len(rows)}] {name or fp_hash}: rank={rank} "
                f"res={res_final:.4e} res_k={'None' if res_k is None else f'{res_k:.4e}'}"
            )
        except Exception as exc:
            print(f"  skip {name or fp_hash}: {exc}")

    print(f"Relabeled {converted} DB tensors -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ML samples by re-labeling tensors stored in algebras.sqlite against T333."
    )
    parser.add_argument("--db-path", default="ade/db/algebras.sqlite")
    parser.add_argument("--out", default="ml_seed/data/db_bootstrap.jsonl")
    parser.add_argument("--include-known", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=[19, 20, 21, 22])
    parser.add_argument("--restarts-per-rank", type=int, default=2)
    parser.add_argument("--als-iters", type=int, default=40)
    parser.add_argument("--lbfgs-maxiter", type=int, default=150)
    parser.add_argument("--k-checkpoint", type=int, default=K_CHECKPOINT)
    args = parser.parse_args()
    run(
        db_path=args.db_path,
        out=args.out,
        novel_only=not args.include_known,
        limit=args.limit,
        ranks=args.ranks,
        restarts_per_rank=args.restarts_per_rank,
        als_iters=args.als_iters,
        lbfgs_maxiter=args.lbfgs_maxiter,
        k_checkpoint=args.k_checkpoint,
    )


if __name__ == "__main__":
    main()