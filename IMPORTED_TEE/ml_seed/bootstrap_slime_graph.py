from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import os
import sqlite3
import time

import numpy as np

from ml_seed.bootstrap_db import _score_tensor_against_t333
from ml_seed.dataset import DatasetWriter
from ml_seed.encode import seed_to_image
from ml_seed.schema import K_CHECKPOINT, TrainingSample


def _load_unique_rows(
    db_path: str,
    limit: int | None,
) -> list[tuple[int, str, str, str | None, int | None, bytes]]:
    con = sqlite3.connect(db_path)
    try:
        query = (
            "SELECT id, label, seed_name, best_rank, C_blob FROM nodes "
            "WHERE C_blob IS NOT NULL ORDER BY updated_at ASC, id ASC"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    unique_rows: list[tuple[int, str, str, str | None, int | None, bytes]] = []
    seen_blob_hashes: set[str] = set()
    for index, (node_id, label, seed_name, stored_rank, c_blob) in enumerate(rows):
        blob_hash = hashlib.md5(c_blob).hexdigest()
        if blob_hash in seen_blob_hashes:
            continue
        seen_blob_hashes.add(blob_hash)
        unique_rows.append((index, blob_hash, node_id, label, seed_name, stored_rank, c_blob))
    return unique_rows


def _score_row(task: tuple[int, str, str, str | None, str | None, int | None, bytes, list[int], int, int, int, int]) -> tuple[str, str, str | None, int | None, str | None, float, int, float | None, np.ndarray]:
    (
        index,
        blob_hash,
        node_id,
        label,
        seed_name,
        stored_rank,
        c_blob,
        ranks,
        restarts_per_rank,
        als_iters,
        lbfgs_maxiter,
        k_checkpoint,
    ) = task
    C = np.load(io.BytesIO(c_blob), allow_pickle=False).astype(np.float64)
    if C.shape != (9, 9, 9):
        raise ValueError(f"Expected (9, 9, 9), got {C.shape}")
    seed_base = int(blob_hash[:8], 16) ^ (index * 1009)
    res_final, rank, res_k = _score_tensor_against_t333(
        C=C,
        ranks=ranks,
        restarts_per_rank=restarts_per_rank,
        als_iters=als_iters,
        lbfgs_maxiter=lbfgs_maxiter,
        k_checkpoint=k_checkpoint,
        seed_base=seed_base,
    )
    return (
        blob_hash,
        node_id,
        label,
        stored_rank,
        seed_name,
        res_final,
        rank,
        res_k,
        seed_to_image(C),
    )


def run(
    db_path: str,
    out: str,
    limit: int | None = None,
    ranks: list[int] | None = None,
    restarts_per_rank: int = 2,
    als_iters: int = 40,
    lbfgs_maxiter: int = 150,
    k_checkpoint: int = K_CHECKPOINT,
    workers: int | None = None,
) -> None:
    if ranks is None:
        ranks = [19, 20, 21, 22]
    if workers is None:
        workers = min(10, os.cpu_count() or 1)
    workers = max(1, int(workers))

    writer = DatasetWriter(out)
    unique_rows = _load_unique_rows(db_path=db_path, limit=limit)

    print(
        f"Found {len(unique_rows)} unique slime graph tensors to relabel "
        f"using {workers} worker(s)."
    )
    converted = 0

    tasks = [
        (
            index,
            blob_hash,
            node_id,
            label,
            seed_name,
            stored_rank,
            c_blob,
            ranks,
            restarts_per_rank,
            als_iters,
            lbfgs_maxiter,
            k_checkpoint,
        )
        for index, blob_hash, node_id, label, seed_name, stored_rank, c_blob in unique_rows
    ]

    if workers == 1:
        result_iter = map(_score_row, tasks)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        result_iter = executor.map(_score_row, tasks, chunksize=4)

    try:
        for blob_hash, node_id, label, stored_rank, seed_name, res_final, rank, res_k, image in result_iter:
            sample = TrainingSample(
                image=image,
                res_k=res_k,
                res_final=res_final,
                rank=rank,
                node_name=label or node_id,
                ts=time.time(),
            )
            writer.append(sample)
            converted += 1
            print(
                f"  [{converted}] {label or node_id}: relabeled_rank={rank} "
                f"stored_rank={stored_rank} res={res_final:.4e} "
                f"res_k={'None' if res_k is None else f'{res_k:.4e}'} seed={seed_name}"
            )
    except Exception as exc:
        print(f"  fatal relabel error: {exc}")
        raise
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)

    print(
        f"Relabeled {converted} unique slime graph tensors "
        f"({len(unique_rows)} unique blobs seen) -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill ML samples by re-labeling tensors stored in a recovered "
            "slime graph SQLite database against T333."
        )
    )
    parser.add_argument(
        "--db-path",
        default="tmp/git_recovery/extracted/bini_slime/slime_graph.db",
    )
    parser.add_argument(
        "--out",
        default="ml_seed/data/slime_graph_bootstrap.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=[19, 20, 21, 22])
    parser.add_argument("--restarts-per-rank", type=int, default=2)
    parser.add_argument("--als-iters", type=int, default=40)
    parser.add_argument("--lbfgs-maxiter", type=int, default=150)
    parser.add_argument("--k-checkpoint", type=int, default=K_CHECKPOINT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run(
        db_path=args.db_path,
        out=args.out,
        limit=args.limit,
        ranks=args.ranks,
        restarts_per_rank=args.restarts_per_rank,
        als_iters=args.als_iters,
        lbfgs_maxiter=args.lbfgs_maxiter,
        k_checkpoint=args.k_checkpoint,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()