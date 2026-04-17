from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.encode import seed_to_image
from ml_seed.schema import TrainingSample


def _load_graph_tensor(graph_db: Path, node_id: str | None) -> np.ndarray | None:
    if not node_id or not graph_db.exists():
        return None
    con = sqlite3.connect(str(graph_db))
    try:
        row = con.execute("SELECT C_blob FROM nodes WHERE id = ?", (node_id,)).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    return np.load(io.BytesIO(row[0]))


def _load_checkpoint_tensor(record: dict, checkpoints_dir: Path) -> np.ndarray | None:
    checkpoint_name = record.get("checkpoint")
    if checkpoint_name:
        checkpoint_path = checkpoints_dir / checkpoint_name
        if checkpoint_path.exists():
            with np.load(str(checkpoint_path)) as data:
                return np.asarray(data["C"], dtype=np.float64)
    return None


def run(discoveries: str, checkpoints: str, out: str, graph_db: str) -> None:
    writer = DatasetWriter(out)
    discoveries_path = Path(discoveries)
    checkpoints_dir = Path(checkpoints)
    graph_db_path = Path(graph_db)

    if not discoveries_path.exists():
        raise FileNotFoundError(f"Discoveries file not found: {discoveries_path}")

    with discoveries_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    print(f"Found {len(records)} discovery records.")
    converted = 0

    for record in records:
        try:
            C = _load_checkpoint_tensor(record, checkpoints_dir)
            if C is None:
                C = _load_graph_tensor(graph_db_path, record.get("node_id"))
            if C is None:
                continue
            sample = TrainingSample(
                image=seed_to_image(C),
                res_k=None,
                res_final=float(record["residual"]),
                rank=int(record["rank"]),
                node_name=str(record.get("node_id") or record.get("novel_id") or ""),
                ts=float(record.get("timestamp", time.time())),
            )
            writer.append(sample)
            converted += 1
        except Exception as exc:
            print(f"  skip {record.get('novel_id', 'unknown')}: {exc}")

    print(f"Bootstrapped {converted} samples -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discoveries", default="bini_slime/discoveries.jsonl")
    parser.add_argument("--checkpoints", default="bini_slime/results/checkpoints")
    parser.add_argument("--out", default="ml_seed/data/bootstrap.jsonl")
    parser.add_argument("--graph-db", default="bini_slime/slime_graph.db")
    args = parser.parse_args()
    run(args.discoveries, args.checkpoints, args.out, args.graph_db)


if __name__ == "__main__":
    main()