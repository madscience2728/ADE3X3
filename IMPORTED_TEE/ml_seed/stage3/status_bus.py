from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time

import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.stage2.data_prep import load_samples

STATUS_PATH = pathlib.Path("ml_seed/data/status.json")
DEFAULT_SAMPLE_FILES = [
    "ml_seed/data/samples.jsonl",
    "ml_seed/data/db_bootstrap.jsonl",
    "ml_seed/data/slime_graph_bootstrap_parallel.jsonl",
    "ml_seed/data/slime_graph_live_bootstrap_parallel.jsonl",
]
DEFAULT_DEPLOYMENT_PATH = pathlib.Path("ml_seed/data/model_checkpoints/deployment.json")


def read() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_section(section: str, data: dict) -> None:
    current = read()
    current[section] = data
    current["updated_at"] = time.time()
    _atomic_write(STATUS_PATH, json.dumps(current, indent=2))


def append_history(entry: dict, max_entries: int = 20) -> None:
    current = read()
    history = list(current.get("history", []))
    history.append(entry)
    current["history"] = history[-max_entries:]
    current["updated_at"] = time.time()
    _atomic_write(STATUS_PATH, json.dumps(current, indent=2))


def existing_sample_files(paths: list[str] | None = None) -> list[str]:
    candidates = DEFAULT_SAMPLE_FILES if paths is None else paths
    return [path for path in candidates if pathlib.Path(path).exists()]


def write_dataset_status(paths: list[str] | None = None) -> dict:
    files = existing_sample_files(paths)
    live_path = pathlib.Path("ml_seed/data/samples.jsonl")
    live_samples = len(DatasetWriter.load_all(str(live_path))) if live_path.exists() else 0
    bootstrap_paths = [path for path in files if pathlib.Path(path) != live_path]
    bootstrap_samples = len(load_samples(bootstrap_paths)) if bootstrap_paths else 0
    combined_samples = load_samples(files) if files else []

    residuals = np.array([sample.res_final for sample in combined_samples], dtype=np.float64)
    payload = {
        "live_samples": live_samples,
        "bootstrap_samples": bootstrap_samples,
        "combined": len(combined_samples),
        "rank_distribution": {
            str(rank): sum(1 for sample in combined_samples if sample.rank == rank)
            for rank in sorted({sample.rank for sample in combined_samples})
        },
        "residual_min": float(residuals.min()) if residuals.size else None,
        "residual_max": float(residuals.max()) if residuals.size else None,
        "residual_mean": float(residuals.mean()) if residuals.size else None,
        "residual_std": float(residuals.std()) if residuals.size else None,
    }
    write_section("dataset", payload)
    return payload


def sync_model_status_from_deployment(deployment_path: str | pathlib.Path | None = None) -> dict | None:
    path = DEFAULT_DEPLOYMENT_PATH if deployment_path is None else pathlib.Path(deployment_path)
    if not path.exists():
        return None
    try:
        deployment = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_payload = {
        "kendall_tau": deployment.get("kendall_tau"),
        "val_loss": deployment.get("val_loss", deployment.get("best_val_loss")),
        "deployed": bool(deployment.get("deployable", False)),
        "n_samples_at_train": deployment.get("n_samples", 0),
        "trained_at": deployment.get("trained_at"),
        "checkpoint_path": deployment.get("checkpoint_path") or "ml_seed/data/model_checkpoints/best.pt",
    }
    write_section("model", model_payload)

    current = read()
    history = current.get("history", [])
    if not history:
        append_history(
            {
                "ts": deployment.get("trained_at", time.time()),
                "n_samples": deployment.get("n_samples", 0),
                "kendall_tau": deployment.get("kendall_tau"),
                "val_loss": deployment.get("val_loss", deployment.get("best_val_loss")),
                "deployed": bool(deployment.get("deployable", False)),
            }
        )
    return model_payload


def _atomic_write(path: pathlib.Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(12):
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp, path)
            return True
        except PermissionError as exc:
            last_error = exc
            try:
                os.unlink(tmp)
            except OSError:
                pass
            time.sleep(min(0.05 * (attempt + 1), 0.5))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    if last_error is not None:
        print(f"[status_bus] warning: skipped status write after repeated file-lock failures: {last_error}", flush=True)
        return False
    return False