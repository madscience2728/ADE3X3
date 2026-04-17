from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.encode import seed_to_image
from ml_seed.schema import TrainingSample

_lock = threading.Lock()
_instance: "SampleCollector | _NoOpCollector | None" = None


class _NoOpCollector:
    def record(self, **_: Any) -> None:
        return None


class SampleCollector:
    def __init__(self, path: str | None = None):
        target = Path(path) if path is not None else Path(__file__).resolve().parent / "data" / "samples.jsonl"
        self._writer = DatasetWriter(str(target))
        self._last_status_refresh = 0.0

    def record(
        self,
        seed_C: np.ndarray,
        res_k: float | None,
        res_final: float,
        rank: int,
        node_name: str = "",
    ) -> None:
        try:
            image = seed_to_image(seed_C)
            sample = TrainingSample(
                image=image,
                res_k=res_k,
                res_final=float(res_final),
                rank=int(rank),
                node_name=node_name,
                ts=time.time(),
            )
            self._writer.append(sample)
            self._maybe_refresh_status()
        except Exception:
            pass

    def _maybe_refresh_status(self) -> None:
        now = time.time()
        if now - self._last_status_refresh < 5.0:
            return
        self._last_status_refresh = now
        try:
            from ml_seed.stage3.status_bus import write_dataset_status

            write_dataset_status()
        except Exception:
            pass


def get_collector() -> SampleCollector | _NoOpCollector:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                try:
                    _instance = SampleCollector()
                except Exception:
                    _instance = _NoOpCollector()
    return _instance