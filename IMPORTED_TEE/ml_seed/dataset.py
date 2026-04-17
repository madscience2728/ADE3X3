from __future__ import annotations

import pathlib
import threading

from ml_seed.schema import TrainingSample


class DatasetWriter:
    """Append-only JSONL writer with dedup support at load time."""

    def __init__(self, path: str):
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def append(self, sample: TrainingSample) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(sample.to_json_line() + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @staticmethod
    def load_all(path: str) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        seen_keys: set[tuple[str, int, float]] = set()
        sample_path = pathlib.Path(path)
        if not sample_path.exists():
            return samples
        with sample_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                sample = TrainingSample.from_json_line(stripped)
                key = (sample.node_name, sample.rank, round(sample.res_final, 6))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                samples.append(sample)
        return samples