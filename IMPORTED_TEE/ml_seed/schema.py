from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass

import numpy as np

K_CHECKPOINT = 20


@dataclass
class TrainingSample:
    image: np.ndarray
    res_k: float | None
    res_final: float
    rank: int
    node_name: str
    ts: float

    def to_json_line(self) -> str:
        data = asdict(self)
        data["image"] = base64.b64encode(
            np.asarray(self.image, dtype=np.float32).tobytes()
        ).decode("ascii")
        return json.dumps(data)

    @classmethod
    def from_json_line(cls, line: str) -> "TrainingSample":
        data = json.loads(line)
        raw = base64.b64decode(data["image"])
        data["image"] = np.frombuffer(raw, dtype=np.float32).reshape(27, 27)
        return cls(**data)