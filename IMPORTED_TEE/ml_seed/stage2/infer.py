from __future__ import annotations

import json
import pathlib

import numpy as np

from ml_seed.encode import seed_to_image

try:
    import torch
except ImportError:
    torch = None

from ml_seed.stage2.model import ConvRanker, get_default_device

_MODEL = None
_MODEL_DIR = pathlib.Path("ml_seed/data/model_checkpoints")
_MODEL_PATH = _MODEL_DIR / "best.pt"
_DEPLOYMENT_PATH = _MODEL_DIR / "deployment.json"


def _load_deployment_metadata() -> dict | None:
    if not _DEPLOYMENT_PATH.exists():
        return None
    try:
        return json.loads(_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_model_deployable(tau_threshold: float = 0.6) -> bool:
    if torch is None or not _MODEL_PATH.exists():
        return False
    metadata = _load_deployment_metadata()
    if not metadata:
        return False
    tau = float(metadata.get("kendall_tau", 0.0))
    deployable = bool(metadata.get("deployable", False))
    min_samples = int(metadata.get("min_samples_for_deploy", 300))
    n_samples = int(metadata.get("n_samples", 0))
    return deployable and tau >= tau_threshold and n_samples >= min_samples


def _load_model() -> ConvRanker | None:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if torch is None or not is_model_deployable():
        return None
    try:
        device = get_default_device()
        model = ConvRanker()
        state = torch.load(_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception:
        return None


def score_seeds(
    C_list: list[np.ndarray],
    n_mc: int = 30,
    lam: float = 1.0,
) -> list[dict]:
    model = _load_model()
    if model is None or torch is None:
        return [
            {"score": 0.0, "uncertainty": 0.0, "acquisition": 0.0}
            for _ in C_list
        ]

    device = next(model.parameters()).device
    images = torch.stack(
        [torch.from_numpy(seed_to_image(C)).unsqueeze(0) for C in C_list]
    ).to(device=device, dtype=torch.float32)
    mean_scores, std_scores = model.score_with_uncertainty(images, n_samples=n_mc)
    mean_scores = mean_scores.detach().cpu()
    std_scores = std_scores.detach().cpu()

    results = []
    for score, uncertainty in zip(mean_scores.tolist(), std_scores.tolist()):
        results.append(
            {
                "score": float(score),
                "uncertainty": float(uncertainty),
                "acquisition": float(score - lam * uncertainty),
            }
        )
    return results