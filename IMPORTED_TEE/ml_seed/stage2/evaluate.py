from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau

try:
    import torch
except ImportError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def kendall_tau(model, val_dl) -> float:
    """Compute Kendall tau within rank groups to avoid cross-rank contamination."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for ml_seed.stage2.evaluate. Install torch in the active environment."
        ) from _TORCH_IMPORT_ERROR
    device = next(model.parameters()).device
    per_rank_scores: dict[int, list[float]] = defaultdict(list)
    per_rank_residuals: dict[int, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for image_a, image_b, res_a, res_b, ranks in val_dl:
            image_a = image_a.to(device=device, dtype=torch.float32, non_blocking=True)
            image_b = image_b.to(device=device, dtype=torch.float32, non_blocking=True)
            score_a = model(image_a).squeeze(-1).detach().cpu()
            score_b = model(image_b).squeeze(-1).detach().cpu()
            for rank, score, residual in zip(ranks.tolist(), score_a.tolist(), res_a.tolist()):
                per_rank_scores[int(rank)].append(float(score))
                per_rank_residuals[int(rank)].append(float(residual))
            for rank, score, residual in zip(ranks.tolist(), score_b.tolist(), res_b.tolist()):
                per_rank_scores[int(rank)].append(float(score))
                per_rank_residuals[int(rank)].append(float(residual))

    taus: list[float] = []
    weights: list[int] = []
    for rank, scores in per_rank_scores.items():
        residuals = per_rank_residuals[rank]
        if len(scores) < 2 or np.std(scores) == 0.0 or np.std(residuals) == 0.0:
            continue
        tau, _ = kendalltau(scores, residuals)
        if tau is None or np.isnan(tau):
            continue
        taus.append(float(tau))
        weights.append(len(scores))

    if not taus:
        return 0.0
    return float(np.average(taus, weights=weights))