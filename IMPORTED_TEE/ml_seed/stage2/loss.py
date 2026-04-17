from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def bradley_terry_loss(
    score_a,
    score_b,
    res_a,
    res_b,
    margin: float = 0.1,
):
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is required for ml_seed.stage2.loss. Install torch in the active environment."
        ) from _TORCH_IMPORT_ERROR

    diff = res_a - res_b
    ambiguous = diff.abs() < margin
    signed_score_diff = torch.where(diff < 0, score_a - score_b, score_b - score_a)
    loss = F.softplus(signed_score_diff)
    loss = loss[~ambiguous]
    if loss.numel() == 0:
        return (score_a.sum() * 0.0) + (score_b.sum() * 0.0)
    return loss.mean()