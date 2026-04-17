from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> tuple[object, object]:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for ml_seed.stage2. Install torch in the active environment."
        ) from _TORCH_IMPORT_ERROR
    return torch, nn


def get_default_device() -> "torch.device":
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if nn is not None:
    class ConvRanker(nn.Module):
        """ConvNet that scores 27x27 seed images by predicted basin quality.

        Architecture: 3-block conv backbone (64→128→256 channels) with a
        3-layer MLP head (256→512→256→1).  Dropout2d in conv blocks and
        standard Dropout in the head enable MC-Dropout uncertainty estimates
        during generation without any extra forward-pass overhead.
        """

        def __init__(self, dropout_p: float = 0.2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(dropout_p),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(dropout_p),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(dropout_p),
                nn.MaxPool2d(2),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.pool(self.features(x)))

        def score_with_uncertainty(
            self, x: "torch.Tensor", n_samples: int = 30
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            self.train()
            with torch.no_grad():
                scores = torch.stack(
                    [self.forward(x).squeeze(-1) for _ in range(n_samples)], dim=0
                )
            return scores.mean(0), scores.std(0)
else:
    class ConvRanker:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            require_torch()