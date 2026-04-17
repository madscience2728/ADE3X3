from __future__ import annotations

import numpy as np


def seed_to_image(C: np.ndarray) -> np.ndarray:
    """Convert a (9, 9, 9) structure tensor to a normalized (27, 27) image."""
    tensor = np.asarray(C, dtype=np.float64)
    if tensor.shape != (9, 9, 9):
        raise ValueError(f"Expected (9, 9, 9), got {tensor.shape}")
    image = tensor.transpose(2, 0, 1).reshape(27, 27)
    scale = float(np.abs(image).max())
    if scale > 1e-12:
        image = image / scale
    return image.astype(np.float32, copy=False)