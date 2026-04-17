from __future__ import annotations

import numpy as np


def image_to_C(image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convert a normalized 27x27 image back into a 9x9x9 tensor."""
    image_arr = np.asarray(image, dtype=np.float32)
    if image_arr.shape != (27, 27):
        raise ValueError(f"Expected (27, 27), got {image_arr.shape}")
    C_kij = image_arr.reshape(9, 9, 9).astype(np.float64, copy=False)
    return C_kij.transpose(1, 2, 0) * float(alpha)


def round_trip_error(C: np.ndarray) -> float:
    """Encode and decode a tensor, restoring the lost amplitude analytically."""
    from ml_seed.encode import seed_to_image

    tensor = np.asarray(C, dtype=np.float64)
    if tensor.shape != (9, 9, 9):
        raise ValueError(f"Expected (9, 9, 9), got {tensor.shape}")
    image = seed_to_image(tensor)
    alpha = float(np.abs(tensor).max()) or 1.0
    recovered = image_to_C(image, alpha=alpha)
    return float(np.linalg.norm(tensor - recovered))