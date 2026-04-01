"""CPU-side tensor operations: reconstruction, fitness, Frobenius residual."""

import numpy as np

from .config import TARGET_TENSOR, DIM


def reconstruct(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Reconstruct 9×9×9 tensor from factor matrices via CP decomposition.

    alpha, beta, gamma: each (R, 9) where R is the number of rank-1 terms.
    """
    return np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)


def residual(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
             target: np.ndarray | None = None) -> np.ndarray:
    """Compute element-wise residual: reconstructed - target."""
    if target is None:
        target = TARGET_TENSOR
    return reconstruct(alpha, beta, gamma) - target


def fitness_maxabs(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                   target: np.ndarray | None = None) -> float:
    """Compute max-abs fitness (minimax objective)."""
    return float(np.max(np.abs(residual(alpha, beta, gamma, target))))


def fitness_frobenius(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                      target: np.ndarray | None = None) -> float:
    """Compute Frobenius norm of the residual."""
    return float(np.linalg.norm(residual(alpha, beta, gamma, target)))
