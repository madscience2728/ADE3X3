"""
data.py — Matrix pair generator for Keth-Varai Machine training.

Generates random float32 3x3 matrix pairs (A, B) and their exact products C = A*B.
Ground truth is computed in float64 then cast to float32 to minimize label noise.
"""
import torch


def sample_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (A, B, C) each of shape (batch_size, 9), float32.
    C = A @ B computed exactly in float64.
    """
    A = torch.randn(batch_size, 9, dtype=torch.float64)
    B = torch.randn(batch_size, 9, dtype=torch.float64)

    A_mat = A.view(batch_size, 3, 3)
    B_mat = B.view(batch_size, 3, 3)
    C_mat = torch.bmm(A_mat, B_mat)

    A32 = A.float().to(device)
    B32 = B.float().to(device)
    C32 = C_mat.view(batch_size, 9).float().to(device)

    return A32, B32, C32


def frobenius_relative_error(C_pred: torch.Tensor, C_true: torch.Tensor) -> torch.Tensor:
    """
    Mean relative Frobenius error over a batch:
      mean_i( ||C_pred_i - C_true_i||_F / ||C_true_i||_F )
    """
    diff = (C_pred - C_true).view(-1, 9)
    true_norm = C_true.view(-1, 9).norm(dim=1).clamp(min=1e-12)
    return (diff.norm(dim=1) / true_norm).mean()
