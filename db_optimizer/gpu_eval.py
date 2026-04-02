"""GPU-accelerated batch evaluation (Tier 1 + Tier 2)."""

import numpy as np
import torch

from .config import TARGET_TENSOR, DIM, RANK, GPU_MINIMAX_SWEEPS, GPU_MINIMAX_N_TRIALS, GPU_MINIMAX_FINE_RANGE


def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


# Target tensor on GPU (lazy init)
_TARGET_GPU: dict[torch.device, torch.Tensor] = {}


def _target_on(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Get target tensor on the given device, cached."""
    key = (device, dtype)
    if key not in _TARGET_GPU:
        _TARGET_GPU[key] = torch.tensor(TARGET_TENSOR, dtype=dtype, device=device)
    return _TARGET_GPU[key]


def gpu_batch_fitness(
    factors_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Tier 1: Compute max-abs fitness for a batch of candidates on GPU.

    Parameters
    ----------
    factors_list : list of (alpha, beta, gamma) each (RANK, DIM) numpy arrays
    device : torch device (default: auto-detect)
    dtype : torch dtype for GPU computation (default: float32)

    Returns
    -------
    np.ndarray of shape (N,) with max-abs fitness per candidate (float32)
    """
    if device is None:
        device = _get_device()

    N = len(factors_list)
    if N == 0:
        return np.array([], dtype=np.float32)

    # Pack into batched tensors: (N, RANK, DIM) for each factor
    alpha_np = np.stack([f[0] for f in factors_list])  # (N, R, 9)
    beta_np = np.stack([f[1] for f in factors_list])
    gamma_np = np.stack([f[2] for f in factors_list])

    return gpu_batch_fitness_stacked(alpha_np, beta_np, gamma_np, device=device, dtype=dtype)


def gpu_batch_fitness_stacked(
    alpha_np: np.ndarray,
    beta_np: np.ndarray,
    gamma_np: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Tier 1: Compute max-abs fitness from pre-stacked (N, RANK, DIM) arrays."""
    if device is None:
        device = _get_device()

    N = alpha_np.shape[0]
    if N == 0:
        return np.array([], dtype=np.float32)

    with torch.no_grad():
        alpha_t = torch.tensor(alpha_np, dtype=dtype, device=device)
        beta_t = torch.tensor(beta_np, dtype=dtype, device=device)
        gamma_t = torch.tensor(gamma_np, dtype=dtype, device=device)

        # Batch reconstruct: einsum over rank dimension
        # (N, R, 9) × (N, R, 9) × (N, R, 9) → (N, 9, 9, 9)
        recon = torch.einsum('nra,nrb,nrc->nabc', alpha_t, beta_t, gamma_t)

        # Residual
        target = _target_on(device, dtype)
        res = recon - target.unsqueeze(0)  # (N, 9, 9, 9)

        # Max-abs per candidate
        fitness = res.abs().reshape(N, -1).max(dim=1).values  # (N,)

        result = fitness.cpu().numpy().astype(np.float32)

    return result


def gpu_batch_fitness_from_blobs(
    blobs: list[bytes],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Tier 1: Compute max-abs fitness directly from blob list.

    Avoids intermediate Python unpacking — decodes blobs in bulk.
    """
    from .blob import blob_to_factors
    factors_list = [blob_to_factors(b) for b in blobs]
    return gpu_batch_fitness(factors_list, device=device, dtype=dtype)


def gpu_batch_frobenius(
    factors_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Compute Frobenius norm of residual for a batch of candidates on GPU."""
    if device is None:
        device = _get_device()

    N = len(factors_list)
    if N == 0:
        return np.array([], dtype=np.float32)

    alpha_np = np.stack([f[0] for f in factors_list])
    beta_np = np.stack([f[1] for f in factors_list])
    gamma_np = np.stack([f[2] for f in factors_list])

    with torch.no_grad():
        alpha_t = torch.tensor(alpha_np, dtype=dtype, device=device)
        beta_t = torch.tensor(beta_np, dtype=dtype, device=device)
        gamma_t = torch.tensor(gamma_np, dtype=dtype, device=device)

        recon = torch.einsum('nra,nrb,nrc->nabc', alpha_t, beta_t, gamma_t)
        target = _target_on(device, dtype)
        res = recon - target.unsqueeze(0)

        fro = res.reshape(N, -1).norm(dim=1)  # (N,)
        result = fro.cpu().numpy().astype(np.float32)

    return result


# ── Tier 2: GPU Minimax Refinement ───────────────────────────────────

def gpu_minimax_refine(
    factors_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    sweeps: int = GPU_MINIMAX_SWEEPS,
    n_trials: int = GPU_MINIMAX_N_TRIALS,
    fine_range: float = GPU_MINIMAX_FINE_RANGE,
    device: torch.device | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Tier 2: GPU-accelerated minimax coordinate descent (list-of-tuples API)."""
    N = len(factors_list)
    if N == 0:
        return []
    alpha_np = np.stack([f[0] for f in factors_list])
    beta_np = np.stack([f[1] for f in factors_list])
    gamma_np = np.stack([f[2] for f in factors_list])
    return gpu_minimax_refine_stacked(alpha_np, beta_np, gamma_np,
                                      sweeps=sweeps, n_trials=n_trials,
                                      fine_range=fine_range, device=device)


def gpu_minimax_refine_stacked(
    alpha_np: np.ndarray,
    beta_np: np.ndarray,
    gamma_np: np.ndarray,
    sweeps: int = GPU_MINIMAX_SWEEPS,
    n_trials: int = GPU_MINIMAX_N_TRIALS,
    fine_range: float = GPU_MINIMAX_FINE_RANGE,
    device: torch.device | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Tier 2: GPU-accelerated minimax coordinate descent from pre-stacked arrays.

    For each candidate, performs ``sweeps`` passes over all 513 coefficients.
    Each coefficient is perturbed with ``n_trials`` linearly-spaced trial values
    and the best-improving move is accepted.

    Uses slice-update trick: changing one coefficient only modifies a single
    (DIM, DIM) slice of the (DIM, DIM, DIM) residual, enabling efficient
    incremental max-abs tracking.

    Parameters
    ----------
    alpha_np, beta_np, gamma_np : pre-stacked (N, RANK, DIM) float64 arrays
    sweeps : coordinate descent passes over all 513 coefficients
    n_trials : trial perturbations per coefficient
    fine_range : half-width of perturbation grid around current value
    device : torch device (default: auto-detect)

    Returns
    -------
    list of (alpha, beta, gamma, fitness) tuples with improved factors
    """
    if device is None:
        device = _get_device()

    N = alpha_np.shape[0]
    if N == 0:
        return []

    with torch.no_grad():
        alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)
        beta = torch.tensor(beta_np, dtype=torch.float32, device=device)
        gamma = torch.tensor(gamma_np, dtype=torch.float32, device=device)
        factors_gpu = [alpha, beta, gamma]

        target = _target_on(device, torch.float32)

        # Initial residuals: (N, DIM, DIM, DIM)
        R = torch.einsum('nra,nrb,nrc->nabc', alpha, beta, gamma) - target.unsqueeze(0)

        # Fine-grid offsets: (n_trials,)
        offsets = torch.linspace(-fine_range, fine_range, n_trials, device=device, dtype=torch.float32)

        arange_N = torch.arange(N, device=device)

        for _sweep in range(sweeps):
            # TDR prevention: sync after each sweep so the driver sees progress
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # Process one factor-mode at a time so per-slice max-abs stays valid
            for fi in range(3):
                # Compute per-slice max-abs for this mode: S[n, d] = max over other dims
                S = _mode_slice_maxabs(R, fi)  # (N, DIM)

                # Shuffle coefficient order within this mode
                perm = torch.randperm(RANK * DIM, device=device)

                for ci_val in perm:
                    ci = ci_val.item()
                    ri = ci // DIM
                    di = ci % DIM

                    old_vals = factors_gpu[fi][:, ri, di]  # (N,)

                    # Trial values: (N, n_trials)
                    trials = old_vals.unsqueeze(1) + offsets.unsqueeze(0)
                    delta = trials - old_vals.unsqueeze(1)  # (N, T)

                    # Outer product of the OTHER two factor vectors at rank ri
                    f1, f2 = _other_factor_vecs(factors_gpu, fi, ri)  # (N, DIM) each
                    outer2 = f1.unsqueeze(2) * f2.unsqueeze(1)  # (N, DIM, DIM)

                    # Current slice of R for this mode/dim
                    old_slice = _get_slice(R, fi, di)  # (N, DIM, DIM)

                    # New slice for each trial: (N, T, DIM, DIM)
                    new_slice = old_slice.unsqueeze(1) + delta.unsqueeze(-1).unsqueeze(-1) * outer2.unsqueeze(1)

                    # Max-abs of each trial's slice: (N, T)
                    max_new_slice = new_slice.abs().reshape(N, n_trials, -1).max(dim=2).values

                    # Max of OTHER slices (exclude di) — use save/restore to avoid clone
                    saved_col = S[:, di].clone()
                    S[:, di] = -1.0
                    max_other = S.max(dim=1).values  # (N,)
                    S[:, di] = saved_col

                    # Overall fitness per trial: (N, T)
                    trial_fitness = torch.maximum(max_other.unsqueeze(1), max_new_slice)

                    # Current fitness
                    current_fitness = torch.maximum(max_other, saved_col)  # (N,)

                    # Best trial per candidate
                    best_trial_fitness, best_idx = trial_fitness.min(dim=1)

                    # Accept improvements (FP32 margin)
                    accept = best_trial_fitness < current_fitness - 1e-7

                    if accept.any():
                        new_vals = trials[arange_N, best_idx]
                        delta_accept = new_vals[accept] - old_vals[accept]

                        # Update factor
                        factors_gpu[fi][accept, ri, di] = new_vals[accept]

                        # Update residual slice
                        outer_acc = outer2[accept]  # (n_acc, DIM, DIM)
                        delta_outer = delta_accept.unsqueeze(-1).unsqueeze(-1) * outer_acc
                        _update_slice(R, fi, di, accept, delta_outer)

                        # Update S for modified slice
                        updated_slice = _get_slice(R, fi, di)  # (N, DIM, DIM)
                        S[accept, di] = updated_slice[accept].abs().reshape(-1, DIM * DIM).max(dim=1).values

        # Final fitness
        final = R.abs().reshape(N, -1).max(dim=1).values

        # Unpack to CPU numpy
        alpha_out = factors_gpu[0].cpu().numpy().astype(np.float64)
        beta_out = factors_gpu[1].cpu().numpy().astype(np.float64)
        gamma_out = factors_gpu[2].cpu().numpy().astype(np.float64)
        fitness_out = final.cpu().numpy().astype(np.float64)

    return [(alpha_out[i], beta_out[i], gamma_out[i], float(fitness_out[i])) for i in range(N)]


def _mode_slice_maxabs(R: torch.Tensor, fi: int) -> torch.Tensor:
    """Per-slice max-abs for a given tensor mode.

    Returns (N, DIM) where entry [n, d] is max(|R|) over the other two dims
    at mode ``fi`` index ``d``.
    """
    N = R.shape[0]
    if fi == 0:
        return R.abs().reshape(N, DIM, -1).max(dim=2).values
    elif fi == 1:
        return R.abs().permute(0, 2, 1, 3).reshape(N, DIM, -1).max(dim=2).values
    else:
        return R.abs().permute(0, 3, 1, 2).reshape(N, DIM, -1).max(dim=2).values


def _other_factor_vecs(factors_gpu, fi, ri):
    """Return the two OTHER factor vectors at rank ``ri``."""
    if fi == 0:
        return factors_gpu[1][:, ri, :], factors_gpu[2][:, ri, :]
    elif fi == 1:
        return factors_gpu[0][:, ri, :], factors_gpu[2][:, ri, :]
    else:
        return factors_gpu[0][:, ri, :], factors_gpu[1][:, ri, :]


def _get_slice(R: torch.Tensor, fi: int, di: int) -> torch.Tensor:
    """Extract the (N, DIM, DIM) slice of R at mode ``fi`` index ``di``."""
    if fi == 0:
        return R[:, di, :, :]
    elif fi == 1:
        return R[:, :, di, :]
    else:
        return R[:, :, :, di]


def _update_slice(R: torch.Tensor, fi: int, di: int, mask: torch.Tensor, delta: torch.Tensor):
    """In-place add ``delta`` to the slice of R at mode ``fi`` index ``di`` for masked rows."""
    if fi == 0:
        R[mask, di, :, :] += delta
    elif fi == 1:
        R[mask, :, di, :] += delta
    else:
        R[mask, :, :, di] += delta
