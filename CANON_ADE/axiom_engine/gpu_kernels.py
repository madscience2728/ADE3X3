"""
gpu_kernels.py — GPU-accelerated batch evaluation of axioms via PyTorch/CUDA.
Falls back to CPU if CUDA unavailable.
"""
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def batch_step51_gpu(alphas, betas, device=DEVICE):
    """
    Compute Step-51 coords for a batch of candidates.
    alphas: (B, R, 3, 3) numpy
    betas:  (B, R, 3, 3) numpy
    Returns: Sigma (B,R,9), H (B,R,18), Delta (B,R,54) as numpy
    """
    a = torch.as_tensor(alphas, dtype=torch.float64, device=device)
    b = torch.as_tensor(betas, dtype=torch.float64, device=device)
    B, R = a.shape[:2]
    
    # Sigma = sum_s a[r,s]*b[s,u] → einsum over s
    Sigma = torch.einsum('bkrs,bksu->bkru', a, b).reshape(B, R, 9)
    
    # Eta1 = a[:,r,0]*b[:,0,u] - a[:,r,1]*b[:,1,u]
    Eta1 = (a[..., 0:1] * b[..., 0:1, :] - a[..., 1:2] * b[..., 1:2, :]).reshape(B, R, 9)
    Eta2 = (a[..., 1:2] * b[..., 1:2, :] - a[..., 2:3] * b[..., 2:3, :]).reshape(B, R, 9)
    H = torch.cat([Eta1, Eta2], dim=-1)
    
    # Delta: s≠t off-diagonal
    dead_pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    parts = []
    for s, t in dead_pairs:
        parts.append((a[..., s:s+1] * b[..., t:t+1, :]).reshape(B, R, 9))
    Delta = torch.cat(parts, dim=-1)
    
    return Sigma.cpu().numpy(), H.cpu().numpy(), Delta.cpu().numpy()


def batch_rank_H_gpu(H_batch, device=DEVICE):
    """
    Compute rank(H) for a batch of H matrices.
    H_batch: (B, R, 18) numpy
    Returns: ranks (B,) numpy int
    """
    H = torch.as_tensor(H_batch, dtype=torch.float64, device=device)
    B = H.shape[0]
    
    # batched SVD
    _, S, _ = torch.linalg.svd(H, full_matrices=False)
    
    # Rank per sample
    tol = 1e-10
    S_max = S[:, 0:1]
    thresholds = tol * S_max.clamp(min=1e-30)
    ranks = (S > thresholds).sum(dim=1)
    
    return ranks.cpu().numpy()


def batch_relation_module_gpu(alphas, betas, gammas, device=DEVICE):
    """
    Compute relation module dimensions for a batch.
    Returns: dim_KA, dim_KB, dim_KC for each sample.
    """
    B, R = alphas.shape[:2]
    
    A = torch.as_tensor(alphas.reshape(B, R, 9), dtype=torch.float64, device=device)
    Bt = torch.as_tensor(betas.reshape(B, R, 9), dtype=torch.float64, device=device)
    C = torch.as_tensor(gammas.reshape(B, R, 9), dtype=torch.float64, device=device)
    
    def batch_null_dim(M):
        # M: (B, R, 9) → transpose to (B, 9, R), SVD
        Mt = M.transpose(1, 2)  # (B, 9, R)
        _, S, _ = torch.linalg.svd(Mt, full_matrices=False)
        tol = 1e-10
        S_max = S[:, 0:1].clamp(min=1e-30)
        ranks = (S > tol * S_max).sum(dim=1)
        return R - ranks  # null space dim = R - rank
    
    dim_KA = batch_null_dim(A).cpu().numpy()
    dim_KB = batch_null_dim(Bt).cpu().numpy()
    dim_KC = batch_null_dim(C).cpu().numpy()
    
    return dim_KA, dim_KB, dim_KC


def batch_fitness_gpu(alphas, betas, gammas, T_target, device=DEVICE):
    """
    Compute reconstruction fitness ||T_recon - T_target||_max for a batch.
    alphas, betas, gammas: (B, R, 3, 3) numpy
    T_target: (9,9,9) numpy
    Returns: fitness (B,) numpy
    """
    a = torch.as_tensor(alphas.reshape(-1, alphas.shape[1], 9), dtype=torch.float64, device=device)
    b = torch.as_tensor(betas.reshape(-1, betas.shape[1], 9), dtype=torch.float64, device=device)
    c = torch.as_tensor(gammas.reshape(-1, gammas.shape[1], 9), dtype=torch.float64, device=device)
    T = torch.as_tensor(T_target.ravel(), dtype=torch.float64, device=device)
    
    B, R, _ = a.shape
    
    # T_recon[b] = sum_k outer(a[b,k], outer(b[b,k], c[b,k]))
    # = einsum('bki,bkj,bkl->bijl') reshaped
    T_recon = torch.einsum('bki,bkj,bkl->bijl', a, b, c).reshape(B, 729)
    fitness = (T_recon - T.unsqueeze(0)).abs().max(dim=1).values
    
    return fitness.cpu().numpy()


def batch_gram_eigenvalues_gpu(factors, device=DEVICE):
    """
    Compute Gram matrix eigenvalues for a batch of factor matrices.
    factors: (B, R, 9)
    Returns: eigenvalues (B, 9) sorted descending
    """
    F = torch.as_tensor(factors, dtype=torch.float64, device=device)
    G = F.transpose(1, 2) @ F  # (B, 9, 9)
    evals = torch.linalg.eigvalsh(G)  # (B, 9), ascending
    return evals.flip(dims=[1]).cpu().numpy()  # descending
