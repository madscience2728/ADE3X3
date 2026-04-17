"""Analyze checkpoint ckpt_N19_s0.pt — collapse ratios, weight structure, error distribution."""
import torch
import numpy as np
from model import KethVaraiMachine
from data import sample_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ckpt = torch.load("checkpoints/ckpt_N19_s0.pt", map_location=device, weights_only=False)
print(f"Step: {ckpt['step']}, Best rel_err: {ckpt['best_rel_err']:.4e}")

# Rebuild model
model = KethVaraiMachine(N=19, latent_dim=256, encoder_depth=8, encoder_width=972,
                         linear_heads=False).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# --- 1. Collapse ratios on large validation batch ---
print("\n=== COLLAPSE RATIOS (16k samples) ===")
with torch.no_grad():
    A, B, C = sample_batch(16384, device)
    x = torch.cat([A, B], dim=1)
    latent = model._encode(x)
    
    U = model.head_U(latent).view(-1, 19, 9)
    V = model.head_V(latent).view(-1, 19, 9)
    W = model.head_W(latent).view(-1, 9, 19)
    
    u_ratio = (U.std(dim=0).mean() / U.abs().mean().clamp(min=1e-12)).item()
    v_ratio = (V.std(dim=0).mean() / V.abs().mean().clamp(min=1e-12)).item()
    w_ratio = (W.std(dim=0).mean() / W.abs().mean().clamp(min=1e-12)).item()
    print(f"  U_ratio: {u_ratio:.4f}")
    print(f"  V_ratio: {v_ratio:.4f}")
    print(f"  W_ratio: {w_ratio:.4f}")

# --- 2. Per-channel collapse: which channels are input-dependent? ---
print("\n=== PER-CHANNEL COLLAPSE (std/mean per channel) ===")
with torch.no_grad():
    # U: (batch, 19, 9) — per channel k, compute std across batch / mean abs
    for name, mat in [("U", U), ("V", V)]:
        ch_ratios = []
        for k in range(19):
            ch = mat[:, k, :]  # (batch, 9)
            r = (ch.std(dim=0).mean() / ch.abs().mean().clamp(min=1e-12)).item()
            ch_ratios.append(r)
        sorted_ch = sorted(enumerate(ch_ratios), key=lambda x: -x[1])
        print(f"  {name} channels (most input-dependent first):")
        for idx, r in sorted_ch[:5]:
            print(f"    ch{idx:2d}: {r:.4f}")
        print(f"    ... min: ch{sorted_ch[-1][0]}: {sorted_ch[-1][1]:.4f}")

    # W: (batch, 9, 19)
    w_ch_ratios = []
    for k in range(19):
        ch = W[:, :, k]  # (batch, 9)
        r = (ch.std(dim=0).mean() / ch.abs().mean().clamp(min=1e-12)).item()
        w_ch_ratios.append(r)
    sorted_wch = sorted(enumerate(w_ch_ratios), key=lambda x: -x[1])
    print(f"  W channels (most input-dependent first):")
    for idx, r in sorted_wch[:5]:
        print(f"    ch{idx:2d}: {r:.4f}")
    print(f"    ... min: ch{sorted_wch[-1][0]}: {sorted_wch[-1][1]:.4f}")

# --- 3. Error distribution ---
print("\n=== ERROR DISTRIBUTION (16k samples) ===")
with torch.no_grad():
    C_hat = model(A, B)
    diff = (C_hat.float() - C.float()).abs().view(-1, 9)
    scale = C.float().view(-1, 9).norm(dim=1, keepdim=True).clamp(min=1e-12)
    per_entry = diff / scale  # (batch, 9)
    per_sample = diff.norm(dim=1) / scale.squeeze().clamp(min=1e-12)  # Frobenius per sample
    
    print(f"  Mean rel Frobenius: {per_sample.mean():.4e}")
    print(f"  Median:            {per_sample.median():.4e}")
    print(f"  90th percentile:   {per_sample.quantile(0.9):.4e}")
    print(f"  99th percentile:   {per_sample.quantile(0.99):.4e}")
    print(f"  Max:               {per_sample.max():.4e}")
    print(f"  Max entry error:   {per_entry.max():.4e}")
    
    print("\n  Per-element mean relative error:")
    elem_means = per_entry.mean(dim=0)
    for j in range(9):
        i_row, i_col = j // 3, j % 3
        print(f"    C[{i_row},{i_col}]: {elem_means[j]:.4e}")

# --- 4. Latent space statistics ---
print("\n=== LATENT SPACE ===")
with torch.no_grad():
    lat = latent.float()
    print(f"  Shape: {lat.shape}")
    print(f"  Mean: {lat.mean():.4f}, Std: {lat.std():.4f}")
    print(f"  Dead dims (<1e-4): {(lat.abs() < 1e-4).float().mean():.2%}")
    
    # Effective dimensionality via explained variance
    lat_centered = lat - lat.mean(dim=0)
    svd = torch.linalg.svdvals(lat_centered)
    var_explained = (svd ** 2) / (svd ** 2).sum()
    cum_var = var_explained.cumsum(0)
    eff_dim_90 = (cum_var < 0.9).sum().item() + 1
    eff_dim_99 = (cum_var < 0.99).sum().item() + 1
    print(f"  Effective dim (90% var): {eff_dim_90}")
    print(f"  Effective dim (99% var): {eff_dim_99}")
    print(f"  Top 5 singular values: {svd[:5].tolist()}")

# --- 5. Mean weight matrices (the "point solution") ---
print("\n=== MEAN WEIGHT MATRICES (Frobenius norms) ===")
with torch.no_grad():
    U_mean = U.mean(dim=0)  # (19, 9)
    V_mean = V.mean(dim=0)
    W_mean = W.mean(dim=0)  # (9, 19)
    print(f"  ||U_mean||_F = {U_mean.norm():.4f}")
    print(f"  ||V_mean||_F = {V_mean.norm():.4f}")
    print(f"  ||W_mean||_F = {W_mean.norm():.4f}")
    
    # How good is the mean point solution alone?
    # C_approx = W_mean @ diag(U_mean @ a * V_mean @ b) for each sample
    p = (A.view(-1, 1, 9) * U_mean.unsqueeze(0)).sum(dim=2)  # (batch, 19)
    q = (B.view(-1, 1, 9) * V_mean.unsqueeze(0)).sum(dim=2)  # (batch, 19)
    # Actually: p = A @ U_mean^T, q = B @ V_mean^T
    p = A @ U_mean.T  # (batch, 19)
    q = B @ V_mean.T  # (batch, 19)
    m = p * q  # (batch, 19)
    C_approx = m @ W_mean.T  # (batch, 9)
    
    mean_frob = ((C_approx - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)).mean()
    print(f"  Mean-weights rel error: {mean_frob:.4e}")
    print(f"  (vs full model:         {per_sample.mean():.4e})")
    gap = mean_frob / per_sample.mean()
    print(f"  Gap ratio (mean/full):  {gap:.2f}x")

# --- 6. Training log trajectory ---
print("\n=== TRAINING LOG (sampled) ===")
log = ckpt.get("log", [])
if log:
    for entry in log:
        s = entry["step"]
        if s <= 5000 and s % 1000 == 0:
            print(f"  step {s:6d}: rel={entry['rel_err']:.3e}  best={entry['best_rel_err']:.3e}")
        elif s <= 20000 and s % 5000 == 0:
            print(f"  step {s:6d}: rel={entry['rel_err']:.3e}  best={entry['best_rel_err']:.3e}")
        elif s % 10000 == 0:
            print(f"  step {s:6d}: rel={entry['rel_err']:.3e}  best={entry['best_rel_err']:.3e}")
    last = log[-1]
    print(f"  step {last['step']:6d}: rel={last['rel_err']:.3e}  best={last['best_rel_err']:.3e}  (final)")
