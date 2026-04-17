"""Visualize manifold channel structure as a series of PNGs."""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import KethVaraiMachine
from data import sample_batch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/ckpt_N19_s0.pt", map_location=device, weights_only=False)
model = KethVaraiMachine(N=19, latent_dim=256, encoder_depth=8, encoder_width=972,
                         linear_heads=False).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

outdir = "manifold_plots"
os.makedirs(outdir, exist_ok=True)

N_SAMPLES = 8192
A, B, C = sample_batch(N_SAMPLES, device)

with torch.no_grad():
    x = torch.cat([A, B], dim=1)
    latent = model._encode(x).float()
    U = model.head_U(latent).view(-1, 19, 9)
    V = model.head_V(latent).view(-1, 19, 9)
    W = model.head_W(latent).view(-1, 9, 19)
    C_hat = model(A, B)
    
    p = (A.unsqueeze(1) * U).sum(dim=2)  # (batch, 19)
    q = (B.unsqueeze(1) * V).sum(dim=2)
    m = p * q  # channel activations (batch, 19)
    
    err = (C_hat - C).view(-1, 9).norm(dim=1) / C.view(-1, 9).norm(dim=1).clamp(min=1e-12)

m_cpu = m.cpu().numpy()
err_cpu = err.cpu().numpy()
U_cpu = U.cpu().numpy()
V_cpu = V.cpu().numpy()
W_cpu = W.cpu().numpy()

# Sort channels by importance
ch_importance = np.abs(m_cpu).mean(axis=0)
ch_order = np.argsort(-ch_importance)

# ============================================================
# 1. Channel activation heatmap — sorted by importance
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
# Sort samples by error for structure
sample_order = np.argsort(err_cpu)
im = ax.imshow(m_cpu[sample_order][:, ch_order].T, aspect='auto', cmap='RdBu_r',
               vmin=-np.percentile(np.abs(m_cpu), 99), vmax=np.percentile(np.abs(m_cpu), 99))
ax.set_ylabel("Channel (sorted by importance)")
ax.set_xlabel("Samples (sorted by error, low→high)")
ax.set_yticks(range(19))
ax.set_yticklabels([f"ch{ch_order[i]}" for i in range(19)])
plt.colorbar(im, ax=ax, label="Channel activation m_k = (U_k·A)(V_k·B)")
ax.set_title("Channel Activations Across Input Space")
plt.tight_layout()
plt.savefig(f"{outdir}/01_channel_activations.png", dpi=150)
plt.close()
print("1/8 done")

# ============================================================
# 2. Channel correlation matrix
# ============================================================
m_n = (m_cpu - m_cpu.mean(axis=0)) / (m_cpu.std(axis=0) + 1e-12)
corr = (m_n.T @ m_n) / N_SAMPLES
# Reorder by importance
corr_sorted = corr[ch_order][:, ch_order]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_sorted, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(19))
ax.set_yticks(range(19))
ax.set_xticklabels([f"ch{ch_order[i]}" for i in range(19)], rotation=45, ha='right', fontsize=8)
ax.set_yticklabels([f"ch{ch_order[i]}" for i in range(19)], fontsize=8)
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Channel Co-activation Correlation")
plt.tight_layout()
plt.savefig(f"{outdir}/02_channel_correlation.png", dpi=150)
plt.close()
print("2/8 done")

# ============================================================
# 3. Per-channel weight matrices U_k (3x3 tiles) — mean + top variation modes
# ============================================================
fig, axes = plt.subplots(4, 19, figsize=(28, 6))
U_mean = U_cpu.mean(axis=0)  # (19, 9)
U_centered = U_cpu - U_mean
U_flat = U_centered.reshape(N_SAMPLES, -1)  # (batch, 171)
Uu, Us, Uvh = np.linalg.svd(U_flat, full_matrices=False)

row_labels = ["Mean U_k", "U mode 1", "U mode 2", "U mode 3"]
for k_idx in range(19):
    k = ch_order[k_idx]
    # Mean
    axes[0, k_idx].imshow(U_mean[k].reshape(3,3) if k < 19 else np.zeros((3,3)), 
                          cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, k_idx].set_title(f"ch{k}", fontsize=7)
    axes[0, k_idx].set_xticks([]); axes[0, k_idx].set_yticks([])

# Top 3 variation modes of U, shown per-channel
for mode in range(3):
    mode_vec = Uvh[mode].reshape(19, 9)  # how this mode affects each channel
    mode_vec *= Us[mode] * 0.1  # scale for visibility
    for k_idx in range(19):
        k = ch_order[k_idx]
        axes[mode+1, k_idx].imshow(mode_vec[k].reshape(3,3), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[mode+1, k_idx].set_xticks([]); axes[mode+1, k_idx].set_yticks([])

for i, lbl in enumerate(row_labels):
    axes[i, 0].set_ylabel(lbl, fontsize=8)
fig.suptitle("U weight matrices per channel (sorted by importance) + top 3 variation modes", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/03_U_channel_weights.png", dpi=150)
plt.close()
print("3/8 done")

# ============================================================
# 4. Same for V
# ============================================================
fig, axes = plt.subplots(4, 19, figsize=(28, 6))
V_mean = V_cpu.mean(axis=0)
V_centered = V_cpu - V_mean
V_flat = V_centered.reshape(N_SAMPLES, -1)
Vu, Vs, Vvh = np.linalg.svd(V_flat, full_matrices=False)

row_labels = ["Mean V_k", "V mode 1", "V mode 2", "V mode 3"]
for k_idx in range(19):
    k = ch_order[k_idx]
    axes[0, k_idx].imshow(V_mean[k].reshape(3,3), cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, k_idx].set_title(f"ch{k}", fontsize=7)
    axes[0, k_idx].set_xticks([]); axes[0, k_idx].set_yticks([])

for mode in range(3):
    mode_vec = Vvh[mode].reshape(19, 9) * Vs[mode] * 0.1
    for k_idx in range(19):
        k = ch_order[k_idx]
        axes[mode+1, k_idx].imshow(mode_vec[k].reshape(3,3), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[mode+1, k_idx].set_xticks([]); axes[mode+1, k_idx].set_yticks([])

for i, lbl in enumerate(row_labels):
    axes[i, 0].set_ylabel(lbl, fontsize=8)
fig.suptitle("V weight matrices per channel (sorted by importance) + top 3 variation modes", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/04_V_channel_weights.png", dpi=150)
plt.close()
print("4/8 done")

# ============================================================
# 5. W decoder weights — mean per channel
# ============================================================
fig, axes = plt.subplots(2, 19, figsize=(28, 4))
W_mean = W_cpu.mean(axis=0)  # (9, 19)
W_centered = W_cpu - W_mean
W_flat = W_centered.reshape(N_SAMPLES, -1)
Wu, Ws, Wvh = np.linalg.svd(W_flat, full_matrices=False)

wmax = np.abs(W_mean).max()
for k_idx in range(19):
    k = ch_order[k_idx]
    axes[0, k_idx].imshow(W_mean[:, k].reshape(3,3), cmap='RdBu_r', vmin=-wmax, vmax=wmax)
    axes[0, k_idx].set_title(f"ch{k}", fontsize=7)
    axes[0, k_idx].set_xticks([]); axes[0, k_idx].set_yticks([])

mode_vec = Wvh[0].reshape(9, 19) * Ws[0] * 0.1
for k_idx in range(19):
    k = ch_order[k_idx]
    axes[1, k_idx].imshow(mode_vec[:, k].reshape(3,3), cmap='RdBu_r', vmin=-wmax, vmax=wmax)
    axes[1, k_idx].set_xticks([]); axes[1, k_idx].set_yticks([])

axes[0, 0].set_ylabel("Mean W_k", fontsize=8)
axes[1, 0].set_ylabel("W mode 1", fontsize=8)
fig.suptitle("W decoder weights per channel + top variation mode", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/05_W_channel_weights.png", dpi=150)
plt.close()
print("5/8 done")

# ============================================================
# 6. Effective rank distribution + error vs rank
# ============================================================
m_abs = np.abs(m_cpu)
m_sorted = np.sort(m_abs, axis=1)[:, ::-1]
m_cumfrac = np.cumsum(m_sorted, axis=1) / (m_sorted.sum(axis=1, keepdims=True) + 1e-12)
eff_rank_90 = (m_cumfrac < 0.9).sum(axis=1) + 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(eff_rank_90, bins=np.arange(0.5, 20.5, 1), color='steelblue', edgecolor='white')
ax1.set_xlabel("Effective Rank (90% energy)")
ax1.set_ylabel("Count")
ax1.set_title("Effective Rank Distribution")
ax1.axvline(np.mean(eff_rank_90), color='red', ls='--', label=f"mean={np.mean(eff_rank_90):.1f}")
ax1.legend()

# Error vs effective rank
ranks = np.arange(int(eff_rank_90.min()), int(eff_rank_90.max())+1)
for r in ranks:
    mask = eff_rank_90 == r
    if mask.sum() > 10:
        ax2.scatter(r, np.mean(err_cpu[mask]), s=60, c='steelblue', zorder=5)
        ax2.errorbar(r, np.mean(err_cpu[mask]), yerr=np.std(err_cpu[mask]), c='steelblue', capsize=3)
ax2.set_xlabel("Effective Rank (90% energy)")
ax2.set_ylabel("Relative Frobenius Error")
ax2.set_title("Error vs Effective Rank")
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(f"{outdir}/06_effective_rank.png", dpi=150)
plt.close()
print("6/8 done")

# ============================================================
# 7. Latent PCA colored by different channel activations
# ============================================================
lat_cpu = latent.cpu().numpy()
lat_centered = lat_cpu - lat_cpu.mean(axis=0)
Ul, Sl, Vhl = np.linalg.svd(lat_centered, full_matrices=False)
lat_pcs = Ul[:, :3] * Sl[:3]  # top 3 PCs

# 4 panels: colored by ch3, ch4 (top 2), ch8, ch11 (most input-dependent W)
highlight_channels = [ch_order[0], ch_order[1], ch_order[2], ch_order[3]]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, ch in enumerate(highlight_channels):
    ax = axes[idx // 2, idx % 2]
    color_val = m_cpu[:, ch]
    vmax = np.percentile(np.abs(color_val), 95)
    sc = ax.scatter(lat_pcs[:, 0], lat_pcs[:, 1], c=color_val, cmap='RdBu_r',
                    s=1, alpha=0.3, vmin=-vmax, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=ax, label=f"m_{{ch{ch}}}")
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.set_title(f"Channel {ch} activation (importance rank {idx+1})")
fig.suptitle("Latent Space PC1 vs PC2, colored by channel activations", fontsize=12)
plt.tight_layout()
plt.savefig(f"{outdir}/07_latent_pca_channels.png", dpi=150)
plt.close()
print("7/8 done")

# ============================================================
# 8. Per-channel collapse ratio + the bilinear form as 3x3 tiles
#    showing how each (U_k, V_k) pair encodes a rank-1 bilinear map
# ============================================================
fig = plt.figure(figsize=(24, 10))
gs = gridspec.GridSpec(3, 19, hspace=0.4, wspace=0.3)

# Row 0: outer product U_k^T V_k (mean) — the "bilinear form" each channel computes
# Row 1: collapse ratio bar
# Row 2: channel activation distribution

collapse_U = []
collapse_V = []
collapse_W = []
for k in range(19):
    cu = (U_cpu[:, k, :].std(axis=0).mean() / (np.abs(U_cpu[:, k, :]).mean() + 1e-12))
    cv = (V_cpu[:, k, :].std(axis=0).mean() / (np.abs(V_cpu[:, k, :]).mean() + 1e-12))
    cw = (W_cpu[:, :, k].std(axis=0).mean() / (np.abs(W_cpu[:, :, k]).mean() + 1e-12))
    collapse_U.append(cu)
    collapse_V.append(cv)
    collapse_W.append(cw)

for k_idx in range(19):
    k = ch_order[k_idx]
    
    # Row 0: outer product of mean U_k and V_k → 9x9? No, 3x3 × 3x3 outer = 9×9
    # Actually the bilinear form is: a^T (U_k^T ⊗ V_k^T) b reshaped
    # But simpler: just show U_k as 3×3 and V_k as 3×3 side by side? 
    # Even better: the "effective bilinear" M_k[i,j, s,t] = U_k[row(i,s)] * V_k[row(t,j)] * W_k[row(i,j)]
    # Just show U_k, V_k as 3x3 with W weight annotation
    
    ax = fig.add_subplot(gs[0, k_idx])
    # Show outer product of mean u_k and v_k as a proxy for the bilinear structure
    uk = U_mean[k].reshape(3, 3)  # rows of A that channel k reads
    vk = V_mean[k].reshape(3, 3)  # rows of B that channel k reads
    # The rank-1 contribution: W[:,k] ⊗ (U[k,:] ⊗ V[k,:]) 
    # Visualize as: for output C[i,j], contribution = W[3i+j, k] * sum_s U[k, 3i+s] * V[k, 3s+j]... 
    # Simpler: just show U_k as a heatmap
    bilinear_form = np.outer(uk.flatten(), vk.flatten()).reshape(9, 9)
    ax.imshow(bilinear_form, cmap='RdBu_r', vmin=-np.percentile(np.abs(bilinear_form), 95),
              vmax=np.percentile(np.abs(bilinear_form), 95))
    ax.set_title(f"ch{k}\n|m|={ch_importance[k]:.2f}", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    if k_idx == 0:
        ax.set_ylabel("U⊗V\nbilinear", fontsize=7)

    # Row 1: collapse bars
    ax2 = fig.add_subplot(gs[1, k_idx])
    bars = ax2.bar([0, 1, 2], [collapse_U[k], collapse_V[k], collapse_W[k]], 
                   color=['#4C72B0', '#DD8452', '#55A868'], width=0.8)
    ax2.set_ylim(0, max(max(collapse_W), 1.2))
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['U', 'V', 'W'], fontsize=6)
    ax2.tick_params(axis='y', labelsize=5)
    if k_idx == 0:
        ax2.set_ylabel("Collapse\nratio", fontsize=7)

    # Row 2: activation distribution
    ax3 = fig.add_subplot(gs[2, k_idx])
    ax3.hist(m_cpu[:, k], bins=50, color='steelblue', edgecolor='none', alpha=0.8)
    ax3.axvline(0, color='red', lw=0.5, alpha=0.5)
    ax3.set_xticks([]); ax3.set_yticks([])
    if k_idx == 0:
        ax3.set_ylabel("Activation\ndist", fontsize=7)

fig.suptitle(f"Channel Manifold Structure — N=19, step {ckpt['step']}, best={ckpt['best_rel_err']:.3e}", fontsize=13)
plt.savefig(f"{outdir}/08_channel_manifold_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("8/8 done")

print(f"\nAll plots saved to {outdir}/")
