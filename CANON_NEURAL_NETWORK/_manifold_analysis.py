"""Manifold structure analysis — what patterns is the encoder learning in U,V,W?"""
import torch
import numpy as np
from model import KethVaraiMachine
from data import sample_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("checkpoints/ckpt_N19_s0.pt", map_location=device, weights_only=False)
model = KethVaraiMachine(N=19, latent_dim=256, encoder_depth=8, encoder_width=972,
                         linear_heads=False).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Step {ckpt['step']}, best {ckpt['best_rel_err']:.4e}\n")

N_SAMPLES = 16384
A, B, C = sample_batch(N_SAMPLES, device)

with torch.no_grad():
    x = torch.cat([A, B], dim=1)
    latent = model._encode(x).float()
    U = model.head_U(latent).view(-1, 19, 9)  # (batch, 19, 9)
    V = model.head_V(latent).view(-1, 19, 9)
    W = model.head_W(latent).view(-1, 9, 19)  # (batch, 9, 19)
    C_hat = model(A, B)

# ============================================================
# 1. PCA of the weight variations — what directions in U,V,W space
#    does the encoder actually use?
# ============================================================
print("=== WEIGHT VARIATION PCA ===")
for name, mat in [("U", U.view(-1, 19*9)), ("V", V.view(-1, 19*9)), ("W", W.view(-1, 9*19))]:
    centered = mat - mat.mean(dim=0)
    svd = torch.linalg.svdvals(centered)
    var = svd ** 2
    cumvar = var.cumsum(0) / var.sum()
    dim90 = (cumvar < 0.9).sum().item() + 1
    dim99 = (cumvar < 0.99).sum().item() + 1
    top5_pct = (var[:5].sum() / var.sum()).item() * 100
    print(f"  {name}: eff_dim(90%)={dim90}, eff_dim(99%)={dim99}, top5={top5_pct:.1f}%")
    print(f"       sv spectrum: {svd[:8].tolist()}")

# ============================================================
# 2. What input features drive the weight variation?
#    Correlate latent PCA directions with input properties.
# ============================================================
print("\n=== INPUT FEATURES vs LATENT DIRECTIONS ===")
lat_centered = latent - latent.mean(dim=0)
U_lat, S_lat, Vh_lat = torch.linalg.svd(lat_centered, full_matrices=False)
lat_pcs = U_lat[:, :10] * S_lat[:10]  # top 10 latent PCs, (batch, 10)

# Input features: det(A), det(B), det(C), tr(A), tr(B), ||A||, ||B||,
# A·B frobenius inner product, condition-related
A3 = A.view(-1, 3, 3)
B3 = B.view(-1, 3, 3)
C3 = C.view(-1, 3, 3)

detA = torch.linalg.det(A3)
detB = torch.linalg.det(B3)
detC = torch.linalg.det(C3)
trA = A3.diagonal(dim1=1, dim2=2).sum(dim=1)
trB = B3.diagonal(dim1=1, dim2=2).sum(dim=1)
normA = A.norm(dim=1)
normB = B.norm(dim=1)
normC = C.norm(dim=1)
# A^T B Frobenius inner product
AB_inner = (A * B).sum(dim=1)
# Symmetry: ||A - A^T|| / ||A||
symA = (A3 - A3.transpose(1,2)).view(-1,9).norm(dim=1) / normA.clamp(min=1e-12)
symB = (B3 - B3.transpose(1,2)).view(-1,9).norm(dim=1) / normB.clamp(min=1e-12)

features = torch.stack([detA, detB, detC, trA, trB, normA, normB, normC, 
                        AB_inner, symA, symB], dim=1)
feat_names = ["det(A)", "det(B)", "det(C)", "tr(A)", "tr(B)", 
              "||A||", "||B||", "||C||", "A·B", "asym(A)", "asym(B)"]

# Correlation matrix: features vs latent PCs
features_n = (features - features.mean(dim=0)) / features.std(dim=0).clamp(min=1e-12)
lat_pcs_n = (lat_pcs - lat_pcs.mean(dim=0)) / lat_pcs.std(dim=0).clamp(min=1e-12)
corr = (features_n.T @ lat_pcs_n) / N_SAMPLES  # (11, 10)

print("  Strongest correlations (|r| > 0.1):")
corr_np = corr.cpu().numpy()
for i, fn in enumerate(feat_names):
    for j in range(10):
        if abs(corr_np[i, j]) > 0.1:
            print(f"    {fn:8s} ↔ PC{j}: r={corr_np[i,j]:+.3f}")

# ============================================================
# 3. Channel activity patterns — which channels activate together?
# ============================================================
print("\n=== CHANNEL CO-ACTIVATION ===")
with torch.no_grad():
    p = (A.unsqueeze(1) * U).sum(dim=2)  # (batch, 19)
    q = (B.unsqueeze(1) * V).sum(dim=2)  # (batch, 19)
    m = p * q  # (batch, 19) — channel activations
    
    # Channel norms
    ch_norms = m.abs().mean(dim=0)
    sorted_ch = ch_norms.argsort(descending=True)
    print("  Channel importance (mean |m_k|):")
    for k in sorted_ch:
        print(f"    ch{k.item():2d}: {ch_norms[k].item():.4f}")
    
    # Correlation between channel activations
    m_n = (m - m.mean(dim=0)) / m.std(dim=0).clamp(min=1e-12)
    ch_corr = (m_n.T @ m_n) / N_SAMPLES
    
    # Find strongest off-diagonal correlations
    ch_corr_np = ch_corr.cpu().numpy()
    np.fill_diagonal(ch_corr_np, 0)
    pairs = []
    for i in range(19):
        for j in range(i+1, 19):
            if abs(ch_corr_np[i,j]) > 0.15:
                pairs.append((i, j, ch_corr_np[i,j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    print(f"\n  Correlated channel pairs (|r|>0.15): {len(pairs)} found")
    for i, j, r in pairs[:15]:
        print(f"    ch{i:2d} ↔ ch{j:2d}: r={r:+.3f}")

# ============================================================
# 4. Error vs input properties — where does the manifold fail?
# ============================================================
print("\n=== ERROR vs INPUT PROPERTIES ===")
with torch.no_grad():
    err = (C_hat - C).view(-1, 9).norm(dim=1) / C.view(-1, 9).norm(dim=1).clamp(min=1e-12)
    
    # Bin by each feature and show mean error
    for i, fn in enumerate(feat_names):
        feat = features[:, i]
        # Quintile analysis
        quantiles = torch.quantile(feat, torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device))
        errs_by_q = []
        for q_idx in range(5):
            mask = (feat >= quantiles[q_idx]) & (feat <= quantiles[q_idx+1])
            if mask.sum() > 0:
                errs_by_q.append(err[mask].mean().item())
            else:
                errs_by_q.append(float('nan'))
        # Only print if there's significant variation
        if max(errs_by_q) / (min(errs_by_q) + 1e-15) > 1.5:
            print(f"  {fn:8s} quintile errors: {' '.join(f'{e:.3e}' for e in errs_by_q)}  (ratio {max(errs_by_q)/min(errs_by_q):.1f}x)")

# ============================================================
# 5. Rank structure of the bilinear decomposition per sample
# ============================================================
print("\n=== EFFECTIVE RANK PER SAMPLE ===")
with torch.no_grad():
    # For each sample, the contribution matrix is outer(p, q) weighted by W
    # Effective rank = how many channels contribute meaningfully
    m_abs = m.abs()  # (batch, 19)
    m_sorted = m_abs.sort(dim=1, descending=True).values
    m_cumfrac = m_sorted.cumsum(dim=1) / m_sorted.sum(dim=1, keepdim=True).clamp(min=1e-12)
    
    eff_rank_90 = (m_cumfrac < 0.9).sum(dim=1).float() + 1  # per sample
    eff_rank_99 = (m_cumfrac < 0.99).sum(dim=1).float() + 1
    
    print(f"  Effective rank (90% energy): mean={eff_rank_90.mean():.1f}, std={eff_rank_90.std():.1f}, min={eff_rank_90.min():.0f}, max={eff_rank_90.max():.0f}")
    print(f"  Effective rank (99% energy): mean={eff_rank_99.mean():.1f}, std={eff_rank_99.std():.1f}, min={eff_rank_99.min():.0f}, max={eff_rank_99.max():.0f}")
    
    # Histogram of effective rank
    for r in range(1, 20):
        count_90 = (eff_rank_90 == r).sum().item()
        count_99 = (eff_rank_99 == r).sum().item()
        if count_90 > 0 or count_99 > 0:
            bar90 = '#' * (count_90 * 60 // N_SAMPLES)
            print(f"    rank={r:2d}: 90%: {count_90:5d} {bar90}")

# ============================================================
# 6. Symmetry: is the encoder exploiting A↔B transpose symmetry?
# ============================================================
print("\n=== A↔B SYMMETRY TEST ===")
with torch.no_grad():
    # If C = A@B, then C^T = B^T @ A^T
    # Feed (B^T, A^T) — does the encoder produce consistent U↔V swap?
    At = A3.transpose(1, 2).reshape(-1, 9)
    Bt = B3.transpose(1, 2).reshape(-1, 9)
    C_hat_swap = model(Bt, At)  # should give C^T
    Ct = C3.transpose(1, 2).reshape(-1, 9)
    
    err_orig = (C_hat - C).view(-1,9).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    err_swap = (C_hat_swap - Ct).view(-1,9).norm(dim=1) / Ct.norm(dim=1).clamp(min=1e-12)
    
    print(f"  Original (A,B)→C      mean err: {err_orig.mean():.4e}")
    print(f"  Swapped  (B^T,A^T)→C^T mean err: {err_swap.mean():.4e}")
    print(f"  Ratio: {err_swap.mean() / err_orig.mean():.2f}x")
