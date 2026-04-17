"""
Analyze: Is the strategy code s learning anisotropy structure?

Key question: When (A, B) has anisotropic structure (one singular value
dominates), does the strategy code s systematically encode this?

We probe:
1. Strategy code PCA -- what directions carry variance?
2. Correlation between s and input anisotropy measures (condition number,
   singular value ratios, entry magnitudes)
3. Does s predict WHICH matmul entries are hard vs easy?
4. Per-dimension mutual information (linear proxy: R^2 of s_i vs anisotropy)
5. Controlled experiment: isotropic vs rank-1 inputs -> how does s change?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from model_kolmogorov import BundleMachine
from data import sample_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load checkpoint ──────────────────────────────────────────────
ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_K_N9_s9_b0.001_seed42.pt")
ckpt = torch.load(ckpt_path, map_location=device)
print(f"Checkpoint: step={ckpt['step']}  best={ckpt['best_rel_err']:.4e}")

model = BundleMachine(N=9, strategy_dim=9, stochastic=True).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# ── Generate data ────────────────────────────────────────────────
N_SAMPLES = 50_000
torch.manual_seed(999)
A, B, C = sample_batch(N_SAMPLES, device)

with torch.no_grad():
    s, kl = model.encode(A, B)
    C_hat, _ = model(A, B)

s_np = s.cpu().numpy()
A_np = A.cpu().numpy().reshape(-1, 3, 3)
B_np = B.cpu().numpy().reshape(-1, 3, 3)
C_np = C.cpu().numpy().reshape(-1, 3, 3)
Chat_np = C_hat.cpu().numpy().reshape(-1, 3, 3)

print(f"\ns shape: {s_np.shape}")
print(f"s mean: {s_np.mean(0)}")
print(f"s std:  {s_np.std(0)}")

# ── 1. Strategy code PCA ────────────────────────────────────────
print("\n" + "="*60)
print("  1. STRATEGY CODE PCA")
print("="*60)
s_centered = s_np - s_np.mean(0)
cov = np.cov(s_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]
total_var = eigvals.sum()
for i, ev in enumerate(eigvals):
    print(f"  PC{i+1}: eigenvalue={ev:.4f}  var_explained={ev/total_var*100:.1f}%  cumulative={eigvals[:i+1].sum()/total_var*100:.1f}%")

# ── 2. Input anisotropy measures ────────────────────────────────
print("\n" + "="*60)
print("  2. CORRELATION: s vs INPUT ANISOTROPY")
print("="*60)

# Compute anisotropy features for each (A, B) pair
svA = np.linalg.svd(A_np, compute_uv=False)  # (N, 3)
svB = np.linalg.svd(B_np, compute_uv=False)

# Anisotropy measures
condA = svA[:, 0] / (svA[:, 2] + 1e-12)
condB = svB[:, 0] / (svB[:, 2] + 1e-12)
ratio_A = svA[:, 0] / (svA[:, 1] + 1e-12)  # top/second sv ratio
ratio_B = svB[:, 0] / (svB[:, 1] + 1e-12)
normA = np.linalg.norm(A_np.reshape(-1, 9), axis=1)
normB = np.linalg.norm(B_np.reshape(-1, 9), axis=1)
normC = np.linalg.norm(C_np.reshape(-1, 9), axis=1)

# det(A), det(B) -- measures "how degenerate"
detA = np.linalg.det(A_np)
detB = np.linalg.det(B_np)

features = {
    "cond(A)": condA, "cond(B)": condB,
    "sv1/sv2(A)": ratio_A, "sv1/sv2(B)": ratio_B,
    "||A||": normA, "||B||": normB, "||C||": normC,
    "det(A)": detA, "det(B)": detB,
    "log|det(A)|": np.log(np.abs(detA) + 1e-12),
    "log|det(B)|": np.log(np.abs(detB) + 1e-12),
}

# Linear R^2 of each s_i vs each feature
from numpy.polynomial.polynomial import polyfit

print(f"\n  {'feature':>15s}  " + "  ".join(f"s{i}" for i in range(9)) + "   best_R2  best_dim")
print("  " + "-" * 120)

for fname, fvals in features.items():
    r2s = []
    for i in range(9):
        corr = np.corrcoef(fvals, s_np[:, i])[0, 1]
        r2s.append(corr ** 2)
    best_i = np.argmax(r2s)
    r2_strs = "  ".join(f"{r:.3f}" for r in r2s)
    print(f"  {fname:>15s}  {r2_strs}   {r2s[best_i]:.3f}     s{best_i}")

# ── 3. Multi-feature R^2: can s jointly predict anisotropy? ─────
print("\n" + "="*60)
print("  3. JOINT PREDICTION: s -> anisotropy (linear regression)")
print("="*60)

from numpy.linalg import lstsq

X = np.column_stack([s_np, np.ones(len(s_np))])  # (N, 10)
for fname, fvals in features.items():
    beta, res, _, _ = lstsq(X, fvals, rcond=None)
    pred = X @ beta
    ss_res = ((fvals - pred) ** 2).sum()
    ss_tot = ((fvals - fvals.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  R^2(s -> {fname:>15s}) = {r2:.4f}")

# ── 4. Controlled experiment: isotropic vs rank-1 ───────────────
print("\n" + "="*60)
print("  4. CONTROLLED: isotropic vs near-rank-1 inputs")
print("="*60)

torch.manual_seed(777)
n_ctrl = 10_000

# Isotropic: entries ~ N(0,1)
A_iso = torch.randn(n_ctrl, 9, device=device)
B_iso = torch.randn(n_ctrl, 9, device=device)

# Near rank-1: A = u v^T + eps, dominant rank-1 structure
u = torch.randn(n_ctrl, 3, 1, device=device)
v = torch.randn(n_ctrl, 1, 3, device=device)
A_r1 = (u @ v).view(n_ctrl, 9) + 0.01 * torch.randn(n_ctrl, 9, device=device)
u2 = torch.randn(n_ctrl, 3, 1, device=device)
v2 = torch.randn(n_ctrl, 1, 3, device=device)
B_r1 = (u2 @ v2).view(n_ctrl, 9) + 0.01 * torch.randn(n_ctrl, 9, device=device)

with torch.no_grad():
    s_iso, _ = model.encode(A_iso, B_iso)
    s_r1, _ = model.encode(A_r1, B_r1)

s_iso_np = s_iso.cpu().numpy()
s_r1_np = s_r1.cpu().numpy()

print(f"\n  Isotropic s:  mean_norm={np.linalg.norm(s_iso_np, axis=1).mean():.3f}  per-dim std={s_iso_np.std(0)}")
print(f"  Rank-1 s:     mean_norm={np.linalg.norm(s_r1_np, axis=1).mean():.3f}  per-dim std={s_r1_np.std(0)}")

# How separable are the two distributions?
iso_mean = s_iso_np.mean(0)
r1_mean = s_r1_np.mean(0)
diff = r1_mean - iso_mean
pooled_std = np.sqrt(0.5 * (s_iso_np.var(0) + s_r1_np.var(0)))
d_cohen = diff / (pooled_std + 1e-12)

print(f"\n  Mean shift (rank1 - iso): {diff}")
print(f"  Cohen's d per dimension:  {d_cohen}")
print(f"  Max |d|: s{np.argmax(np.abs(d_cohen))} = {d_cohen[np.argmax(np.abs(d_cohen))]:.3f}")

# Linear classifier separability
labels = np.concatenate([np.zeros(n_ctrl), np.ones(n_ctrl)])
all_s = np.vstack([s_iso_np, s_r1_np])
X_cls = np.column_stack([all_s, np.ones(len(all_s))])
beta_cls, _, _, _ = lstsq(X_cls, labels, rcond=None)
pred_cls = (X_cls @ beta_cls > 0.5).astype(float)
acc = (pred_cls == labels).mean()
print(f"  Linear classifier accuracy (iso vs rank1): {acc:.3f}")

# ── 5. Per-element error vs strategy code ────────────────────────
print("\n" + "="*60)
print("  5. DOES s PREDICT WHICH ENTRIES ARE HARD?")
print("="*60)

err_per_elem = np.abs(Chat_np - C_np) / (np.abs(C_np) + 1e-8)  # (N, 3, 3)
err_flat = err_per_elem.reshape(-1, 9)  # (N, 9)

print(f"\n  Mean per-element error: {err_flat.mean(0).reshape(3,3)}")
print(f"  Std per-element error:  {err_flat.std(0).reshape(3,3)}")

# Can s predict which element has worst error?
worst_elem = err_flat.argmax(axis=1)  # (N,)
# Multinomial logistic proxy: just check if s separates worst-element classes
for elem in range(9):
    mask = worst_elem == elem
    frac = mask.mean()
    if frac < 0.01:
        continue
    s_elem = s_np[mask]
    s_other = s_np[~mask]
    d = (s_elem.mean(0) - s_other.mean(0)) / (np.sqrt(0.5*(s_elem.var(0) + s_other.var(0))) + 1e-12)
    print(f"  C[{elem//3},{elem%3}] worst in {frac*100:.1f}% of samples  max|d|={np.abs(d).max():.3f} at s{np.argmax(np.abs(d))}")

# ── 6. Strategy code vs reconstruction quality ──────────────────
print("\n" + "="*60)
print("  6. DOES ||s|| PREDICT ERROR MAGNITUDE?")
print("="*60)

s_norms = np.linalg.norm(s_np, axis=1)
total_err = np.linalg.norm(err_flat, axis=1)

corr_norm_err = np.corrcoef(s_norms, total_err)[0, 1]
print(f"  corr(||s||, total_err) = {corr_norm_err:.4f}")

# R^2 of s -> total error
X_err = np.column_stack([s_np, np.ones(len(s_np))])
beta_err, _, _, _ = lstsq(X_err, total_err, rcond=None)
pred_err = X_err @ beta_err
ss_res = ((total_err - pred_err)**2).sum()
ss_tot = ((total_err - total_err.mean())**2).sum()
r2_err = 1 - ss_res / ss_tot
print(f"  R^2(s -> total_err) = {r2_err:.4f}")

# Quartile analysis
quartiles = np.percentile(s_norms, [25, 50, 75])
for lo, hi, label in [(0, quartiles[0], "Q1 (small ||s||)"),
                       (quartiles[1], quartiles[2], "Q2-Q3 (mid)"),
                       (quartiles[2], s_norms.max()+1, "Q4 (large ||s||)")]:
    mask = (s_norms >= lo) & (s_norms < hi)
    print(f"  {label}: mean_err={total_err[mask].mean():.4f}  std_err={total_err[mask].std():.4f}  n={mask.sum()}")

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
