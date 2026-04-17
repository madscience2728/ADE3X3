"""
compare_N1_N11.py -- Side-by-side manifold comparison of N=1 vs N=11 checkpoints.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from scipy import stats
from model import KethVaraiMachine
from data import sample_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50_000
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def load_model(N, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = KethVaraiMachine(
        N=N, latent_dim=128, encoder_depth=2, encoder_width=192,
        n_heads=4, attn_every=4, expanded_products=True,
        head_depth=1, head_width=256,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    best = ckpt.get("best_rel_err", "?")
    return model, step, best


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


@torch.no_grad()
def analyze_model(label, model, N, A, B, C):
    """Return a dict of manifold properties."""
    batch = A.shape[0]
    latent = model._encode(torch.cat([A, B], dim=1))
    U = model.head_U(latent).view(batch, N, 9)
    V = model.head_V(latent).view(batch, N, 9)
    W = model.head_W(latent).view(batch, 9, N)
    C_hat = model(A, B)

    # Norms and variation
    U_mean = U.mean(dim=0)
    V_mean = V.mean(dim=0)
    W_mean = W.mean(dim=0)
    U_cv = (U.std(dim=0) / U_mean.abs().clamp(min=1e-10)).mean().item()
    V_cv = (V.std(dim=0) / V_mean.abs().clamp(min=1e-10)).mean().item()
    W_cv = (W.std(dim=0) / W_mean.abs().clamp(min=1e-10)).mean().item()
    U_var_ratio = U.std(dim=0).norm().item() / U_mean.norm().item()
    V_var_ratio = V.std(dim=0).norm().item() / V_mean.norm().item()
    W_var_ratio = W.std(dim=0).norm().item() / W_mean.norm().item()

    # Model error
    err = (C_hat - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    mean_err = err.mean().item()
    max_err = err.max().item()

    # Fixed decomposition error
    p_fixed = A @ U_mean.T
    q_fixed = B @ V_mean.T
    m_fixed = p_fixed * q_fixed
    C_hat_fixed = m_fixed @ W_mean.T
    err_fixed = (C_hat_fixed - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    mean_err_fixed = err_fixed.mean().item()
    input_dep_factor = mean_err_fixed / mean_err if mean_err > 0 else float('inf')

    # Tensor reconstruction
    T_learned = torch.einsum('ck,ka,kb->cab', W_mean, U_mean, V_mean)
    T_true = torch.zeros(9, 9, 9, device=A.device)
    for i in range(3):
        for j in range(3):
            for s in range(3):
                T_true[3*i+j, 3*i+s, 3*s+j] = 1.0
    tensor_err = (T_learned - T_true).norm().item() / T_true.norm().item()

    # Latent PCA
    lat_np = latent.cpu().numpy()
    lat_centered = lat_np - lat_np.mean(axis=0)
    cov = np.cov(lat_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total_var = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total_var
    dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    eff_dim = float(np.exp(-np.sum((eigvals/total_var) * np.log(eigvals/total_var + 1e-30))))

    # W-space PCA
    W_flat = W.view(batch, -1).cpu().numpy()
    W_centered = W_flat - W_flat.mean(axis=0)
    cov_W = np.cov(W_centered, rowvar=False)
    eigvals_W = np.linalg.eigvalsh(cov_W)[::-1]
    total_var_W = eigvals_W.sum()
    cumvar_W = np.cumsum(eigvals_W) / total_var_W
    w_dims_90 = int(np.searchsorted(cumvar_W, 0.90) + 1)
    w_dims_95 = int(np.searchsorted(cumvar_W, 0.95) + 1)

    # Polynomial fit of W
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    input_flat = np.concatenate([A_np, B_np], axis=1)
    products = (A_np[:, :, None] * B_np[:, None, :]).reshape(batch, 81)
    input_quad = np.concatenate([input_flat, products, np.ones((batch, 1))], axis=1)
    M, _, _, _ = np.linalg.lstsq(input_quad, W_flat, rcond=None)
    W_pred = input_quad @ M
    quad_r2 = 1 - (np.linalg.norm(W_flat - W_pred) / np.linalg.norm(W_centered))**2

    # Polynomial fit of C_hat
    C_hat_np = C_hat.cpu().numpy()
    M2, _, _, _ = np.linalg.lstsq(input_quad, C_hat_np, rcond=None)
    C_pred = input_quad @ M2
    C_centered = C_hat_np - C_hat_np.mean(axis=0)
    c_quad_r2 = 1 - (np.linalg.norm(C_hat_np - C_pred) / np.linalg.norm(C_centered))**2

    # Local curvature (quick sample)
    from sklearn.neighbors import NearestNeighbors
    n_wpc = min(w_dims_95 + 5, W_flat.shape[1])
    eigvecs_W = np.linalg.eigh(cov_W)[1][:, ::-1][:, :n_wpc]
    W_proj = W_centered @ eigvecs_W
    nn = NearestNeighbors(n_neighbors=11)
    nn.fit(W_proj[:10000])
    dists, indices = nn.kneighbors(W_proj[:500])
    curvatures = []
    local_dims = []
    for i in range(500):
        deltas = W_proj[indices[i, 1:]] - W_proj[indices[i, 0]]
        sv = np.linalg.svd(deltas, compute_uv=False)
        local_dims.append((sv / sv[0] > 0.1).sum())
        U_s, S_s, Vt_s = np.linalg.svd(deltas, full_matrices=False)
        k = min(5, len(sv))
        proj = deltas @ Vt_s[:k].T @ Vt_s[:k]
        curvatures.append(np.linalg.norm(deltas - proj) / np.linalg.norm(deltas))
    mean_curv = np.mean(curvatures)
    median_local_dim = int(np.median(local_dims))

    # Symmetry test (row swap)
    n_sym = 5000
    A_s = A[:n_sym].clone()
    B_s = B[:n_sym].clone()
    A_sw = A_s.clone()
    A_sw[:, 0:3] = A_s[:, 3:6]
    A_sw[:, 3:6] = A_s[:, 0:3]
    W_o = model.head_W(model._encode(torch.cat([A_s, B_s], dim=1))).view(-1, 9, N).cpu().numpy()
    W_s = model.head_W(model._encode(torch.cat([A_sw, B_s], dim=1))).view(-1, 9, N).cpu().numpy()
    W_exp = W_o.copy()
    W_exp[:, 0:3, :] = W_o[:, 3:6, :]
    W_exp[:, 3:6, :] = W_o[:, 0:3, :]
    sym_err = (np.linalg.norm(W_s - W_exp, axis=(1, 2)) / np.linalg.norm(W_o, axis=(1, 2))).mean()

    return {
        'mean_err': mean_err,
        'max_err': max_err,
        'mean_err_fixed': mean_err_fixed,
        'input_dep_factor': input_dep_factor,
        'tensor_err': tensor_err,
        'U_cv': U_cv, 'V_cv': V_cv, 'W_cv': W_cv,
        'U_var': U_var_ratio, 'V_var': V_var_ratio, 'W_var': W_var_ratio,
        'latent_dim_90': dims_90, 'latent_dim_95': dims_95, 'latent_eff_dim': eff_dim,
        'w_dims_90': w_dims_90, 'w_dims_95': w_dims_95,
        'quad_r2_W': quad_r2, 'quad_r2_C': c_quad_r2,
        'mean_curvature': mean_curv, 'median_local_dim': median_local_dim,
        'sym_err': sym_err,
    }


def main():
    ckpt_n1 = os.path.join(CKPT_DIR, "ckpt_N1_s42.pt")
    ckpt_n11 = os.path.join(CKPT_DIR, "ckpt_N11_s42.pt")

    if not os.path.exists(ckpt_n1):
        print(f"ERROR: {ckpt_n1} not found. Train N=1 first.")
        return
    if not os.path.exists(ckpt_n11):
        print(f"ERROR: {ckpt_n11} not found.")
        return

    model1, step1, best1 = load_model(1, ckpt_n1)
    model11, step11, best11 = load_model(11, ckpt_n11)

    print(f"N=1  checkpoint: step={step1}, best_rel_err={best1}")
    print(f"N=11 checkpoint: step={step11}, best_rel_err={best11}")

    torch.manual_seed(999)
    A, B, C = sample_batch(N_SAMPLES, torch.device(DEVICE))

    section("ANALYZING N=1")
    r1 = analyze_model("N=1", model1, 1, A, B, C)

    section("ANALYZING N=11")
    r11 = analyze_model("N=11", model11, 11, A, B, C)

    # ================================================================
    section("SIDE-BY-SIDE COMPARISON")
    # ================================================================

    def row(label, key, fmt=".6f"):
        v1 = r1[key]
        v11 = r11[key]
        ratio = v1 / v11 if v11 != 0 else float('inf')
        print(f"  {label:35s}  {v1:{fmt}}  {v11:{fmt}}  {ratio:8.2f}x")

    print(f"  {'Metric':35s}  {'N=1':>10}  {'N=11':>10}  {'Ratio':>8}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")

    row("Model rel error (mean)", "mean_err", ">10.4e")
    row("Model rel error (max)", "max_err", ">10.4e")
    row("Fixed decomp error", "mean_err_fixed", ">10.4e")
    row("Input-dependence factor", "input_dep_factor", ">10.2f")
    row("Tensor reconstruction error", "tensor_err", ">10.4f")
    print()
    row("U coefficient of variation", "U_cv", ">10.4f")
    row("V coefficient of variation", "V_cv", ">10.4f")
    row("W coefficient of variation", "W_cv", ">10.4f")
    row("U variation/mean ratio", "U_var", ">10.4f")
    row("V variation/mean ratio", "V_var", ">10.4f")
    row("W variation/mean ratio", "W_var", ">10.4f")
    print()
    row("Latent effective dim", "latent_eff_dim", ">10.1f")
    row("Latent dims @90%", "latent_dim_90", ">10d")
    row("Latent dims @95%", "latent_dim_95", ">10d")
    row("W-manifold dims @90%", "w_dims_90", ">10d")
    row("W-manifold dims @95%", "w_dims_95", ">10d")
    print()
    row("Quadratic R^2 of W", "quad_r2_W", ">10.4f")
    row("Quadratic R^2 of C_hat", "quad_r2_C", ">10.4f")
    row("Mean local curvature", "mean_curvature", ">10.4f")
    row("Median local dimension", "median_local_dim", ">10d")
    row("Row-swap symmetry error", "sym_err", ">10.4f")

    # ================================================================
    section("INTERPRETATION")
    # ================================================================

    print("Key differences:")
    if r1['input_dep_factor'] < 1.5 and r11['input_dep_factor'] > 2:
        print("  - N=1 is essentially a FIXED decomposition; N=11 is a true hypernetwork")
    elif r1['input_dep_factor'] > 2 and r11['input_dep_factor'] > 2:
        print("  - BOTH are true hypernetworks (input-dependent)")
    if r1['quad_r2_W'] > 0.5 and r11['quad_r2_W'] < 0.1:
        print("  - N=1 W is polynomial; N=11 W is deeply nonlinear")
    if r1['mean_curvature'] < 0.1 and r11['mean_curvature'] > 0.2:
        print("  - N=1 manifold is FLAT; N=11 manifold is CURVED")
    if r1['sym_err'] < 0.2 and r11['sym_err'] > 0.5:
        print("  - N=1 respects matmul symmetry; N=11 breaks it")

    print(f"\n  N=1 uses {r1['w_dims_95']} W-dims to get {r1['mean_err']:.4e} error")
    print(f"  N=11 uses {r11['w_dims_95']} W-dims to get {r11['mean_err']:.4e} error")
    ratio = r1['mean_err'] / r11['mean_err'] if r11['mean_err'] > 0 else float('inf')
    print(f"  N=11 is {ratio:.1f}x better in accuracy")

    print("\n" + "=" * 72)
    print("  COMPARISON COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
