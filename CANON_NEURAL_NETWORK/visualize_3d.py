"""
visualize_3d.py — Interactive 3D latent manifold visualization.

Opens in browser. Color by condition number, reconstruction error, etc.
Rotate/zoom with mouse.

Usage:
    python visualize_3d.py
    python visualize_3d.py --method umap
"""

import argparse, os, glob
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model import KethVaraiMachine
from data import sample_batch


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def run(ckpt_path, n_samples=20000, method="both", device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step, best_err = ckpt["step"], ckpt["best_rel_err"]
    state = ckpt["model"]

    head_U_w = state["head_U.2.weight"]
    N = head_U_w.shape[0] // 9
    latent_dim = state["head_U.0.weight"].shape[1]
    encoder_width = state["encoder.0.weight"].shape[0]
    block_keys = [k for k in state if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight")]
    encoder_depth = len(block_keys)
    attn_keys = [k for k in state if "attn.in_proj_weight" in k]
    n_attn = len(attn_keys)
    attn_every = encoder_depth // max(n_attn, 1) if n_attn else 0

    print(f"Step={step}, best={best_err:.4e}, N={N}, latent={latent_dim}")

    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
        encoder_width=encoder_width, n_heads=8, attn_every=attn_every,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    all_latent, all_A, all_B, all_err = [], [], [], []
    bs = min(n_samples, 4096)
    with torch.no_grad():
        for start in range(0, n_samples, bs):
            b = min(bs, n_samples - start)
            A, B, C = sample_batch(b, device)
            latent = model._encode(torch.cat([A, B], dim=1)).float()
            C_hat = model(A, B)
            diff = (C_hat.float() - C.float()).view(b, 9)
            tn = C.float().view(b, 9).norm(dim=1).clamp(min=1e-12)
            all_latent.append(latent.cpu().numpy())
            all_A.append(A.cpu().numpy())
            all_B.append(B.cpu().numpy())
            all_err.append((diff.norm(dim=1) / tn).cpu().numpy())

    latent_np = np.concatenate(all_latent)
    A_mat = np.concatenate(all_A).reshape(-1, 3, 3)
    B_mat = np.concatenate(all_B).reshape(-1, 3, 3)
    err_np = np.concatenate(all_err)

    A_svd = np.linalg.svd(A_mat, compute_uv=False)
    B_svd = np.linalg.svd(B_mat, compute_uv=False)
    C_mat = A_mat @ B_mat
    C_svd = np.linalg.svd(C_mat, compute_uv=False)
    BA = B_mat @ A_mat
    C_frob = np.linalg.norm(C_mat.reshape(-1, 9), axis=1)
    comm = np.linalg.norm((C_mat - BA).reshape(-1, 9), axis=1) / np.maximum(C_frob, 1e-12)

    props = {
        "log₁₀(cond A)": np.log10(A_svd[:, 0] / np.maximum(A_svd[:, -1], 1e-12)),
        "log₁₀(cond C)": np.log10(C_svd[:, 0] / np.maximum(C_svd[:, -1], 1e-12)),
        "recon error": err_np,
        "commutator": comm,
        "det(A)": np.linalg.det(A_mat),
    }

    scaler = StandardScaler()
    latent_sc = scaler.fit_transform(latent_np)

    coords = {}

    if method in ("pca", "both"):
        print("PCA...")
        pca = PCA(n_components=3)
        coords["PCA"] = pca.fit_transform(latent_sc)
        print(f"  Variance: {pca.explained_variance_ratio_[:3]}")

    if method in ("umap", "both"):
        try:
            from umap import UMAP
            print("UMAP (3D)...")
            coords["UMAP"] = UMAP(n_components=3, n_neighbors=30, min_dist=0.3,
                                   random_state=42).fit_transform(latent_sc)
        except ImportError:
            print("UMAP not available, skipping")

    for proj_name, xyz in coords.items():
        for prop_name, vals in props.items():
            vmin, vmax = np.percentile(vals, [2, 98])
            v = np.clip(vals, vmin, vmax)

            # Subsample for performance
            n = min(len(v), 15000)
            idx = np.random.RandomState(0).choice(len(v), n, replace=False)

            fig = go.Figure(data=[go.Scatter3d(
                x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2],
                mode="markers",
                marker=dict(size=1.5, color=v[idx], colorscale="Viridis",
                           colorbar=dict(title=prop_name), opacity=0.5),
                text=[f"{prop_name}={v[i]:.3f}" for i in idx],
                hoverinfo="text",
            )])
            fig.update_layout(
                title=f"{proj_name} — {prop_name} (step {step}, err={best_err:.3e})",
                scene=dict(xaxis_title=f"{proj_name}1",
                          yaxis_title=f"{proj_name}2",
                          zaxis_title=f"{proj_name}3"),
                width=900, height=700,
            )
            fname = f"latent3d_{proj_name}_{prop_name.replace(' ', '_').replace('(', '').replace(')', '')}.html"
            fig.write_html(fname)
            print(f"  Saved {fname}")

    # Also make one combined figure with dropdown
    main_proj = "UMAP" if "UMAP" in coords else "PCA"
    xyz = coords[main_proj]
    n = min(len(err_np), 15000)
    idx = np.random.RandomState(0).choice(len(err_np), n, replace=False)

    fig = go.Figure()
    buttons = []
    for i, (pname, vals) in enumerate(props.items()):
        vmin, vmax = np.percentile(vals, [2, 98])
        v = np.clip(vals, vmin, vmax)
        fig.add_trace(go.Scatter3d(
            x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2],
            mode="markers", visible=(i == 0),
            marker=dict(size=1.5, color=v[idx], colorscale="Viridis",
                       colorbar=dict(title=pname), opacity=0.5),
            text=[f"{pname}={v[j]:.3f}" for j in idx],
            hoverinfo="text", name=pname,
        ))
        vis = [False] * len(props)
        vis[i] = True
        buttons.append(dict(label=pname, method="update",
                           args=[{"visible": vis}]))

    fig.update_layout(
        title=f"{main_proj} 3D — Step {step}, Best={best_err:.3e}",
        updatemenus=[dict(buttons=buttons, direction="down",
                         x=0.0, xanchor="left", y=1.15, yanchor="top")],
        scene=dict(xaxis_title=f"{main_proj}1",
                  yaxis_title=f"{main_proj}2",
                  zaxis_title=f"{main_proj}3"),
        width=1000, height=800,
    )
    combo = f"latent3d_interactive_step{step}.html"
    fig.write_html(combo)
    print(f"\n  ★ Interactive combined: {combo}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--method", choices=["pca", "umap", "both"], default="both")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ckpt = a.ckpt or find_latest_checkpoint()
    if not ckpt:
        print("No checkpoint found!"); exit(1)
    print(f"Checkpoint: {ckpt}")
    run(ckpt, a.n_samples, a.method, a.device)
