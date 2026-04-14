"""
visualize_latent.py — Visualize the encoder's latent manifold.

Projects the 1024-dim latent space to 2D/3D via UMAP and PCA,
colored by input properties (condition number, determinant, etc.)

Usage:
    python visualize_latent.py
    python visualize_latent.py --ckpt checkpoints/ckpt_N19_s42.pt --n_samples 20000
"""

import argparse
import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model import KethVaraiMachine
from data import sample_batch


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_properties(A_mat, B_mat):
    """Returns dict of property_name -> (batch,) array."""
    batch = A_mat.shape[0]
    props = {}

    A_svd = np.linalg.svd(A_mat, compute_uv=False)
    B_svd = np.linalg.svd(B_mat, compute_uv=False)
    C_mat = A_mat @ B_mat
    C_svd = np.linalg.svd(C_mat, compute_uv=False)

    props["A_cond"] = np.log10(A_svd[:, 0] / np.maximum(A_svd[:, -1], 1e-12))
    props["B_cond"] = np.log10(B_svd[:, 0] / np.maximum(B_svd[:, -1], 1e-12))
    props["C_cond"] = np.log10(C_svd[:, 0] / np.maximum(C_svd[:, -1], 1e-12))
    props["A_det"] = np.linalg.det(A_mat)
    props["B_det"] = np.linalg.det(B_mat)
    props["C_frob"] = np.linalg.norm(C_mat.reshape(batch, -1), axis=1)
    props["A_trace"] = np.trace(A_mat, axis1=1, axis2=2)

    BA_mat = B_mat @ A_mat
    comm = np.linalg.norm((C_mat - BA_mat).reshape(batch, -1), axis=1)
    props["commutator"] = comm / np.maximum(props["C_frob"], 1e-12)

    props["A_spectral_gap"] = A_svd[:, 0] - A_svd[:, 1]
    props["A_rank1_ness"] = A_svd[:, 0] / np.maximum(A_svd.sum(axis=1), 1e-12)

    return props


def visualize(ckpt_path, n_samples=20000, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt["step"]
    best_err = ckpt["best_rel_err"]
    state = ckpt["model"]

    # Infer model params
    head_U_weight = state["head_U.2.weight"]
    N = head_U_weight.shape[0] // 9
    latent_dim = state["head_U.0.weight"].shape[1]
    encoder_width = state["encoder.0.weight"].shape[0]
    block_keys = [k for k in state if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight")]
    encoder_depth = len(block_keys)
    attn_keys = [k for k in state if "attn.in_proj_weight" in k]
    n_attn = len(attn_keys)
    attn_every = encoder_depth // max(n_attn, 1) if n_attn > 0 else 0
    n_heads = 8  # from CLI default

    print(f"Checkpoint: step={step}, best_rel_err={best_err:.4e}")
    print(f"Model: N={N}, latent={latent_dim}, depth={encoder_depth}, width={encoder_width}")

    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
        encoder_width=encoder_width, n_heads=n_heads, attn_every=attn_every,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Extract latents and channel norms
    print(f"Generating {n_samples} samples...")
    all_latent = []
    all_cn = []
    all_A = []
    all_B = []
    all_err = []

    batch_size = min(n_samples, 4096)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - start)
            A, B, C = sample_batch(bs, device)
            latent = model._encode(torch.cat([A, B], dim=1)).float()
            C_hat = model(A, B)

            # Per-sample error
            diff = (C_hat.float() - C.float()).view(bs, 9)
            true_norm = C.float().view(bs, 9).norm(dim=1).clamp(min=1e-12)
            err = (diff.norm(dim=1) / true_norm).cpu().numpy()

            cn = model.channel_norms(A, B)

            all_latent.append(latent.cpu().numpy())
            all_cn.append(cn.cpu().numpy())
            all_A.append(A.cpu().numpy())
            all_B.append(B.cpu().numpy())
            all_err.append(err)

    latent_np = np.concatenate(all_latent)
    cn_np = np.concatenate(all_cn)
    A_mat = np.concatenate(all_A).reshape(-1, 3, 3)
    B_mat = np.concatenate(all_B).reshape(-1, 3, 3)
    err_np = np.concatenate(all_err)
    props = extract_properties(A_mat, B_mat)
    props["recon_error"] = err_np
    props["total_channel_norm"] = cn_np.sum(axis=1)

    # Add top channel dominance
    sorted_cn = np.sort(cn_np, axis=1)[:, ::-1]
    props["top1_frac"] = sorted_cn[:, 0] / np.maximum(cn_np.sum(axis=1), 1e-12)

    # PCA
    print("Running PCA...")
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_np)
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(latent_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_[:3]}")

    # Try UMAP
    try:
        from umap import UMAP
        print("Running UMAP...")
        umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        umap_coords = umap.fit_transform(latent_scaled)
        have_umap = True
    except ImportError:
        print("UMAP not installed (pip install umap-learn), using PCA only")
        have_umap = False

    # Plot
    color_props = ["A_cond", "B_cond", "C_cond", "commutator", "recon_error",
                   "A_det", "total_channel_norm", "top1_frac", "A_rank1_ness"]

    n_props = len(color_props)
    n_methods = 2 if have_umap else 1

    fig, axes = plt.subplots(n_props, n_methods, figsize=(7 * n_methods, 4 * n_props),
                              squeeze=False)
    fig.suptitle(f"Latent Manifold — Step {step}, Best={best_err:.3e}", fontsize=14, y=1.0)

    for i, prop_name in enumerate(color_props):
        vals = props[prop_name]
        # Clip outliers for better color range
        vmin, vmax = np.percentile(vals, [2, 98])
        vals_clipped = np.clip(vals, vmin, vmax)

        # PCA plot
        ax = axes[i, 0]
        sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=vals_clipped,
                       cmap="viridis", s=1, alpha=0.3, rasterized=True)
        ax.set_title(f"PCA — {prop_name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax)

        # UMAP plot
        if have_umap:
            ax = axes[i, 1]
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=vals_clipped,
                           cmap="viridis", s=1, alpha=0.3, rasterized=True)
            ax.set_title(f"UMAP — {prop_name}")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    out_path = f"latent_manifold_step{step}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()

    # Also: PCA variance explained curve
    pca_full = PCA(n_components=min(50, latent_dim))
    pca_full.fit(latent_scaled)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=3)
    ax2.axhline(0.9, color="r", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(0.95, color="orange", linestyle="--", alpha=0.5, label="95%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title(f"PCA Variance — Step {step}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    out_path2 = f"pca_variance_step{step}.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path2}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoint found!")
        exit(1)

    print(f"Using checkpoint: {ckpt_path}")
    visualize(ckpt_path, args.n_samples, args.device)
