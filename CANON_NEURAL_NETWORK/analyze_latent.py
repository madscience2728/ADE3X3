"""
analyze_latent.py — Cluster the encoder's latent space from a checkpoint.

Loads a checkpoint, generates random (A, B) pairs, extracts latent vectors
and U/V/W matrices, clusters, and reports what input properties correlate
with cluster membership.

Usage:
    python analyze_latent.py                          # uses latest checkpoint
    python analyze_latent.py --ckpt checkpoints/ckpt_N19_s42.pt
    python analyze_latent.py --n_samples 100000 --max_k 30
"""

import argparse
import os
import glob

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from model import KethVaraiMachine
from data import sample_batch


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_input_properties(A_mat, B_mat):
    """Compute interpretable properties of (A, B) matrix pairs.
    A_mat, B_mat: (batch, 3, 3) numpy arrays.
    Returns dict of property_name -> (batch,) array.
    """
    batch = A_mat.shape[0]
    props = {}

    # A properties
    props["A_det"] = np.linalg.det(A_mat)
    props["A_trace"] = np.trace(A_mat, axis1=1, axis2=2)
    props["A_frob"] = np.linalg.norm(A_mat.reshape(batch, -1), axis=1)
    A_svd = np.linalg.svd(A_mat, compute_uv=False)
    props["A_cond"] = A_svd[:, 0] / np.maximum(A_svd[:, -1], 1e-12)
    props["A_rank_gap"] = A_svd[:, 0] - A_svd[:, 1]  # gap between top 2 singular values
    props["A_spectral_ratio"] = A_svd[:, 2] / np.maximum(A_svd[:, 0], 1e-12)  # smallest/largest sv

    # B properties
    props["B_det"] = np.linalg.det(B_mat)
    props["B_trace"] = np.trace(B_mat, axis1=1, axis2=2)
    props["B_frob"] = np.linalg.norm(B_mat.reshape(batch, -1), axis=1)
    B_svd = np.linalg.svd(B_mat, compute_uv=False)
    props["B_cond"] = B_svd[:, 0] / np.maximum(B_svd[:, -1], 1e-12)
    props["B_rank_gap"] = B_svd[:, 0] - B_svd[:, 1]
    props["B_spectral_ratio"] = B_svd[:, 2] / np.maximum(B_svd[:, 0], 1e-12)

    # Joint properties
    C_mat = A_mat @ B_mat
    props["C_det"] = np.linalg.det(C_mat)
    props["C_frob"] = np.linalg.norm(C_mat.reshape(batch, -1), axis=1)
    C_svd = np.linalg.svd(C_mat, compute_uv=False)
    props["C_cond"] = C_svd[:, 0] / np.maximum(C_svd[:, -1], 1e-12)

    # Commutativity measure: ||AB - BA||_F / ||AB||_F
    BA_mat = B_mat @ A_mat
    comm = np.linalg.norm((C_mat - BA_mat).reshape(batch, -1), axis=1)
    props["commutator_norm"] = comm / np.maximum(props["C_frob"], 1e-12)

    # Alignment: how aligned are A and B's singular spaces?
    # Use abs cos of angle between top singular vectors
    A_U = np.linalg.svd(A_mat, compute_uv=True)[0][:, :, 0]  # (batch, 3)
    B_U = np.linalg.svd(B_mat, compute_uv=True)[0][:, :, 0]  # (batch, 3)
    props["sv_alignment"] = np.abs(np.sum(A_U * B_U, axis=1))

    return props


def analyze(ckpt_path, n_samples=50000, max_k=20, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt["step"]
    best_rel_err = ckpt["best_rel_err"]
    print(f"Loaded checkpoint: step={step}, best_rel_err={best_rel_err:.4e}")

    # Infer model params from state dict
    # head_U output: (latent_dim*2, N*9) for last linear
    state = ckpt["model"]
    head_U_weight = state["head_U.2.weight"]  # (N*9, head_hidden)
    N_times_9 = head_U_weight.shape[0]
    N = N_times_9 // 9
    head_hidden = head_U_weight.shape[1]
    latent_dim = state["head_U.0.weight"].shape[1]
    encoder_width = state["encoder.0.weight"].shape[0]

    # Count encoder depth from state dict keys
    block_keys = [k for k in state.keys() if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight")]
    encoder_depth = len(block_keys)

    # Count attention blocks to infer attn_every and n_heads
    attn_keys = [k for k in state.keys() if k.startswith("encoder_blocks.") and "attn.in_proj_weight" in k]
    n_attn_blocks = len(attn_keys)
    attn_every = encoder_depth // max(n_attn_blocks, 1) if n_attn_blocks > 0 else 0

    if n_attn_blocks > 0:
        first_attn_key = [k for k in state.keys() if "attn.in_proj_weight" in k][0]
        attn_weight_size = state[first_attn_key].shape[0]
        d_token = encoder_width // 9
        n_heads = attn_weight_size // (3 * d_token)
    else:
        n_heads = 4

    print(f"Model: N={N}, latent_dim={latent_dim}, encoder_depth={encoder_depth}, "
          f"encoder_width={encoder_width}, n_heads={n_heads}, attn_every={attn_every}")

    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
        encoder_width=encoder_width, n_heads=n_heads, attn_every=attn_every,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Generate samples and extract features
    print(f"Generating {n_samples} samples...")
    all_latent = []
    all_channel_norms = []
    all_A = []
    all_B = []
    all_U_collapse = []
    all_V_collapse = []
    all_W_collapse = []

    batch_size = min(n_samples, 8192)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - start)
            A, B, C = sample_batch(bs, device)

            latent = model._encode(torch.cat([A, B], dim=1)).float()
            U = model.head_U(latent).view(bs, N, 9)
            V = model.head_V(latent).view(bs, N, 9)
            W = model.head_W(latent).view(bs, 9, N)

            cn = model.channel_norms(A, B)

            all_latent.append(latent.cpu().numpy())
            all_channel_norms.append(cn.cpu().numpy())
            all_A.append(A.cpu().numpy())
            all_B.append(B.cpu().numpy())

    latent_np = np.concatenate(all_latent, axis=0)
    channel_norms_np = np.concatenate(all_channel_norms, axis=0)
    A_flat = np.concatenate(all_A, axis=0)
    B_flat = np.concatenate(all_B, axis=0)
    A_mat = A_flat.reshape(-1, 3, 3)
    B_mat = B_flat.reshape(-1, 3, 3)

    print(f"Latent shape: {latent_np.shape}")
    print(f"Latent stats: mean={latent_np.mean():.4f}, std={latent_np.std():.4f}, "
          f"dead_frac={(np.abs(latent_np) < 1e-4).mean():.4f}")

    # Channel norms analysis
    mean_cn = channel_norms_np.mean(axis=0)
    sorted_idx = np.argsort(mean_cn)[::-1]
    print(f"\nChannel norms (sorted, top 10):")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  ch{idx:2d}: {mean_cn[idx]:.4f}")
    active = (mean_cn > mean_cn.max() * 0.01).sum()
    print(f"Active channels (>1% of max): {active}/{N}")

    # Extract input properties
    print(f"\nComputing input properties...")
    props = extract_input_properties(A_mat, B_mat)

    # Cluster sweep
    print(f"\nClustering latent space (k=2..{max_k})...")
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_np)

    best_k = 2
    best_score = -1
    inertias = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        labels = km.fit_predict(latent_scaled)
        inertia = km.inertia_
        inertias.append(inertia)

        # Silhouette score on a subsample for speed
        if k <= 10 or k % 5 == 0:
            from sklearn.metrics import silhouette_score
            sub = min(5000, len(latent_scaled))
            idx = np.random.RandomState(42).choice(len(latent_scaled), sub, replace=False)
            sil = silhouette_score(latent_scaled[idx], labels[idx])
            if sil > best_score:
                best_score = sil
                best_k = k
            print(f"  k={k:3d}  inertia={inertia:.0f}  silhouette={sil:.4f}")
        else:
            print(f"  k={k:3d}  inertia={inertia:.0f}")

    print(f"\nBest k by silhouette: k={best_k} (score={best_score:.4f})")

    # Detailed analysis at best k
    km_best = KMeans(n_clusters=best_k, n_init=10, max_iter=300, random_state=42)
    labels = km_best.fit_predict(latent_scaled)

    print(f"\n{'='*70}")
    print(f"CLUSTER ANALYSIS (k={best_k})")
    print(f"{'='*70}")

    for c in range(best_k):
        mask = labels == c
        count = mask.sum()
        pct = 100 * count / len(labels)
        print(f"\n--- Cluster {c} ({count} samples, {pct:.1f}%) ---")

        # Channel norms for this cluster
        cn_cluster = channel_norms_np[mask].mean(axis=0)
        active_c = (cn_cluster > cn_cluster.max() * 0.01).sum()
        print(f"  Active channels: {active_c}/{N}")
        top3 = np.argsort(cn_cluster)[::-1][:3]
        print(f"  Top channels: {', '.join(f'ch{i}={cn_cluster[i]:.3f}' for i in top3)}")

        # Input properties for this cluster
        print(f"  Input properties (mean ± std):")
        for name, vals in props.items():
            cluster_vals = vals[mask]
            global_mean = vals.mean()
            cluster_mean = cluster_vals.mean()
            cluster_std = cluster_vals.std()
            # Highlight if cluster mean differs significantly from global
            diff = abs(cluster_mean - global_mean) / max(abs(global_mean), 1e-8)
            marker = " <<<" if diff > 0.2 else ""
            print(f"    {name:20s}: {cluster_mean:+.4f} ± {cluster_std:.4f}"
                  f"  (global: {global_mean:+.4f}){marker}")

    # Elbow plot data
    print(f"\n{'='*70}")
    print(f"INERTIA CURVE (for elbow detection)")
    print(f"{'='*70}")
    for k, inertia in zip(range(2, max_k + 1), inertias):
        bar = "#" * int(50 * inertia / inertias[0])
        print(f"  k={k:3d}  {bar}  {inertia:.0f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze encoder latent space")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: latest)")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--max_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found!")
        exit(1)

    print(f"Using checkpoint: {ckpt_path}")
    analyze(ckpt_path, args.n_samples, args.max_k, args.device)
