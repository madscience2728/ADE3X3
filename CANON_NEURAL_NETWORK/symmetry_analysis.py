"""
symmetry_analysis.py — Detect S_3 and other discrete symmetries in latent space.

Analyzes the PCA-projected latent vectors for:
1. N-fold rotational symmetry in each PC pair
2. Triangular prism detection (3-fold cross-section + continuous axis)
3. Correlation of symmetry axes with input permutation structure

Usage:
    python symmetry_analysis.py --ckpt checkpoints/ckpt_N19_s45.pt
"""

import argparse
import os
import glob
import itertools

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model import KethVaraiMachine
from data import sample_batch


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def angular_histogram_score(points_2d, n_fold, n_bins=360):
    """
    Score how well a 2D point cloud has n-fold rotational symmetry.
    Returns correlation of angular histogram with itself rotated by 2π/n.
    Score near 1 = strong n-fold symmetry.
    """
    angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-12

    # Rotate histogram by 2π/n
    shift = n_bins // n_fold
    hist_rot = np.roll(hist, shift)

    # Correlation
    return np.corrcoef(hist, hist_rot)[0, 1]


def detect_vertices(points_2d, n_fold, radius_frac=0.5):
    """
    Detect n_fold vertices in a 2D projection by finding
    density peaks in angular bins after filtering by radius.
    Returns angles of detected peaks.
    """
    radii = np.linalg.norm(points_2d, axis=1)
    r_thresh = np.percentile(radii, radius_frac * 100)
    outer = points_2d[radii > r_thresh]

    if len(outer) < 100:
        return np.array([])

    angles = np.arctan2(outer[:, 1], outer[:, 0])
    n_bins = 72
    hist, edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(hist.astype(float), sigma=2, mode='wrap')

    # Find peaks
    peaks = []
    for i in range(len(smooth)):
        prev_i = (i - 1) % len(smooth)
        next_i = (i + 1) % len(smooth)
        if smooth[i] > smooth[prev_i] and smooth[i] > smooth[next_i]:
            peaks.append(centers[i])

    return np.array(peaks)


def test_permutation_correlation(latent_np, A_flat, B_flat, device, model):
    """
    Test whether permuting rows/cols of A,B maps to predictable
    transformations in latent space (evidence of S_3 encoding).
    """
    n_test = min(2000, len(A_flat))
    A = A_flat[:n_test].reshape(-1, 3, 3)
    B = B_flat[:n_test].reshape(-1, 3, 3)

    # The 6 elements of S_3 as 3x3 permutation matrices
    perms = []
    for p in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(p):
            P[i, j] = 1.0
        perms.append(P)

    print(f"\nTesting {len(perms)} permutations on {n_test} samples...")

    # Get base latent vectors
    base_latent = latent_np[:n_test]

    results = []
    with torch.no_grad():
        for pi, P in enumerate(perms):
            # Permute rows of A and cols of B: P @ A, B @ P^T
            # This leaves C = A @ B invariant as a similarity transform
            # Actually: (PAP^T)(PBP^T) = P(AB)P^T — same matmul, permuted output
            A_perm = (P @ A)  # permute rows of A
            B_perm = (B @ P.T)  # permute cols of B

            A_t = torch.tensor(A_perm.reshape(-1, 9), dtype=torch.float32, device=device)
            B_t = torch.tensor(B_perm.reshape(-1, 9), dtype=torch.float32, device=device)

            perm_latent = model._encode(torch.cat([A_t, B_t], dim=1)).float().cpu().numpy()

            # How does the permuted latent relate to the base?
            # Compute displacement vectors
            displacements = perm_latent - base_latent
            mean_disp = np.mean(displacements, axis=0)
            disp_norm = np.linalg.norm(mean_disp)
            disp_std = np.std(np.linalg.norm(displacements, axis=1))

            # Cosine similarity of displacement directions across samples
            d_norms = np.linalg.norm(displacements, axis=1, keepdims=True)
            d_normed = displacements / np.maximum(d_norms, 1e-12)
            mean_d_normed = d_normed.mean(axis=0)
            consistency = np.linalg.norm(mean_d_normed)  # 1 = all same direction

            perm_label = ''.join(str(j) for j in [list(row).index(1.0) for row in P])
            results.append({
                'perm': perm_label,
                'mean_disp_norm': disp_norm,
                'disp_std': disp_std,
                'direction_consistency': consistency,
                'is_identity': np.allclose(P, np.eye(3)),
            })

            print(f"  Perm {perm_label}: |mean_disp|={disp_norm:.3f}, "
                  f"std={disp_std:.3f}, consistency={consistency:.3f}"
                  f"{'  (identity)' if results[-1]['is_identity'] else ''}")

    return results


def run(ckpt_path, n_samples=20000, n_pcs=15, device_str="cuda"):
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

    all_latent, all_A, all_B = [], [], []
    bs = min(n_samples, 4096)
    with torch.no_grad():
        for start in range(0, n_samples, bs):
            b = min(bs, n_samples - start)
            A, B, C = sample_batch(b, device)
            latent = model._encode(torch.cat([A, B], dim=1)).float()
            all_latent.append(latent.cpu().numpy())
            all_A.append(A.cpu().numpy())
            all_B.append(B.cpu().numpy())

    latent_np = np.concatenate(all_latent)
    A_flat = np.concatenate(all_A)
    B_flat = np.concatenate(all_B)

    # PCA
    scaler = StandardScaler()
    latent_sc = scaler.fit_transform(latent_np)
    pca = PCA(n_components=n_pcs)
    pc_coords = pca.fit_transform(latent_sc)
    var_ratios = pca.explained_variance_ratio_

    print(f"\nPCA variance ratios:")
    for i in range(n_pcs):
        print(f"  PC{i+1}: {var_ratios[i]*100:.1f}%  (cumul: {np.sum(var_ratios[:i+1])*100:.1f}%)")

    # ── Test 1: N-fold rotational symmetry in all PC pairs ──
    print(f"\n{'='*60}")
    print("TEST 1: Rotational symmetry scores in PC pairs")
    print(f"{'='*60}")

    top_k = 10  # check top 10 PCs
    best_3fold = []

    for i in range(top_k):
        for j in range(i + 1, top_k):
            pts = pc_coords[:, [i, j]]
            s2 = angular_histogram_score(pts, 2)
            s3 = angular_histogram_score(pts, 3)
            s4 = angular_histogram_score(pts, 4)
            s6 = angular_histogram_score(pts, 6)

            if s3 > 0.5:  # notable 3-fold symmetry
                best_3fold.append((i, j, s3))

            if max(s3, s4, s6) > 0.5:
                print(f"  PC{i+1}-PC{j+1}: 2-fold={s2:.3f}  3-fold={s3:.3f}  "
                      f"4-fold={s4:.3f}  6-fold={s6:.3f}")

    if best_3fold:
        best_3fold.sort(key=lambda x: -x[2])
        print(f"\n  Top 3-fold pairs:")
        for i, j, s in best_3fold[:5]:
            print(f"    PC{i+1}-PC{j+1}: score={s:.3f}")
    else:
        print("  No strong 3-fold symmetry found in top PC pairs.")

    # ── Test 2: Vertex detection ──
    print(f"\n{'='*60}")
    print("TEST 2: Angular vertex detection")
    print(f"{'='*60}")

    for i, j, s in best_3fold[:3]:
        pts = pc_coords[:, [i, j]]
        peaks = detect_vertices(pts, 3)
        if len(peaks) > 0:
            print(f"  PC{i+1}-PC{j+1}: {len(peaks)} angular peaks at "
                  f"{np.degrees(peaks).round(1)}°")
            if len(peaks) == 3:
                separations = np.diff(np.sort(peaks))
                print(f"    Separations: {np.degrees(separations).round(1)}°  "
                      f"(ideal: 120°)")

    # ── Test 3: Triangular prism detection ──
    print(f"\n{'='*60}")
    print("TEST 3: Prism detection (triangular cross-section + continuous axis)")
    print(f"{'='*60}")

    for i, j, s in best_3fold[:3]:
        # For each remaining PC, check if it's independent (forms the prism axis)
        for k in range(top_k):
            if k == i or k == j:
                continue
            pts_cross = pc_coords[:, [i, j]]
            axis_vals = pc_coords[:, k]

            # If PC_k is independent of the triangular cross-section,
            # the angular structure should be present at all axis values
            lo_mask = axis_vals < np.percentile(axis_vals, 25)
            hi_mask = axis_vals > np.percentile(axis_vals, 75)

            s_lo = angular_histogram_score(pts_cross[lo_mask], 3)
            s_hi = angular_histogram_score(pts_cross[hi_mask], 3)

            if s_lo > 0.4 and s_hi > 0.4:
                print(f"  Prism: cross=PC{i+1}-PC{j+1}, axis=PC{k+1}  "
                      f"(3-fold lo={s_lo:.3f}, hi={s_hi:.3f})")

    # ── Test 4: S_3 permutation correlation ──
    print(f"\n{'='*60}")
    print("TEST 4: S_3 permutation responses in latent space")
    print(f"{'='*60}")

    perm_results = test_permutation_correlation(latent_np, A_flat, B_flat, device, model)

    # Check if non-identity permutations produce consistent displacements
    non_id = [r for r in perm_results if not r['is_identity']]
    mean_consistency = np.mean([r['direction_consistency'] for r in non_id])
    print(f"\n  Mean direction consistency (non-identity): {mean_consistency:.3f}")
    if mean_consistency > 0.5:
        print("  → STRONG: Permutations map to consistent directions in latent space")
        print("    The encoder has learned S_3 symmetry structure!")
    elif mean_consistency > 0.2:
        print("  → MODERATE: Some permutation structure detected")
    else:
        print("  → WEAK: Permutations don't map to consistent latent directions")

    # Check if 3-cycles (order 3) form triangular orbits
    # Permutations: identity=012, 3-cycles=120,201, transpositions=021,102,210
    three_cycles = [r for r in perm_results if r['perm'] in ('120', '201')]
    if len(three_cycles) == 2:
        d1 = three_cycles[0]['mean_disp_norm']
        d2 = three_cycles[1]['mean_disp_norm']
        print(f"\n  3-cycle displacement norms: {d1:.3f}, {d2:.3f}")
        if abs(d1 - d2) / max(d1, d2, 1e-12) < 0.3:
            print("  → 3-cycles have similar displacement magnitude (consistent with S_3)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if best_3fold:
        print(f"  3-fold symmetry: YES (best score={best_3fold[0][2]:.3f} in PC{best_3fold[0][0]+1}-PC{best_3fold[0][1]+1})")
    else:
        print(f"  3-fold symmetry: NOT DETECTED in PC projections")
    print(f"  S_3 permutation consistency: {mean_consistency:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_pcs", type=int, default=15)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ckpt = a.ckpt or find_latest_checkpoint()
    if not ckpt:
        print("No checkpoint found!"); exit(1)
    print(f"Checkpoint: {ckpt}")
    run(ckpt, a.n_samples, a.n_pcs, a.device)
