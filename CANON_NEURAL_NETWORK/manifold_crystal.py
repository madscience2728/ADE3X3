"""
manifold_crystal.py -- Crystallize the latent manifold.

The network maps 18-dim input (A,B) -> 25-dim effective latent -> W(99 params).
V is frozen, U barely moves. The manifold IS the W-variation.

We characterize:
  1. Manifold shape: curvature, topology, boundary
  2. Discrete vs continuous structure
  3. Algebraic relations between latent coords
  4. Minimal parametrization: can we express W(input) in closed form?
  5. Symmetry orbits: does the manifold respect matmul symmetries?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from model import KethVaraiMachine
from data import sample_batch

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_N11_s42.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 50_000


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = KethVaraiMachine(
        N=11, latent_dim=128, encoder_depth=2, encoder_width=192,
        n_heads=4, attn_every=4, expanded_products=True,
        head_depth=1, head_width=256,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


@torch.no_grad()
def analyze():
    model = load_model()
    torch.manual_seed(42)
    A, B, C = sample_batch(N, torch.device(DEVICE))

    # Extract W for all samples (the manifold lives in W-space)
    latent = model._encode(torch.cat([A, B], dim=1))
    W = model.head_W(latent)  # (N, 99)
    U = model.head_U(latent)  # (N, 99)

    W_np = W.cpu().numpy()
    U_np = U.cpu().numpy()
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    W_mean = W_np.mean(axis=0)
    W_centered = W_np - W_mean

    # PCA of W-space (the actual manifold)
    cov_W = np.cov(W_centered, rowvar=False)
    eigvals_W, eigvecs_W = np.linalg.eigh(cov_W)
    eigvals_W = eigvals_W[::-1]
    eigvecs_W = eigvecs_W[:, ::-1]
    total_var_W = eigvals_W.sum()

    # ================================================================
    section("1. MANIFOLD DIMENSIONALITY IN W-SPACE")
    # ================================================================
    cumvar = np.cumsum(eigvals_W) / total_var_W
    dims_90 = np.searchsorted(cumvar, 0.90) + 1
    dims_95 = np.searchsorted(cumvar, 0.95) + 1
    dims_99 = np.searchsorted(cumvar, 0.99) + 1

    print(f"W lives in R^99. Effective manifold dimension:")
    print(f"  90% variance: {dims_90} dims")
    print(f"  95% variance: {dims_95} dims")
    print(f"  99% variance: {dims_99} dims")

    print(f"\nTop 20 W-space eigenvalues:")
    for i in range(20):
        pct = eigvals_W[i] / total_var_W * 100
        cum = cumvar[i] * 100
        print(f"  WPC{i+1:2d}: {eigvals_W[i]:10.2f} ({pct:5.1f}%, cum {cum:5.1f}%)")

    # Project W onto its top PCs
    n_wpc = dims_99
    W_proj = W_centered @ eigvecs_W[:, :n_wpc]  # (N, n_wpc)

    # ================================================================
    section("2. MANIFOLD SHAPE: GAUSSIANITY TEST")
    # ================================================================
    # If the manifold were a flat linear subspace with Gaussian input,
    # each WPC coordinate would be Gaussian. Test this.
    from scipy import stats

    print("Normality test for each W-PC coordinate (Jarque-Bera):")
    print(f"{'WPC':>5} {'skew':>8} {'kurtosis':>10} {'JB stat':>10} {'p-value':>10} {'Verdict':>10}")
    n_nongauss = 0
    for i in range(min(n_wpc, 30)):
        x = W_proj[:, i]
        jb, p = stats.jarque_bera(x)
        sk = stats.skew(x)
        ku = stats.kurtosis(x)
        verdict = "GAUSSIAN" if p > 0.01 else "NON-GAUSS"
        if p <= 0.01:
            n_nongauss += 1
        print(f"  WPC{i+1:2d} {sk:+8.3f}  {ku:+10.3f}  {jb:10.1f}  {p:10.4f}  {verdict}")

    print(f"\n{n_nongauss}/{min(n_wpc, 30)} dimensions are non-Gaussian")
    print("=> Non-Gaussianity indicates nonlinear manifold structure (curves, folds, boundaries)")

    # ================================================================
    section("3. PAIRWISE STRUCTURE: SCATTERPLOT STATISTICS")
    # ================================================================
    # Check for nonlinear dependencies between WPC pairs
    print("Nonlinear dependence between top WPC pairs:")
    print("(Linear corr vs mutual info estimate via binning)\n")

    def mutual_info_binned(x, y, bins=30):
        """Estimate MI via histogram."""
        c_xy = np.histogram2d(x, y, bins=bins)[0]
        c_xy = c_xy / c_xy.sum()
        c_x = c_xy.sum(axis=1)
        c_y = c_xy.sum(axis=0)
        mask = c_xy > 0
        mi = np.sum(c_xy[mask] * np.log(c_xy[mask] / (c_x[:, None] * c_y[None, :])[mask]))
        return mi

    print(f"{'Pair':>12} {'lin_corr':>8} {'MI':>8} {'MI/H':>8}  Note")
    interesting_pairs = []
    for i in range(min(10, n_wpc)):
        for j in range(i + 1, min(10, n_wpc)):
            x, y = W_proj[:, i], W_proj[:, j]
            r = np.corrcoef(x, y)[0, 1]
            mi = mutual_info_binned(x, y)
            # Entropy of marginals
            hx = stats.entropy(np.histogram(x, bins=30)[0] + 1)
            hy = stats.entropy(np.histogram(y, bins=30)[0] + 1)
            mi_norm = mi / min(hx, hy) if min(hx, hy) > 0 else 0
            note = ""
            if abs(r) < 0.05 and mi_norm > 0.05:
                note = "NONLINEAR DEP"
                interesting_pairs.append((i, j, r, mi_norm))
            elif abs(r) > 0.3:
                note = "LINEAR DEP"
                interesting_pairs.append((i, j, r, mi_norm))
            if note:
                print(f"  WPC{i+1}-WPC{j+1} {r:+8.4f} {mi:8.4f} {mi_norm:8.4f}  {note}")

    if not interesting_pairs:
        print("  No strong pairwise dependencies found -- dimensions are largely independent")

    # ================================================================
    section("4. IS W A LINEAR FUNCTION OF INPUT?")
    # ================================================================
    # Test: W = M @ [A, B] + bias? If so, manifold is a linear subspace.
    input_flat = np.concatenate([A_np, B_np], axis=1)  # (N, 18)

    # Least-squares fit: W = input @ M + bias
    input_aug = np.concatenate([input_flat, np.ones((N, 1))], axis=1)  # (N, 19)
    M, residuals, rank, sv = np.linalg.lstsq(input_aug, W_np, rcond=None)

    W_linear_pred = input_aug @ M
    linear_err = np.linalg.norm(W_np - W_linear_pred) / np.linalg.norm(W_centered)
    print(f"Linear model W = M @ [A,B] + b:")
    print(f"  Relative error: {linear_err:.4f}")
    print(f"  R^2: {1 - linear_err**2:.4f}")

    # Quadratic model: include all 81 bilinear products
    products = (A_np[:, :, None] * B_np[:, None, :]).reshape(N, 81)
    input_quad = np.concatenate([input_flat, products, np.ones((N, 1))], axis=1)  # (N, 100)

    M2, _, _, _ = np.linalg.lstsq(input_quad, W_np, rcond=None)
    W_quad_pred = input_quad @ M2
    quad_err = np.linalg.norm(W_np - W_quad_pred) / np.linalg.norm(W_centered)
    print(f"\nQuadratic model W = M1 @ [A,B] + M2 @ (A ox B) + b:")
    print(f"  Relative error: {quad_err:.4f}")
    print(f"  R^2: {1 - quad_err**2:.4f}")

    # What about just the 27 matmul-relevant products?
    matmul_prods = []
    for i in range(3):
        for j in range(3):
            for s in range(3):
                matmul_prods.append(A_np[:, 3*i+s] * B_np[:, 3*s+j])
    matmul_prods = np.stack(matmul_prods, axis=1)  # (N, 27)

    input_matmul = np.concatenate([input_flat, matmul_prods, np.ones((N, 1))], axis=1)  # (N, 46)
    M3, _, _, _ = np.linalg.lstsq(input_matmul, W_np, rcond=None)
    W_matmul_pred = input_matmul @ M3
    matmul_err = np.linalg.norm(W_np - W_matmul_pred) / np.linalg.norm(W_centered)
    print(f"\nMatmul-product model W = M1 @ [A,B] + M2 @ [27 matmul products] + b:")
    print(f"  Relative error: {matmul_err:.4f}")
    print(f"  R^2: {1 - matmul_err**2:.4f}")

    # ================================================================
    section("5. IS W A LINEAR FUNCTION OF A*B PRODUCTS?")
    # ================================================================
    # Pure bilinear: W depends only on the 81 A_i * B_j products?
    input_pure_bilin = np.concatenate([products, np.ones((N, 1))], axis=1)  # (N, 82)
    M4, _, _, _ = np.linalg.lstsq(input_pure_bilin, W_np, rcond=None)
    W_bilin_pred = input_pure_bilin @ M4
    bilin_err = np.linalg.norm(W_np - W_bilin_pred) / np.linalg.norm(W_centered)
    print(f"Pure bilinear model W = M @ (A ox B) + b:")
    print(f"  Relative error: {bilin_err:.4f}")
    print(f"  R^2: {1 - bilin_err**2:.4f}")

    # Do the same for C_hat directly
    C_hat = model(A, B).cpu().numpy()
    C_np = C.cpu().numpy()

    M5, _, _, _ = np.linalg.lstsq(input_quad, C_hat, rcond=None)
    C_quad_pred = input_quad @ M5
    c_quad_err = np.linalg.norm(C_hat - C_quad_pred) / np.linalg.norm(C_hat - C_hat.mean(axis=0))
    print(f"\nFor reference -- quadratic model of C_hat directly:")
    print(f"  Relative error: {c_quad_err:.4f}")
    print(f"  R^2: {1 - c_quad_err**2:.4f}")

    # ================================================================
    section("6. MANIFOLD CURVATURE")
    # ================================================================
    # Sample triplets of nearby points and measure how much the manifold curves
    # between them (ratio of geodesic to chord length)

    # Use W-space for this
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11, algorithm='auto')
    nn.fit(W_proj[:10000])

    dists, indices = nn.kneighbors(W_proj[:1000])

    # For each point, fit a local linear subspace to its 10 neighbors
    # and measure residual (how much neighbors deviate from the tangent plane)
    local_dims = []
    local_curvatures = []
    for i in range(1000):
        nbrs = W_proj[indices[i]]  # (11, n_wpc)
        center = nbrs[0]
        deltas = nbrs[1:] - center  # (10, n_wpc)
        sv = np.linalg.svd(deltas, compute_uv=False)
        # Local dimension: number of significant SVs
        sv_norm = sv / sv[0]
        local_dim = (sv_norm > 0.1).sum()
        local_dims.append(local_dim)

        # Curvature: residual of best-fit plane
        U_svd, S_svd, Vt_svd = np.linalg.svd(deltas, full_matrices=False)
        # Project onto top-k plane
        k = min(5, len(sv))
        proj = deltas @ Vt_svd[:k].T @ Vt_svd[:k]
        residual = np.linalg.norm(deltas - proj) / np.linalg.norm(deltas)
        local_curvatures.append(residual)

    local_dims = np.array(local_dims)
    local_curvatures = np.array(local_curvatures)

    print(f"Local manifold analysis (1000 neighborhoods, 10 neighbors each):")
    print(f"  Local dimension (at 10% SV threshold):")
    for d in range(1, min(local_dims.max() + 1, 20)):
        count = (local_dims == d).sum()
        if count > 0:
            print(f"    dim={d}: {count} points ({100*count/len(local_dims):.1f}%)")

    print(f"\n  Local curvature (residual from tangent plane):")
    print(f"    Mean: {local_curvatures.mean():.4f}")
    print(f"    Median: {local_curvatures.median() if hasattr(local_curvatures, 'median') else np.median(local_curvatures):.4f}")
    print(f"    Max: {local_curvatures.max():.4f}")
    print(f"    <1% residual: {(local_curvatures < 0.01).sum()} points (FLAT)")
    print(f"    >10% residual: {(local_curvatures > 0.10).sum()} points (CURVED)")

    # ================================================================
    section("7. SYMMETRY ORBITS")
    # ================================================================
    # Matmul has symmetries. If we permute rows/cols of A,B consistently,
    # the output should transform predictably. Does the manifold respect this?

    # Test: swap rows 0,1 of A and rows 0,1 of C.
    # A' = P @ A => C' = P @ A @ B = P @ C
    # So W should transform covariantly.

    n_sym = 5000
    A_sym = A[:n_sym].clone()
    B_sym = B[:n_sym].clone()

    # Swap rows 0 and 1 of A (in flattened form: swap entries 0,1,2 with 3,4,5)
    A_swapped = A_sym.clone()
    A_swapped[:, 0:3] = A_sym[:, 3:6]
    A_swapped[:, 3:6] = A_sym[:, 0:3]

    # Get W for original and swapped
    lat_orig = model._encode(torch.cat([A_sym, B_sym], dim=1))
    lat_swap = model._encode(torch.cat([A_swapped, B_sym], dim=1))
    W_orig = model.head_W(lat_orig).cpu().numpy()
    W_swap = model.head_W(lat_swap).cpu().numpy()

    # For exact matmul, swapping rows 0,1 of A should swap rows 0,1 of C.
    # In W-space (9x11), this means swapping rows 0,1,2 with 3,4,5
    # (since C is flattened as [C00,C01,C02,C10,C11,C12,...])
    W_orig_mat = W_orig.reshape(-1, 9, 11)
    W_swap_mat = W_swap.reshape(-1, 9, 11)

    # Expected: W_swap[0:3,:] = W_orig[3:6,:] and vice versa
    W_expected = W_orig_mat.copy()
    W_expected[:, 0:3, :] = W_orig_mat[:, 3:6, :]
    W_expected[:, 3:6, :] = W_orig_mat[:, 0:3, :]

    sym_err = np.linalg.norm(W_swap_mat - W_expected, axis=(1, 2))
    W_norm = np.linalg.norm(W_orig_mat, axis=(1, 2))
    sym_rel = (sym_err / W_norm).mean()

    print(f"Row-swap symmetry test (swap rows 0,1 of A):")
    print(f"  Expected: W transforms by swapping C-rows 0,1")
    print(f"  Relative error from equivariance: {sym_rel:.4f}")
    if sym_rel < 0.05:
        print(f"  => EQUIVARIANT -- manifold respects row permutation symmetry")
    elif sym_rel < 0.2:
        print(f"  => APPROXIMATELY equivariant")
    else:
        print(f"  => NOT equivariant -- symmetry is broken")

    # Test column swap of B
    B_swapped = B_sym.clone()
    B_swapped[:, [0, 3, 6]] = B_sym[:, [1, 4, 7]]
    B_swapped[:, [1, 4, 7]] = B_sym[:, [0, 3, 6]]

    lat_bswap = model._encode(torch.cat([A_sym, B_swapped], dim=1))
    W_bswap = model.head_W(lat_bswap).cpu().numpy().reshape(-1, 9, 11)

    # Swapping cols 0,1 of B swaps cols 0,1 of C
    # C flattened: swap entries {0,3,6} with {1,4,7}
    W_b_expected = W_orig_mat.copy()
    for row in range(3):
        W_b_expected[:, 3*row+0, :] = W_orig_mat[:, 3*row+1, :]
        W_b_expected[:, 3*row+1, :] = W_orig_mat[:, 3*row+0, :]

    bsym_err = np.linalg.norm(W_bswap - W_b_expected, axis=(1, 2))
    bsym_rel = (bsym_err / W_norm).mean()

    print(f"\nColumn-swap symmetry test (swap cols 0,1 of B):")
    print(f"  Relative error from equivariance: {bsym_rel:.4f}")
    if bsym_rel < 0.05:
        print(f"  => EQUIVARIANT")
    elif bsym_rel < 0.2:
        print(f"  => APPROXIMATELY equivariant")
    else:
        print(f"  => NOT equivariant -- symmetry is broken")

    # ================================================================
    section("8. CRYSTALLIZATION: CLOSED-FORM W(A,B)")
    # ================================================================
    # Since W varies with input and we've tested linear/quadratic models,
    # let's extract the best closed-form approximation and measure its
    # quality for the ACTUAL matmul task.

    # Best model so far: quadratic W = M1@[A,B] + M2@(AxB) + b
    # Use this to predict C_hat and compare to true C.
    # C_hat = W @ (U_mean * V_mean stuff) but W is input-dependent...
    # Actually, need to go through the full bilinear computation.

    # Let's try: use the quadratic W model + mean U,V to compute C
    U_mean = U.mean(dim=0).view(11, 9)  # (11, 9)
    V_mean = model.head_V(latent).mean(dim=0).view(11, 9)

    # For each sample: W_approx from quadratic model, U=U_mean, V=V_mean
    W_approx = torch.from_numpy(W_quad_pred.astype(np.float32)).to(DEVICE).view(-1, 9, 11)
    p = A @ U_mean.T  # (N, 11)
    q = B @ V_mean.T  # (N, 11)
    m = p * q  # (N, 11)
    C_crystal = torch.bmm(W_approx, m.unsqueeze(-1)).squeeze(-1)  # (N, 9)

    err_crystal = (C_crystal - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    err_model = (model(A, B) - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)

    print(f"Crystallized approximation: W_quad(A,B) @ (U_mean . A) * (V_mean . B)")
    print(f"  Mean rel error (crystal): {err_crystal.mean():.6e}")
    print(f"  Mean rel error (model):   {err_model.mean():.6e}")
    print(f"  Mean rel error (exact):   0.0")
    print(f"  Crystal captures {100*(1 - err_crystal.mean()/err_model.mean()):.1f}% of model's remaining error")

    # What's the crystal formula explicitly?
    # W_ij(A,B) = sum_a M1[a, ij]*A_a + sum_b M1[9+b, ij]*B_b
    #           + sum_{a,b} M2[18+9a+b, ij]*A_a*B_b + bias[ij]
    # Then C_c = sum_k W[c,k] * (sum_a U[k,a]*A_a) * (sum_b V[k,b]*B_b)
    #
    # This is a CUBIC function of A,B (W is bilinear, times bilinear UxV = quartic total)
    # Wait: W is linear+bilinear in (A,B), and the products p*q are bilinear.
    # So C_hat is: (linear+bilinear) * bilinear = cubic + quartic in the entries.
    # True matmul is bilinear. So the cubic/quartic terms should cancel or be negligible.

    print(f"\nThe crystal formula is a degree-4 polynomial in A,B entries.")
    print(f"True matmul is degree-2 (bilinear). The extra degrees provide correction.")

    # Decompose error by degree: how much does each order contribute?
    # Constant (mean W, mean U, mean V):
    W_const = torch.from_numpy(W_mean.astype(np.float32)).to(DEVICE).view(9, 11)
    C_const = (W_const @ (p * q).T).T
    err_const = (C_const - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)

    # Linear W only:
    W_lin = torch.from_numpy((input_aug @ M).astype(np.float32)).to(DEVICE).view(-1, 9, 11)
    C_lin = torch.bmm(W_lin, m.unsqueeze(-1)).squeeze(-1)
    err_lin = (C_lin - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)

    print(f"\nError by approximation order:")
    print(f"  Constant W (degree-2 output):  {err_const.mean():.4e}")
    print(f"  Linear W (degree-3 output):    {err_lin.mean():.4e}")
    print(f"  Quadratic W (degree-4 output): {err_crystal.mean():.4e}")
    print(f"  Full neural net:               {err_model.mean():.4e}")
    print(f"  Exact matmul:                  0.0")

    # ================================================================
    section("9. RESIDUAL ANALYSIS: WHAT THE CRYSTAL MISSES")
    # ================================================================
    residual = (C_crystal - C).cpu().numpy()
    C_np = C.cpu().numpy()

    # Per-entry analysis
    print(f"Per-entry crystal error (mean abs / mean abs of entry):")
    for c in range(9):
        i, j = c // 3, c % 3
        entry_err = np.abs(residual[:, c]).mean()
        entry_scale = np.abs(C_np[:, c]).mean()
        print(f"  C[{i},{j}]: err={entry_err:.4e}, scale={entry_scale:.4e}, "
              f"ratio={entry_err/entry_scale:.4e}")

    # ================================================================
    section("10. SUMMARY: MANIFOLD IDENTITY")
    # ================================================================
    print("The manifold M is the image of the map:")
    print("  phi: R^18 -> R^99")
    print("  (A, B) |-> W(A, B)")
    print()
    print(f"Properties:")
    print(f"  - Ambient dimension: 99 (W has 9x11 entries)")
    print(f"  - Intrinsic dimension: ~{dims_95} (95% variance)")
    print(f"  - W is well-approximated as degree-2 polynomial in (A,B)")
    print(f"  - Quadratic R^2 of W: {1 - quad_err**2:.4f}")
    print(f"  - Bilinear R^2 of W: {1 - bilin_err**2:.4f}")
    print(f"  - The crystal (quad W + fixed U,V) achieves rel err {err_crystal.mean():.4e}")
    print(f"  - Full model achieves rel err {err_model.mean():.4e}")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    analyze()
