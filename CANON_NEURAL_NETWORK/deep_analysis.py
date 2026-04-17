"""
deep_analysis.py -- Autopsy of a trained N=11 Keth-Varai checkpoint.

Questions we answer:
  1. Is U,V,W input-dependent or effectively constant? (hypernetwork test)
  2. What is the effective rank of the predicted U,V,W across the input manifold?
  3. What do the "mean" U,V,W look like as fixed matrices? How close is C_hat to C?
  4. What is the latent manifold geometry? (PCA, variance explained, clustering)
  5. Per-channel analysis: which bilinear channels are active, dead, or redundant?
  6. Symmetry: does the network exploit any structure of matmul?
  7. Sensitivity: which input directions cause the most variation in U,V,W?
  8. Can we extract a closed-form rank-11 decomposition from the mean weights?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from model import KethVaraiMachine
from data import sample_batch

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_N11_s42.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50_000  # large sample for statistics


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    # Reconstruct model from checkpoint
    model = KethVaraiMachine(
        N=11, latent_dim=128, encoder_depth=2, encoder_width=192,
        n_heads=4, attn_every=4, expanded_products=True,
        head_depth=1, head_width=256,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    best = ckpt.get("best_rel_err", "?")
    print(f"Loaded checkpoint: step={step}, best_rel_err={best}")
    return model


@torch.no_grad()
def extract_UVW(model, A, B):
    """Run encoder+heads, return U,V,W tensors and latent."""
    batch = A.shape[0]
    latent = model._encode(torch.cat([A, B], dim=1))
    U = model.head_U(latent).view(batch, 11, 9)
    V = model.head_V(latent).view(batch, 11, 9)
    W = model.head_W(latent).view(batch, 9, 11)
    return U, V, W, latent


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze():
    model = load_model()
    torch.manual_seed(999)
    A, B, C = sample_batch(N_SAMPLES, torch.device(DEVICE))

    # ================================================================
    # 1. IS U,V,W INPUT-DEPENDENT OR EFFECTIVELY CONSTANT?
    # ================================================================
    section("1. INPUT-DEPENDENCE OF U, V, W")

    U, V, W, latent = extract_UVW(model, A, B)

    U_mean = U.mean(dim=0)  # (11, 9)
    V_mean = V.mean(dim=0)
    W_mean = W.mean(dim=0)  # (9, 11)

    U_std = U.std(dim=0)    # per-element std across batch
    V_std = V.std(dim=0)
    W_std = W.std(dim=0)

    # Coefficient of variation: std / |mean|
    U_cv = (U_std / U_mean.abs().clamp(min=1e-10)).mean().item()
    V_cv = (V_std / V_mean.abs().clamp(min=1e-10)).mean().item()
    W_cv = (W_std / W_mean.abs().clamp(min=1e-10)).mean().item()

    print(f"U: mean_norm={U_mean.norm():.4f}, mean_element_std={U_std.mean():.6f}, CV={U_cv:.4f}")
    print(f"V: mean_norm={V_mean.norm():.4f}, mean_element_std={V_std.mean():.6f}, CV={V_cv:.4f}")
    print(f"W: mean_norm={W_mean.norm():.4f}, mean_element_std={W_std.mean():.6f}, CV={W_cv:.4f}")

    # Frobenius norm of variation vs mean
    U_var_ratio = U_std.norm() / U_mean.norm()
    V_var_ratio = V_std.norm() / V_mean.norm()
    W_var_ratio = W_std.norm() / W_mean.norm()
    print(f"\nVariation / Mean (Frobenius):")
    print(f"  U: {U_var_ratio:.6f}")
    print(f"  V: {V_var_ratio:.6f}")
    print(f"  W: {W_var_ratio:.6f}")

    if max(U_cv, V_cv, W_cv) < 0.1:
        print("\n=> U,V,W are EFFECTIVELY CONSTANT across inputs.")
        print("  The encoder learned to ignore the input and produce fixed weights.")
        print("  This is a FIXED decomposition, not a manifold of decompositions.")
    else:
        print("\n=> U,V,W vary meaningfully with input -- true hypernetwork behavior.")

    # ================================================================
    # 2. LATENT SPACE GEOMETRY
    # ================================================================
    section("2. LATENT SPACE GEOMETRY")

    lat_np = latent.cpu().numpy()
    lat_mean = lat_np.mean(axis=0)
    lat_centered = lat_np - lat_mean

    # PCA
    cov = np.cov(lat_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]  # descending
    eigvecs = eigvecs[:, ::-1]
    total_var = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total_var

    print(f"Latent dim: {latent.shape[1]}")
    print(f"Total variance: {total_var:.6f}")
    print(f"\nTop eigenvalues (variance explained):")
    for i in range(min(20, len(eigvals))):
        pct = eigvals[i] / total_var * 100
        cum = cumvar[i] * 100
        bar = '#' * int(pct / 2)
        print(f"  PC{i+1:2d}: {eigvals[i]:.6f} ({pct:5.1f}%, cum {cum:5.1f}%) {bar}")

    dims_90 = np.searchsorted(cumvar, 0.90) + 1
    dims_99 = np.searchsorted(cumvar, 0.99) + 1
    print(f"\nDims for 90% variance: {dims_90}")
    print(f"Dims for 99% variance: {dims_99}")

    effective_dim = np.exp(-np.sum((eigvals/total_var) * np.log(eigvals/total_var + 1e-30)))
    print(f"Effective dimensionality (exp entropy): {effective_dim:.1f}")

    # ================================================================
    # 3. MEAN-WEIGHT DECOMPOSITION QUALITY
    # ================================================================
    section("3. FIXED (MEAN) DECOMPOSITION QUALITY")

    # Use mean U,V,W as a fixed decomposition and evaluate
    A_mat = A.view(-1, 3, 3)
    B_mat = B.view(-1, 3, 3)

    # Fixed decomposition: C_hat[i,j] = sum_k (U_k . A_col_?) * (V_k . B_row_?) * W[ij,k]
    # In the network's parametrization:
    # p_k = U_mean[k,:] @ A  (dot product with flattened A)
    # q_k = V_mean[k,:] @ B
    # C_hat = W_mean @ (p * q)
    p_fixed = A @ U_mean.T  # (batch, 11)
    q_fixed = B @ V_mean.T  # (batch, 11)
    m_fixed = p_fixed * q_fixed
    C_hat_fixed = m_fixed @ W_mean.T  # (batch, 9)

    err_fixed = (C_hat_fixed - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    print(f"Fixed decomposition (mean weights):")
    print(f"  Mean relative error: {err_fixed.mean():.6e}")
    print(f"  Median relative error: {err_fixed.median():.6e}")
    print(f"  Max relative error: {err_fixed.max():.6e}")
    print(f"  Std relative error: {err_fixed.std():.6e}")

    # Also evaluate the actual model
    C_hat_model = model(A, B)
    err_model = (C_hat_model - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
    print(f"\nActual model (input-dependent weights):")
    print(f"  Mean relative error: {err_model.mean():.6e}")
    print(f"  Median relative error: {err_model.median():.6e}")
    print(f"  Max relative error: {err_model.max():.6e}")

    improvement = err_fixed.mean() / err_model.mean()
    print(f"\nInput-dependence helps by factor: {improvement:.2f}x")
    if improvement < 1.5:
        print("  => Negligible. The learned solution is essentially a fixed decomposition.")

    # ================================================================
    # 4. PER-CHANNEL ANALYSIS
    # ================================================================
    section("4. PER-CHANNEL DECOMPOSITION ANALYSIS")

    # Using mean weights, analyze each bilinear channel
    print("Channel analysis using MEAN weights:")
    print(f"{'Ch':>3} {'||u_k||':>8} {'||v_k||':>8} {'||w_k||':>8} {'Scale':>10} {'u·v cos':>8}")
    total_scale = 0
    for k in range(11):
        u_k = U_mean[k]  # (9,)
        v_k = V_mean[k]  # (9,)
        w_k = W_mean[:, k]  # (9,)
        u_norm = u_k.norm().item()
        v_norm = v_k.norm().item()
        w_norm = w_k.norm().item()
        scale = u_norm * v_norm * w_norm
        total_scale += scale
        # Cosine similarity between u and v (if aligned, channel is symmetric)
        cos_uv = (u_k @ v_k / (u_norm * v_norm + 1e-10)).item()
        print(f"  {k:2d}  {u_norm:8.4f} {v_norm:8.4f} {w_norm:8.4f} {scale:10.4f}  {cos_uv:+8.4f}")

    print(f"\n  Total scale: {total_scale:.4f}")

    # ================================================================
    # 5. WHAT MATMUL STRUCTURE DID IT LEARN?
    # ================================================================
    section("5. MATMUL STRUCTURE IN LEARNED WEIGHTS")

    # True 3x3 matmul: C[i,j] = sum_s A[i,s] * B[s,j]
    # In flattened form: C[3i+j] = sum_s A[3i+s] * B[3s+j]
    # This is a bilinear map with 27 terms (standard algorithm).
    # The network's bilinear form: C[3i+j] ≈ sum_k W[3i+j, k] * (sum_a U[k,a]*A[a]) * (sum_b V[k,b]*B[b])
    #
    # For channel k, the "tensor" it represents is:
    #   T_k[c, a, b] = W[c, k] * U[k, a] * V[k, b]
    #
    # The full tensor is T[c, a, b] = sum_k T_k[c, a, b]
    # For exact matmul, T[3i+j, 3i'+s, 3t+j'] = delta(i,i') * delta(s,t) * delta(j,j')

    # Reconstruct the full 9x9x9 tensor
    # T[c, a, b] = sum_k W_mean[c,k] * U_mean[k,a] * V_mean[k,b]
    T_learned = torch.einsum('ck,ka,kb->cab', W_mean, U_mean, V_mean)  # (9, 9, 9)

    # True matmul tensor
    T_true = torch.zeros(9, 9, 9, device=A.device)
    for i in range(3):
        for j in range(3):
            for s in range(3):
                c = 3 * i + j
                a = 3 * i + s
                b = 3 * s + j
                T_true[c, a, b] = 1.0

    err_tensor = (T_learned - T_true).norm() / T_true.norm()
    print(f"Tensor reconstruction error (mean weights): {err_tensor:.6e}")
    print(f"  T_learned Frobenius norm: {T_learned.norm():.4f}")
    print(f"  T_true Frobenius norm: {T_true.norm():.4f} (should be sqrt(27)~={27**0.5:.4f})")

    # Decompose the error: which output entries are worst?
    per_output_err = (T_learned - T_true).view(9, -1).norm(dim=1)
    print(f"\nPer-output-entry tensor error:")
    for c in range(9):
        i, j = c // 3, c % 3
        print(f"  C[{i},{j}]: error={per_output_err[c]:.6f}")

    # ================================================================
    # 6. RANK ANALYSIS OF THE LEARNED TENSOR
    # ================================================================
    section("6. RANK ANALYSIS")

    # The learned tensor T has tensor rank ≤ 11 by construction.
    # But what's its matrix rank when unfolded?
    # Unfolding-1: T as (9, 81) matrix
    T_unfold1 = T_learned.view(9, 81)
    T_unfold2 = T_learned.permute(1, 0, 2).reshape(9, 81)
    T_unfold3 = T_learned.permute(2, 0, 1).reshape(9, 81)

    s1 = torch.linalg.svdvals(T_unfold1)
    s2 = torch.linalg.svdvals(T_unfold2)
    s3 = torch.linalg.svdvals(T_unfold3)

    print("Singular values of mode unfoldings:")
    print(f"  Mode-1 (C-mode, 9×81): {s1.cpu().numpy().round(4)}")
    print(f"  Mode-2 (A-mode, 9×81): {s2.cpu().numpy().round(4)}")
    print(f"  Mode-3 (B-mode, 9×81): {s3.cpu().numpy().round(4)}")

    # For true matmul: all 9 singular values should be 1.0 for each unfolding
    T_true_u1 = T_true.view(9, 81)
    s1_true = torch.linalg.svdvals(T_true_u1)
    print(f"\n  True matmul mode-1 SVs: {s1_true.cpu().numpy().round(4)}")

    # ================================================================
    # 7. INPUT SENSITIVITY ANALYSIS
    # ================================================================
    section("7. INPUT SENSITIVITY -- JACOBIAN OF LATENT W.R.T. INPUT")

    # Take a small batch and compute Jacobian of latent w.r.t. input
    n_jac = 100
    A_jac = A[:n_jac].clone().requires_grad_(True)
    B_jac = B[:n_jac].clone().requires_grad_(True)

    # Forward through encoder only
    inp = torch.cat([A_jac, B_jac], dim=1)
    lat = model._encode(inp)  # (100, 128)

    # Compute Jacobian via backward passes on each latent dim
    jac_norms = []
    for d in range(lat.shape[1]):
        lat_d = lat[:, d].sum()
        grads = torch.autograd.grad(lat_d, [A_jac, B_jac], retain_graph=True)
        grad_A = grads[0]  # (100, 9)
        grad_B = grads[1]  # (100, 9)
        jac_norm = (grad_A.norm(dim=1) + grad_B.norm(dim=1)).mean().item()
        jac_norms.append(jac_norm)

    jac_norms = np.array(jac_norms)
    print(f"Jacobian ||d(latent_d)/d(input)||, averaged over {n_jac} samples:")
    print(f"  Mean: {jac_norms.mean():.6f}")
    print(f"  Max:  {jac_norms.max():.6f}")
    print(f"  Min:  {jac_norms.min():.6f}")
    print(f"  Std:  {jac_norms.std():.6f}")

    # How many latent dims are actually responsive to input?
    responsive = (jac_norms > 0.01 * jac_norms.max()).sum()
    print(f"  Responsive dims (>1% of max): {responsive} / {len(jac_norms)}")

    # Sensitivity to A vs B
    grad_A_total = torch.zeros(n_jac, device=A.device)
    grad_B_total = torch.zeros(n_jac, device=A.device)
    for d in range(lat.shape[1]):
        lat_d = lat[:, d].sum()
        grads = torch.autograd.grad(lat_d, [A_jac, B_jac], retain_graph=(d < lat.shape[1]-1))
        grad_A_total += grads[0].norm(dim=1)
        grad_B_total += grads[1].norm(dim=1)

    ratio_AB = (grad_A_total / grad_B_total.clamp(min=1e-10)).mean().item()
    print(f"\n  Sensitivity ratio |d/dA| / |d/dB|: {ratio_AB:.4f}")
    if abs(ratio_AB - 1.0) < 0.1:
        print("  => Symmetric sensitivity to A and B (expected for matmul)")
    else:
        print(f"  => Asymmetric! The network is {ratio_AB:.1f}x more sensitive to {'A' if ratio_AB > 1 else 'B'}")

    # ================================================================
    # 8. CAN WE EXTRACT A RANK-11 DECOMPOSITION?
    # ================================================================
    section("8. EXTRACTED RANK-11 DECOMPOSITION")

    print("The mean weights define a rank-<=11 decomposition of the matmul tensor:")
    print(f"  C[i,j] ~= Sum_k W[3i+j, k] * (U[k,:] . A_flat) * (V[k,:] . B_flat)")
    print()

    # Print the decomposition
    U_np = U_mean.cpu().numpy()
    V_np = V_mean.cpu().numpy()
    W_np = W_mean.cpu().numpy()

    for k in range(11):
        u = U_np[k]
        v = V_np[k]
        w = W_np[:, k]
        print(f"  Channel {k}:")
        print(f"    u = [{', '.join(f'{x:+.4f}' for x in u)}]")
        print(f"    v = [{', '.join(f'{x:+.4f}' for x in v)}]")
        print(f"    w = [{', '.join(f'{x:+.4f}' for x in w)}]")

    # ================================================================
    # 9. IS THE INPUT-DEPENDENT PART JUST NOISE OR STRUCTURED?
    # ================================================================
    section("9. STRUCTURE OF INPUT-DEPENDENT RESIDUAL")

    # U(input) = U_mean + delta_U(input)
    # What does delta_U look like? Is it low-rank?
    delta_U = (U - U_mean.unsqueeze(0)).view(N_SAMPLES, -1)  # (N, 99)
    delta_V = (V - V_mean.unsqueeze(0)).view(N_SAMPLES, -1)
    delta_W = (W - W_mean.unsqueeze(0)).view(N_SAMPLES, -1)

    for name, delta in [("delta_U", delta_U), ("delta_V", delta_V), ("delta_W", delta_W)]:
        sv = torch.linalg.svdvals(delta.float())
        total = sv.sum().item()
        cumfrac = torch.cumsum(sv, 0) / total
        rank_90 = (cumfrac < 0.90).sum().item() + 1
        rank_99 = (cumfrac < 0.99).sum().item() + 1
        print(f"{name}: Frobenius={delta.norm():.6f}, top SV={sv[0]:.6f}, "
              f"rank@90%={rank_90}, rank@99%={rank_99}")

    # ================================================================
    # 10. CORRELATION BETWEEN INPUT AND WEIGHT VARIATION
    # ================================================================
    section("10. WHAT INPUT FEATURES DRIVE WEIGHT VARIATION?")

    # Correlate the 81 bilinear products with each PC of the latent
    products = (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, 81)  # (N, 81)
    lat_proj = torch.from_numpy(lat_centered @ eigvecs).to(DEVICE)  # (N, 128)

    # Correlation matrix between products and top latent PCs
    n_pcs = min(10, latent.shape[1])
    print(f"Top correlations between 81 bilinear products and latent PCs:")
    for pc in range(n_pcs):
        corr = torch.corrcoef(torch.stack([lat_proj[:, pc], *[products[:, j] for j in range(81)]]))[0, 1:]
        top_idx = corr.abs().topk(3)
        entries = []
        for idx, val in zip(top_idx.indices, top_idx.values):
            # Decode product index to (i,s,t,j)
            i_idx = idx.item()
            a_idx, b_idx = i_idx // 9, i_idx % 9
            ai, as_ = a_idx // 3, a_idx % 3
            bt, bj = b_idx // 3, b_idx % 3
            entries.append(f"A[{ai},{as_}]*B[{bt},{bj}] (r={val:.3f})")
        var_pct = eigvals[pc] / total_var * 100
        print(f"  PC{pc+1} ({var_pct:.1f}% var): {', '.join(entries)}")

    # ================================================================
    # 11. COMPARISON: HOW CLOSE TO STRASSEN-LIKE?
    # ================================================================
    section("11. PROXIMITY TO KNOWN DECOMPOSITIONS")

    # Standard matmul uses 27 multiplications. Network uses 11.
    # Check: is the learned tensor exactly matmul, or an approximation?
    T_err = T_learned - T_true
    print(f"Absolute tensor error: {T_err.norm():.6e}")
    print(f"Relative tensor error: {err_tensor:.6e}")

    # Check sparsity of the learned factors
    for name, mat in [("U_mean", U_mean), ("V_mean", V_mean), ("W_mean", W_mean)]:
        flat = mat.abs().flatten()
        near_zero = (flat < 0.01 * flat.max()).sum().item()
        total = flat.numel()
        print(f"  {name}: {near_zero}/{total} entries near-zero ({100*near_zero/total:.0f}%)")

    # Check if any channels are (anti-)Strassen-like:
    # In Strassen for 2x2, each u,v,w has entries in {-1, 0, 1}.
    # For 3x3, check how "integer-like" the scaled factors are.
    print(f"\nInteger-likeness of factors (after scaling each channel to max=1):")
    for k in range(11):
        u = U_mean[k] / U_mean[k].abs().max().clamp(min=1e-10)
        v = V_mean[k] / V_mean[k].abs().max().clamp(min=1e-10)
        w = W_mean[:, k] / W_mean[:, k].abs().max().clamp(min=1e-10)
        # Distance to nearest integer
        u_int_err = (u - u.round()).abs().mean().item()
        v_int_err = (v - v.round()).abs().mean().item()
        w_int_err = (w - w.round()).abs().mean().item()
        print(f"  Ch{k:2d}: u_int_err={u_int_err:.4f} v_int_err={v_int_err:.4f} w_int_err={w_int_err:.4f}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    analyze()
