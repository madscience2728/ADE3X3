"""
latent_pc_anatomy.py -- What does each latent PC dimension control?

For each of the top ~25 PCs, we compute:
  1. How it modulates U, V, W (dU/dPC, dV/dPC, dW/dPC via finite differences)
  2. Which output entries C[i,j] it most affects
  3. Whether it corresponds to a recognizable matmul sub-structure
  4. Correlation with input features (A entries, B entries, A*B products)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from model import KethVaraiMachine
from data import sample_batch

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ckpt_N11_s42.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50_000
N_PCS = 25


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


@torch.no_grad()
def extract_all(model, A, B):
    """Return U, V, W, latent, C_hat."""
    batch = A.shape[0]
    latent = model._encode(torch.cat([A, B], dim=1))
    U = model.head_U(latent).view(batch, 11, 9)
    V = model.head_V(latent).view(batch, 11, 9)
    W = model.head_W(latent).view(batch, 9, 11)
    C_hat = model(A, B)
    return U, V, W, latent, C_hat


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


def analyze():
    model = load_model()
    torch.manual_seed(999)
    A, B, C = sample_batch(N_SAMPLES, torch.device(DEVICE))

    U, V, W, latent, C_hat = extract_all(model, A, B)

    # PCA of latent space
    lat_np = latent.cpu().numpy()
    lat_mean = lat_np.mean(axis=0)
    lat_centered = lat_np - lat_mean
    cov = np.cov(lat_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    total_var = eigvals.sum()

    # Project latent onto PCs
    pc_scores = lat_centered @ eigvecs[:, :N_PCS]  # (N, 25)

    # ================================================================
    # 1. PC -> OUTPUT EFFECT: which C[i,j] entries does each PC control?
    # ================================================================
    section("1. PC -> OUTPUT ENTRY MAPPING")
    print("For each PC, correlation with each of the 9 output entries C[i,j]:")
    print("(Using actual model output error per entry as proxy)\n")

    C_hat_np = C_hat.cpu().numpy()
    C_np = C.cpu().numpy()
    err_per_entry = np.abs(C_hat_np - C_np)  # (N, 9)

    # But more useful: correlation of PC score with each C_hat entry
    C_hat_centered = C_hat_np - C_hat_np.mean(axis=0)

    print(f"{'PC':>4} {'var%':>5}  ", end="")
    for i in range(3):
        for j in range(3):
            print(f"C[{i},{j}]  ", end="")
    print("  Dominant entries")

    for pc in range(N_PCS):
        pct = eigvals[pc] / total_var * 100
        corrs = []
        for entry in range(9):
            r = np.corrcoef(pc_scores[:, pc], C_hat_centered[:, entry])[0, 1]
            corrs.append(r)
        corrs = np.array(corrs)

        print(f"PC{pc+1:2d} {pct:5.1f}  ", end="")
        for entry in range(9):
            r = corrs[entry]
            print(f"{r:+.3f}  ", end="")

        # Top 2 entries by absolute correlation
        top = np.argsort(np.abs(corrs))[::-1][:3]
        labels = [f"C[{t//3},{t%3}]({corrs[t]:+.3f})" for t in top]
        print(f"  {', '.join(labels)}")

    # ================================================================
    # 2. PC -> U, V, W MODULATION
    # ================================================================
    section("2. PC -> WEIGHT MODULATION")
    print("How much each PC modulates U, V, W (correlation of PC score with weight Frobenius)")
    print("Plus: which channel k is most affected\n")

    U_flat = U.view(N_SAMPLES, -1).cpu().numpy()  # (N, 99)
    V_flat = V.view(N_SAMPLES, -1).cpu().numpy()
    W_flat = W.view(N_SAMPLES, -1).cpu().numpy()

    U_centered = U_flat - U_flat.mean(axis=0)
    V_centered = V_flat - V_flat.mean(axis=0)
    W_centered = W_flat - W_flat.mean(axis=0)

    print(f"{'PC':>4} {'var%':>5}  {'|dU|':>7} {'|dV|':>7} {'|dW|':>7}  {'U chan':>6} {'V chan':>6} {'W chan':>6}")

    for pc in range(N_PCS):
        pct = eigvals[pc] / total_var * 100
        sc = pc_scores[:, pc]

        # Regression coefficient: project weight variation onto this PC
        # dWeight/dPC = cov(weight, PC) / var(PC)
        pc_var = sc.var()
        dU = (U_centered.T @ sc) / (len(sc) * pc_var + 1e-30)  # (99,)
        dV = (V_centered.T @ sc) / (len(sc) * pc_var + 1e-30)
        dW = (W_centered.T @ sc) / (len(sc) * pc_var + 1e-30)

        dU_norm = np.linalg.norm(dU)
        dV_norm = np.linalg.norm(dV)
        dW_norm = np.linalg.norm(dW)

        # Which channel k is most affected?
        dU_ch = np.linalg.norm(dU.reshape(11, 9), axis=1)
        dV_ch = np.linalg.norm(dV.reshape(11, 9), axis=1)
        dW_ch = np.linalg.norm(dW.reshape(9, 11), axis=0)

        u_top = np.argmax(dU_ch)
        v_top = np.argmax(dV_ch)
        w_top = np.argmax(dW_ch)

        print(f"PC{pc+1:2d} {pct:5.1f}  {dU_norm:7.4f} {dV_norm:7.4f} {dW_norm:7.4f}  "
              f"k={u_top:2d}    k={v_top:2d}    k={w_top:2d}")

    # ================================================================
    # 3. PC -> INPUT FEATURE MAPPING
    # ================================================================
    section("3. PC -> INPUT FEATURES")
    print("Correlation of each PC with raw input entries and key bilinear products\n")

    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    # Raw entries: 18 features
    raw_features = np.concatenate([A_np, B_np], axis=1)  # (N, 18)
    raw_labels = [f"A[{i//3},{i%3}]" for i in range(9)] + [f"B[{i//3},{i%3}]" for i in range(9)]

    # Key bilinear products: the 27 that appear in standard matmul
    # C[i,j] = sum_s A[i,s]*B[s,j], so relevant products are A[i,s]*B[s,j]
    matmul_products = []
    matmul_labels = []
    for i in range(3):
        for j in range(3):
            for s in range(3):
                prod = A_np[:, 3*i+s] * B_np[:, 3*s+j]
                matmul_products.append(prod)
                matmul_labels.append(f"A[{i},{s}]*B[{s},{j}]")
    matmul_products = np.stack(matmul_products, axis=1)  # (N, 27)

    print(f"{'PC':>4} {'var%':>5}  Top raw inputs                    Top matmul products")

    for pc in range(N_PCS):
        pct = eigvals[pc] / total_var * 100
        sc = pc_scores[:, pc]

        # Raw input correlations
        raw_corrs = np.array([np.corrcoef(sc, raw_features[:, j])[0, 1] for j in range(18)])
        raw_top = np.argsort(np.abs(raw_corrs))[::-1][:3]
        raw_str = ", ".join(f"{raw_labels[t]}({raw_corrs[t]:+.3f})" for t in raw_top)

        # Matmul product correlations
        mp_corrs = np.array([np.corrcoef(sc, matmul_products[:, j])[0, 1] for j in range(27)])
        mp_top = np.argsort(np.abs(mp_corrs))[::-1][:2]
        mp_str = ", ".join(f"{matmul_labels[t]}({mp_corrs[t]:+.3f})" for t in mp_top)

        print(f"PC{pc+1:2d} {pct:5.1f}  {raw_str:38s} {mp_str}")

    # ================================================================
    # 4. FINITE-DIFFERENCE: PERTURB EACH PC, MEASURE C_hat CHANGE
    # ================================================================
    section("4. CAUSAL EFFECT: PERTURBING EACH PC")
    print("Inject +/- 2*sigma perturbation in latent PC, measure output change\n")

    # Use a fixed reference: mean latent
    lat_mean_t = torch.from_numpy(lat_mean).float().to(DEVICE).unsqueeze(0)  # (1, 128)
    eigvecs_t = torch.from_numpy(eigvecs[:, :N_PCS].astype(np.float32)).to(DEVICE)  # (128, 25)

    # Decode mean latent
    with torch.no_grad():
        U0 = model.head_U(lat_mean_t).view(1, 11, 9)
        V0 = model.head_V(lat_mean_t).view(1, 11, 9)
        W0 = model.head_W(lat_mean_t).view(1, 9, 11)

    # For reference: need A, B to compute C_hat. Use identity-like inputs
    # Actually, perturbation of latent changes U,V,W which changes C_hat for ALL inputs.
    # Let's measure the tensor change instead.

    print(f"{'PC':>4} {'var%':>5}  {'|dU|':>8} {'|dV|':>8} {'|dW|':>8}  "
          f"{'|dTensor|':>9}  Top C[i,j] affected")

    for pc in range(N_PCS):
        pct = eigvals[pc] / total_var * 100
        sigma = eigvals[pc] ** 0.5
        direction = eigvecs_t[:, pc]  # (128,)

        with torch.no_grad():
            lat_plus = lat_mean_t + 2 * sigma * direction.unsqueeze(0)
            lat_minus = lat_mean_t - 2 * sigma * direction.unsqueeze(0)

            U_p = model.head_U(lat_plus).view(11, 9)
            V_p = model.head_V(lat_plus).view(11, 9)
            W_p = model.head_W(lat_plus).view(9, 11)

            U_m = model.head_U(lat_minus).view(11, 9)
            V_m = model.head_V(lat_minus).view(11, 9)
            W_m = model.head_W(lat_minus).view(9, 11)

            dU = (U_p - U_m).norm().item()
            dV = (V_p - V_m).norm().item()
            dW = (W_p - W_m).norm().item()

            # Reconstruct tensors T[c,a,b] = sum_k W[c,k]*U[k,a]*V[k,b]
            T_p = torch.einsum('ck,ka,kb->cab', W_p, U_p, V_p)
            T_m = torch.einsum('ck,ka,kb->cab', W_m, U_m, V_m)
            dT = T_p - T_m  # (9, 9, 9)

            dT_norm = dT.norm().item()

            # Which output entries C[i,j] change most?
            dT_per_output = dT.view(9, -1).norm(dim=1).cpu().numpy()
            top_out = np.argsort(dT_per_output)[::-1][:3]
            out_str = ", ".join(f"C[{t//3},{t%3}]({dT_per_output[t]:.3f})" for t in top_out)

        print(f"PC{pc+1:2d} {pct:5.1f}  {dU:8.4f} {dV:8.4f} {dW:8.4f}  "
              f"{dT_norm:9.4f}  {out_str}")

    # ================================================================
    # 5. SEMANTIC CLUSTERING: GROUP PCs BY FUNCTION
    # ================================================================
    section("5. FUNCTIONAL GROUPING OF PCs")

    # For each PC, compute the output-entry correlation profile (9-dim vector)
    # and cluster them
    profiles = np.zeros((N_PCS, 9))
    for pc in range(N_PCS):
        for entry in range(9):
            profiles[pc, entry] = np.corrcoef(pc_scores[:, pc], C_hat_centered[:, entry])[0, 1]

    # Cosine similarity matrix between PC profiles
    norms = np.linalg.norm(profiles, axis=1, keepdims=True).clip(1e-10)
    normed = profiles / norms
    sim = normed @ normed.T

    # Simple grouping: which output entries does each PC primarily serve?
    print("Assignment of PCs to output entries (by strongest correlation):\n")

    entry_pcs = {c: [] for c in range(9)}
    for pc in range(N_PCS):
        best = np.argmax(np.abs(profiles[pc]))
        entry_pcs[best].append((pc, profiles[pc, best], eigvals[pc] / total_var * 100))

    for c in range(9):
        i, j = c // 3, c % 3
        pcs = entry_pcs[c]
        if pcs:
            pc_strs = [f"PC{p+1}({r:+.3f}, {v:.1f}%)" for p, r, v in pcs]
            print(f"  C[{i},{j}]: {', '.join(pc_strs)}")
        else:
            print(f"  C[{i},{j}]: (none)")

    total_assigned = sum(sum(v for _, _, v in pcs) for pcs in entry_pcs.values())
    print(f"\n  Total variance assigned: {total_assigned:.1f}%")

    # ================================================================
    # 6. INDEPENDENCE CHECK: ARE PCs DOING REDUNDANT WORK?
    # ================================================================
    section("6. REDUNDANCY CHECK")
    print("Cosine similarity between PC output-effect profiles:")
    print("(Values near +/-1 mean redundant PCs)\n")

    # Print top redundant pairs
    pairs = []
    for i in range(N_PCS):
        for j in range(i+1, N_PCS):
            pairs.append((abs(sim[i, j]), i, j, sim[i, j]))

    pairs.sort(reverse=True)
    print("Top 10 most similar PC pairs:")
    for _, i, j, s in pairs[:10]:
        print(f"  PC{i+1} <-> PC{j+1}: cosine={s:+.4f}")

    # ================================================================
    # 7. MATMUL ROW ANALYSIS
    # ================================================================
    section("7. MATMUL ROW DECOMPOSITION")
    print("Matrix multiplication has 3 output rows, each needing 3 inner products.")
    print("How does the network distribute its capacity?\n")

    # For each output row (0,1,2), measure total PC variance allocated
    row_var = np.zeros(3)
    row_pcs = {0: [], 1: [], 2: []}
    for pc in range(N_PCS):
        # Which row does this PC most affect?
        row_corrs = np.zeros(3)
        for row in range(3):
            row_corrs[row] = np.linalg.norm(profiles[pc, 3*row:3*row+3])
        best_row = np.argmax(row_corrs)
        row_var[best_row] += eigvals[pc] / total_var * 100
        row_pcs[best_row].append(pc + 1)

    for row in range(3):
        print(f"  Row {row} (C[{row},*]): {row_var[row]:.1f}% variance, "
              f"PCs: {row_pcs[row]}")

    print(f"\n  Row imbalance: max/min = {row_var.max()/row_var.min():.2f}x")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    analyze()
