"""
encoder_deep_analysis.py
Deep comparative analysis of the encoder across checkpoints:
  ckpt_N1_s42069.pt, ckpt_N11_s42069.pt, ckpt_N81_s42069.pt

Sections:
  1. Checkpoint metadata
  2. Encoder weight statistics (per-layer norm, spectral norm, rank)
  3. Latent space geometry (PCA variance, effective dim, isotropy)
  4. Input sensitivity / Jacobian analysis
  5. Collapse ratios (how much U/V/W depend on input)
  6. Channel utilisation in U, V, W
  7. Error distribution comparison
  8. Attention pattern analysis (per block)
  9. Encoder representation alignment (CKA between pairs)
 10. Summary table
"""

import sys
import os
import torch
import numpy as np

# ── make sure we can import from the CANON_NEURAL_NETWORK dir ───────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from model import KethVaraiMachine
from data import sample_batch

# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(THIS_DIR, "checkpoints")
BATCH = 8192
SEED = 42

TARGETS = [
    ("N1",  1,  "ckpt_N1_s42069.pt"),
    ("N11", 11, "ckpt_N11_s42069.pt"),
    ("N81", 81, "ckpt_N81_s42069.pt"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
def sep(title="", width=72):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("\n" + "─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def infer_arch(state_dict, N):
    """Infer latent_dim, encoder_width, encoder_depth from state_dict keys."""
    # encoder_head.1.bias shape → latent_dim
    latent_dim = state_dict["encoder_head.1.bias"].shape[0]
    # encoder.0.bias shape → encoder_width
    encoder_width = state_dict["encoder.0.bias"].shape[0]
    # count ResBlock layers: each ResBlock has net.1 and net.4 Linear layers
    depth = 0
    for k in state_dict:
        # encoder_blocks.X.net.1.weight — one per ResBlock
        if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight"):
            depth += 1
    return latent_dim, encoder_width, depth


def load(name, N, fname):
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["model"]
    latent_dim, encoder_width, encoder_depth = infer_arch(sd, N)
    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
        encoder_width=encoder_width, linear_heads=False
    ).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Shared validation set
torch.manual_seed(SEED)
A_val, B_val, C_val = sample_batch(BATCH, DEVICE)
# 81-product input for encoder
AB_val = torch.cat([A_val, B_val], dim=1)  # (B, 18)


def products(A, B):
    return (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, 81)


X_val = products(A_val, B_val)  # (B, 81)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint metadata
sep("1. CHECKPOINT METADATA")
print(f"{'Tag':<6} {'N':>4}  {'Step':>8}  {'BestRelErr':>12}  {'Params(M)':>10}")
sep()
models = {}
ckpts = {}
for tag, N, fname in TARGETS:
    m, c = load(tag, N, fname)
    models[tag] = (m, N)
    ckpts[tag] = c
    nparams = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{tag:<6} {N:>4}  {c['step']:>8}  {c['best_rel_err']:>12.4e}  {nparams:>10.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Encoder weight statistics
sep("2. ENCODER WEIGHT STATISTICS (per linear layer in encoder trunk)")

def encoder_weight_stats(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "head_" not in name:
            W = module.weight.detach().float()
            frob = W.norm().item()
            # spectral norm via SVD (truncated for speed)
            k = min(W.shape[0], W.shape[1], 10)
            try:
                sv = torch.linalg.svdvals(W)
                spec = sv[0].item()
                stable_rank = (W.norm()**2 / sv[0]**2).item()
                eff_rank = torch.exp(-(sv/sv.sum() * torch.log(sv/sv.sum() + 1e-12)).sum()).item()
            except Exception:
                spec, stable_rank, eff_rank = float("nan"), float("nan"), float("nan")
            rows.append((name, W.shape, frob, spec, stable_rank))
    return rows

for tag, (model, N) in models.items():
    sep(f"  [{tag}]")
    print(f"  {'Layer':<50} {'Shape':<16} {'||W||_F':>10} {'σ_max':>10} {'StabRk':>8}")
    for name, shape, frob, spec, sr in encoder_weight_stats(model):
        print(f"  {name:<50} {str(list(shape)):<16} {frob:>10.3f} {spec:>10.4f} {sr:>8.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Latent space geometry
sep("3. LATENT SPACE GEOMETRY")

def latent_geometry(model):
    with torch.no_grad():
        latent = model._encode(AB_val)  # (B, 256)
    L = latent.float().cpu().numpy()
    L -= L.mean(axis=0, keepdims=True)
    _, sv, _ = np.linalg.svd(L, full_matrices=False)
    sv2 = sv**2
    total = sv2.sum()
    cumvar = np.cumsum(sv2) / total
    # effective dimensionality (participation ratio)
    eff_dim = (sv2.sum()**2) / (sv2**2).sum()
    # fraction of variance in top-k
    top5  = cumvar[4]
    top10 = cumvar[9]
    top50 = cumvar[49]
    isotropy = sv[-1] / sv[0]  # ratio smallest/largest singular value
    return sv, eff_dim, top5, top10, top50, isotropy, L, latent

print(f"{'Tag':<6}  {'EffDim':>8}  {'Top5%':>8}  {'Top10%':>8}  {'Top50%':>8}  {'Isotropy':>10}")
sep()
latent_data = {}
for tag, (model, N) in models.items():
    sv, eff_dim, t5, t10, t50, iso, L, latent_t = latent_geometry(model)
    latent_data[tag] = (sv, L, latent_t)
    print(f"{tag:<6}  {eff_dim:>8.1f}  {t5*100:>7.1f}%  {t10*100:>7.1f}%  {t50*100:>7.1f}%  {iso:>10.4e}")

# Singular value profiles
sep("  Singular value profile (top-20 cumulative variance %)")
print(f"  {'k':>4}", end="")
for tag in [t for t, _, _ in TARGETS]:
    print(f"  {tag:>8}", end="")
print()
for k in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 256, 512, 1024]:
    sv0, *_ = latent_data[list(latent_data.keys())[0]]
    if k > len(sv0):
        continue
    print(f"  {k:>4}", end="")
    for tag in [t for t, _, _ in TARGETS]:
        sv, *_ = latent_data[tag]
        sv2 = sv**2
        cumvar = np.cumsum(sv2) / sv2.sum()
        print(f"  {cumvar[k-1]*100:>7.1f}%", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Input sensitivity via Hutchinson estimator of ||J||_F
#    Uses K random unit vectors v: E[||J^T v||^2] = ||J||_F^2 / latent_dim
#    Only K backward passes per sample instead of latent_dim — ~100× faster.
sep("4. ENCODER INPUT SENSITIVITY  (Hutchinson ||J||_F estimate, K=32 probes)")

def encoder_jacobian_stats(model, n_samples=512, K=32):
    """
    Estimate mean Frobenius norm of d(latent)/d(products) via Hutchinson trace estimator.
    ||J||_F^2 = E_v[ ||J^T v||^2 ] * latent_dim,  v ~ Rademacher.
    Cost: K backward passes total (not per-sample), very fast.
    """
    torch.manual_seed(0)
    A_s, B_s, _ = sample_batch(n_samples, DEVICE)
    prods = products(A_s, B_s)  # (n, 81)

    latent_dim = model.encoder_head[-1].out_features
    frob2_estimates = []

    for _ in range(K):
        p = prods.detach().requires_grad_(True)
        lat = model.encoder_head(model.encoder_blocks(model.encoder(p)))  # (n, latent_dim)
        # Rademacher probe: (n, latent_dim)
        v = torch.randint(0, 2, lat.shape, device=DEVICE).float() * 2 - 1
        # vjp: J^T v for each sample simultaneously
        grad = torch.autograd.grad((lat * v).sum(), p)[0]  # (n, 81)
        frob2_estimates.append((grad ** 2).sum(dim=1).detach())  # (n,)

    # stack K estimates, average over K → estimate of ||J||_F^2 / latent_dim * latent_dim
    frob2 = torch.stack(frob2_estimates, dim=0).mean(dim=0) * latent_dim  # (n,)
    frob  = frob2.sqrt().cpu().numpy()
    return float(frob.mean()), float(frob.std())

print(f"  {'Tag':<6}  {'Mean ||J||_F':>14}  {'Std':>10}")
sep()
for tag, (model, N) in models.items():
    print(f"  Computing {tag}...", end="\r", flush=True)
    mu, sigma = encoder_jacobian_stats(model)
    print(f"  {tag:<6}  {mu:>14.4f}  {sigma:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Collapse ratios: how much do U/V/W depend on the input?
sep("5. COLLAPSE RATIOS  (std_across_batch / mean_abs: higher = more input-dependent)")

def collapse_ratios(model, N):
    with torch.no_grad():
        latent = model._encode(AB_val)
        U = model.head_U(latent).view(-1, N, 9)
        V = model.head_V(latent).view(-1, N, 9)
        W = model.head_W(latent).view(-1, 9, N)
    def ratio(t):
        return (t.float().std(dim=0).mean() / t.float().abs().mean().clamp(min=1e-12)).item()
    return ratio(U), ratio(V), ratio(W), U, V, W

print(f"{'Tag':<6}  {'U_ratio':>10}  {'V_ratio':>10}  {'W_ratio':>10}")
sep()
uvw_data = {}
for tag, (model, N) in models.items():
    ru, rv, rw, U, V, W = collapse_ratios(model, N)
    uvw_data[tag] = (U, V, W, N)
    print(f"{tag:<6}  {ru:>10.4f}  {rv:>10.4f}  {rw:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Channel utilisation
sep("6. CHANNEL UTILISATION  (mean |u_k|·|v_k| over batch, sorted)")

def channel_utilisation(model, N):
    with torch.no_grad():
        latent = model._encode(AB_val)
        U = model.head_U(latent).view(-1, N, 9)  # (B, N, 9)
        V = model.head_V(latent).view(-1, N, 9)
        W = model.head_W(latent).view(-1, 9, N)  # (B, 9, N)
    # per-channel "importance": ||u_k||_2 * ||v_k||_2 * ||w_k||_2 averaged over batch
    u_norm = U.float().norm(dim=2)  # (B, N)
    v_norm = V.float().norm(dim=2)  # (B, N)
    w_norm = W.float().norm(dim=1)  # (B, N)
    importance = (u_norm * v_norm * w_norm).mean(dim=0).cpu().numpy()  # (N,)
    sorted_imp = np.sort(importance)[::-1]
    # effective channel count (participation ratio)
    eff_ch = (importance.sum()**2) / ((importance**2).sum() + 1e-12)
    return sorted_imp, eff_ch, importance

print(f"{'Tag':<6}  {'EffChannels':>12}  {'Top1 imp':>10}  {'Top5 imp%':>12}  {'Dead(<1%)':>10}")
sep()
for tag, (model, N) in models.items():
    si, eff, imp = channel_utilisation(model, N)
    top5_frac = si[:5].sum() / (si.sum() + 1e-12)
    dead = (si < 0.01 * si[0]).sum()
    print(f"{tag:<6}  {eff:>12.1f}  {si[0]:>10.3f}  {top5_frac*100:>11.1f}%  {dead:>10d}")

# Per-channel profile for N11
sep("  N11 per-channel importance (all 11 channels)")
_, _, imp11 = channel_utilisation(models["N11"][0], 11)
sorted_idx = np.argsort(imp11)[::-1]
print(f"  {'Ch':>4}  {'Importance':>12}  {'Fraction%':>10}")
total_imp = imp11.sum()
for i, idx in enumerate(sorted_idx):
    print(f"  {idx:>4}  {imp11[idx]:>12.4f}  {imp11[idx]/total_imp*100:>9.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Error distribution
sep("7. ERROR DISTRIBUTION  (Frobenius relative error over 8192 samples)")

def error_distribution(model):
    with torch.no_grad():
        C_hat = model(A_val, B_val)
    diff = (C_hat.float() - C_val.float()).view(-1, 9)
    scale = C_val.float().view(-1, 9).norm(dim=1).clamp(min=1e-12)
    per_sample = diff.norm(dim=1) / scale
    per_entry  = diff.abs() / scale.unsqueeze(1)
    return per_sample.cpu().numpy(), per_entry.cpu().numpy()

print(f"{'Tag':<6}  {'Mean':>10}  {'Median':>10}  {'p90':>10}  {'p99':>10}  {'Max':>10}")
sep()
err_data = {}
for tag, (model, N) in models.items():
    ps, pe = error_distribution(model)
    err_data[tag] = ps
    print(f"{tag:<6}  {ps.mean():>10.4e}  {np.median(ps):>10.4e}  "
          f"{np.percentile(ps,90):>10.4e}  {np.percentile(ps,99):>10.4e}  {ps.max():>10.4e}")

# Error histogram bucketing
sep("  Error histogram (fraction of samples in each bucket)")
buckets = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, np.inf]
bucket_labels = ["<1e-6","1e-6→5","1e-5→4","1e-4→3","1e-3→2","1e-2→1","1e-1→0",">1"]
print(f"  {'Bucket':<12}", end="")
for tag in [t for t, _, _ in TARGETS]:
    print(f"  {tag:>8}", end="")
print()
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    label = f"{lo:.0e}–{hi:.0e}" if hi < np.inf else f">{lo:.0e}"
    print(f"  {label:<12}", end="")
    for tag in [t for t, _, _ in TARGETS]:
        ps = err_data[tag]
        frac = ((ps >= lo) & (ps < hi)).mean()
        print(f"  {frac*100:>7.1f}%", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Attention pattern analysis
sep("8. ATTENTION ENTROPY  (mean attention entropy per AttnBlock head)")

def attn_entropy(model):
    """
    Hook into each AttnBlock's self-attention to capture attention weights,
    compute per-head entropy (bits), average over batch and tokens.
    """
    from model import AttnBlock
    import math

    attn_weights_store = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, AttnBlock):
            def make_hook(n, mod):
                def hook(m, inp, out):
                    # re-run with need_weights=True
                    x_in = inp[0]
                    batch = x_in.shape[0]
                    tokens = x_in.view(batch, mod.n_tokens, mod.d_token)
                    normed = mod.norm(tokens)
                    _, w = mod.attn(normed, normed, normed, need_weights=True, average_attn_weights=False)
                    # w: (batch, n_heads, n_tokens, n_tokens) — may be None for flash attn
                    if w is not None:
                        attn_weights_store[n] = w.detach().cpu()
                return hook
            h = module.register_forward_hook(make_hook(name, module))
            hooks.append(h)

    with torch.no_grad():
        model._encode(AB_val[:512])

    for h in hooks:
        h.remove()

    results = {}
    for name, w in attn_weights_store.items():
        # w: (batch, n_heads, n_tokens, n_tokens)
        # entropy per head per query token: -sum(p log p)
        p = w.float().clamp(min=1e-9)
        ent = -(p * torch.log2(p)).sum(dim=-1)  # (batch, n_heads, n_tokens)
        results[name] = {
            "mean_entropy": ent.mean().item(),
            "per_head": ent.mean(dim=(0,2)).numpy(),  # (n_heads,)
            "max_entropy": math.log2(w.shape[-1]),
        }
    return results

for tag, (model, N) in models.items():
    sep(f"  [{tag}]  (max entropy = log2(n_tokens) bits)")
    print(f"  Computing attention entropy for {tag}...", flush=True)
    res = attn_entropy(model)
    if not res:
        print("  No attention blocks captured (flash attn / no weights returned).")
        continue
    for block_name, info in res.items():
        max_h = info["max_entropy"]
        mean_h = info["mean_entropy"]
        per_h = info["per_head"]
        print(f"  {block_name}")
        print(f"    Mean entropy: {mean_h:.3f} / {max_h:.3f} bits  ({mean_h/max_h*100:.1f}% of max)")
        heads_str = "  ".join(f"h{i}:{v:.2f}" for i,v in enumerate(per_h))
        print(f"    Per-head: {heads_str}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Encoder representation alignment (CKA)
sep("9. LINEAR CKA BETWEEN ENCODER REPRESENTATIONS")

def gram(X):
    """Centered Gram matrix."""
    X = X - X.mean(0, keepdim=True)
    return X @ X.T

def linear_cka(X, Y):
    """Linear CKA between two (n, d) representation matrices."""
    X = X.float().cpu()
    Y = Y.float().cpu()
    Kx = gram(X)
    Ky = gram(Y)
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-12)).item()

n_cka = 2048
latents_cka = {}
for tag, (model, N) in models.items():
    with torch.no_grad():
        lat = model._encode(AB_val[:n_cka])
    latents_cka[tag] = lat

tags = [t for t, _, _ in TARGETS]
print(f"  {'':>6}", end="")
for t in tags:
    print(f"  {t:>8}", end="")
print()
for ti in tags:
    print(f"  {ti:<6}", end="")
    for tj in tags:
        cka = linear_cka(latents_cka[ti], latents_cka[tj])
        print(f"  {cka:>8.4f}", end="")
    print()

# Also compare intermediate layer representations
sep("  Per-layer CKA: N1 vs N11 vs N81 (encoder trunk layers)")

def get_layer_reps(model, n_samples=2048):
    """Return dict of layer_name → representation (n, d) for encoder trunk."""
    reps = {}
    handles = []
    x_in = AB_val[:n_samples]

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                reps[name] = out.detach().float()
        return hook

    for name, module in model.encoder_blocks.named_modules():
        if isinstance(module, (torch.nn.Linear,)):
            handles.append(module.register_forward_hook(make_hook(f"blocks.{name}")))

    with torch.no_grad():
        model._encode(x_in)
    for h in handles:
        h.remove()
    return reps

rep_n1  = get_layer_reps(models["N1"][0])
rep_n11 = get_layer_reps(models["N11"][0])
rep_n81 = get_layer_reps(models["N81"][0])

# Only compare layers that exist in all three
common_keys = [k for k in rep_n1 if k in rep_n11 and k in rep_n81]
print(f"  {'Layer':<45}  {'N1↔N11':>8}  {'N1↔N81':>8}  {'N11↔N81':>8}")
print(f"  ({len(common_keys)} layers to compare...)", flush=True)
sep()
for k in common_keys:
    r1  = rep_n1[k]
    r11 = rep_n11[k]
    r81 = rep_n81[k]
    # must have same second dim for meaningful CKA; skip mismatches
    if r1.shape[1] != r11.shape[1] or r1.shape[1] != r81.shape[1]:
        print(f"  {k:<45}  {'(dim mismatch)':>26}")
        continue
    c1_11  = linear_cka(r1,  r11)
    c1_81  = linear_cka(r1,  r81)
    c11_81 = linear_cka(r11, r81)
    print(f"  {k:<45}  {c1_11:>8.4f}  {c1_81:>8.4f}  {c11_81:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary
sep("10. SUMMARY")

print("""
  Key dimensions of comparison:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Metric              N=1            N=11           N=81                 │
  └─────────────────────────────────────────────────────────────────────────┘
""")

for tag, (model, N) in models.items():
    sv, eff_dim, t5, t10, t50, iso, L, lat = latent_geometry(model)
    ru, rv, rw, U, V, W = collapse_ratios(model, N)
    si, eff_ch, imp = channel_utilisation(model, N)
    ps, pe = error_distribution(model)
    print(f"  ── {tag} (N={N}) ──")
    print(f"    Best rel err:      {ckpts[tag]['best_rel_err']:.4e}")
    print(f"    Mean val err:      {ps.mean():.4e}")
    print(f"    Latent eff dim:    {eff_dim:.1f} / {lat.shape[1]}")
    print(f"    Latent top-10 var: {t10*100:.1f}%")
    print(f"    Latent isotropy:   {iso:.4e}")
    print(f"    Collapse (U/V/W):  {ru:.3f} / {rv:.3f} / {rw:.3f}")
    print(f"    Channel eff count: {eff_ch:.1f} / {N}")
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Params:            {nparams:.2f}M")
    print()
