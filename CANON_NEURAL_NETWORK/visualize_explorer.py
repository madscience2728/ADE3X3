"""
visualize_explorer.py — Interactive higher-dimensional latent explorer.

Opens a Plotly Dash app in browser. Dropdown selectors let you pick which
3 principal components to project onto (from top 20 PCs), plus color-by
property selector. Rotate/zoom the 3D scatter with mouse.

Usage:
    python visualize_explorer.py
    python visualize_explorer.py --ckpt checkpoints/ckpt_N19_s45.pt
"""

import argparse
import os
import glob

import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from model import KethVaraiMachine
from data import sample_batch


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def build_explorer(ckpt_path, n_samples=20000, n_pcs=20, device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step, best_err = ckpt["step"], ckpt["best_rel_err"]
    state = ckpt["model"]

    # Infer model config from checkpoint
    head_U_w = state["head_U.2.weight"]
    N = head_U_w.shape[0] // 9
    latent_dim = state["head_U.0.weight"].shape[1]
    encoder_width = state["encoder.0.weight"].shape[0]
    block_keys = [k for k in state if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight")]
    encoder_depth = len(block_keys)
    attn_keys = [k for k in state if "attn.in_proj_weight" in k]
    n_attn = len(attn_keys)
    attn_every = encoder_depth // max(n_attn, 1) if n_attn else 0

    print(f"Step={step}, best={best_err:.4e}, N={N}, latent={latent_dim}, width={encoder_width}")

    model = KethVaraiMachine(
        N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
        encoder_width=encoder_width, n_heads=8, attn_every=attn_every,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Collect latent vectors and properties
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

    # Compute properties
    A_svd = np.linalg.svd(A_mat, compute_uv=False)
    B_svd = np.linalg.svd(B_mat, compute_uv=False)
    C_mat = A_mat @ B_mat
    C_svd = np.linalg.svd(C_mat, compute_uv=False)
    BA = B_mat @ A_mat
    C_frob = np.linalg.norm(C_mat.reshape(-1, 9), axis=1)
    comm = np.linalg.norm((C_mat - BA).reshape(-1, 9), axis=1) / np.maximum(C_frob, 1e-12)

    # Channel norms
    with torch.no_grad():
        A_t = torch.tensor(np.concatenate(all_A)[:bs], dtype=torch.float32, device=device)
        B_t = torch.tensor(np.concatenate(all_B)[:bs], dtype=torch.float32, device=device)
        ch_norms = model.channel_norms(A_t, B_t).cpu().numpy()
        total_ch = ch_norms.sum(axis=1)
        top1_frac = ch_norms.max(axis=1) / np.maximum(total_ch, 1e-12)

    # Pad channel props to full dataset size (only computed for first batch)
    total_ch_full = np.full(len(err_np), np.nan)
    top1_frac_full = np.full(len(err_np), np.nan)
    total_ch_full[:len(total_ch)] = total_ch
    top1_frac_full[:len(top1_frac)] = top1_frac

    props = {
        "recon_error": err_np,
        "log10_cond_A": np.log10(A_svd[:, 0] / np.maximum(A_svd[:, -1], 1e-12)),
        "log10_cond_B": np.log10(B_svd[:, 0] / np.maximum(B_svd[:, -1], 1e-12)),
        "log10_cond_C": np.log10(C_svd[:, 0] / np.maximum(C_svd[:, -1], 1e-12)),
        "commutator": comm,
        "det_A": np.linalg.det(A_mat),
        "det_B": np.linalg.det(B_mat),
        "C_frob": C_frob,
        "total_channel_norm": total_ch_full,
        "top1_channel_frac": top1_frac_full,
    }

    # PCA
    scaler = StandardScaler()
    latent_sc = scaler.fit_transform(latent_np)
    pca = PCA(n_components=n_pcs)
    pc_coords = pca.fit_transform(latent_sc)  # (n_samples, n_pcs)
    var_ratios = pca.explained_variance_ratio_

    print(f"PCA variance (top 5): {var_ratios[:5]}")
    print(f"Cumulative: {np.cumsum(var_ratios)[[4, 9, 14, 19]]}")

    # Subsample for browser performance
    n_disp = min(len(err_np), 12000)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(err_np), n_disp, replace=False)

    pc_sub = pc_coords[idx]  # (n_disp, n_pcs)
    props_sub = {k: v[idx] for k, v in props.items()}

    # Build HTML with JavaScript-based axis selectors
    # Pre-embed all PC coordinates and properties as JSON arrays
    import json

    pc_data = {f"PC{i+1}": pc_sub[:, i].tolist() for i in range(n_pcs)}
    prop_data = {}
    for k, v in props_sub.items():
        arr = v.copy()
        arr = np.nan_to_num(arr, nan=0.0)
        vlo, vhi = np.nanpercentile(arr, [2, 98])
        arr = np.clip(arr, vlo, vhi)
        prop_data[k] = arr.tolist()

    pc_labels = [f"PC{i+1} ({var_ratios[i]*100:.1f}%)" for i in range(n_pcs)]

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Latent Explorer — Step {step}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  .controls {{ display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap; align-items: center; }}
  .controls label {{ font-weight: bold; font-size: 14px; }}
  .controls select {{ padding: 5px 10px; font-size: 14px; background: #16213e; color: #e0e0e0;
                       border: 1px solid #0f3460; border-radius: 4px; }}
  h1 {{ color: #e94560; margin-bottom: 5px; }}
  h3 {{ color: #0f3460; margin-top: 0; }}
  #plot {{ width: 100%; height: 80vh; }}
</style>
</head>
<body>
<h1>Latent Explorer — Step {step}, Best={best_err:.3e}</h1>
<h3>N={N}, latent={latent_dim}, width={encoder_width}, depth={encoder_depth} | {n_disp:,} points</h3>

<div class="controls">
  <div><label>X axis:</label><br><select id="sel_x"></select></div>
  <div><label>Y axis:</label><br><select id="sel_y"></select></div>
  <div><label>Z axis:</label><br><select id="sel_z"></select></div>
  <div><label>Color:</label><br><select id="sel_color"></select></div>
</div>

<div id="plot"></div>

<script>
const pcData = {json.dumps(pc_data)};
const propData = {json.dumps(prop_data)};
const pcLabels = {json.dumps(pc_labels)};
const pcKeys = {json.dumps([f"PC{i+1}" for i in range(n_pcs)])};
const propKeys = {json.dumps(list(prop_data.keys()))};

// Populate dropdowns
function populateSelect(id, keys, labels, defaultIdx) {{
  const sel = document.getElementById(id);
  for (let i = 0; i < keys.length; i++) {{
    const opt = document.createElement('option');
    opt.value = keys[i];
    opt.text = labels[i];
    if (i === defaultIdx) opt.selected = true;
    sel.appendChild(opt);
  }}
}}

populateSelect('sel_x', pcKeys, pcLabels, 0);
populateSelect('sel_y', pcKeys, pcLabels, 1);
populateSelect('sel_z', pcKeys, pcLabels, 2);
populateSelect('sel_color', propKeys, propKeys, 0);

function updatePlot() {{
  const xk = document.getElementById('sel_x').value;
  const yk = document.getElementById('sel_y').value;
  const zk = document.getElementById('sel_z').value;
  const ck = document.getElementById('sel_color').value;

  const xlab = pcLabels[pcKeys.indexOf(xk)];
  const ylab = pcLabels[pcKeys.indexOf(yk)];
  const zlab = pcLabels[pcKeys.indexOf(zk)];

  const trace = {{
    x: pcData[xk], y: pcData[yk], z: pcData[zk],
    mode: 'markers',
    type: 'scatter3d',
    marker: {{
      size: 1.5,
      color: propData[ck],
      colorscale: 'Viridis',
      colorbar: {{ title: ck, tickfont: {{ color: '#e0e0e0' }}, titlefont: {{ color: '#e0e0e0' }} }},
      opacity: 0.6,
    }},
    text: propData[ck].map((v, i) => ck + '=' + v.toFixed(4)),
    hoverinfo: 'text',
  }};

  const layout = {{
    scene: {{
      xaxis: {{ title: xlab, color: '#e0e0e0', gridcolor: '#333' }},
      yaxis: {{ title: ylab, color: '#e0e0e0', gridcolor: '#333' }},
      zaxis: {{ title: zlab, color: '#e0e0e0', gridcolor: '#333' }},
      bgcolor: '#16213e',
    }},
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#1a1a2e',
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
  }};

  Plotly.react('plot', [trace], layout);
}}

['sel_x', 'sel_y', 'sel_z', 'sel_color'].forEach(id =>
  document.getElementById(id).addEventListener('change', updatePlot));

updatePlot();
</script>
</body>
</html>"""

    out_path = f"latent_explorer_step{step}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n★ Saved: {out_path}")
    print(f"  Open in browser to explore. Select any 3 of {n_pcs} PCs as axes,")
    print(f"  color by any property. Rotate/zoom with mouse.")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_pcs", type=int, default=20, help="Number of PCA components to keep")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ckpt = a.ckpt or find_latest_checkpoint()
    if not ckpt:
        print("No checkpoint found!"); exit(1)
    print(f"Checkpoint: {ckpt}")
    path = build_explorer(ckpt, a.n_samples, a.n_pcs, a.device)

    # Try to open in browser
    import webbrowser
    webbrowser.open(os.path.abspath(path))
