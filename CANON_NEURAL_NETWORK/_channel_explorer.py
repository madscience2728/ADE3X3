"""Interactive 3D channel explorer — select any channel for X/Y/Z axis."""
import torch
import numpy as np
import plotly.graph_objects as go
import argparse
from model import KethVaraiMachine
from data import sample_batch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/ckpt_N19_s0.pt")
parser.add_argument("--n_samples", type=int, default=12000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
state = ckpt["model"]
step, best_err = ckpt["step"], ckpt["best_rel_err"]

# Infer arch
if "head_U.4.weight" in state:
    N = state["head_U.4.weight"].shape[0] // 9
    latent_dim = state["head_U.0.weight"].shape[1]
else:
    N = state["head_U.weight"].shape[0] // 9
    latent_dim = state["head_U.weight"].shape[1]
encoder_width = state["encoder.0.weight"].shape[0]
block_keys = [k for k in state if k.startswith("encoder_blocks.") and k.endswith(".net.1.weight")]
encoder_depth = len(block_keys)
attn_keys = [k for k in state if "attn.in_proj_weight" in k]
n_attn = len(attn_keys)
attn_every = encoder_depth // max(n_attn, 1) if n_attn else 0
n_tokens = 81
d_token = encoder_width // n_tokens
n_heads = 4
for c in [8, 4, 2, 1]:
    if d_token % c == 0:
        n_heads = c
        break

linear_heads = "bias_U" in state  # legacy compat
model = KethVaraiMachine(N=N, latent_dim=latent_dim, encoder_depth=encoder_depth,
                         encoder_width=encoder_width, n_heads=n_heads, attn_every=attn_every,
                         linear_heads=linear_heads).to(device)
model.load_state_dict(state)
model.eval()
print(f"N={N}, step={step}, best={best_err:.4e}")

N_SAMPLES = args.n_samples
A, B, C = sample_batch(N_SAMPLES, device)

with torch.no_grad():
    x = torch.cat([A, B], dim=1)
    latent = model._encode(x).float()
    U = model.head_U(latent).view(-1, N, 9)
    V = model.head_V(latent).view(-1, N, 9)
    W = model.head_W(latent).view(-1, 9, N)
        W = model.head_W(latent).view(-1, 9, N)
    C_hat = model(A, B)
    
    p = (A.unsqueeze(1) * U).sum(dim=2)  # (batch, N)
    q = (B.unsqueeze(1) * V).sum(dim=2)
    m = p * q  # channel activations
    
    err = (C_hat - C).view(-1, 9).norm(dim=1) / C.view(-1, 9).norm(dim=1).clamp(min=1e-12)

m_np = m.cpu().numpy()  # (samples, N)
err_np = err.cpu().numpy()
latent_np = latent.cpu().numpy()

# Also compute latent PCs
from sklearn.decomposition import PCA
pca = PCA(n_components=min(10, latent_np.shape[1]))
lat_pcs = pca.fit_transform(latent_np)

# Channel importance order
ch_importance = np.abs(m_np).mean(axis=0)
ch_order = np.argsort(-ch_importance)

# Build axis options: channels + latent PCs
axis_names = [f"ch{k} (|m|={ch_importance[k]:.2f})" for k in range(N)]
axis_names += [f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)" for i in range(min(10, lat_pcs.shape[1]))]

# All data columns
all_data = np.hstack([m_np, lat_pcs])  # (samples, N + n_pcs)
n_axes = all_data.shape[1]

# Color options
A3 = A.cpu().numpy().reshape(-1, 3, 3)
B3 = B.cpu().numpy().reshape(-1, 3, 3)
C3 = (A3 @ B3)
detA = np.linalg.det(A3)
detC = np.linalg.det(C3)
normC = np.linalg.norm(C3.reshape(-1, 9), axis=1)
comm = np.linalg.norm((C3 - B3 @ A3).reshape(-1, 9), axis=1) / np.maximum(normC, 1e-12)

color_options = {
    "Recon Error": err_np,
    "log₁₀(error)": np.log10(err_np + 1e-15),
    "det(A)": detA,
    "det(C)": detC,
    "||C||": normC,
    "Commutator": comm,
}
for k in ch_order[:6]:
    color_options[f"ch{k} activation"] = m_np[:, k]

# Subsample for performance
MAX_POINTS = 8000
if N_SAMPLES > MAX_POINTS:
    idx = np.random.choice(N_SAMPLES, MAX_POINTS, replace=False)
    all_data = all_data[idx]
    for key in color_options:
        color_options[key] = color_options[key][idx]
    n_pts = MAX_POINTS
else:
    n_pts = N_SAMPLES

# Default axes: top 3 channels by importance
default_x, default_y, default_z = ch_order[0], ch_order[1], ch_order[2]

# Build the HTML with JavaScript dropdowns
# Pre-serialize all data as JSON arrays
import json

data_json = json.dumps(all_data.tolist())
axis_names_json = json.dumps(axis_names)
color_data = {k: v.tolist() for k, v in color_options.items()}
color_json = json.dumps(color_data)
color_names = list(color_options.keys())
color_names_json = json.dumps(color_names)

html = f"""<!DOCTYPE html>
<html>
<head>
<title>Channel Explorer — N={N}, step {step}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 10px; background: #1a1a2e; color: #eee; }}
.controls {{ display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; padding: 10px; background: #16213e; border-radius: 8px; }}
.controls label {{ font-size: 13px; color: #aaa; }}
.controls select {{ background: #0f3460; color: #eee; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 13px; }}
h2 {{ margin: 5px 0; color: #e94560; }}
.info {{ font-size: 12px; color: #888; margin-bottom: 5px; }}
</style>
</head>
<body>
<h2>Channel Manifold Explorer — N={N}, step {step}, best={best_err:.3e}</h2>
<div class="info">{n_pts} samples · Select channels or latent PCs for each axis · Color by property</div>
<div class="controls">
  <div><label>X axis:</label><br><select id="sel_x" onchange="updatePlot()"></select></div>
  <div><label>Y axis:</label><br><select id="sel_y" onchange="updatePlot()"></select></div>
  <div><label>Z axis:</label><br><select id="sel_z" onchange="updatePlot()"></select></div>
  <div><label>Color:</label><br><select id="sel_color" onchange="updatePlot()"></select></div>
  <div><label>Point size:</label><br><input type="range" id="pt_size" min="1" max="8" value="2" oninput="updatePlot()"></div>
</div>
<div id="plot" style="width:100%; height:85vh;"></div>

<script>
const allData = {data_json};
const axisNames = {axis_names_json};
const colorData = {color_json};
const colorNames = {color_names_json};

// Populate dropdowns
const selX = document.getElementById('sel_x');
const selY = document.getElementById('sel_y');
const selZ = document.getElementById('sel_z');
const selColor = document.getElementById('sel_color');

axisNames.forEach((name, i) => {{
    selX.add(new Option(name, i));
    selY.add(new Option(name, i));
    selZ.add(new Option(name, i));
}});
selX.value = {default_x};
selY.value = {default_y};
selZ.value = {default_z};

colorNames.forEach(name => {{
    selColor.add(new Option(name, name));
}});
selColor.value = "log₁₀(error)";

function updatePlot() {{
    const xi = parseInt(selX.value);
    const yi = parseInt(selY.value);
    const zi = parseInt(selZ.value);
    const cname = selColor.value;
    const ptSize = parseInt(document.getElementById('pt_size').value);
    
    const x = allData.map(r => r[xi]);
    const y = allData.map(r => r[yi]);
    const z = allData.map(r => r[zi]);
    const c = colorData[cname];
    
    const trace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: x, y: y, z: z,
        marker: {{
            size: ptSize,
            color: c,
            colorscale: 'RdBu_r',
            colorbar: {{ title: cname, thickness: 15 }},
            opacity: 0.6,
            reversescale: cname.includes('error'),
        }},
        hovertemplate: axisNames[xi] + ': %{{x:.3f}}<br>' + axisNames[yi] + ': %{{y:.3f}}<br>' + axisNames[zi] + ': %{{z:.3f}}<br>' + cname + ': %{{marker.color:.4f}}<extra></extra>',
    }};
    
    const layout = {{
        scene: {{
            xaxis: {{ title: axisNames[xi], color: '#aaa', gridcolor: '#333' }},
            yaxis: {{ title: axisNames[yi], color: '#aaa', gridcolor: '#333' }},
            zaxis: {{ title: axisNames[zi], color: '#aaa', gridcolor: '#333' }},
            bgcolor: '#0a0a1a',
        }},
        paper_bgcolor: '#1a1a2e',
        margin: {{ l: 0, r: 0, t: 0, b: 0 }},
    }};
    
    Plotly.react('plot', [trace], layout);
}}

updatePlot();
</script>
</body>
</html>"""

outpath = f"latent3d_channel_explorer_step{step}.html"
with open(outpath, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved: {outpath}")
