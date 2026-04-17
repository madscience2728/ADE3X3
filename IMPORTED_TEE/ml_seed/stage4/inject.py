from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import numpy as np

from ml_seed.stage2.model import ConvRanker, get_default_device, require_torch
from ml_seed.stage4.generator import generate_batch

torch, _nn = require_torch()

warnings.filterwarnings(
    "ignore",
    message=r"Attempting to run cuBLAS, but there was no current CUDA context!.*",
    category=UserWarning,
)

DEFAULT_OUT = pathlib.Path("bini_slime/generated_seeds")
DEFAULT_MODEL_PATH = pathlib.Path("ml_seed/data/model_checkpoints/best.pt")


def _warm_up_cuda(device: torch.device) -> None:
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return

    device_index = getattr(device, "index", None)
    if device_index is None:
        device_index = torch.cuda.current_device()

    torch.cuda.set_device(device_index)
    warmup_device = torch.device("cuda", device_index)
    # Force CUDA context creation and initialize the cuBLAS/autograd path
    # before the real generator backward pass runs.
    left = torch.ones((1, 1), device=warmup_device, requires_grad=True)
    right = torch.ones((1, 1), device=warmup_device)
    (torch.mm(left, right).sum()).backward()
    torch.cuda.synchronize(device_index)


def inject(
    n_seeds: int = 10,
    out_dir: pathlib.Path = DEFAULT_OUT,
    model_path: pathlib.Path = DEFAULT_MODEL_PATH,
    n_steps: int = 400,
    lr: float = 0.01,
    noise_scale: float = 0.1,
    ucb_beta: float = 2.0,
    ucb_samples: int = 8,
    grad_noise_scale: float = 0.005,
) -> list[dict]:
    out_dir = pathlib.Path(out_dir)
    model_path = pathlib.Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_default_device()
    _warm_up_cuda(device)

    model = ConvRanker()
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception as exc:
        print(
            f"[inject] WARNING: checkpoint architecture mismatch ({exc}). "
            "Skipping injection — will retry after next training run.",
            flush=True,
        )
        return []
    model.to(device)

    print(
        f"[inject] generating {n_seeds} seeds from {model_path} on {device} ...",
        flush=True,
    )
    batch = generate_batch(
        model,
        n_seeds=n_seeds,
        n_steps=n_steps,
        lr=lr,
        noise_scale=noise_scale,
        ucb_beta=ucb_beta,
        ucb_samples=ucb_samples,
        grad_noise_scale=grad_noise_scale,
    )

    manifest: list[dict] = []
    timestamp = time.time_ns()
    for index, result in enumerate(batch):
        name = f"gen_{timestamp}_{index:02d}"
        npz_path = out_dir / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            C=result["C"],
            image=result["image"],
            score_final=np.array(result["score_final"], dtype=np.float32),
            alpha=np.array(result["alpha"], dtype=np.float32),
            best_step=np.array(result["best_step"], dtype=np.int32),
        )
        entry = {
            "name": name,
            "path": str(npz_path),
            "score_final": float(result["score_final"]),
            "alpha": float(result["alpha"]),
            "best_step": int(result["best_step"]),
        }
        manifest.append(entry)
        print(
            f"  [{index + 1}/{n_seeds}] score={entry['score_final']:.4f} "
            f"alpha={entry['alpha']:.2f} -> {npz_path.name}",
            flush=True,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[inject] wrote {len(manifest)} seeds to {out_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 4 adversarial seeds.")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--ucb-samples", type=int, default=8)
    parser.add_argument("--grad-noise-scale", type=float, default=0.005)
    args = parser.parse_args()
    inject(
        n_seeds=args.n_seeds,
        out_dir=pathlib.Path(args.out),
        model_path=pathlib.Path(args.model),
        n_steps=args.steps,
        lr=args.lr,
        noise_scale=args.noise_scale,
        ucb_beta=args.ucb_beta,
        ucb_samples=args.ucb_samples,
        grad_noise_scale=args.grad_noise_scale,
    )


if __name__ == "__main__":
    main()