"""plot_lambda.py — Visualise the residual landscape from results.jsonl.

Usage:
    python -m fusion_search.plot_lambda [--out fusion_search/results.jsonl]
                                        [--save fusion_search/lambda_landscape.png]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent


def _load(path: pathlib.Path) -> tuple[list[float], list[float], list[int]]:
    lambdas, residuals, ranks = [], [], []
    if not path.exists():
        return lambdas, residuals, ranks
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                lambdas.append(float(rec["lambda"]))
                residuals.append(float(rec["residual"]))
                ranks.append(int(rec["rank_estimate"]))
            except (json.JSONDecodeError, KeyError):
                continue
    # Sort by lambda
    order = sorted(range(len(lambdas)), key=lambda i: lambdas[i])
    lambdas   = [lambdas[i]   for i in order]
    residuals = [residuals[i] for i in order]
    ranks     = [ranks[i]     for i in order]
    return lambdas, residuals, ranks


def plot(
    src: str = str(_HERE / "results.jsonl"),
    save: str | None = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    lambdas, residuals, ranks = _load(pathlib.Path(src))
    if not lambdas:
        print(f"No data found in {src}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "T₃₃₃ rank landscape via scalar curvature fusion law", fontsize=14
    )

    # ---- top panel: residual vs λ ----
    ax = axes[0]
    # colour by rank
    rank_vals  = sorted(set(ranks))
    cmap       = cm.get_cmap("viridis", len(rank_vals))
    rank_to_c  = {r: cmap(i) for i, r in enumerate(rank_vals)}
    colours    = [rank_to_c[r] for r in ranks]

    ax.scatter(lambdas, residuals, c=colours, s=20, zorder=3, label="_nolegend_")
    ax.plot(lambdas, residuals, linewidth=0.6, color="grey", alpha=0.5, zorder=2)

    # Mark best point
    best_i = int(min(range(len(residuals)), key=lambda i: residuals[i]))
    ax.scatter(
        [lambdas[best_i]], [residuals[best_i]],
        marker="*", s=200, zorder=5, color="red",
        label=f"best  λ={lambdas[best_i]:+.4f}  res={residuals[best_i]:.3e}",
    )
    ax.axvline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.4,
               label="λ=0  (flat)")
    ax.set_ylabel("ALS residual  ‖T₃₃₃ − T̂‖")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Colourbar for rank
    norm = mcolors.BoundaryNorm(
        boundaries=[r - 0.5 for r in rank_vals] + [rank_vals[-1] + 0.5],
        ncolors=len(rank_vals),
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01)
    cb.set_label("rank estimate", fontsize=8)
    cb.set_ticks(rank_vals)

    # ---- bottom panel: rank estimate vs λ ----
    ax2 = axes[1]
    ax2.scatter(lambdas, ranks, c=colours, s=20, zorder=3)
    ax2.plot(lambdas, ranks, linewidth=0.6, color="grey", alpha=0.5, zorder=2)
    ax2.axvline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.4)
    ax2.set_xlabel("λ  (scalar curvature parameter)")
    ax2.set_ylabel("rank estimate")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.grid(True, which="major", alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save}")

    if show:
        plt.show()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot λ residual landscape")
    p.add_argument("--out",  type=str, default=str(_HERE / "results.jsonl"),
                   help="Path to results.jsonl")
    p.add_argument("--save", type=str, default=None,
                   help="Save figure to this path (PNG/PDF/SVG)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful for headless runs)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    plot(src=args.out, save=args.save, show=not args.no_show)
