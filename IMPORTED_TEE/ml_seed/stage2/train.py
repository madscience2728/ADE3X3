from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from ml_seed.stage2.data_prep import PairDataset, dataset_summary
from ml_seed.stage2.evaluate import kendall_tau
from ml_seed.stage2.loss import bradley_terry_loss
from ml_seed.stage2.model import ConvRanker, get_default_device, require_torch


def _write_deployment_metadata(out_dir: pathlib.Path, payload: dict) -> None:
    (out_dir / "deployment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_dataset_summary(summary: dict, margin: float) -> None:
    histogram = summary["residual_histogram"]
    pair_summary = summary["pair_summary"]
    print(f"Samples: {summary['n_samples']} | ranks: {summary['rank_counts']}")
    if histogram["counts"]:
        print(
            "Residual summary: "
            f"min={histogram['min']:.4f} max={histogram['max']:.4f} "
            f"mean={histogram['mean']:.4f} std={histogram['std']:.4f}"
        )
        print(f"Residual histogram counts: {histogram['counts']}")
    usable_fraction = pair_summary["usable_fraction"]
    print(
        f"Pair summary @ margin={margin:.3f}: usable {pair_summary['usable_pairs']} / "
        f"{pair_summary['total_pairs']} ({usable_fraction:.1%})"
    )
    if pair_summary["total_pairs"] and usable_fraction < 0.25:
        print(
            "WARNING: margin discarded most pairs. Inspect residual histogram and consider lowering --margin.")


def train(
    samples: list[str],
    out: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    val_frac: float = 0.15,
    margin: float = 0.1,
    dropout_p: float = 0.2,
    tau_threshold: float = 0.6,
    max_pairs_per_rank: int = 10_000,
    min_samples_for_deploy: int = 300,
    candidate_name: str = "best.pt",
    warm_start: bool = True,
    warm_start_path: str | None = None,
    progress_callback=None,
) -> dict:
    torch, _nn = require_torch()
    from torch.utils.data import DataLoader, random_split
    device = get_default_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dataset_summary(samples, margin=margin, max_pairs_per_rank=max_pairs_per_rank)
    _print_dataset_summary(summary, margin)

    dataset = PairDataset(samples, max_pairs_per_rank=max_pairs_per_rank)
    if len(dataset) == 0:
        raise RuntimeError("No same-rank pairs available for Stage 2 training.")

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise RuntimeError("Not enough pairs to create a non-empty training split.")
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    pin_memory = device.type == "cuda"
    num_workers = min(4, os.cpu_count() or 1) if device.type == "cuda" else 0
    _dl_kwargs: dict = dict(
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **_dl_kwargs,
    )

    warm_start_used = False
    warm_start_source = None
    checkpoint_path = out_dir / candidate_name
    model = ConvRanker(dropout_p=dropout_p).to(device)
    if warm_start:
        source_path = pathlib.Path(warm_start_path) if warm_start_path else (out_dir / "best.pt")
        if source_path.exists():
            try:
                state = torch.load(source_path, map_location=device)
                model.load_state_dict(state)
                warm_start_used = True
                warm_start_source = str(source_path).replace("\\", "/")
                print(f"Warm-starting from {warm_start_source}")
            except Exception as exc:
                print(
                    f"WARNING: warm start from {source_path} failed ({exc}); training from scratch.",
                    flush=True,
                )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_loss = float("inf")
    best_state = None

    print(f"Training on {n_train} pairs, validating on {n_val} pairs")
    print(f"Dataset total: {len(dataset)} pairs across ranks {dataset.rank_summary}")
    print(f"Using device: {device}")

    last_epoch_tau = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for image_a, image_b, res_a, res_b, _ranks in train_dl:
            image_a = image_a.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            image_b = image_b.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            res_a = res_a.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            res_b = res_b.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                score_a = model(image_a).squeeze(-1)
                score_b = model(image_b).squeeze(-1)
                loss = bradley_terry_loss(score_a, score_b, res_a, res_b, margin=margin)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss_total += float(loss.item()) * len(image_a)
        train_loss = train_loss_total / n_train
        sched.step()

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for image_a, image_b, res_a, res_b, _ranks in val_dl:
                image_a = image_a.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                image_b = image_b.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                res_a = res_a.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                res_b = res_b.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
                score_a = model(image_a).squeeze(-1)
                score_b = model(image_b).squeeze(-1)
                loss = bradley_terry_loss(score_a, score_b, res_a, res_b, margin=margin)
                val_loss_total += float(loss.item()) * len(image_a)
        val_loss = val_loss_total / n_val

        epoch_tau = None
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            epoch_tau = kendall_tau(model, val_dl)
            last_epoch_tau = epoch_tau
            print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | tau={epoch_tau:.3f}")

        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "kendall_tau": None if epoch_tau is None else float(epoch_tau),
                    "last_known_tau": None if last_epoch_tau is None else float(last_epoch_tau),
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint state.")

    checkpoint_path = out_dir / candidate_name
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    model.eval()
    tau = kendall_tau(model, val_dl)
    deployment = {
        "trained_at": time.time(),
        "samples": samples,
        "n_samples": summary["n_samples"],
        "n_pairs": len(dataset),
        "n_train_pairs": n_train,
        "n_val_pairs": n_val,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "val_fraction": val_frac,
        "margin": margin,
        "dropout_p": dropout_p,
        "tau_threshold": tau_threshold,
        "min_samples_for_deploy": min_samples_for_deploy,
        "kendall_tau": tau,
        "val_loss": best_val_loss,
        "deployable": bool(
            tau >= tau_threshold and summary["n_samples"] >= min_samples_for_deploy
        ),
        "best_val_loss": best_val_loss,
        "checkpoint_name": candidate_name,
        "checkpoint_path": str(checkpoint_path),
        "warm_start": warm_start,
        "warm_start_used": warm_start_used,
        "warm_start_path": warm_start_source,
        "dataset_summary": summary,
    }
    _write_deployment_metadata(out_dir, deployment)

    print(f"\nBest val loss: {best_val_loss:.4f} | kendall_tau={tau:.3f}")
    if tau >= tau_threshold and summary["n_samples"] >= min_samples_for_deploy:
        print(f"Deployment gate passed (tau >= {tau_threshold:.2f}).")
    else:
        print(
            "Deployment gate blocked "
            f"(tau={tau:.3f}, samples={summary['n_samples']}, "
            f"min_samples={min_samples_for_deploy}); inference remains neutral."
        )
    print(f"Artifacts written to {out_dir}")
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=["ml_seed/data/samples.jsonl", "ml_seed/data/bootstrap.jsonl"])
    parser.add_argument("--out", default="ml_seed/data/model_checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--tau-threshold", type=float, default=0.6)
    parser.add_argument("--max-pairs-per-rank", type=int, default=10_000)
    parser.add_argument("--min-samples-for-deploy", type=int, default=300)
    parser.add_argument("--candidate-name", default="best.pt")
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--warm-start-path", default=None)
    args = parser.parse_args()
    train(
        samples=args.samples,
        out=args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        margin=args.margin,
        dropout_p=args.dropout_p,
        tau_threshold=args.tau_threshold,
        max_pairs_per_rank=args.max_pairs_per_rank,
        min_samples_for_deploy=args.min_samples_for_deploy,
        candidate_name=args.candidate_name,
        warm_start=not args.cold_start,
        warm_start_path=args.warm_start_path,
    )


if __name__ == "__main__":
    main()