from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
import time
from threading import Event

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ml_seed.stage2.train import train
from ml_seed.stage3.atomic_swap import swap_model
from ml_seed.stage3.status_bus import append_history, write_dataset_status, write_section

_STOP_REQUESTED = Event()


def _install_signal_handlers() -> None:
    def _request_stop(signum, _frame) -> None:
        if not _STOP_REQUESTED.is_set():
            print(f"[trainer] received signal {signum}; stopping", flush=True)
        _STOP_REQUESTED.set()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)


def _write_training_progress(progress: dict) -> None:
    from ml_seed.stage3.status_bus import read

    watcher = dict(read().get("watcher", {}))
    watcher.update(
        {
            "currently_training": True,
            "training_epoch": progress["epoch"],
            "training_epochs_total": progress["epochs"],
            "training_progress": progress["epoch"] / progress["epochs"],
            "training_train_loss": progress["train_loss"],
            "training_val_loss": progress["val_loss"],
            "training_tau": progress["kendall_tau"],
            "training_last_known_tau": progress["last_known_tau"],
            "training_updated_at": time.time(),
        }
    )
    write_section("watcher", watcher)


def _clear_training_progress(duration: float) -> None:
    from ml_seed.stage3.status_bus import read

    watcher = dict(read().get("watcher", {}))
    watcher.update(
        {
            "currently_training": False,
            "last_train_duration_s": round(duration, 1),
            "training_epoch": None,
            "training_epochs_total": None,
            "training_progress": None,
            "training_train_loss": None,
            "training_val_loss": None,
            "training_tau": None,
            "training_last_known_tau": None,
            "training_updated_at": time.time(),
        }
    )
    write_section("watcher", watcher)


def main() -> None:
    _install_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=0)
    args = parser.parse_args()

    candidate_path = pathlib.Path(args.out_pt)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    result = train(
        samples=args.samples,
        out=str(candidate_path.parent),
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        dropout_p=args.dropout_p,
        candidate_name=candidate_path.name,
        progress_callback=_write_training_progress,
    )
    duration = time.time() - started_at

    deployed = bool(result["deployable"])
    if deployed:
        deployed = swap_model(candidate_path)

    write_dataset_status(args.samples)
    write_section(
        "model",
        {
            "kendall_tau": result["kendall_tau"],
            "val_loss": result["val_loss"],
            "deployed": deployed,
            "n_samples_at_train": args.n_samples,
            "trained_at": time.time(),
            "checkpoint_path": (
                "ml_seed/data/model_checkpoints/best.pt"
                if deployed
                else str(candidate_path).replace("\\", "/")
            ),
        },
    )
    append_history(
        {
            "ts": time.time(),
            "n_samples": args.n_samples,
            "kendall_tau": result["kendall_tau"],
            "val_loss": result["val_loss"],
            "deployed": deployed,
            "duration_s": round(duration, 1),
        }
    )
    _clear_training_progress(duration)
    print(
        json.dumps(
            {
                "tau": result["kendall_tau"],
                "deployed": deployed,
                "duration_s": round(duration, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()