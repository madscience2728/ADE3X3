from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
from threading import Event

from ml_seed.stage3.status_bus import existing_sample_files, read as read_status, sync_model_status_from_deployment, write_dataset_status, write_section

RETRAIN_EVERY = 50
POLL_INTERVAL = 30
EPOCHS = 30
BATCH_SIZE = 512
MARGIN = 0.1
DROPOUT_P = 0.2
_SHUTDOWN_REQUESTED = Event()
_TRAINER_PROCESS: subprocess.Popen | None = None


def _install_signal_handlers() -> None:
    def _request_shutdown(signum, _frame) -> None:
        if not _SHUTDOWN_REQUESTED.is_set():
            print(f"[watcher] received signal {signum}; shutting down", flush=True)
        _SHUTDOWN_REQUESTED.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_shutdown)


def _spawn_process(command: list[str]) -> subprocess.Popen:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def _terminate_trainer() -> None:
    global _TRAINER_PROCESS
    process = _TRAINER_PROCESS
    if process is None or process.poll() is not None:
        return
    print(f"[watcher] stopping trainer pid={process.pid}", flush=True)
    try:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                process.terminate()
        else:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[watcher] force-killing trainer", flush=True)
        process.kill()
        process.wait(timeout=5)
    finally:
        _TRAINER_PROCESS = None


def _write_idle_watcher(last_duration: float | None = None) -> None:
    payload = {
        "running": not _SHUTDOWN_REQUESTED.is_set(),
        "samples_since_last_train": 0,
        "retrain_threshold": RETRAIN_EVERY,
        "next_retrain_at": None,
        "currently_training": False,
        "last_train_duration_s": _get_last_duration() if last_duration is None else last_duration,
        "training_epoch": None,
        "training_epochs_total": None,
        "training_progress": None,
        "training_train_loss": None,
        "training_val_loss": None,
        "training_tau": None,
        "training_last_known_tau": None,
    }
    write_section("watcher", payload)


def count_samples(sample_files: list[str]) -> int:
    return write_dataset_status(sample_files)["combined"]


def main() -> None:
    _install_signal_handlers()
    print("[watcher] starting", flush=True)
    sync_model_status_from_deployment()
    last_train_count = _get_last_train_count()
    last_duration = _get_last_duration()
    print(f"[watcher] last train was at n={last_train_count}", flush=True)

    try:
        while not _SHUTDOWN_REQUESTED.is_set():
            sample_files = existing_sample_files()
            n_samples = count_samples(sample_files)
            since_last = max(0, n_samples - last_train_count)
            next_at = last_train_count + RETRAIN_EVERY
            write_section(
                "watcher",
                {
                    "running": True,
                    "samples_since_last_train": since_last,
                    "retrain_threshold": RETRAIN_EVERY,
                    "next_retrain_at": next_at,
                    "currently_training": False,
                    "last_train_duration_s": last_duration,
                    "training_epoch": None,
                    "training_epochs_total": None,
                    "training_progress": None,
                    "training_train_loss": None,
                    "training_val_loss": None,
                    "training_tau": None,
                    "training_last_known_tau": None,
                },
            )

            if since_last >= RETRAIN_EVERY:
                print(
                    f"[watcher] {n_samples} samples (+{since_last} since last train) — retraining",
                    flush=True,
                )
                last_duration, completed = _run_trainer(n_samples, sample_files)
                if completed:
                    last_train_count = n_samples
                if _SHUTDOWN_REQUESTED.is_set():
                    break
            else:
                print(
                    f"[watcher] {n_samples} samples — {RETRAIN_EVERY - since_last} until retrain",
                    flush=True,
                )

            _SHUTDOWN_REQUESTED.wait(POLL_INTERVAL)
    finally:
        _terminate_trainer()
        _write_idle_watcher(last_duration)


def _run_trainer(n_samples: int, sample_files: list[str]) -> tuple[float, bool]:
    global _TRAINER_PROCESS
    write_section(
        "watcher",
        {
            "running": True,
            "samples_since_last_train": 0,
            "retrain_threshold": RETRAIN_EVERY,
            "next_retrain_at": n_samples + RETRAIN_EVERY,
            "currently_training": True,
            "last_train_duration_s": _get_last_duration(),
            "training_epoch": 0,
            "training_epochs_total": EPOCHS,
            "training_progress": 0.0,
            "training_train_loss": None,
            "training_val_loss": None,
            "training_tau": None,
            "training_last_known_tau": None,
            "training_started_at": time.time(),
        },
    )
    started_at = time.time()
    cmd = [
        sys.executable,
        "-m",
        "ml_seed.stage3.trainer_proc",
        "--out-pt",
        "ml_seed/data/model_checkpoints/candidate.pt",
        "--samples",
        *[path for path in sample_files if pathlib.Path(path).exists()],
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--margin",
        str(MARGIN),
        "--dropout-p",
        str(DROPOUT_P),
        "--n-samples",
        str(n_samples),
    ]
    _TRAINER_PROCESS = _spawn_process(cmd)
    while _TRAINER_PROCESS.poll() is None and not _SHUTDOWN_REQUESTED.wait(1.0):
        pass

    if _SHUTDOWN_REQUESTED.is_set() and _TRAINER_PROCESS.poll() is None:
        _terminate_trainer()
        duration = time.time() - started_at
        print(f"[watcher] training interrupted after {duration:.1f}s", flush=True)
        return round(duration, 1), False

    return_code = 0 if _TRAINER_PROCESS is None else int(_TRAINER_PROCESS.returncode or 0)
    _TRAINER_PROCESS = None
    duration = time.time() - started_at
    print(
        f"[watcher] training done in {duration:.1f}s — exit code {return_code}",
        flush=True,
    )
    write_section(
        "watcher",
        {
            "running": True,
            "samples_since_last_train": 0,
            "retrain_threshold": RETRAIN_EVERY,
            "next_retrain_at": n_samples + RETRAIN_EVERY,
            "currently_training": False,
            "last_train_duration_s": round(duration, 1),
            "training_epoch": None,
            "training_epochs_total": None,
            "training_progress": None,
            "training_train_loss": None,
            "training_val_loss": None,
            "training_tau": None,
            "training_last_known_tau": None,
        },
    )
    return round(duration, 1), return_code == 0


def _get_last_train_count() -> int:
    status = read_status()
    return int(status.get("model", {}).get("n_samples_at_train", 0) or 0)


def _get_last_duration() -> float:
    history = read_status().get("history", [])
    if history:
        return float(history[-1].get("duration_s", 0.0) or 0.0)
    return 0.0


if __name__ == "__main__":
    main()