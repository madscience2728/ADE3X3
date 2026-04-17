from __future__ import annotations

import os
import pathlib
import shutil

CHECKPOINT_DIR = pathlib.Path("ml_seed/data/model_checkpoints")


def swap_model(new_pt_path: pathlib.Path) -> bool:
    dest = CHECKPOINT_DIR / "best.pt"
    bak = CHECKPOINT_DIR / "best.pt.bak"
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.copy2(dest, bak)
        os.replace(new_pt_path, dest)
        return True
    except Exception as exc:
        print(f"[atomic_swap] failed: {exc}", flush=True)
        return False