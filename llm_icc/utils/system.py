from __future__ import annotations

import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_memory_report() -> str:
    if not torch.cuda.is_available():
        return "no CUDA device"
    i = torch.cuda.current_device()
    return (f"{torch.cuda.get_device_name(i)}: allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB, "
            f"peak {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB, "
            f"total {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GiB")


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def make_run_dir(results_dir: str, name: str, timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(results_dir) / name / ts
    os.makedirs(path, exist_ok=True)
    return path
