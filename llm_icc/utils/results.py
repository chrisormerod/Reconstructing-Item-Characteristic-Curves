"""Write/read result tables. Long format: one row per (item, descriptor, option)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def icc_table(item, matrix: np.ndarray, labels: list[str], theta_k: np.ndarray, omega_k: np.ndarray,
              value_name: str) -> pd.DataFrame:
    N, n = matrix.shape
    return pd.DataFrame({
        "item_id": item.item_id,
        "descriptor_index": np.repeat(np.arange(N), n),
        "descriptor": np.repeat(labels, n),
        "theta_k": np.repeat(theta_k, n),
        "omega_k": np.repeat(omega_k, n),
        "option": np.tile([o.key for o in item.options], N),
        "is_correct": np.tile([o.is_correct for o in item.options], N),
        value_name: matrix.reshape(-1),
    })


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_config(config, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="python")
    data["descriptors"] = {k: list(v) for k, v in data["descriptors"].items()}
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
