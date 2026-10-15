"""YAML loading with `base:` inheritance, CLI overrides and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from llm_icc.config.schema import Config


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "descriptors":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base:
        raw = _deep_merge(_load_raw((path.parent / base).resolve()), raw)
    return raw


def apply_overrides(raw: dict, overrides: Iterable[str]) -> dict:
    """`--set training.learning_rate=1e-5` -> nested assignment; values parsed as YAML."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed: Any = yaml.safe_load(value)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return raw


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    raw = apply_overrides(_load_raw(Path(path).resolve()), overrides)
    return Config.model_validate(raw)
