from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: str, run_name: str) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("llm_icc")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(Path(logs_dir) / f"{run_name}.log", maxBytes=10_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    return root


def make_tracker(run_cfg, run_dir: Path):
    """Optional TensorBoard / W&B sink: returns `fn(metrics, step)` or None."""
    sinks = []
    if run_cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(str(run_dir / "tb"))
        sinks.append(lambda m, s: [w.add_scalar(k, v, s) for k, v in m.items()])
    if run_cfg.wandb:
        import wandb
        wandb.init(project="llm-icc", name=run_cfg.name, dir=str(run_dir))
        sinks.append(lambda m, s: wandb.log(m, step=s))
    if not sinks:
        return None
    return lambda m, s: [f(m, s) for f in sinks]
