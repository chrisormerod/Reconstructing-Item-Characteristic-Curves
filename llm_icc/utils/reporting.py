"""ICC plots (paper Fig. 3 style) and summary tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_icc(item, theta_k, target: np.ndarray | None, pred: np.ndarray, path: Path, title: str = "") -> None:
    """Correct-option curve (target vs LLM) with the distractors drawn thin."""
    c = item.correct_index
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if target is not None:
        ax.plot(theta_k, target[:, c], "-", color="0.25", lw=2, label="target (correct)")
    ax.plot(theta_k, pred[:, c], "o-", color="#1f77b4", lw=2, ms=4, label="LLM (correct)")
    for i in range(item.n_options):
        if i != c:
            ax.plot(theta_k, pred[:, i], "-", color="#d62728", lw=0.8, alpha=0.5,
                    label="LLM (distractors)" if i == (0 if c != 0 else 1) else None)
    ax.set_xlabel("ability θ (descriptor mean)")
    ax.set_ylabel("option probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or f"item {item.item_id}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(pred, true, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(true, pred, s=14, alpha=0.7)
    lo, hi = min(np.min(true), np.min(pred)), max(np.max(true), np.max(pred))
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    ax.set_xlabel("reference Difficulty")
    ax.set_ylabel("predicted Difficulty")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_table(metrics_by_split: dict[str, dict]) -> pd.DataFrame:
    """Rows = splits (+ mean/std across splits); columns = flattened test metrics."""
    rows = {name: {k: v for k, v in m["test"].items() if isinstance(v, (int, float))}
            for name, m in metrics_by_split.items()}
    df = pd.DataFrame(rows).T
    if len(df) > 1:
        df.loc["mean"] = df.mean()
        df.loc["std"] = df.iloc[:-1].std()
    return df
