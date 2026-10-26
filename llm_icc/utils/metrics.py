from __future__ import annotations

import numpy as np


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


def bias(pred, ref) -> float:
    """Mean signed error: mean(pred - ref)."""
    return float(np.mean(np.asarray(pred, float) - np.asarray(ref, float)))


_BOOT_STATS = {
    "rmse": lambda p, r: np.sqrt(np.mean((p - r) ** 2, axis=-1)),
    "bias": lambda p, r: np.mean(p - r, axis=-1),
}


def bootstrap_ci(pred, ref, stat: str = "rmse", n_boot: int = 10_000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat` ('rmse' | 'bias'), resampling (pred, ref) pairs, i.e. items, with replacement.

    Only item sampling is resampled; the model is not retrained. The same seed gives the same resampled item indices
    for every statistic and model, so intervals are paired across calls on equal-length inputs.
    """
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    n = len(pred)
    if n < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    vals = _BOOT_STATS[stat](pred[idx], ref[idx])
    lo, hi = np.percentile(vals, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def weighted_mse(pred, target, weights) -> float:
    """pred/target (..., N, n) or (N, n); weights (N,). omega-weighted sum over options, per item mean."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    w = np.asarray(weights, float)
    sq = ((pred - target) ** 2).sum(-1)  # (..., N)
    return float((sq * w).sum(-1).mean() / w.sum())
