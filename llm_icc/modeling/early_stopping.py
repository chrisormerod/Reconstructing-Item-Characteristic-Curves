from __future__ import annotations

import math


class EarlyStopping:
    """Tracks the best dev metric (lower is better) and the matching adapter state."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.best_step = -1
        self.best_state = None
        self.bad_evals = 0

    def update(self, metric: float, step: int, state_fn) -> bool:
        """Record an evaluation. Returns True if it is a new best. `state_fn()` snapshots the weights."""
        if metric < self.best - self.min_delta:
            self.best, self.best_step, self.bad_evals = metric, step, 0
            self.best_state = state_fn()
            return True
        self.bad_evals += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_evals >= self.patience
