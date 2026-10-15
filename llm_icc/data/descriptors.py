"""Ability descriptors and the ordered scale built from them (paper §3.1, Table 2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class AbilityDescriptor:
    label: str
    lower: float  # c_{k-1}, may be -inf
    upper: float  # c_k, may be +inf


class DescriptorScale:
    """Ordered, contiguous, non-overlapping set of descriptors covering (-inf, +inf)."""

    def __init__(self, descriptors: Sequence[AbilityDescriptor]):
        self.descriptors = list(descriptors)
        self._validate()
        self.labels = [d.label for d in self.descriptors]
        self._uppers = np.array([d.upper for d in self.descriptors])

    def _validate(self) -> None:
        ds = self.descriptors
        if not ds:
            raise ValueError("DescriptorScale needs at least one descriptor")
        labels = [d.label for d in ds]
        if len(set(labels)) != len(labels):
            raise ValueError("Descriptor labels must be unique")
        if ds[0].lower != -math.inf:
            raise ValueError(f"First descriptor '{ds[0].label}' must start at -inf")
        if ds[-1].upper != math.inf:
            raise ValueError(f"Last descriptor '{ds[-1].label}' must end at +inf")
        for d in ds:
            if not d.lower < d.upper:
                raise ValueError(f"Descriptor '{d.label}' has lower >= upper ({d.lower}, {d.upper})")
        for prev, cur in zip(ds, ds[1:]):
            if cur.lower > prev.upper:
                raise ValueError(
                    f"Gap between '{prev.label}' (upper={prev.upper}) and '{cur.label}' (lower={cur.lower})")
            if cur.lower < prev.upper:
                raise ValueError(
                    f"Overlap between '{prev.label}' (upper={prev.upper}) and '{cur.label}' (lower={cur.lower})")

    @classmethod
    def from_config(cls, mapping: Mapping[str, tuple[float, float]]) -> "DescriptorScale":
        return cls([AbilityDescriptor(k, float(lo), float(hi)) for k, (lo, hi) in mapping.items()])

    def __len__(self) -> int:
        return len(self.descriptors)

    def label_for(self, theta: float) -> int:
        """Index k with c_{k-1} < theta <= c_k."""
        return int(np.searchsorted(self._uppers, theta, side="left"))

    def assign(self, thetas: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._uppers, np.asarray(thetas), side="left")

    def _alphas(self, mu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        lo = (np.array([d.lower for d in self.descriptors]) - mu) / sigma
        hi = (np.array([d.upper for d in self.descriptors]) - mu) / sigma
        return lo, hi

    def conditional_means(self, mu: float, sigma: float) -> np.ndarray:
        """theta_k = E[theta | c_{k-1} < theta <= c_k] for theta ~ N(mu, sigma^2)."""
        lo, hi = self._alphas(mu, sigma)
        mass = norm.cdf(hi) - norm.cdf(lo)
        return mu + sigma * (norm.pdf(lo) - norm.pdf(hi)) / mass

    def weights(self, mu: float, sigma: float) -> np.ndarray:
        """omega_k = P(c_{k-1} < theta <= c_k)."""
        lo, hi = self._alphas(mu, sigma)
        return norm.cdf(hi) - norm.cdf(lo)
