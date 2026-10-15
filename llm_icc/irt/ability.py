"""theta_k and omega_k for a descriptor scale under theta ~ N(mu, sigma^2)."""
from __future__ import annotations

import numpy as np

from llm_icc.data.descriptors import DescriptorScale


def conditional_means(scale: DescriptorScale, mu: float, sigma: float) -> np.ndarray:
    return scale.conditional_means(mu, sigma)


def descriptor_weights(scale: DescriptorScale, mu: float, sigma: float) -> np.ndarray:
    return scale.weights(mu, sigma)
