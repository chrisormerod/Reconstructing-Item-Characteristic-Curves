from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from llm_icc.data.descriptors import DescriptorScale
from llm_icc.data.options import Option
from llm_icc.data.parameters import LogisticParameters


@dataclass
class Item:
    item_id: str
    stem: str
    options: list[Option]
    parameters: LogisticParameters | None = None  # b on the model scale (after the difficulty transform)
    reference_difficulty: float | None = None     # Difficulty as given in the data (original scale)
    metadata: dict = field(default_factory=dict)

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def correct_index(self) -> int:
        for i, o in enumerate(self.options):
            if o.is_correct:
                return i
        raise ValueError(f"Item {self.item_id} has no correct option")

    @property
    def correct_option(self) -> Option:
        return self.options[self.correct_index]

    def is_ready(self) -> bool:
        return self.parameters is not None

    def target_distribution(self, scale: DescriptorScale, mu: float, sigma: float) -> np.ndarray:
        """(N, n) target option probabilities at theta_k via eq. 4. Raises if not ready."""
        if not self.is_ready():
            raise ValueError(f"Item {self.item_id} has no parameters; cannot build targets")
        nrm = self.parameters.to_nrm(self.n_options, self.correct_index)
        return nrm.option_probabilities(scale.conditional_means(mu, sigma))
