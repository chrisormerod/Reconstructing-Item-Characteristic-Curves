from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from llm_icc.data.descriptors import DescriptorScale
from llm_icc.data.item import Item


class ResponseSimulator(ABC):
    """Wraps a pretrained LM; exposes option-level probabilities, not text."""

    @abstractmethod
    def answer_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, V) full-vocabulary logits at the answer slot (used by SFT)."""

    @abstractmethod
    def option_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, n_max) logits at the answer slot, gathered at option_token_ids; masked = -inf."""

    def option_probabilities(self, batch: dict[str, Tensor]) -> Tensor:
        """Restricted softmax over option tokens (paper eq. 16, LHS)."""
        return torch.softmax(self.option_logits(batch), dim=-1)

    @abstractmethod
    def item_batches(self, item: Item, scale: DescriptorScale) -> Iterable[dict[str, Tensor]]:
        """Batches covering all N descriptors of one item, in descriptor order."""

    @torch.no_grad()
    def predict_icc(self, item: Item, scale: DescriptorScale) -> np.ndarray:
        """(N, n) option probabilities for one item at every descriptor."""
        self.eval_mode()
        probs = [self.option_probabilities(b)[:, : item.n_options].float().cpu()
                 for b in self.item_batches(item, scale)]
        return torch.cat(probs).numpy()

    @abstractmethod
    def eval_mode(self) -> None: ...

    @abstractmethod
    def train_mode(self) -> None: ...

    @abstractmethod
    def trainable_parameters(self) -> Iterable[nn.Parameter]: ...

    @abstractmethod
    def get_trainable_state(self) -> dict[str, Tensor]: ...

    @abstractmethod
    def set_trainable_state(self, state: dict[str, Tensor]) -> None: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, config) -> "ResponseSimulator": ...
