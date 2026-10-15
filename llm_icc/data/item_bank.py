from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import numpy as np

from llm_icc.data.item import Item


@dataclass
class Split:
    name: str  # e.g. "predefined" or "fold_0"
    train: list[str]
    dev: list[str]
    test: list[str]


class ItemBank:
    """Collection of items. Each item's `metadata['source']` is 'train' or 'test'."""

    def __init__(self, items: dict[str, Item] | list[Item]):
        if not isinstance(items, dict):
            items = {i.item_id: i for i in items}
        self.items: dict[str, Item] = items

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item_id: str) -> Item:
        return self.items[item_id]

    def ids(self, source: str | None = None) -> list[str]:
        return [i.item_id for i in self if source is None or i.metadata.get("source") == source]

    def filter(self, predicate: Callable[[Item], bool]) -> "ItemBank":
        return ItemBank({k: v for k, v in self.items.items() if predicate(v)})

    def split(self, strategy: Literal["predefined", "kfold"], seed: int = 0, dev_fraction: float = 0.1,
              k: int = 5) -> list[Split]:
        train_ids = self.ids("train")
        test_ids = self.ids("test")
        rng = np.random.default_rng(seed)
        perm = [train_ids[i] for i in rng.permutation(len(train_ids))]
        if strategy == "predefined":
            n_dev = max(1, int(round(dev_fraction * len(perm))))
            return [Split("predefined", train=perm[n_dev:], dev=perm[:n_dev], test=test_ids)]
        if strategy == "kfold":
            if k < 2 or k > len(perm):
                raise ValueError(f"Invalid k={k} for {len(perm)} training items")
            folds = [list(f) for f in np.array_split(np.array(perm, dtype=object), k)]
            out = []
            for i in range(k):
                train = [x for j, f in enumerate(folds) if j != i for x in f]
                out.append(Split(f"fold_{i}", train=train, dev=folds[i], test=test_ids))
            return out
        raise ValueError(f"Unknown split strategy {strategy!r}")
