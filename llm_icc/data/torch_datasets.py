"""Datasets: one example per (item, descriptor) pair."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from llm_icc.data.descriptors import DescriptorScale
from llm_icc.data.item import Item


@dataclass
class DistributionExample:
    item_id: str
    descriptor_index: int
    input_ids: list[int]          # prompt truncated at answer slot
    option_token_ids: list[int]   # length n
    target: np.ndarray            # (n,)
    weight: float                 # omega_k


@dataclass
class SFTExample:
    item_id: str
    descriptor_index: int
    input_ids: list[int]
    label_id: int                 # sampled option token id


def build_examples(items: list[Item], scale: DescriptorScale, mu: float, sigma: float, prompt_builder,
                   option_tokenizer, max_length: int, with_target: bool = True) -> list[DistributionExample]:
    """Tokenize every (item, descriptor) pair. Fails early if a prompt exceeds `max_length`."""
    omega = scale.weights(mu, sigma)
    out = []
    for item in items:
        opt_ids = option_tokenizer.ids_for([o.key for o in item.options])
        target = item.target_distribution(scale, mu, sigma) if with_target else None
        for k, label in enumerate(scale.labels):
            ids = prompt_builder.build_ids(item, label)
            if len(ids) > max_length:
                raise ValueError(f"Prompt for item {item.item_id} is {len(ids)} tokens > max_length={max_length}")
            out.append(DistributionExample(
                item.item_id, k, ids, opt_ids,
                target[k] if target is not None else np.full(len(opt_ids), np.nan), float(omega[k])))
    return out


class DistributionDataset(Dataset):
    def __init__(self, examples: list[DistributionExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> DistributionExample:
        return self.examples[i]


class SFTDataset(Dataset):
    """Resamples one answer per (item, descriptor) from the target distribution each epoch."""

    def __init__(self, examples: list[DistributionExample], seed: int = 0):
        self.examples = examples
        self.seed = seed
        self._cur: list[SFTExample] = []
        self.resample(0)

    def resample(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + 1000 * epoch)
        self._cur = []
        for ex in self.examples:
            p = ex.target / ex.target.sum()
            j = int(rng.choice(len(p), p=p))
            self._cur.append(SFTExample(ex.item_id, ex.descriptor_index, ex.input_ids, ex.option_token_ids[j]))

    def __len__(self):
        return len(self._cur)

    def __getitem__(self, i) -> SFTExample:
        return self._cur[i]


def _pad_left(seqs: list[list[int]], pad_id: int):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        mask[i, T - len(s):] = 1
    return ids, mask


def distribution_collate(batch: list[DistributionExample], pad_id: int) -> dict[str, torch.Tensor]:
    """Left-padded so the answer slot is always the last position."""
    ids, mask = _pad_left([b.input_ids for b in batch], pad_id)
    n_max = max(len(b.option_token_ids) for b in batch)
    B = len(batch)
    opt = torch.zeros((B, n_max), dtype=torch.long)
    omask = torch.zeros((B, n_max), dtype=torch.bool)
    tgt = torch.zeros((B, n_max), dtype=torch.float32)
    for i, b in enumerate(batch):
        n = len(b.option_token_ids)
        opt[i, :n] = torch.tensor(b.option_token_ids)
        omask[i, :n] = True
        tgt[i, :n] = torch.tensor(np.nan_to_num(b.target), dtype=torch.float32)
    w = torch.tensor([b.weight for b in batch], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": mask, "option_token_ids": opt,
            "option_mask": omask, "target": tgt, "weight": w}


def sft_collate(batch: list[SFTExample], pad_id: int) -> dict[str, torch.Tensor]:
    ids, mask = _pad_left([b.input_ids for b in batch], pad_id)
    return {"input_ids": ids, "attention_mask": mask,
            "labels": torch.tensor([b.label_id for b in batch], dtype=torch.long)}
