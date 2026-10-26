"""Regression baseline: a transformer encoder with a one-output head predicts the item parameter b from the
item text and is fine-tuned with MSE. It has no notion of ability or option probabilities, so it is compared
with the ICC simulators only through the predicted Difficulty."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from llm_icc.data.item import Item

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class RegressionExample:
    item_id: str
    input_ids: list[int]
    target: float  # b on the model scale; nan when unknown


class RegressionDataset(Dataset):
    def __init__(self, examples: list[RegressionExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> RegressionExample:
        return self.examples[i]


def regression_collate(batch: list[RegressionExample], pad_id: int) -> dict[str, Tensor]:
    L = max(len(b.input_ids) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b.input_ids)] = torch.tensor(b.input_ids)
        mask[i, : len(b.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "target": torch.tensor([b.target for b in batch], dtype=torch.float32)}


def encode_item(tokenizer, item: Item, max_length: int, include_correct: bool) -> tuple[list[int], bool]:
    """`[CLS] stem [SEP] options (+ correct answer) [SEP]`; an over-long stem loses its beginning."""
    options = "\n".join(f"{o.key}. {o.text.strip()}" for o in item.options)
    if include_correct:
        options += f"\nCorrect answer: {item.correct_option.key}"
    stem = tokenizer.encode(item.stem.strip(), add_special_tokens=False)
    tail = tokenizer.encode(options, add_special_tokens=False)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    budget = max_length - len(tail) - 3
    if budget < 1:
        raise ValueError(f"Options of item {item.item_id} are {len(tail)} tokens; max_length={max_length}")
    truncated = len(stem) > budget
    return [cls, *stem[-budget:], sep, *tail, sep], truncated


class RegressionModel:
    """Sequence-classification model with `num_labels=1`. Mirrors the simulator interface the trainers use."""
    distribution_collate = staticmethod(regression_collate)

    def __init__(self, config, model_path: Path | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        pm = config.pretrained_model
        src = str(model_path) if model_path is not None else pm.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.pad_id = self.tokenizer.pad_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = _DTYPES[pm.dtype] if self.device.type == "cuda" else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=1, problem_type="regression", attn_implementation=pm.attn_implementation)
        if pm.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """(B,) predicted b, float32."""
        with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
            out = self.model(input_ids=batch["input_ids"].to(self.device),
                             attention_mask=batch["attention_mask"].to(self.device))
        return out.logits.float().squeeze(-1)

    def build_examples(self, items: list[Item], with_target: bool = True) -> list[RegressionExample]:
        rc, out, n_trunc = self.config.regression, [], 0
        for item in items:
            ids, truncated = encode_item(self.tokenizer, item, rc.max_length, rc.include_correct_answer)
            n_trunc += truncated
            if with_target and not item.is_ready():
                raise ValueError(f"Item {item.item_id} has no parameters; cannot build targets")
            out.append(RegressionExample(item.item_id, ids,
                                         item.parameters.b if with_target else float("nan")))
        if n_trunc:
            log.warning("%d of %d items had their stem truncated to fit max_length", n_trunc, len(items))
        return out

    @torch.no_grad()
    def predict_b(self, items: list[Item]) -> np.ndarray:
        self.eval_mode()
        exs, bs, preds = self.build_examples(items, with_target=False), self.config.pretrained_model.predict_batch_size, []
        for i in range(0, len(exs), bs):
            preds.append(self.forward(regression_collate(exs[i:i + bs], self.pad_id)).cpu())
        return torch.cat(preds).numpy()

    # ---- interface used by the trainers ----
    def eval_mode(self) -> None:
        self.model.eval()

    def train_mode(self) -> None:
        self.model.train()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_trainable_state(self) -> dict[str, Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def set_trainable_state(self, state: dict[str, Tensor]) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v.to(params[n].device, params[n].dtype))

    def save(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def load(cls, path: Path, config) -> "RegressionModel":
        return cls(config, model_path=Path(path))
