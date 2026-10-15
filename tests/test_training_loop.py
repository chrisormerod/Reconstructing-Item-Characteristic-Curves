"""Exercise the manual loops, early stopping, and evaluate_split with a toy simulator (no LLM download)."""
import logging
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from llm_icc.app import build_targets, evaluate_split
from llm_icc.config import load_config
from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option
from llm_icc.data.torch_datasets import DistributionDataset, SFTDataset, build_examples
from llm_icc.modeling.base import ResponseSimulator
from llm_icc.modeling.training_loop import DistributionTrainer, SFTTrainer

CFG = load_config("config/default.yaml", ["training.gradient_accumulation_steps=2",
                                           "training.eval_every_steps=2", "training.epochs=4",
                                           "training.learning_rate=0.05", "training.sft.learning_rate=0.05"])


class Tok:
    def encode(self, t, add_special_tokens=False):
        return [ord(c) % 50 + 1 for c in t]


class PB:
    def build_ids(self, item, level):
        return [1 + (hash(level) % 7), 2, 3]


class OT:
    def ids_for(self, keys):
        return [10 + "ABCDEFGHIJ".index(k) for k in keys]


class Toy(ResponseSimulator):
    """Logits at the answer slot are a learned function of the (descriptor) first token only."""
    pad_id = 0
    device = torch.device("cpu")

    def __init__(self):
        self.emb = nn.Embedding(10, 32)
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Embedding(10, 16), nn.Linear(16, 32))

    def answer_logits(self, batch):
        return self.net(batch["input_ids"][:, 0])

    def option_logits(self, batch):
        g = self.answer_logits(batch).gather(1, batch["option_token_ids"])
        return g.masked_fill(~batch["option_mask"], float("-inf"))

    def item_batches(self, item, scale): return []
    def eval_mode(self): self.net.eval()
    def train_mode(self): self.net.train()
    def trainable_parameters(self): return self.net.parameters()
    def get_trainable_state(self): return {k: v.clone() for k, v in self.net.state_dict().items()}
    def set_trainable_state(self, s): self.net.load_state_dict(s)
    def save(self, path): pass
    @classmethod
    def load(cls, path, config): return cls()


def _bank(n=12):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        k = int(rng.integers(4, 8))
        items.append(Item(str(i), "s", [Option("ABCDEFGHIJ"[j], "t", j == 0) for j in range(k)],
                          reference_difficulty=float(rng.uniform(0.1, 1.2)),
                          metadata={"source": "train" if i < 8 else "test", "item_type": "Text", "exam": "S1"}))
    return ItemBank(items)


def test_loops_reduce_dev_loss_and_restore_best():
    bank = _bank()
    transform, _ = build_targets(bank, CFG)
    scale = CFG.scale()
    ex = build_examples([bank[str(i)] for i in range(6)], scale, 0, 1, PB(), OT(), 100)
    dev = build_examples([bank[str(i)] for i in (6, 7)], scale, 0, 1, PB(), OT(), 100)
    sim = Toy()
    log = logging.getLogger("t")
    SFTTrainer(sim, SFTDataset(ex), CFG, log).train()
    tr = DistributionTrainer(sim, DistributionDataset(ex), DistributionDataset(dev), CFG, log)
    start = tr.evaluate(DistributionDataset(dev))["dev_loss"]
    res = tr.train()
    assert res.best_dev_loss <= start and res.history[0]["step"] == 0
    assert abs(tr.evaluate(DistributionDataset(dev))["dev_loss"] - res.best_dev_loss) < 1e-6  # best restored


def test_sft_dataset_samples_from_target():
    bank = _bank(2)
    build_targets(bank, CFG)
    ex = build_examples([bank["0"]], CFG.scale(), 0, 1, PB(), OT(), 100)
    ds = SFTDataset(ex, seed=3)
    valid = set(ex[0].option_token_ids)
    assert all(s.label_id in valid for s in ds) and len(ds) == 20


def test_evaluate_split_recovers_difficulty_from_perfect_predictions():
    bank = _bank(30)
    # make 20 train, 10 "test"; split: dev = a few train items
    for i, it in enumerate(bank):
        it.metadata["source"] = "train" if i < 20 else "test"
    transform, _ = build_targets(bank, CFG)
    scale = CFG.scale()
    split = SimpleNamespace(name="s", train=[str(i) for i in range(14)], dev=[str(i) for i in range(14, 20)],
                            test=[str(i) for i in range(20, 30)])
    preds = {i: bank[i].target_distribution(scale, 0, 1) for i in split.dev + split.test}
    m, df = evaluate_split(preds, bank, split, CFG, transform)
    assert m["test"]["rmse"] < 1e-3 and m["test"]["pearson"] > 0.9999 and m["test"]["icc_mse"] < 1e-12
