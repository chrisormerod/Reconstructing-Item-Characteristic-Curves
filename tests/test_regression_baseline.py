"""Regression baseline on a tiny random ModernBERT and a toy tokenizer (no download)."""
import logging

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

from llm_icc.app import build_targets
from llm_icc.config import load_config
from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option
from llm_icc.modeling.regression_baseline import RegressionDataset, RegressionModel, encode_item
from llm_icc.modeling.training_loop import RegressionTrainer

WORDS = ["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]"] + [f"w{i}" for i in range(40)] + ["A.", "B.", "C.", "D.", "Correct", "answer:", "A", "B", "C", "D"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_modernbert")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tok, cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
                            unk_token="[UNK]", mask_token="[MASK]").save_pretrained(d)
    torch.manual_seed(0)
    ModernBertForSequenceClassification(ModernBertConfig(
        vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        pad_token_id=1, cls_token_id=0, sep_token_id=2, bos_token_id=0, eos_token_id=2, num_labels=1,
        global_attn_every_n_layers=2, local_attention=16)).save_pretrained(d)
    return str(d)


def _cfg(model_dir, *extra):
    return load_config("config/default.yaml", [
        "run.method=regression", f"pretrained_model.name_or_path={model_dir}",
        "pretrained_model.gradient_checkpointing=false", "pretrained_model.attn_implementation=eager",
        "pretrained_model.predict_batch_size=4", "regression.max_length=48",
        "training.gradient_accumulation_steps=1", "training.eval_every_steps=2", "training.epochs=6",
        "training.batch_size=4", "training.learning_rate=3e-3", *extra])


def _bank(n=12):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        items.append(Item(str(i), " ".join(f"w{j}" for j in rng.integers(0, 40, 10)),
                          [Option("ABCD"[j], f"w{rng.integers(0, 40)}", j == 0) for j in range(4)],
                          reference_difficulty=float(rng.uniform(0.1, 1.2)),
                          metadata={"source": "train", "item_type": "Text", "exam": "S1"}))
    return ItemBank(items)


def test_encode_keeps_options_and_truncates_stem_from_the_left(model_dir):
    m = RegressionModel(_cfg(model_dir))
    item = Item("x", " ".join(f"w{i}" for i in range(60)), [Option("A", "w1", True), Option("B", "w2", False)])
    ids, truncated = encode_item(m.tokenizer, item, 32, True)
    assert truncated and len(ids) == 32 and ids[0] == m.tokenizer.cls_token_id
    assert m.tokenizer.encode("w59", add_special_tokens=False)[0] in ids
    assert m.tokenizer.encode("w0", add_special_tokens=False)[0] not in ids


def test_training_fits_targets_and_checkpoint_roundtrips(model_dir, tmp_path):
    cfg = _cfg(model_dir)
    bank = _bank()
    build_targets(bank, cfg)
    model = RegressionModel(cfg)
    items = lambda ids: [bank[str(i)] for i in ids]  # noqa: E731
    tr, dev = RegressionDataset(model.build_examples(items(range(9)))), RegressionDataset(model.build_examples(items(range(9, 12))))
    trainer = RegressionTrainer(model, tr, dev, cfg, logging.getLogger("t"))
    start = trainer.evaluate(dev)["dev_loss"]
    res = trainer.train()
    assert res.best_dev_loss <= start
    assert abs(trainer.evaluate(dev)["dev_loss"] - res.best_dev_loss) < 1e-5  # best restored
    model.save(tmp_path / "ck")
    again = RegressionModel.load(tmp_path / "ck", cfg)
    np.testing.assert_allclose(again.predict_b(items(range(12))), model.predict_b(items(range(12))), atol=1e-4)
