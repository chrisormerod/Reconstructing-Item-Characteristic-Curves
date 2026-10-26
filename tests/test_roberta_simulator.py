"""RoBERTa simulator on a tiny random model and a toy WordLevel tokenizer (no download)."""
import logging

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMultipleChoice

from llm_icc.app import build_targets, make_simulator
from llm_icc.config import load_config
from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option
from llm_icc.data.torch_datasets import DistributionDataset, SFTDataset
from llm_icc.modeling.roberta_simulator import RobertaResponseSimulator, encode_item
from llm_icc.modeling.training_loop import DistributionTrainer, SFTTrainer

WORDS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + [f"w{i}" for i in range(40)]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_roberta")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers.processors import RobertaProcessing
    tok.post_processor = RobertaProcessing(("</s>", 2), ("<s>", 0))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", sep_token="</s>",
                                   cls_token="<s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    fast.save_pretrained(d)
    torch.manual_seed(0)
    RobertaForMultipleChoice(RobertaConfig(
        vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=66, pad_token_id=1, bos_token_id=0, eos_token_id=2)).save_pretrained(d)
    return str(d)


def _cfg(model_dir, *extra):
    return load_config("config/default.yaml", [
        "pretrained_model.architecture=roberta_mc", f"pretrained_model.name_or_path={model_dir}",
        "pretrained_model.max_length=64", "pretrained_model.gradient_checkpointing=false",
        "pretrained_model.predict_batch_size=7", "roberta.ablation_samples=3",
        "training.gradient_accumulation_steps=2", "training.eval_every_steps=2", "training.epochs=2",
        "training.batch_size=2", *extra])


def _bank(n=6):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        k = int(rng.integers(3, 6))
        items.append(Item(str(i), " ".join(f"w{j}" for j in rng.integers(0, 40, 12)),
                          [Option("ABCDEFGHIJ"[j], f"w{rng.integers(0, 40)} w{rng.integers(0, 40)}", j == 0)
                           for j in range(k)],
                          reference_difficulty=float(rng.uniform(0.1, 1.2)),
                          metadata={"source": "train", "item_type": "Text", "exam": "S1"}))
    return ItemBank(items)


def test_keep_fraction_is_monotone_and_ablation_zeroes_embeddings(model_dir):
    sim = RobertaResponseSimulator(_cfg(model_dir))
    kf = sim.keep_fraction.cpu().numpy()
    assert (np.diff(kf) > 0).all() and kf[0] < 0.01 and kf[-1] > 0.99
    sim.eval_mode()
    bank = _bank(2)
    ex = sim.build_examples([bank["0"], bank["1"]], sim.config.scale(), 0, 1, with_target=False)
    top = [e for e in ex if e.descriptor_index == len(sim.config.scale()) - 1]  # keeps ~all words
    b = sim.distribution_collate(top, sim.pad_id)
    out = sim.option_logits(b)[:, :3]
    assert not torch.allclose(out[0], out[1])           # content matters when words are remembered
    sim.keep_fraction[:] = 0.0
    sim.protected[:] = False                              # forget everything: same-shaped inputs are identical
    out = sim.option_logits(b)[:, :3]
    torch.testing.assert_close(out[0], out[1])


def test_stem_truncated_from_the_left(model_dir):
    sim = RobertaResponseSimulator(_cfg(model_dir))
    item = Item("x", " ".join(f"w{i}" for i in range(30)), [Option("A", "w1 w2", True), Option("B", "w3", False)])
    seqs, truncated = encode_item(sim.tokenizer, item, 16)
    assert truncated and all(len(s) <= 16 for s in seqs)
    stem_ids = sim.tokenizer.encode("w29", add_special_tokens=False)
    assert stem_ids[0] in seqs[0]            # question end kept
    assert sim.tokenizer.encode("w0", add_special_tokens=False)[0] not in seqs[0]


def test_predict_icc_is_a_deterministic_distribution(model_dir):
    sim = RobertaResponseSimulator(_cfg(model_dir))
    item = _bank(1)["0"]
    p1, p2 = sim.predict_icc(item, sim.config.scale()), sim.predict_icc(item, sim.config.scale())
    assert p1.shape == (21, item.n_options)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p1, p2)


def test_training_runs_and_checkpoint_roundtrips(model_dir, tmp_path):
    cfg = _cfg(model_dir, "training.learning_rate=1e-3", "training.sft.learning_rate=1e-3")
    bank = _bank()
    build_targets(bank, cfg)
    sim = make_simulator(cfg)
    scale, log = cfg.scale(), logging.getLogger("t")
    tr_ex = sim.build_examples([bank[str(i)] for i in range(4)], scale, 0, 1)
    dev_ex = sim.build_examples([bank[str(i)] for i in (4, 5)], scale, 0, 1)
    SFTTrainer(sim, SFTDataset(tr_ex), cfg, log).train()
    tr = DistributionTrainer(sim, DistributionDataset(tr_ex), DistributionDataset(dev_ex), cfg, log)
    start = tr.evaluate(DistributionDataset(dev_ex))["dev_loss"]
    res = tr.train()
    assert res.best_dev_loss <= start
    sim.save(tmp_path / "ck")
    again = make_simulator(cfg, str(tmp_path / "ck"))
    item = bank["5"]
    np.testing.assert_allclose(again.predict_icc(item, scale), sim.predict_icc(item, scale), atol=1e-4)
