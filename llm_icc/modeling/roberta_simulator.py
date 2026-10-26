"""RoBERTa multiple-choice simulator after Maeda (2023), "Field-testing items using artificial
intelligence: Natural language processing with transformers" (arXiv:2310.11655).

The paper makes RoBERTa-large answer an exam and lowers its "ability" by zeroing a random proportion of
its word embeddings, so that it forgets those words. Here the same device is tied to the descriptor scale:
descriptor k keeps a fraction q_k = Phi((theta_k - mu) / sigma) of the vocabulary, where theta_k is the
conditional mean of the descriptor, so the lowest descriptors keep almost no words and the highest almost
all. The model is a `RobertaForMultipleChoice` that scores each (stem, option) pair; its softmax over the
options is the response distribution.

Adaptations to this pipeline:
  * `roberta-large` has no multiple-choice head (the paper used a RACE checkpoint), so the head and encoder
    are trained with the same two stages as the causal-LM simulator (SFT, then distribution MSE).
  * the ICC at a descriptor is the option distribution averaged over several random ablations, rather
    than one sampled response per simulated examinee.
  * ablation is Bernoulli per token (keep with probability q_k) instead of an exact random count.
  * stems longer than the 512-token limit are truncated from the left, keeping the question, rather
    than the item being dropped.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor, nn

from llm_icc.data.descriptors import DescriptorScale
from llm_icc.data.item import Item
from llm_icc.data.torch_datasets import DistributionExample
from llm_icc.modeling.base import ResponseSimulator

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# --------------------------------------------------------------------------- encoding and batching
def encode_item(tokenizer, item: Item, max_length: int,
                include_correct: bool = False) -> tuple[list[list[int]], bool]:
    """One `<s> context </s></s> option </s>` sequence per option; returns (sequences, stem_truncated).

    The context is the stem, followed by "Correct answer: <text>" when `include_correct` (the causal-LM
    prompt also reveals it; without it the model cannot tell which option the ability-dependent target favours).
    An over-long stem loses its beginning, since the question is at the end of a USMLE stem.
    """
    stem = tokenizer.encode(item.stem.strip(), add_special_tokens=False)
    tail = (tokenizer.encode(f" Correct answer: {item.correct_option.text.strip()}", add_special_tokens=False)
            if include_correct else [])
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    n_special = 4  # <s> stem </s></s> option </s>
    seqs, truncated = [], False
    for o in item.options:
        opt = tokenizer.encode(o.text.strip(), add_special_tokens=False)
        budget = max_length - n_special - len(opt) - len(tail)
        if budget < 1:
            raise ValueError(f"Option {o.key} of item {item.item_id} is {len(opt)} tokens; "
                             f"no room for the stem within max_length={max_length}")
        s = stem
        if len(s) > budget:
            s, truncated = s[-budget:], True
        seqs.append([cls, *s, *tail, sep, sep, *opt, sep])
    return seqs, truncated


def _pad_options(batch: list[DistributionExample], pad_id: int) -> dict[str, Tensor]:
    """Right-pad to (B, n_max, L). Padded options attend to their first token so no row is fully masked."""
    B = len(batch)
    n_max = max(len(b.input_ids) for b in batch)
    L = max(len(s) for b in batch for s in b.input_ids)
    ids = torch.full((B, n_max, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, n_max, L), dtype=torch.long)
    mask[:, :, 0] = 1
    omask = torch.zeros((B, n_max), dtype=torch.bool)
    for i, b in enumerate(batch):
        omask[i, : len(b.input_ids)] = True
        for j, s in enumerate(b.input_ids):
            ids[i, j, : len(s)] = torch.tensor(s)
            mask[i, j, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask, "option_mask": omask,
            "descriptor_index": torch.tensor([b.descriptor_index for b in batch], dtype=torch.long)}


def roberta_distribution_collate(batch: list[DistributionExample], pad_id: int) -> dict[str, Tensor]:
    out = _pad_options(batch, pad_id)
    tgt = torch.zeros(out["option_mask"].shape, dtype=torch.float32)
    for i, b in enumerate(batch):
        tgt[i, : len(b.target)] = torch.tensor(np.nan_to_num(b.target), dtype=torch.float32)
    out["target"] = tgt
    out["weight"] = torch.tensor([b.weight for b in batch], dtype=torch.float32)
    return out


def roberta_sft_collate(batch, pad_id: int) -> dict[str, Tensor]:
    """SFT examples carry the sampled option *index* as `label_id`."""
    out = _pad_options(batch, pad_id)
    out["labels"] = torch.tensor([b.label_id for b in batch], dtype=torch.long)
    return out


# --------------------------------------------------------------------------- simulator
class RobertaResponseSimulator(ResponseSimulator):
    distribution_collate = staticmethod(roberta_distribution_collate)
    sft_collate = staticmethod(roberta_sft_collate)

    def __init__(self, config, model_path: Path | None = None):
        from transformers import AutoTokenizer, RobertaForMultipleChoice

        self.config = config
        pm, rc = config.pretrained_model, config.roberta
        if pm.max_length > 512:
            raise ValueError("RoBERTa supports at most 512 tokens; set pretrained_model.max_length <= 512")
        src = str(model_path) if model_path is not None else pm.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.pad_id = self.tokenizer.pad_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = _DTYPES[pm.dtype] if self.device.type == "cuda" else None
        if self.autocast_dtype is torch.float16:
            log.warning("float16 autocast runs without loss scaling; bfloat16 is recommended for training")
        self.model = RobertaForMultipleChoice.from_pretrained(src, attn_implementation=pm.attn_implementation)
        self.model.config.use_cache = False
        if pm.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        # descriptor k keeps this fraction of the vocabulary
        mu, sigma = config.population.mu, config.population.sigma
        theta_k = config.scale().conditional_means(mu, sigma)
        self.keep_fraction = torch.tensor(norm.cdf((theta_k - mu) / sigma), dtype=torch.float32,
                                          device=self.device)
        self.vocab_size = self.model.get_input_embeddings().num_embeddings
        self.protected = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        if rc.protect_special_tokens:
            self.protected[[i for i in self.tokenizer.all_special_ids if i < self.vocab_size]] = True

        self._gen = torch.Generator(device=self.device)  # fixed ablations whenever the model is in eval mode
        self._gen.manual_seed(config.run.seed)
        self._token_keep: Tensor | None = None
        self.model.roberta.embeddings.word_embeddings.register_forward_hook(self._ablate)

    # ---- vocabulary ablation ----
    def _ablate(self, module, args, output: Tensor) -> Tensor:
        """Zero the embeddings of ablated tokens (the paper's "forgetting")."""
        if self._token_keep is None:
            return output
        return output * self._token_keep.to(output.dtype)

    def keep_mask(self, descriptor_index: Tensor) -> Tensor:
        """(B, V) bool; True where the word is remembered. Fresh draws in train mode, seeded draws in eval."""
        gen = None if self.model.training else self._gen
        u = torch.rand(len(descriptor_index), self.vocab_size, device=self.device, generator=gen)
        return (u < self.keep_fraction[descriptor_index.to(self.device)][:, None]) | self.protected

    # ---- forward ----
    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def option_logits(self, batch: dict[str, Tensor]) -> Tensor:
        b = self._to_device(batch)
        ids = b["input_ids"]
        B, n, L = ids.shape
        keep = self.keep_mask(b["descriptor_index"]).gather(1, ids.view(B, n * L))
        self._token_keep = keep.view(B * n, L, 1)  # the multiple-choice head flattens (B, n) -> B * n
        try:
            with torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
                logits = self.model(input_ids=ids, attention_mask=b["attention_mask"]).logits
        finally:
            self._token_keep = None
        return logits.float().masked_fill(~b["option_mask"], float("-inf"))

    def answer_logits(self, batch: dict[str, Tensor]) -> Tensor:
        return self.option_logits(batch)  # the head's option scores are the answer slot; labels are option indices

    def build_examples(self, items, scale, mu, sigma, with_target=True) -> list[DistributionExample]:
        omega = scale.weights(mu, sigma)
        out, n_trunc = [], 0
        for item in items:
            seqs, truncated = encode_item(self.tokenizer, item, self.config.pretrained_model.max_length,
                                            self.config.roberta.include_correct_answer)
            n_trunc += truncated
            target = item.target_distribution(scale, mu, sigma) if with_target else None
            for k in range(len(scale)):
                out.append(DistributionExample(
                    item.item_id, k, seqs, list(range(len(seqs))),
                    target[k] if target is not None else np.full(len(seqs), np.nan), float(omega[k])))
        if n_trunc:
            log.warning("%d of %d items had their stem truncated to fit max_length", n_trunc, len(items))
        return out

    @torch.no_grad()
    def predict_icc(self, item: Item, scale: DescriptorScale) -> np.ndarray:
        """(N, n) option probabilities, averaged over `roberta.ablation_samples` random ablations."""
        self.eval_mode()
        exs = self.build_examples([item], scale, 0.0, 1.0, with_target=False)
        bs = self.config.pretrained_model.predict_batch_size
        self._gen.manual_seed(self.config.run.seed + zlib.crc32(item.item_id.encode()))  # order-independent
        total = torch.zeros(len(exs), item.n_options)
        for _ in range(self.config.roberta.ablation_samples):
            rows = []
            for i in range(0, len(exs), bs):
                batch = self.distribution_collate(exs[i:i + bs], self.pad_id)
                rows.append(self.option_probabilities(batch)[:, : item.n_options].float().cpu())
            total += torch.cat(rows)
        return (total / self.config.roberta.ablation_samples).numpy()

    def item_batches(self, item: Item, scale: DescriptorScale):
        exs = self.build_examples([item], scale, 0.0, 1.0, with_target=False)
        bs = self.config.pretrained_model.predict_batch_size
        for i in range(0, len(exs), bs):
            yield self.distribution_collate(exs[i:i + bs], self.pad_id)

    # ---- state ----
    def eval_mode(self) -> None:
        self.model.eval()
        self._gen.manual_seed(self.config.run.seed)  # identical ablations on every dev evaluation

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
    def load(cls, path: Path, config) -> "RobertaResponseSimulator":
        return cls(config, model_path=Path(path))
