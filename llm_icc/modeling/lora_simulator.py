"""HF + PEFT LoRA wrapper that exposes option-token probabilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from llm_icc.data.descriptors import DescriptorScale
from llm_icc.data.item import Item
from llm_icc.data.torch_datasets import build_examples, distribution_collate
from llm_icc.modeling.base import ResponseSimulator
from llm_icc.modeling.option_tokens import OptionTokenizer
from llm_icc.modeling.prompts import PromptBuilder

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


class LoRAResponseSimulator(ResponseSimulator):
    def __init__(self, config, adapter_path: Path | None = None):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        pm = config.pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(pm.name_or_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.option_tokenizer = OptionTokenizer(self.tokenizer)
        self.prompt_builder = PromptBuilder(self.tokenizer, enable_thinking=pm.enable_thinking,
                                            chat_template_kwargs=pm.chat_template_kwargs)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _DTYPES[pm.dtype] if device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            pm.name_or_path, dtype=dtype, attn_implementation=pm.attn_implementation)
        base.config.use_cache = False
        if pm.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
        else:
            lc = config.lora
            self.model = get_peft_model(base, LoraConfig(
                r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout, target_modules=lc.target_modules,
                bias=lc.bias, task_type="CAUSAL_LM"))
        self.model.to(device)
        self.device = torch.device(device)

    # ---- forward ----
    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def answer_logits(self, batch: dict[str, Tensor]) -> Tensor:
        b = self._to_device(batch)
        mask = b["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)  # left padding: keep positions aligned to real tokens
        out = self.model(input_ids=b["input_ids"], attention_mask=mask, position_ids=pos, logits_to_keep=1)
        return out.logits[:, -1, :]  # left-padded => last position is the answer slot

    def option_logits(self, batch: dict[str, Tensor]) -> Tensor:
        logits = self.answer_logits(batch)
        b = self._to_device(batch)
        gathered = logits.gather(1, b["option_token_ids"]).float()
        return gathered.masked_fill(~b["option_mask"], float("-inf"))

    def build_examples(self, items, scale, mu, sigma, with_target=True):
        return build_examples(items, scale, mu, sigma, self.prompt_builder, self.option_tokenizer,
                              self.config.pretrained_model.max_length, with_target=with_target)

    def item_batches(self, item: Item, scale: DescriptorScale) -> Iterable[dict[str, Tensor]]:
        # no targets are needed for prediction
        exs = build_examples([item], scale, 0.0, 1.0, self.prompt_builder, self.option_tokenizer,
                             self.config.pretrained_model.max_length, with_target=False)
        bs = self.config.pretrained_model.predict_batch_size
        for i in range(0, len(exs), bs):
            yield distribution_collate(exs[i:i + bs], self.pad_id)

    # ---- state ----
    def eval_mode(self) -> None:
        self.model.eval()

    def train_mode(self) -> None:
        self.model.train()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
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
        self.model.save_pretrained(str(path))  # adapter weights only

    @classmethod
    def load(cls, path: Path, config) -> "LoRAResponseSimulator":
        return cls(config, adapter_path=Path(path))
