"""Option key <-> single token id."""
from __future__ import annotations

import string

KEYS = list(string.ascii_uppercase[:10])  # A..J


class OptionTokenizer:
    """Maps option keys to token ids and asserts each key is exactly one token.

    The answer slot follows `**Student Answer:**`, so keys are encoded with a leading space (" A")
    by default; `leading_space=False` encodes the bare letter.
    """

    def __init__(self, tokenizer, keys: list[str] | None = None, leading_space: bool = True):
        self.tokenizer = tokenizer
        self.keys = keys or KEYS
        self.leading_space = leading_space
        self.key_to_id: dict[str, int] = {}
        for k in self.keys:
            text = (" " if leading_space else "") + k
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Option key {text!r} is {len(ids)} tokens ({ids}); expected exactly 1")
            self.key_to_id[k] = ids[0]
        if len(set(self.key_to_id.values())) != len(self.keys):
            raise ValueError("Option keys do not map to distinct token ids")
        self.id_to_key = {v: k for k, v in self.key_to_id.items()}

    def ids_for(self, keys: list[str]) -> list[int]:
        return [self.key_to_id[k] for k in keys]
