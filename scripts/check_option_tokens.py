"""Verify option keys A-J are single tokens for a tokenizer (with and without a leading space).

Usage: python scripts/check_option_tokens.py [Qwen/Qwen3-4B]
"""
import sys

from transformers import AutoTokenizer

from llm_icc.modeling.option_tokens import KEYS


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(name)
    ok = True
    for prefix in (" ", ""):
        for k in KEYS:
            ids = tok.encode(prefix + k, add_special_tokens=False)
            good = len(ids) == 1
            ok &= good
            print(f"{(prefix + k)!r:6} -> {ids} {'ok' if good else 'NOT A SINGLE TOKEN'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
