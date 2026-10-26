import pytest

from llm_icc.data.item import Item
from llm_icc.data.options import Option
from llm_icc.modeling.option_tokens import KEYS, OptionTokenizer
from llm_icc.modeling.prompts import ANSWER_PREFILL, PromptBuilder


class StubTokenizer:
    """Character-level stand-in: ' A' -> one id, longer strings -> many."""
    def encode(self, text, add_special_tokens=False):
        if len(text.strip()) == 1 and len(text) <= 2:
            return [ord(text.strip())]
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking, **kwargs):
        assert tokenize is False and add_generation_prompt and enable_thinking is False
        head = f"[{kwargs['date_string']}]" if "date_string" in kwargs else ""
        return head + "".join(f"<{m['role']}>{m['content']}</>" for m in messages) + "<assistant>"


def _item():
    opts = [Option(k, f"text {k}", k == "B") for k in "ABCD"]
    return Item("1", "  A stem.  ", opts)


def test_prompt_lists_keys_correct_answer_and_prefill():
    text = PromptBuilder(StubTokenizer()).build_text(_item(), "Proficient")
    assert "Proficient" in text and "A. text A" in text and "D. text D" in text
    assert "**Correct Answer:** B. text B" in text
    assert text.endswith("<assistant>" + ANSWER_PREFILL)
    assert "PIX" not in text


def test_chat_template_kwargs_are_forwarded():
    text = PromptBuilder(StubTokenizer(), chat_template_kwargs={"date_string": "01 Jan 2025"}).build_text(_item(), "Basic")
    assert text.startswith("[01 Jan 2025]")


def test_option_tokenizer_maps_all_keys_to_distinct_ids():
    ot = OptionTokenizer(StubTokenizer())
    assert list(ot.key_to_id) == KEYS and len(set(ot.key_to_id.values())) == 10


def test_option_tokenizer_rejects_multi_token_keys():
    class Bad(StubTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [1, 2]
    with pytest.raises(ValueError, match="expected exactly 1"):
        OptionTokenizer(Bad())


def test_real_qwen_tokenizer_if_cached():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3 tokenizer not cached")
    assert len(OptionTokenizer(tok).key_to_id) == 10
    ids = PromptBuilder(tok).build_ids(_item(), "Basic")
    assert ids[-1] != tok.eos_token_id
