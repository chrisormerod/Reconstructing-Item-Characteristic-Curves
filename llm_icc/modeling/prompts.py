"""Prompt construction. The assistant turn is pre-filled up to the answer slot."""
from __future__ import annotations

from llm_icc.data.item import Item

SYSTEM_PROMPT = (
    "You are simulating how a test taker responds to a multiple-choice exam item. "
    "You will be told the test taker's ability level. Respond exactly as a student at that ability "
    "level would: students at lower levels often choose plausible but incorrect options, while "
    "students at higher levels usually choose the correct option. Do not try to be correct; "
    "try to be realistic for the stated ability level."
)

USER_TEMPLATE = (
    "**Student Ability Level:** {level}\n\n"
    "**Question:**\n{stem}\n\n"
    "**Options:**\n{options}\n\n"
    "**Correct Answer:** {correct}\n\n"
    "Give the option the student would choose."
)

ANSWER_PREFILL = "**Student Answer:**"


class PromptBuilder:
    """Renders prompts via the tokenizer's chat template (Appendix A structure, adapted so the
    answer slot is a single option-key token)."""

    def __init__(self, tokenizer, enable_thinking: bool = False):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    @staticmethod
    def user_message(item: Item, level: str) -> str:
        options = "\n".join(f"{o.key}. {o.text}" for o in item.options)
        c = item.correct_option
        return USER_TEMPLATE.format(level=level, stem=item.stem.strip(), options=options,
                                    correct=f"{c.key}. {c.text}")

    def build_text(self, item: Item, level: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self.user_message(item, level)}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        return text + ANSWER_PREFILL

    def build_ids(self, item: Item, level: str) -> list[int]:
        """Token ids truncated at the answer slot (the next token is the option key)."""
        return self.tokenizer.encode(self.build_text(item, level), add_special_tokens=False)
