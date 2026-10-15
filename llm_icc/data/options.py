from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Option:
    key: str  # display key, "A".."J"; must map to one token
    text: str
    is_correct: bool
