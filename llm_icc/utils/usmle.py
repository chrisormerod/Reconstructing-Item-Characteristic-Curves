"""USMLE spreadsheets -> ItemBank."""
from __future__ import annotations

import logging
import string
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option

log = logging.getLogger(__name__)

KEYS = list(string.ascii_uppercase[:10])
OPTION_COLS = [f"Answer__{k}" for k in KEYS]
CONTENT_COLS = ["ItemNum", "ItemStem_Text", *OPTION_COLS, "Answer_Key", "Answer_Text", "ItemType", "EXAM"]


class ItemSource(ABC):
    @abstractmethod
    def load(self) -> ItemBank: ...


def _id(x) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).strip()


def _text(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


class USMLEParser(ItemSource):
    """Reads train_final / test_final / gold_final. `Response_Time` is never read."""

    def __init__(self, usmle_dir: str | Path):
        self.dir = Path(usmle_dir)

    def _read(self, name: str, cols: list[str]) -> pd.DataFrame:
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_excel(path, engine="openpyxl", usecols=cols)  # ignores ~$ lock files by name

    def load(self) -> ItemBank:
        train = self._read("train_final.xlsx", CONTENT_COLS + ["Difficulty"])
        test = self._read("test_final.xlsx", CONTENT_COLS)
        gold = self._read("gold_final.xlsx", ["ItemNum", "Difficulty"])

        test_ids = {_id(x) for x in test["ItemNum"]}
        gold_ids = {_id(x) for x in gold["ItemNum"]}
        if test_ids != gold_ids:
            raise ValueError(f"test and gold ItemNum sets differ: {len(test_ids ^ gold_ids)} ids not shared")
        gold_map = {_id(i): float(d) for i, d in zip(gold["ItemNum"], gold["Difficulty"])}

        items: dict[str, Item] = {}
        for source, df in (("train", train), ("test", test)):
            for row in df.to_dict("records"):
                item = self._parse_row(row, source, gold_map)
                if item.item_id in items:
                    raise ValueError(f"Duplicate ItemNum {item.item_id}")
                items[item.item_id] = item
        return ItemBank(items)

    def _parse_row(self, row: dict, source: str, gold_map: dict[str, float]) -> Item:
        item_id = _id(row["ItemNum"])
        texts = [_text(row[c]) for c in OPTION_COLS]
        n = sum(1 for t in texts if t)
        if any(not t for t in texts[:n]) or any(t for t in texts[n:]):
            raise ValueError(f"Item {item_id}: options are not contiguous from A")
        if not 4 <= n <= 10:
            raise ValueError(f"Item {item_id}: {n} options (expected 4-10)")
        key = _text(row["Answer_Key"]).upper()
        if key not in KEYS[:n]:
            raise ValueError(f"Item {item_id}: Answer_Key {key!r} does not point to an option")
        if texts[KEYS.index(key)] != _text(row["Answer_Text"]):
            log.warning("Item %s: Answer_Text differs from option %s text; Answer_Key wins", item_id, key)
        options = [Option(KEYS[i], texts[i], KEYS[i] == key) for i in range(n)]
        difficulty = float(row["Difficulty"]) if source == "train" else gold_map[item_id]
        return Item(
            item_id=item_id, stem=_text(row["ItemStem_Text"]), options=options, parameters=None,
            reference_difficulty=difficulty,
            metadata={"source": source, "item_type": _text(row["ItemType"]), "exam": _text(row["EXAM"])})


def describe_bank(bank: ItemBank) -> dict:
    """Counts for the parse-stage log."""
    df = pd.DataFrame([{"source": i.metadata["source"], "n": i.n_options, "type": i.metadata["item_type"],
                        "exam": i.metadata["exam"]} for i in bank])
    return {
        "n_items": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "option_counts": {s: g["n"].value_counts().sort_index().to_dict() for s, g in df.groupby("source")},
        "item_type": df["type"].value_counts().to_dict(),
        "exam": df["exam"].value_counts().to_dict(),
    }
