"""Tabular item files (USMLE spreadsheets by default) -> ItemBank. File names and columns come from `data.files` / `data.columns`."""
from __future__ import annotations

import logging
import string
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from llm_icc.config.schema import DataColumnsConfig, DataFilesConfig
from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option

log = logging.getLogger(__name__)

KEYS = list(string.ascii_uppercase[:10])


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
    """Reads the train, test and (optional) gold files named in `files`, with columns named in `columns`.

    Defaults are the USMLE layout. Other datasets only need different names; `Response_Time` and any other
    column that is not mapped is never read.
    """

    def __init__(self, data_dir: str | Path, files: DataFilesConfig | None = None,
                 columns: DataColumnsConfig | None = None):
        self.dir = Path(data_dir)
        self.files = files or DataFilesConfig()
        self.cols = columns or DataColumnsConfig()
        self.option_cols = [f"{self.cols.option_prefix}{k}" for k in KEYS]

    def _read(self, name: str, cols: list[str]) -> pd.DataFrame:
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        wanted = set(cols)
        pick = lambda c: c in wanted  # noqa: E731  (only mapped columns are read; ignores ~$ lock files by name)
        ext = path.suffix.lower()
        if ext in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(path, usecols=pick)
        elif ext in (".csv", ".tsv"):
            df = pd.read_csv(path, usecols=pick, sep="\t" if ext == ".tsv" else ",")
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"{path}: unsupported extension {ext!r} (use .xlsx, .csv, .tsv or .parquet)")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        return df

    def _content_cols(self) -> list[str]:
        c = self.cols
        return [x for x in [c.id, c.stem, *self.option_cols, c.key, c.key_text, c.item_type, c.exam] if x]

    def load(self) -> ItemBank:
        c, f = self.cols, self.files
        if f.gold:
            train = self._read(f.train, self._content_cols() + [c.difficulty])
            test = self._read(f.test, self._content_cols())
            gold = self._read(f.gold, [c.id, c.difficulty])
        else:  # the test file carries its own difficulties
            train = self._read(f.train, self._content_cols() + [c.difficulty])
            test = self._read(f.test, self._content_cols() + [c.difficulty])
            gold = test[[c.id, c.difficulty]]

        test_ids = {_id(x) for x in test[c.id]}
        gold_ids = {_id(x) for x in gold[c.id]}
        if test_ids != gold_ids:
            raise ValueError(f"test and gold {c.id} sets differ: {len(test_ids ^ gold_ids)} ids not shared")
        gold_map = {_id(i): float(d) for i, d in zip(gold[c.id], gold[c.difficulty])}

        items: dict[str, Item] = {}
        for source, df in (("train", train), ("test", test)):
            for row in df.to_dict("records"):
                item = self._parse_row(row, source, gold_map)
                if item.item_id in items:
                    raise ValueError(f"Duplicate {c.id} {item.item_id}")
                items[item.item_id] = item
        return ItemBank(items)

    def _parse_row(self, row: dict, source: str, gold_map: dict[str, float]) -> Item:
        c = self.cols
        item_id = _id(row[c.id])
        texts = [_text(row[x]) for x in self.option_cols]
        n = sum(1 for t in texts if t)
        if any(not t for t in texts[:n]) or any(t for t in texts[n:]):
            raise ValueError(f"Item {item_id}: options are not contiguous from A")
        if not 4 <= n <= 10:
            raise ValueError(f"Item {item_id}: {n} options (expected 4-10)")
        key = _text(row[c.key]).upper()
        if key not in KEYS[:n]:
            raise ValueError(f"Item {item_id}: {c.key} {key!r} does not point to an option")
        if c.key_text and texts[KEYS.index(key)] != _text(row[c.key_text]):
            log.warning("Item %s: %s differs from option %s text; %s wins", item_id, c.key_text, key, c.key)
        options = [Option(KEYS[i], texts[i], KEYS[i] == key) for i in range(n)]
        difficulty = float(row[c.difficulty]) if source == "train" else gold_map[item_id]
        return Item(
            item_id=item_id, stem=_text(row[c.stem]), options=options, parameters=None,
            reference_difficulty=difficulty,
            metadata={"source": source, "item_type": _text(row[c.item_type]) if c.item_type else "",
                      "exam": _text(row[c.exam]) if c.exam else ""})


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
