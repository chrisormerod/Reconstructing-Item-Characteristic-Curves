from __future__ import annotations

import argparse

STAGES = ["parse", "targets", "sft", "train", "predict", "evaluate", "all"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-simulated item characteristic curves")
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--stage", choices=STAGES, default="all",
                   help="run the pipeline up to and including this stage (default: all)")
    p.add_argument("--fold", type=int, default=None, help="run a single fold (default: all folds)")
    p.add_argument("--resume", default=None, help="checkpoint (LoRA adapter, or RoBERTa model dir) to load instead of a fresh model")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + build targets + one forward pass, no training")
    return p
