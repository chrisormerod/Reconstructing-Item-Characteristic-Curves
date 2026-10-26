"""Features baseline: TF-IDF of the item text plus readability / length features (textstat) and a few option
statistics, fed to ridge regression on the item parameter b. No neural model, no ability, no ICC; like the
transformer regression baseline it is compared with the simulators only through the predicted Difficulty."""
from __future__ import annotations

import re

import numpy as np

from llm_icc.data.item import Item

_WORD = re.compile(r"[A-Za-z0-9']+")

READABILITY = ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index",
               "automated_readability_index", "coleman_liau_index", "dale_chall_readability_score")
COUNTS = ("syllable_count", "lexicon_count", "sentence_count", "char_count", "difficult_words")


def item_text(item: Item, include_correct: bool = True) -> str:
    """Stem followed by the lettered options (and the key), the same information the encoder baseline sees."""
    text = item.stem.strip() + "\n" + "\n".join(f"{o.key}. {o.text.strip()}" for o in item.options)
    return text + (f"\nCorrect answer: {item.correct_option.key}" if include_correct else "")


def _text_stats(text: str) -> list[float]:
    import textstat

    out = [float(getattr(textstat, name)(text)) if text.strip() else 0.0 for name in READABILITY + COUNTS]
    n_words, n_sent = max(out[-4], 1.0), max(out[-3], 1.0)
    return out + [n_words / n_sent, out[-5] / n_words]  # words per sentence, syllables per word


def feature_names() -> list[str]:
    parts = [f"stem_{n}" for n in READABILITY + COUNTS + ("words_per_sentence", "syllables_per_word")]
    return parts + ["n_options", "mean_option_words", "correct_option_words", "correct_minus_mean_distractor_words",
                    "max_option_words", "mean_option_stem_overlap", "correct_option_stem_overlap", "key_index"]


def numeric_features(item: Item) -> list[float]:
    """Readability/length of the stem followed by option-structure statistics (see `feature_names`)."""
    stem_words = {w.lower() for w in _WORD.findall(item.stem)}
    lens = np.array([len(_WORD.findall(o.text)) for o in item.options], float)
    overlap = np.array([len({w.lower() for w in _WORD.findall(o.text)} & stem_words) / max(len(_WORD.findall(o.text)), 1)
                        for o in item.options])
    c = item.correct_index
    distractors = np.delete(lens, c)
    return _text_stats(item.stem) + [
        float(item.n_options), float(lens.mean()), float(lens[c]),
        float(lens[c] - (distractors.mean() if len(distractors) else lens[c])),
        float(lens.max()), float(overlap.mean()), float(overlap[c]), float(c)]


class FeaturesModel:
    """`fit(items)` on items with parameters, `predict_b(items)` on any items. Ridge strength is chosen by
    efficient leave-one-out CV on the training items only (the dev set is left for the shared recalibration)."""

    def __init__(self, config):
        self.cfg = config.features
        self.vectorizer = self.scaler = self.ridge = None

    def _matrix(self, items: list[Item], fit: bool):
        from scipy import sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        texts = [item_text(i, self.cfg.include_correct_answer) for i in items]
        num = np.array([numeric_features(i) for i in items], float)
        if fit:
            self.vectorizer = TfidfVectorizer(max_features=self.cfg.tfidf_max_features, ngram_range=(1, self.cfg.ngram_max),
                                              min_df=self.cfg.tfidf_min_df, sublinear_tf=True, lowercase=True)
            self.scaler = StandardScaler().fit(num)
            self.vectorizer.fit(texts)
        tfidf = self.vectorizer.transform(texts)
        scaled = self.scaler.transform(num) * self.cfg.numeric_weight
        return sparse.hstack([tfidf, sparse.csr_matrix(scaled)]).tocsr()

    def fit(self, items: list[Item]) -> "FeaturesModel":
        from sklearn.linear_model import RidgeCV

        X = self._matrix(items, fit=True)
        y = np.array([i.parameters.b for i in items], float)
        self.ridge = RidgeCV(alphas=np.logspace(self.cfg.alpha_log10_min, self.cfg.alpha_log10_max, 13)).fit(X, y)
        return self

    def predict_b(self, items: list[Item]) -> np.ndarray:
        return self.ridge.predict(self._matrix(items, fit=False))
