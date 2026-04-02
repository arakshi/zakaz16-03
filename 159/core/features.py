from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from core.preprocessing import TextPreprocessor

RUS_STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
}
INTRO_PHRASES = {
    "возможно",
    "вероятно",
    "кажется",
    "кстати",
    "однако",
    "например",
    "вообще",
    "безусловно",
}


NUMERIC_FEATURE_NAMES: List[str] = [
    "avg_word_len",
    "type_token_ratio",
    "long_word_share",
    "rare_word_share",
    "avg_sentence_len",
    "std_sentence_len",
    "short_sentence_share",
    "long_sentence_share",
    "service_word_freq",
    "repeat_ratio",
    "template_ratio",
    "intro_phrase_share",
    "smoothness",
    "punct_density_100w",
    "comma_freq",
    "dot_freq",
    "colon_freq",
    "dash_freq",
    "brackets_freq",
    "quotes_freq",
    "question_freq",
    "exclamation_freq",
]


@dataclass
class FeatureSet:
    frame: pd.DataFrame


class FeatureExtractor:
    def __init__(self, preprocessor: TextPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.corpus_freq_ = Counter()

    def fit(self, texts: Iterable[str]) -> FeatureExtractor:
        processed = [self.preprocessor.process(t) for t in texts]
        self.corpus_freq_ = self._corpus_word_frequency(processed)
        return self

    def transform(self, texts: Iterable[str]) -> FeatureSet:
        processed = [self.preprocessor.process(t) for t in texts]
        rows = [
            self._single_features(
                doc.clean_text, doc.sentences, doc.tokens, self.corpus_freq_
            )
            for doc in processed
        ]
        return FeatureSet(frame=pd.DataFrame(rows).fillna(0.0))

    def fit_transform(self, texts: Iterable[str]) -> FeatureSet:
        return self.fit(texts).transform(texts)

    def _corpus_word_frequency(self, docs) -> Counter:
        all_words = []
        for doc in docs:
            all_words.extend([t for t in doc.tokens if t.isalpha()])
        return Counter(all_words)

    def _single_features(
        self,
        clean_text: str,
        sentences: List[str],
        tokens: List[str],
        corpus_freq: Counter,
    ) -> Dict[str, float]:
        words = [t for t in tokens if t.isalpha()]
        word_count = len(words)
        unique_count = len(set(words))
        sentence_lens = [
            len([t for t in s.split() if t.strip(string.punctuation)])
            for s in sentences
        ] or [0]

        long_words = [w for w in words if len(w) >= 7]
        rare_words = [w for w in words if corpus_freq.get(w, 0) <= 1]
        repeated_ratio = (word_count - unique_count) / word_count if word_count else 0.0

        punct_marks = ",.:-—()\"'«»!?"
        punct_counts = Counter(ch for ch in clean_text if ch in punct_marks)

        boilerplate_ngrams = self._template_ratio(words)
        intro_count = sum(1 for w in words if w in INTRO_PHRASES)

        return {
            "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
            "type_token_ratio": (unique_count / word_count) if word_count else 0.0,
            "long_word_share": (len(long_words) / word_count) if word_count else 0.0,
            "rare_word_share": (len(rare_words) / word_count) if word_count else 0.0,
            "avg_sentence_len": float(np.mean(sentence_lens)) if sentence_lens else 0.0,
            "std_sentence_len": float(np.std(sentence_lens)) if sentence_lens else 0.0,
            "short_sentence_share": (
                float(np.mean([slen <= 7 for slen in sentence_lens]))
                if sentence_lens
                else 0.0
            ),
            "long_sentence_share": (
                float(np.mean([slen >= 20 for slen in sentence_lens]))
                if sentence_lens
                else 0.0
            ),
            "service_word_freq": self._service_word_freq(words),
            "repeat_ratio": repeated_ratio,
            "template_ratio": boilerplate_ngrams,
            "intro_phrase_share": (intro_count / word_count) if word_count else 0.0,
            "smoothness": self._smoothness(sentence_lens),
            "punct_density_100w": (
                (sum(punct_counts.values()) / word_count * 100) if word_count else 0.0
            ),
            "comma_freq": punct_counts.get(",", 0),
            "dot_freq": punct_counts.get(".", 0),
            "colon_freq": punct_counts.get(":", 0),
            "dash_freq": punct_counts.get("-", 0) + punct_counts.get("—", 0),
            "brackets_freq": punct_counts.get("(", 0) + punct_counts.get(")", 0),
            "quotes_freq": punct_counts.get('"', 0)
            + punct_counts.get("'", 0)
            + punct_counts.get("«", 0)
            + punct_counts.get("»", 0),
            "question_freq": punct_counts.get("?", 0),
            "exclamation_freq": punct_counts.get("!", 0),
        }

    @staticmethod
    def _service_word_freq(words: List[str]) -> float:
        if not words:
            return 0.0
        return sum(1 for w in words if w in RUS_STOPWORDS) / len(words)

    @staticmethod
    def _template_ratio(words: List[str], n: int = 3) -> float:
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(v for v in counts.values() if v > 1)
        return repeated / len(ngrams)

    @staticmethod
    def _smoothness(sentence_lens: List[int]) -> float:
        if len(sentence_lens) < 2:
            return 0.5
        diffs = np.diff(sentence_lens)
        return float(1 / (1 + np.mean(np.abs(diffs))))
