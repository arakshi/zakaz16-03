from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from razdel import sentenize, tokenize


@dataclass
class ProcessedText:
    original: str
    clean_text: str
    sentences: List[str]
    tokens: List[str]


class TextPreprocessor:
    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def process(self, text: str) -> ProcessedText:
        cleaned = self._normalize(text)
        sentences = [s.text.strip() for s in sentenize(cleaned) if s.text.strip()]
        token_list = [t.text for t in tokenize(cleaned) if t.text.strip()]
        return ProcessedText(original=text, clean_text=cleaned, sentences=sentences, tokens=token_list)

    def _normalize(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s.,:;!?()\-—\"'«»]", "", text, flags=re.UNICODE)
        if self.lowercase:
            text = text.lower()
        return text
