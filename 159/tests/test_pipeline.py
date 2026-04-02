from __future__ import annotations

from core.features import FeatureExtractor
from core.preprocessing import TextPreprocessor


def test_preprocessor_basic() -> None:
    p = TextPreprocessor()
    processed = p.process("Привет,   мир!!! Это   тест.")
    assert "  " not in processed.clean_text
    assert len(processed.sentences) >= 1
    assert len(processed.tokens) >= 2


def test_preprocessor_empty_text() -> None:
    p = TextPreprocessor()
    processed = p.process("")
    assert processed.clean_text == ""
    assert processed.sentences == []
    assert processed.tokens == []


def test_preprocessor_whitespace_only() -> None:
    p = TextPreprocessor()
    processed = p.process("   \t\n  ")
    assert processed.clean_text.strip() == ""


def test_preprocessor_special_chars_only() -> None:
    p = TextPreprocessor()
    processed = p.process("@#$%^&*")
    assert isinstance(processed.clean_text, str)


def test_preprocessor_preserves_russian_punctuation() -> None:
    p = TextPreprocessor()
    processed = p.process("Он сказал: «Привет!» — и ушёл.")
    assert "«" in processed.clean_text or "привет" in processed.clean_text


def test_preprocessor_nbsp_replaced() -> None:
    p = TextPreprocessor()
    processed = p.process("слово\xa0слово")
    assert "\xa0" not in processed.clean_text


def test_feature_extractor_output_columns() -> None:
    fx = FeatureExtractor()
    texts = [
        "Это простой тестовый текст.",
        "Еще один текст, немного длиннее и разнообразнее.",
    ]
    frame = fx.fit_transform(texts).frame
    required = {
        "avg_word_len",
        "type_token_ratio",
        "punct_density_100w",
        "smoothness",
    }
    assert required.issubset(set(frame.columns))
    assert len(frame) == 2


def test_feature_extractor_empty_text() -> None:
    fx = FeatureExtractor()
    frame = fx.fit_transform([""]).frame
    assert len(frame) == 1
    assert frame["avg_word_len"].iloc[0] == 0.0


def test_feature_extractor_single_word() -> None:
    fx = FeatureExtractor()
    frame = fx.fit_transform(["слово"]).frame
    assert len(frame) == 1
    assert frame["type_token_ratio"].iloc[0] == 1.0
