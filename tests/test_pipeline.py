from __future__ import annotations

from core.features import FeatureExtractor
from core.preprocessing import TextPreprocessor


def test_preprocessor_basic() -> None:
    p = TextPreprocessor()
    processed = p.process("Привет,   мир!!! Это   тест.")
    assert "  " not in processed.clean_text
    assert len(processed.sentences) >= 1
    assert len(processed.tokens) >= 2


def test_feature_extractor_output_columns() -> None:
    fx = FeatureExtractor()
    frame = fx.fit_transform(["Это простой тестовый текст.", "Еще один текст, немного длиннее и разнообразнее."]).frame
    required = {"word_count", "type_token_ratio", "punct_density_100w", "smoothness"}
    assert required.issubset(set(frame.columns))
    assert len(frame) == 2
