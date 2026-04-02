from __future__ import annotations

from core.features import FeatureExtractor, NUMERIC_FEATURE_NAMES


def test_fit_transform_separate_from_transform() -> None:
    """fit() on corpus A, then transform() on corpus B must use A's frequencies."""
    fx = FeatureExtractor()
    train = ["слово слово слово повторяется часто"]
    test = ["слово появляется один раз"]

    fx.fit(train)
    assert fx.corpus_freq_["слово"] == 3

    result = fx.transform(test).frame
    assert (
        result["rare_word_share"].iloc[0] < 1.0
    ), "After fit on train, 'слово' should not be considered rare in test"


def test_transform_without_fit_uses_empty_corpus() -> None:
    """Calling transform() before fit() should still work (empty corpus)."""
    fx = FeatureExtractor()
    result = fx.transform(["тестовый текст для проверки"]).frame
    assert result["rare_word_share"].iloc[0] == 1.0


def test_feature_names_match_constant() -> None:
    """All features produced by FeatureExtractor must match NUMERIC_FEATURE_NAMES."""
    fx = FeatureExtractor()
    frame = fx.fit_transform(["Пример текста для проверки имён признаков."]).frame
    assert list(frame.columns) == NUMERIC_FEATURE_NAMES


def test_fit_transform_equals_fit_then_transform() -> None:
    texts = ["Первый текст.", "Второй текст с дополнением."]
    fx1 = FeatureExtractor()
    combined = fx1.fit_transform(texts).frame

    fx2 = FeatureExtractor()
    fx2.fit(texts)
    separate = fx2.transform(texts).frame

    assert combined.equals(separate)
