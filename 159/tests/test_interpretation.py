from __future__ import annotations

import pandas as pd
import pytest

from core.features import FeatureExtractor
from core.interpretation import (
    build_hypothesis_conclusion,
    explain_single_prediction,
    extract_feature_importance,
)
from core.modeling import TextModelTrainer


def _train_pipeline():
    """Train a minimal pipeline for feature-importance tests."""
    trainer = TextModelTrainer()
    extractor = FeatureExtractor()
    texts = pd.Series(
        [
            "Человеческий текст с авторским стилем.",
            "Ещё один пример текста от человека.",
            "Сгенерированный текст от модели ИИ.",
            "Ещё один AI-текст для обучения.",
        ]
    )
    labels = pd.Series([0, 0, 1, 1])
    features = extractor.fit_transform(texts).frame
    merged = trainer.merge_text_and_features(texts, features)

    models = trainer.build_models()
    pipelines = {}
    for name, model in models.items():
        pipe = trainer.build_hybrid_pipeline(model)
        pipe.fit(merged, labels.values)
        pipelines[name] = pipe
    return pipelines


@pytest.fixture(scope="module")
def trained_pipelines():
    return _train_pipeline()


def test_extract_feature_importance_coef(trained_pipelines) -> None:
    pipe = trained_pipelines["log_reg"]
    imp = extract_feature_importance(pipe, top_k=5)
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "weight" in imp.columns
    assert len(imp) <= 5
    for name in imp["feature"]:
        assert not name.startswith("tfidf_"), "should use real feature names"


def test_extract_feature_importance_importances(trained_pipelines) -> None:
    pipe = trained_pipelines["random_forest"]
    imp = extract_feature_importance(pipe, top_k=5)
    assert isinstance(imp, pd.DataFrame)
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) <= 5
    for name in imp["feature"]:
        assert not name.startswith("tfidf_"), "should use real feature names"


def test_build_hypothesis_conclusion_weak_better() -> None:
    rows = [
        {"scenario": "human_vs_ai_weak", "f1": 0.95},
        {"scenario": "human_vs_ai_strong", "f1": 0.80},
        {"scenario": "human_vs_ai_all", "f1": 0.88},
    ]
    result = build_hypothesis_conclusion(pd.DataFrame(rows))
    assert "weak" in result.lower() or "гипотез" in result.lower()
    assert isinstance(result, str)


def test_build_hypothesis_conclusion_strong_better() -> None:
    rows = [
        {"scenario": "human_vs_ai_weak", "f1": 0.70},
        {"scenario": "human_vs_ai_strong", "f1": 0.90},
        {"scenario": "human_vs_ai_all", "f1": 0.80},
    ]
    result = build_hypothesis_conclusion(pd.DataFrame(rows))
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_hypothesis_conclusion_no_data() -> None:
    rows = [{"scenario": "human_vs_ai_all", "f1": 0.88}]
    result = build_hypothesis_conclusion(pd.DataFrame(rows))
    assert "недостаточно" in result.lower()


def test_explain_single_prediction_no_anomalies() -> None:
    features = {
        "template_ratio": 0.01,
        "type_token_ratio": 0.7,
        "smoothness": 0.3,
        "punct_density_100w": 8.0,
    }
    result = explain_single_prediction(features)
    assert "аномалий" in result.lower() or "TF-IDF" in result


def test_explain_single_prediction_with_signals() -> None:
    features = {
        "template_ratio": 0.5,
        "type_token_ratio": 0.3,
        "smoothness": 0.9,
        "punct_density_100w": 1.0,
    }
    result = explain_single_prediction(features)
    assert "шаблонность" in result
    assert "разнообразие" in result
