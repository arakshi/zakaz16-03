from __future__ import annotations

import pandas as pd
import pytest

from core.features import FeatureExtractor
from core.modeling import TextModelTrainer


def test_run_experiment_returns_valid_results(sample_df: pd.DataFrame) -> None:
    """run_experiment should return one EvalResult per model with valid metrics."""
    trainer = TextModelTrainer()
    extractor = FeatureExtractor()

    results, trained, full_df = trainer.run_experiment(
        "test_scenario",
        extractor,
        sample_df["target"],
        sample_df["text"],
        n_splits=2,
    )

    model_names = {r.model_name for r in results}
    assert model_names == set(trainer.build_models().keys())

    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert 0.0 <= r.roc_auc <= 1.0
        assert r.scenario == "test_scenario"

    assert len(trained) == len(results)
    assert len(full_df) == len(sample_df)


def test_run_experiment_trains_pipelines_on_full_data(
    sample_df: pd.DataFrame,
) -> None:
    """Trained pipelines returned after CV must be able to predict on new data."""
    trainer = TextModelTrainer()
    extractor = FeatureExtractor()

    _, trained, _ = trainer.run_experiment(
        "test",
        extractor,
        sample_df["target"],
        sample_df["text"],
        n_splits=2,
    )

    new_text = "Совершенно новый текст для проверки предсказания."
    features = extractor.transform([new_text]).frame
    row = trainer.merge_text_and_features(pd.Series([new_text]), features)

    for name, pipeline in trained.items():
        pred = pipeline.predict(row)
        assert pred[0] in (0, 1), f"Pipeline {name} returned unexpected prediction"
