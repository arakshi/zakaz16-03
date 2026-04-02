from __future__ import annotations

from pathlib import Path

import pytest

from core.demo_data import populate_demo_dataset
from core.service import ProjectService


@pytest.fixture()
def service_env(tmp_path: Path) -> tuple[ProjectService, Path]:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    for d in (raw, processed, models, reports):
        d.mkdir()

    populate_demo_dataset(raw, overwrite=True)

    svc = ProjectService(
        raw_data_dir=raw,
        processed_path=processed / "merged.csv",
        models_dir=models,
        reports_dir=reports,
    )
    return svc, models


def test_train_and_evaluate_saves_extractor(
    service_env: tuple[ProjectService, Path],
) -> None:
    svc, models = service_env
    artifacts = svc.train_and_evaluate()

    assert (models / "feature_extractor.joblib").exists()
    assert artifacts.best_model_path.exists()
    assert artifacts.best_model_name


def test_analyze_text_after_training(
    service_env: tuple[ProjectService, Path],
) -> None:
    svc, models = service_env
    artifacts = svc.train_and_evaluate()

    result = svc.analyze_text(
        "Некоторый текст для проверки анализа после обучения.",
        artifacts.best_model_path,
    )
    assert result["predicted_class"] in ("human", "ai")
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert isinstance(result["explanation"], str)


def test_analyze_text_single_word(
    service_env: tuple[ProjectService, Path],
) -> None:
    svc, models = service_env
    artifacts = svc.train_and_evaluate()

    result = svc.analyze_text("слово", artifacts.best_model_path)
    assert result["predicted_class"] in ("human", "ai")
    assert 0.0 <= result["ai_probability"] <= 1.0


def test_analyze_text_digits_only(
    service_env: tuple[ProjectService, Path],
) -> None:
    svc, models = service_env
    artifacts = svc.train_and_evaluate()

    result = svc.analyze_text("12345 67890", artifacts.best_model_path)
    assert result["predicted_class"] in ("human", "ai")
    assert 0.0 <= result["ai_probability"] <= 1.0
