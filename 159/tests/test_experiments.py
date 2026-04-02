from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data_loader import DatasetLoader
from core.demo_data import populate_demo_dataset
from core.experiments import run_all_experiments


def test_run_all_experiments_returns_trained_models(tmp_path: Path) -> None:
    """run_all_experiments must return non-empty trained_models and a fitted extractor."""
    raw = tmp_path / "raw"
    populate_demo_dataset(raw, overwrite=True)

    loader = DatasetLoader(raw)
    df = loader.load().dataframe

    results, trained_models, extractor = run_all_experiments(df, loader)

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "model" in results.columns
    assert "scenario" in results.columns
    assert "f1" in results.columns
    assert "f1_std" in results.columns
    assert "roc_auc_std" in results.columns

    assert len(trained_models) > 0
    for name, pipeline in trained_models.items():
        assert hasattr(pipeline, "predict"), f"{name} pipeline missing predict"

    assert len(extractor.corpus_freq_) > 0
