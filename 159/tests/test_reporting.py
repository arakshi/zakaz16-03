from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.reporting import plot_model_comparison, save_results


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "log_reg",
                "scenario": "human_vs_ai_all",
                "f1": 0.9,
                "accuracy": 0.9,
            },
            {
                "model": "svm",
                "scenario": "human_vs_ai_all",
                "f1": 0.85,
                "accuracy": 0.85,
            },
            {
                "model": "log_reg",
                "scenario": "human_vs_ai_weak",
                "f1": 0.88,
                "accuracy": 0.87,
            },
        ]
    )


def test_save_results_creates_files(tmp_path: Path) -> None:
    df = _sample_results()
    save_results(df, tmp_path)

    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"
    assert csv_path.exists()
    assert json_path.exists()

    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)


def test_save_results_creates_dir(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "reports"
    save_results(_sample_results(), nested)
    assert (nested / "results.csv").exists()


def test_plot_model_comparison_creates_png(tmp_path: Path) -> None:
    df = _sample_results()
    out = plot_model_comparison(df, tmp_path)

    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0
