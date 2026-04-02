from __future__ import annotations

from pathlib import Path

import pytest

from core.data_loader import DatasetLoader


def test_load_txt_files(tmp_path: Path) -> None:
    """DatasetLoader must read .txt files from label subdirectories."""
    human_dir = tmp_path / "human"
    human_dir.mkdir()
    (human_dir / "sample.txt").write_text("Пример текста.", encoding="utf-8")

    ai_dir = tmp_path / "ai_gpt_fast"
    ai_dir.mkdir()
    (ai_dir / "sample.txt").write_text("Сгенерированный текст.", encoding="utf-8")

    loader = DatasetLoader(tmp_path)
    bundle = loader.load()
    df = bundle.dataframe

    assert len(df) == 2
    assert set(df["label"]) == {"human", "ai_gpt_fast"}


def test_load_csv_files(tmp_path: Path) -> None:
    """DatasetLoader must read .csv files with a 'text' column."""
    ai_dir = tmp_path / "ai_deepseek"
    ai_dir.mkdir()
    csv_path = ai_dir / "data.csv"
    csv_path.write_text("text\nПервый текст\nВторой текст\n", encoding="utf-8")

    human_dir = tmp_path / "human"
    human_dir.mkdir()
    (human_dir / "h.txt").write_text("Текст человека.", encoding="utf-8")

    loader = DatasetLoader(tmp_path)
    bundle = loader.load()
    df = bundle.dataframe

    assert len(df) == 3
    assert "ai_deepseek" in set(df["label"])


def test_load_empty_raises(tmp_path: Path) -> None:
    """Loading from an empty directory should raise ValueError."""
    loader = DatasetLoader(tmp_path)
    with pytest.raises(ValueError, match="No data found"):
        loader.load()


def test_build_binary_view_scenarios(tmp_path: Path) -> None:
    human_dir = tmp_path / "human"
    human_dir.mkdir()
    (human_dir / "a.txt").write_text("Человеческий текст.", encoding="utf-8")

    ai_dir = tmp_path / "ai_gpt_fast"
    ai_dir.mkdir()
    (ai_dir / "a.txt").write_text("ИИ текст.", encoding="utf-8")

    loader = DatasetLoader(tmp_path)
    df = loader.load().dataframe

    for scenario in ("human_vs_ai_all", "human_vs_ai_weak"):
        view = loader.build_binary_view(df, scenario)
        assert "target" in view.columns
        assert set(view["target"].unique()) == {0, 1}
