from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_results(results_df: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "results.csv", index=False, encoding="utf-8")
    (reports_dir / "results.json").write_text(results_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


def plot_model_comparison(results_df: pd.DataFrame, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="model", y="f1", hue="scenario")
    plt.title("Сравнение F1 по моделям и сценариям")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out = reports_dir / "f1_comparison.png"
    plt.savefig(out, dpi=140)
    plt.close()
    return out
