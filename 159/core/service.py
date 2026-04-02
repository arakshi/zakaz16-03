from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time

import joblib
import pandas as pd

from core.data_loader import DatasetLoader
from core.demo_data import populate_demo_dataset
from core.experiments import run_all_experiments
from core.features import FeatureExtractor
from core.interpretation import (
    build_hypothesis_conclusion,
    explain_single_prediction,
    extract_feature_importance,
)
from core.modeling import TextModelTrainer
from core.reporting import plot_model_comparison, save_results

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    results: pd.DataFrame
    best_model_name: str
    best_model_path: Path
    chart_path: Path
    hypothesis_text: str


class ProjectService:
    def __init__(
        self,
        raw_data_dir: Path,
        processed_path: Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> None:
        self.loader = DatasetLoader(raw_data_dir)
        self.extractor = FeatureExtractor()
        self.trainer = TextModelTrainer()
        self.processed_path = processed_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.raw_data_dir = raw_data_dir

    def save_text_sample(
        self, text: str, label: str, source_name: str = "ui_text"
    ) -> Path:
        safe_label = label.strip().lower()
        target_dir = self.raw_data_dir / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        file_path = target_dir / f"{source_name}_{ts}.txt"
        file_path.write_text(text, encoding="utf-8")
        return file_path

    def save_uploaded_file(self, filename: str, content: bytes, label: str) -> Path:
        safe_label = label.strip().lower()
        target_dir = self.raw_data_dir / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        ext = Path(filename).suffix.lower() or ".txt"
        out_path = target_dir / f"upload_{ts}{ext}"
        out_path.write_bytes(content)
        return out_path

    def add_test_demo_dataset(self, overwrite: bool = False) -> Dict[str, int]:
        return populate_demo_dataset(self.raw_data_dir, overwrite=overwrite)

    def load_and_prepare(self) -> pd.DataFrame:
        bundle = self.loader.load()
        bundle.save(self.processed_path)
        return bundle.dataframe

    def train_and_evaluate(self) -> TrainingArtifacts:
        df = self.load_and_prepare()
        results, trained_models, extractor = run_all_experiments(df, self.loader)
        self.extractor = extractor
        save_results(results, self.reports_dir)
        chart_path = plot_model_comparison(results, self.reports_dir)

        main_results = results[results["scenario"] == "human_vs_ai_all"]
        ranked = main_results.sort_values("f1", ascending=False)
        best_name = None
        for _, row in ranked.iterrows():
            name = row["model"]
            if name in trained_models and hasattr(
                trained_models[name], "predict_proba"
            ):
                best_name = name
                break
        if best_name is None:
            best_name = ranked.iloc[0]["model"]

        self.models_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.models_dir / f"best_{best_name}.joblib"
        joblib.dump(trained_models[best_name], best_path)

        extractor_path = self.models_dir / "feature_extractor.joblib"
        joblib.dump(self.extractor, extractor_path)

        imp = extract_feature_importance(trained_models[best_name], top_k=20)
        imp.to_csv(
            self.reports_dir / "feature_importance.csv", index=False, encoding="utf-8"
        )

        hypothesis = build_hypothesis_conclusion(results)
        (self.reports_dir / "hypothesis_conclusion.txt").write_text(
            hypothesis, encoding="utf-8"
        )

        return TrainingArtifacts(
            results=results,
            best_model_name=best_name,
            best_model_path=best_path,
            chart_path=chart_path,
            hypothesis_text=hypothesis,
        )

    def analyze_text(self, text: str, model_path: Path) -> Dict[str, str | float]:
        model = joblib.load(model_path)

        extractor_path = self.models_dir / "feature_extractor.joblib"
        if extractor_path.exists():
            self.extractor = joblib.load(extractor_path)
        else:
            logger.warning(
                "feature_extractor.joblib not found in %s — using a fresh "
                "extractor with empty corpus frequencies. Results may be unreliable. "
                "Re-train the model to generate the extractor artifact.",
                self.models_dir,
            )

        feature_row = self.extractor.transform([text]).frame
        row = self.trainer.merge_text_and_features(pd.Series([text]), feature_row)

        pred = int(model.predict(row)[0])
        prob = float(TextModelTrainer.extract_probabilities(model, row)[0])

        explanation = explain_single_prediction(feature_row.iloc[0].to_dict())
        return {
            "predicted_class": "ai" if pred == 1 else "human",
            "ai_probability": round(prob, 4),
            "explanation": explanation,
        }
