from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time

import joblib
import pandas as pd

from core.data_loader import DatasetLoader
from core.experiments import run_all_experiments
from core.features import FeatureExtractor
from core.interpretation import build_hypothesis_conclusion, explain_single_prediction, extract_feature_importance
from core.modeling import TextModelTrainer
from core.reporting import plot_model_comparison, save_results
from core.demo_data import populate_demo_dataset


@dataclass
class TrainingArtifacts:
    results: pd.DataFrame
    best_model_name: str
    best_model_path: Path
    chart_path: Path
    hypothesis_text: str


class ProjectService:
    def __init__(self, raw_data_dir: Path, processed_path: Path, models_dir: Path, reports_dir: Path) -> None:
        self.loader = DatasetLoader(raw_data_dir)
        self.extractor = FeatureExtractor()
        self.trainer = TextModelTrainer()
        self.processed_path = processed_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.raw_data_dir = raw_data_dir

    def save_text_sample(self, text: str, label: str, source_name: str = "ui_text") -> Path:
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
        results = run_all_experiments(df, self.loader)
        if results.empty:
            raise ValueError("Недостаточно сбалансированных данных для экспериментов: добавьте тексты обоих классов в каждый сценарий.")

        save_results(results, self.reports_dir)
        chart_path = plot_model_comparison(results, self.reports_dir)

        base = self.loader.build_binary_view(df, "human_vs_ai_all")
        feats = self.extractor.fit_transform(base["text"]).frame
        experiment, trained_models, _ = self.trainer.run_experiment("human_vs_ai_all", feats, base["target"], base["text"])
        if not experiment:
            raise ValueError("Не удалось обучить базовый сценарий human_vs_ai_all: проверьте баланс классов и объем данных.")

        ranked = sorted(experiment, key=lambda x: x.f1, reverse=True)
        proba_capable = [r for r in ranked if hasattr(trained_models[r.model_name], "predict_proba")]
        best = proba_capable[0] if proba_capable else ranked[0]

        self.models_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.models_dir / f"best_{best.model_name}.joblib"
        joblib.dump(trained_models[best.model_name], best_path)

        imp = extract_feature_importance(trained_models[best.model_name], top_k=20)
        imp.to_csv(self.reports_dir / "feature_importance.csv", index=False, encoding="utf-8")

        hypothesis = build_hypothesis_conclusion(results)
        (self.reports_dir / "hypothesis_conclusion.txt").write_text(hypothesis, encoding="utf-8")

        return TrainingArtifacts(
            results=results,
            best_model_name=best.model_name,
            best_model_path=best_path,
            chart_path=chart_path,
            hypothesis_text=hypothesis,
        )

    def analyze_text(self, text: str, model_path: Path) -> Dict[str, str | float]:
        model = joblib.load(model_path)
        feature_row = self.extractor.transform([text]).frame
        row = pd.DataFrame({"text": [text]}).join(feature_row)

        pred = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(row)[0, 1])
        elif hasattr(model, "decision_function"):
            raw = float(model.decision_function(row)[0])
            prob = 1.0 / (1.0 + pow(2.718281828, -raw))
        else:
            prob = float(pred)

        explanation = explain_single_prediction(feature_row.iloc[0].to_dict())
        return {
            "predicted_class": "ai" if pred == 1 else "human",
            "ai_probability": round(prob, 4),
            "explanation": explanation,
        }
