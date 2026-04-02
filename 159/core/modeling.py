from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from core.features import NUMERIC_FEATURE_NAMES

if TYPE_CHECKING:
    from core.features import FeatureExtractor


@dataclass
class EvalResult:
    model_name: str
    scenario: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    f1_std: float = 0.0
    roc_auc_std: float = 0.0


class TextModelTrainer:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def build_models(self) -> Dict[str, ClassifierMixin]:
        return {
            "log_reg": LogisticRegression(
                max_iter=2000, random_state=self.random_state, class_weight="balanced"
            ),
            "linear_svm": LinearSVC(
                max_iter=10000,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                random_state=self.random_state,
                class_weight="balanced",
            ),
        }

    @staticmethod
    def extract_probabilities(pipeline: Pipeline, X) -> np.ndarray:
        """Extract class-1 probabilities from any fitted pipeline."""
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "decision_function"):
            raw = pipeline.decision_function(X)
            return 1.0 / (1.0 + np.exp(-raw))
        return pipeline.predict(X).astype(float)

    @staticmethod
    def merge_text_and_features(
        texts: pd.Series, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Safely combine text and numeric features without index-alignment NaNs."""
        text_df = pd.DataFrame({"text": texts.values}).reset_index(drop=True)
        feat_df = (
            features.reset_index(drop=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        merged = pd.concat([text_df, feat_df], axis=1)
        return merged

    def run_experiment(
        self,
        scenario: str,
        extractor: FeatureExtractor,
        labels: pd.Series,
        text_series: pd.Series,
        n_splits: int = 5,
    ) -> Tuple[List[EvalResult], Dict[str, Pipeline], pd.DataFrame]:
        """Run stratified k-fold CV experiment.

        Returns (eval_results, trained_pipelines, full_dataset_frame).
        Pipelines in ``trained_pipelines`` are re-trained on the full dataset
        after cross-validation for deployment use.  ``full_dataset_frame``
        contains all labels and texts (not just the test fold).
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        base_models = self.build_models()
        model_metrics = {
            m: {"acc": [], "prec": [], "rec": [], "f1": [], "roc": [], "cm": []}
            for m in base_models
        }

        texts_arr = text_series.values
        labels_arr = labels.values

        for train_idx, test_idx in skf.split(texts_arr, labels_arr):
            x_train_txt = pd.Series(texts_arr[train_idx])
            x_test_txt = pd.Series(texts_arr[test_idx])
            y_train = labels_arr[train_idx]
            y_test = labels_arr[test_idx]

            x_train_f = extractor.fit_transform(x_train_txt).frame
            x_test_f = extractor.transform(x_test_txt).frame

            for model_name, model in base_models.items():
                pipeline = self.build_hybrid_pipeline(clone(model))
                train_df = self.merge_text_and_features(x_train_txt, x_train_f)
                test_df = self.merge_text_and_features(x_test_txt, x_test_f)

                pipeline.fit(train_df, y_train)
                preds = pipeline.predict(test_df)
                probs = self.extract_probabilities(pipeline, test_df)

                cm = confusion_matrix(y_test, preds).tolist()
                roc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else 0.5

                model_metrics[model_name]["acc"].append(accuracy_score(y_test, preds))
                model_metrics[model_name]["prec"].append(
                    precision_score(y_test, preds, zero_division=0)
                )
                model_metrics[model_name]["rec"].append(
                    recall_score(y_test, preds, zero_division=0)
                )
                model_metrics[model_name]["f1"].append(
                    f1_score(y_test, preds, zero_division=0)
                )
                model_metrics[model_name]["roc"].append(roc)
                model_metrics[model_name]["cm"].append(cm)

        results: List[EvalResult] = []
        trained: Dict[str, Pipeline] = {}

        x_all_f = extractor.fit_transform(text_series).frame

        for model_name, model in base_models.items():
            mean_acc = float(np.mean(model_metrics[model_name]["acc"]))
            mean_prec = float(np.mean(model_metrics[model_name]["prec"]))
            mean_rec = float(np.mean(model_metrics[model_name]["rec"]))
            mean_f1 = float(np.mean(model_metrics[model_name]["f1"]))
            mean_roc = float(np.mean(model_metrics[model_name]["roc"]))
            std_f1 = float(np.std(model_metrics[model_name]["f1"]))
            std_roc = float(np.std(model_metrics[model_name]["roc"]))
            sum_cm = np.sum(model_metrics[model_name]["cm"], axis=0).tolist()

            results.append(
                EvalResult(
                    model_name=model_name,
                    scenario=scenario,
                    accuracy=mean_acc,
                    precision=mean_prec,
                    recall=mean_rec,
                    f1=mean_f1,
                    roc_auc=mean_roc,
                    confusion_matrix=sum_cm,
                    f1_std=std_f1,
                    roc_auc_std=std_roc,
                )
            )

            final_pipeline = self.build_hybrid_pipeline(clone(model))
            final_train_df = self.merge_text_and_features(text_series, x_all_f)
            final_pipeline.fit(final_train_df, labels_arr)
            trained[model_name] = final_pipeline

        return results, trained, pd.DataFrame({"y_true": labels_arr, "text": texts_arr})

    def build_hybrid_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        text_pipe = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), max_features=4000, analyzer="word"
                    ),
                ),
            ]
        )

        numeric_pipe = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipe, "text"),
                ("num", numeric_pipe, NUMERIC_FEATURE_NAMES),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

        return Pipeline(steps=[("pre", preprocessor), ("clf", classifier)])
