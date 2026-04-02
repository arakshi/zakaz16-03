from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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


class TextModelTrainer:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def build_models(self) -> Dict[str, ClassifierMixin]:
        return {
            "log_reg": LogisticRegression(max_iter=4000, random_state=self.random_state, class_weight="balanced"),
            "linear_svm": LinearSVC(random_state=self.random_state, max_iter=12000),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
        }

    @staticmethod
    def merge_text_and_features(texts: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Safely combine text and numeric features without index-alignment NaNs."""
        text_df = pd.DataFrame({"text": texts.values}).reset_index(drop=True)
        feat_df = features.reset_index(drop=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged = pd.concat([text_df, feat_df], axis=1)
        return merged

    def run_experiment(
        self,
        scenario: str,
        features: pd.DataFrame,
        labels: pd.Series,
        text_series: pd.Series,
        test_size: float = 0.2,
    ) -> Tuple[List[EvalResult], Dict[str, Pipeline], pd.DataFrame]:
        label_counts = labels.value_counts()
        if labels.nunique() < 2 or label_counts.min() < 2:
            return [], {}, pd.DataFrame()

        stratify_labels = labels if label_counts.min() >= 2 else None
        x_train_f, x_test_f, y_train, y_test, x_train_txt, x_test_txt = train_test_split(
            features,
            labels,
            text_series,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=self.random_state,
        )

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return [], {}, pd.DataFrame()

        results: List[EvalResult] = []
        trained: Dict[str, Pipeline] = {}

        for model_name, model in self.build_models().items():
            pipeline = self._build_hybrid_pipeline(model)
            train_df = self.merge_text_and_features(x_train_txt, x_train_f)
            test_df = self.merge_text_and_features(x_test_txt, x_test_f)

            try:
                pipeline.fit(train_df, y_train.values)
            except ValueError:
                continue

            preds = pipeline.predict(test_df)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(test_df)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                raw = pipeline.decision_function(test_df)
                probs = (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-9)
            else:
                probs = preds

            cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
            roc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else 0.5
            results.append(
                EvalResult(
                    model_name=model_name,
                    scenario=scenario,
                    accuracy=accuracy_score(y_test, preds),
                    precision=precision_score(y_test, preds, zero_division=0),
                    recall=recall_score(y_test, preds, zero_division=0),
                    f1=f1_score(y_test, preds, zero_division=0),
                    roc_auc=roc,
                    confusion_matrix=cm,
                )
            )
            trained[model_name] = pipeline

        return results, trained, pd.DataFrame({"y_true": y_test.values, "text": x_test_txt.values})

    def _build_hybrid_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        text_pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=4000, analyzer="word")),
            ]
        )

        numeric_pipe = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipe, "text"),
                (
                    "num",
                    numeric_pipe,
                    [
                        "word_count",
                        "unique_word_count",
                        "avg_word_len",
                        "type_token_ratio",
                        "long_word_share",
                        "rare_word_share",
                        "sentence_count",
                        "avg_sentence_len",
                        "std_sentence_len",
                        "short_sentence_share",
                        "long_sentence_share",
                        "service_word_freq",
                        "repeat_ratio",
                        "template_ratio",
                        "intro_phrase_share",
                        "smoothness",
                        "punct_density_100w",
                        "comma_freq",
                        "dot_freq",
                        "colon_freq",
                        "dash_freq",
                        "brackets_freq",
                        "quotes_freq",
                        "question_freq",
                        "exclamation_freq",
                    ],
                ),
            ],
            remainder="drop",
        )

        return Pipeline(steps=[("pre", preprocessor), ("clf", classifier)])
