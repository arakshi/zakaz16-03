from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.base import clone
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

from core.config import STRONG_AI_LABELS, WEAK_AI_LABELS
from core.data_loader import DatasetLoader
from core.features import FeatureExtractor
from core.modeling import EvalResult, TextModelTrainer


def run_all_experiments(
    df: pd.DataFrame, loader: DatasetLoader
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], FeatureExtractor]:
    """Run all experiment scenarios and return metrics, trained models, and extractor.

    The trained pipelines and extractor correspond to the ``human_vs_ai_all``
    scenario so that the caller can persist the best model without re-training.
    """
    extractor = FeatureExtractor()
    trainer = TextModelTrainer()
    all_rows: List[Dict] = []
    all_trained: Dict[str, Pipeline] = {}

    for scenario in ["human_vs_ai_all", "human_vs_ai_weak", "human_vs_ai_strong"]:
        subset = loader.build_binary_view(df, scenario)
        results, trained, _ = trainer.run_experiment(
            scenario, extractor, subset["target"], subset["text"]
        )
        all_rows.extend(_eval_to_rows(results))
        if scenario == "human_vs_ai_all":
            all_trained = trained

    all_rows.extend(_cross_generalization(df, extractor, trainer))
    all_rows.extend(_length_robustness(df, extractor, trainer))

    if "artificial_humanized" in set(df["label"]):
        all_rows.extend(_humanized_test(df, extractor, trainer))

    return pd.DataFrame(all_rows), all_trained, extractor


def _eval_to_rows(results: List[EvalResult]) -> List[Dict]:
    rows: List[Dict] = []
    for r in results:
        row = asdict(r)
        row["model"] = row.pop("model_name")
        rows.append(row)
    return rows


def _cross_generalization(
    df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer
) -> List[Dict]:
    out: List[Dict] = []

    human_df = df[df["label"] == "human"]
    human_train, human_test = train_test_split(human_df, test_size=0.5, random_state=42)

    weak_ai = df[df["label"].isin(WEAK_AI_LABELS)]
    strong_ai = df[df["label"].isin(STRONG_AI_LABELS)]

    # Scenario D: train on weak, test on strong
    train_d = pd.concat([human_train, weak_ai]).copy()
    test_d = pd.concat([human_test, strong_ai]).copy()
    out.extend(
        _train_test_manual(
            train_d, test_d, extractor, trainer, "train_weak_test_strong"
        )
    )

    # Scenario E: train on strong, test on weak
    train_e = pd.concat([human_train, strong_ai]).copy()
    test_e = pd.concat([human_test, weak_ai]).copy()
    out.extend(
        _train_test_manual(
            train_e, test_e, extractor, trainer, "train_strong_test_weak"
        )
    )

    return out


def _train_test_manual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extractor: FeatureExtractor,
    trainer: TextModelTrainer,
    scenario: str,
) -> List[Dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = train_df["label"].apply(lambda x: 0 if x == "human" else 1)
    test_df["target"] = test_df["label"].apply(lambda x: 0 if x == "human" else 1)

    x_train = extractor.fit_transform(train_df["text"]).frame
    x_test = extractor.transform(test_df["text"]).frame
    rows: List[Dict] = []
    base_models = trainer.build_models()

    for model_name, model in base_models.items():
        pipeline = trainer.build_hybrid_pipeline(clone(model))
        train_join = trainer.merge_text_and_features(train_df["text"], x_train)
        test_join = trainer.merge_text_and_features(test_df["text"], x_test)
        pipeline.fit(train_join, train_df["target"].values)
        preds = pipeline.predict(test_join)
        probs = TextModelTrainer.extract_probabilities(pipeline, test_join)

        rows.append(
            {
                "model": model_name,
                "scenario": scenario,
                "accuracy": accuracy_score(test_df["target"], preds),
                "precision": precision_score(test_df["target"], preds, zero_division=0),
                "recall": recall_score(test_df["target"], preds, zero_division=0),
                "f1": f1_score(test_df["target"], preds, zero_division=0),
                "roc_auc": (
                    roc_auc_score(test_df["target"], probs)
                    if len(set(test_df["target"])) > 1
                    else 0.5
                ),
                "confusion_matrix": confusion_matrix(test_df["target"], preds).tolist(),
            }
        )
    return rows


def _length_robustness(
    df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer
) -> List[Dict]:
    rows: List[Dict] = []
    base = df.copy()
    base["target"] = base["label"].apply(lambda x: 0 if x == "human" else 1)
    base["wc"] = base["text"].str.split().apply(len)

    bins = {
        "len_le_100": base[base["wc"] <= 100],
        "len_100_300": base[(base["wc"] > 100) & (base["wc"] <= 300)],
        "len_gt_300": base[base["wc"] > 300],
    }

    for name, split in bins.items():
        if split["target"].nunique() < 2 or len(split) < 20:
            continue
        n_folds = min(5, split["target"].value_counts().min())
        if n_folds < 2:
            continue
        results, _, _ = trainer.run_experiment(
            f"length_robustness_{name}",
            extractor,
            split["target"],
            split["text"],
            n_splits=n_folds,
        )
        rows.extend(_eval_to_rows(results))
    return rows


def _humanized_test(
    df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer
) -> List[Dict]:
    human_df = df[df["label"] == "human"]
    human_train, human_test = train_test_split(human_df, test_size=0.5, random_state=42)

    other_train = df[(df["label"] != "human") & (df["label"] != "artificial_humanized")]
    art_hum = df[df["label"] == "artificial_humanized"]

    train_h = pd.concat([human_train, other_train]).copy()
    test_h = pd.concat([human_test, art_hum]).copy()

    return _train_test_manual(train_h, test_h, extractor, trainer, "humanized_ai_test")
