from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_loader import DatasetLoader
from core.features import FeatureExtractor
from core.modeling import EvalResult, TextModelTrainer


def run_all_experiments(df: pd.DataFrame, loader: DatasetLoader) -> pd.DataFrame:
    extractor = FeatureExtractor()
    trainer = TextModelTrainer()
    all_rows: List[Dict] = []

    for scenario in ["human_vs_ai_all", "human_vs_ai_weak", "human_vs_ai_strong"]:
        subset = loader.build_binary_view(df, scenario)
        features = extractor.fit_transform(subset["text"]).frame
        results, _, _ = trainer.run_experiment(scenario, features, subset["target"], subset["text"])
        all_rows.extend(_eval_to_rows(results))

    all_rows.extend(_cross_generalization(df, extractor, trainer))
    all_rows.extend(_length_robustness(df, extractor, trainer))

    if "artificial_humanized" in set(df["label"]):
        all_rows.extend(_humanized_test(df, extractor, trainer))

    return pd.DataFrame(all_rows)


def _eval_to_rows(results: List[EvalResult]) -> List[Dict]:
    rows: List[Dict] = []
    for r in results:
        row = asdict(r)
        row["model"] = row.pop("model_name")
        rows.append(row)
    return rows


def _cross_generalization(df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer) -> List[Dict]:
    out: List[Dict] = []
    # D
    train_d = df[df["label"].isin(["human", "ai_weak"])].copy()
    test_d = df[df["label"].isin(["human", "ai_strong"])].copy()
    out.extend(_train_test_manual(train_d, test_d, extractor, trainer, "train_weak_test_strong"))
    # E
    train_e = df[df["label"].isin(["human", "ai_strong"])].copy()
    test_e = df[df["label"].isin(["human", "ai_weak"])].copy()
    out.extend(_train_test_manual(train_e, test_e, extractor, trainer, "train_strong_test_weak"))
    return out


def _train_test_manual(train_df: pd.DataFrame, test_df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer, scenario: str) -> List[Dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = train_df["label"].apply(lambda x: 0 if x == "human" else 1)
    test_df["target"] = test_df["label"].apply(lambda x: 0 if x == "human" else 1)

    x_train = extractor.fit_transform(train_df["text"]).frame
    x_test = extractor.transform(test_df["text"]).frame
    rows: List[Dict] = []

    for model_name, model in trainer.build_models().items():
        pipeline = trainer._build_hybrid_pipeline(model)
        train_join = pd.DataFrame({"text": train_df["text"].values}).join(x_train.reset_index(drop=True))
        test_join = pd.DataFrame({"text": test_df["text"].values}).join(x_test.reset_index(drop=True))
        pipeline.fit(train_join, train_df["target"].values)
        preds = pipeline.predict(test_join)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(test_join)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            raw = pipeline.decision_function(test_join)
            probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            probs = preds

        rows.append({
            "model": model_name,
            "scenario": scenario,
            "accuracy": accuracy_score(test_df["target"], preds),
            "precision": precision_score(test_df["target"], preds, zero_division=0),
            "recall": recall_score(test_df["target"], preds, zero_division=0),
            "f1": f1_score(test_df["target"], preds, zero_division=0),
            "roc_auc": roc_auc_score(test_df["target"], probs) if len(set(test_df["target"])) > 1 else 0.5,
            "confusion_matrix": confusion_matrix(test_df["target"], preds).tolist(),
        })
    return rows


def _length_robustness(df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer) -> List[Dict]:
    rows: List[Dict] = []
    base = df[df["label"].isin(["human", "ai_weak", "ai_strong"])].copy()
    base["target"] = base["label"].apply(lambda x: 0 if x == "human" else 1)
    base["wc"] = base["text"].str.split().apply(len)

    bins = {
        "len_le_100": base[base["wc"] <= 100],
        "len_100_300": base[(base["wc"] > 100) & (base["wc"] <= 300)],
        "len_gt_300": base[base["wc"] > 300],
    }

    for name, split in bins.items():
        if split["target"].nunique() < 2 or len(split) < 8:
            continue
        features = extractor.fit_transform(split["text"]).frame
        results, _, _ = trainer.run_experiment(f"length_robustness_{name}", features, split["target"], split["text"], test_size=0.3)
        rows.extend(_eval_to_rows(results))
    return rows


def _humanized_test(df: pd.DataFrame, extractor: FeatureExtractor, trainer: TextModelTrainer) -> List[Dict]:
    train_df = df[df["label"].isin(["human", "ai_weak", "ai_strong"])].copy()
    test_df = df[df["label"].isin(["human", "artificial_humanized"])].copy()
    return _train_test_manual(train_df, test_df, extractor, trainer, "humanized_ai_test")
