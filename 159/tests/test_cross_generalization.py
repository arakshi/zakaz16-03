from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import WEAK_AI_LABELS, STRONG_AI_LABELS


def test_cross_generalization_no_human_leakage() -> None:
    """Human texts must not overlap between train and test in cross-generalization."""
    human_texts = [f"human_text_{i}" for i in range(20)]
    human_df = pd.DataFrame({"text": human_texts, "label": "human"})

    human_train, human_test = train_test_split(human_df, test_size=0.5, random_state=42)

    train_set = set(human_train["text"])
    test_set = set(human_test["text"])

    assert train_set.isdisjoint(test_set), "Human texts leaked between train and test"
    assert len(train_set) + len(test_set) == len(human_texts)


def test_humanized_test_no_human_leakage() -> None:
    """Same check for the humanized AI test scenario."""
    human_texts = [f"human_text_{i}" for i in range(20)]
    human_df = pd.DataFrame({"text": human_texts, "label": "human"})

    human_train, human_test = train_test_split(human_df, test_size=0.5, random_state=42)

    other_train = pd.DataFrame(
        {"text": ["ai_text_1", "ai_text_2"], "label": "ai_gpt_fast"}
    )
    art_hum = pd.DataFrame(
        {"text": ["humanized_1", "humanized_2"], "label": "artificial_humanized"}
    )

    train_h = pd.concat([human_train, other_train])
    test_h = pd.concat([human_test, art_hum])

    train_human_texts = set(train_h[train_h["label"] == "human"]["text"])
    test_human_texts = set(test_h[test_h["label"] == "human"]["text"])

    assert train_human_texts.isdisjoint(test_human_texts)
