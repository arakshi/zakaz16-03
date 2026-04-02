from __future__ import annotations

import numpy as np
import pandas as pd

from core.modeling import TextModelTrainer


def test_merge_text_and_features_no_nan_on_misaligned_index() -> None:
    trainer = TextModelTrainer()

    texts = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    features = pd.DataFrame(
        {
            "word_count": [1.0, np.nan, 3.0],
            "unique_word_count": [1.0, 2.0, 3.0],
        },
        index=[100, 200, 300],
    )

    merged = trainer.merge_text_and_features(texts, features)

    assert list(merged["text"]) == ["a", "b", "c"]
    assert not merged.isna().any().any()
    assert merged.loc[1, "word_count"] == 0.0
