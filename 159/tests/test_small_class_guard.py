from __future__ import annotations

import pandas as pd

from core.modeling import TextModelTrainer


def test_run_experiment_returns_empty_when_single_class() -> None:
    trainer = TextModelTrainer(random_state=42)
    features = pd.DataFrame(
        {
            "word_count": [10.0, 12.0, 9.0],
            "unique_word_count": [10.0, 11.0, 9.0],
            "avg_word_len": [5.0, 5.0, 5.0],
            "type_token_ratio": [0.9, 0.9, 0.9],
            "long_word_share": [0.1, 0.1, 0.1],
            "rare_word_share": [0.1, 0.1, 0.1],
            "sentence_count": [2.0, 2.0, 2.0],
            "avg_sentence_len": [5.0, 5.0, 5.0],
            "std_sentence_len": [0.1, 0.1, 0.1],
            "short_sentence_share": [0.5, 0.5, 0.5],
            "long_sentence_share": [0.0, 0.0, 0.0],
            "service_word_freq": [0.2, 0.2, 0.2],
            "repeat_ratio": [0.1, 0.1, 0.1],
            "template_ratio": [0.0, 0.0, 0.0],
            "intro_phrase_share": [0.0, 0.0, 0.0],
            "smoothness": [0.7, 0.7, 0.7],
            "punct_density_100w": [5.0, 5.0, 5.0],
            "comma_freq": [1.0, 1.0, 1.0],
            "dot_freq": [1.0, 1.0, 1.0],
            "colon_freq": [0.0, 0.0, 0.0],
            "dash_freq": [0.0, 0.0, 0.0],
            "brackets_freq": [0.0, 0.0, 0.0],
            "quotes_freq": [0.0, 0.0, 0.0],
            "question_freq": [0.0, 0.0, 0.0],
            "exclamation_freq": [0.0, 0.0, 0.0],
        }
    )
    labels = pd.Series([1, 1, 1])
    texts = pd.Series(["a b c", "d e f", "g h i"])

    results, models, test_df = trainer.run_experiment("single_class", features, labels, texts)
    assert results == []
    assert models == {}
    assert test_df.empty
