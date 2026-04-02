from __future__ import annotations

import pandas as pd
import pytest

from core.features import FeatureExtractor
from core.modeling import TextModelTrainer


HUMAN_TEXTS = [
    "Вечером я пересмотрел конспекты и понял, что прогресс зависит от качества заметок.",
    "На этой неделе я тестировал несколько режимов обучения и пришел к выводу.",
    "Когда пишешь главу диссертации, полезно оставлять альтернативные варианты.",
    "Я сравнил несколько источников по теме стилистического анализа текста.",
    "В библиотеке нашел старые журнальные публикации по лингвистике.",
    "Небольшой перерыв во время работы неожиданно улучшил качество редактуры.",
]

AI_TEXTS = [
    "Развитие образовательных технологий ускоряет доступ к знаниям и возможностям.",
    "При анализе текста важно учитывать не только длину предложений, но и разнообразие.",
    "Модель машинного обучения может показывать хорошие метрики на тестовом наборе.",
    "Для корректной оценки гипотезы нужно сравнить несколько сценариев эксперимента.",
    "В прикладных задачах часто полезно сочетать статистические признаки и TF-IDF.",
    "Если данные сильно неоднородны, модель может ориентироваться на случайные маркеры.",
]


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal balanced dataset with human / AI labels."""
    rows = [{"text": t, "label": "human"} for t in HUMAN_TEXTS] + [
        {"text": t, "label": "ai_gpt_fast"} for t in AI_TEXTS
    ]
    df = pd.DataFrame(rows)
    df["target"] = df["label"].apply(lambda x: 0 if x == "human" else 1)
    return df


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture()
def trainer() -> TextModelTrainer:
    return TextModelTrainer()
