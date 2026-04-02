from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    class_to_folder: Dict[str, str] = field(
        default_factory=lambda: {
            "human": "human",
            "ai_deepseek": "ai_deepseek",
            "ai_gpt_fast": "ai_gpt_fast",
            "ai_gpt_reasoning": "ai_gpt_reasoning",
            "ai_yandex": "ai_yandex",
            "artificial_humanized": "artificial_humanized",
            # legacy compatibility
            "ai_weak": "ai_weak",
            "ai_strong": "ai_strong",
        }
    )
    supported_extensions: List[str] = field(default_factory=lambda: [".txt", ".csv"])


WEAK_AI_LABELS = {"ai_deepseek", "ai_gpt_fast", "ai_yandex", "ai_weak"}
STRONG_AI_LABELS = {"ai_gpt_reasoning", "ai_strong"}
