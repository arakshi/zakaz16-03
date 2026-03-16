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
            "ai_weak": "ai_weak",
            "ai_strong": "ai_strong",
        }
    )
    supported_extensions: List[str] = field(default_factory=lambda: [".txt", ".csv"])


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
