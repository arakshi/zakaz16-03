from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PathsConfig
from core.service import ProjectService


if __name__ == "__main__":
    p = PathsConfig()
    service = ProjectService(
        raw_data_dir=p.data_raw,
        processed_path=p.data_processed / "merged_dataset.csv",
        models_dir=p.models_dir,
        reports_dir=p.reports_dir,
    )
    artifacts = service.train_and_evaluate()
    print("Best model:", artifacts.best_model_name)
    print("Model path:", artifacts.best_model_path)
    print("Hypothesis conclusion:", artifacts.hypothesis_text)
