from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from core.config import PathsConfig
from core.service import ProjectService


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AI/human for one text")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--model", default="", help="Path to model joblib")
    args = parser.parse_args()

    p = PathsConfig()
    model_path = Path(args.model) if args.model else sorted(p.models_dir.glob("best_*.joblib"))[0]
    service = ProjectService(p.data_raw, p.data_processed / "merged_dataset.csv", p.models_dir, p.reports_dir)
    print(service.analyze_text(args.text, model_path))
