from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PathsConfig
from core.demo_data import populate_demo_dataset


if __name__ == "__main__":
    paths = PathsConfig()
    stats = populate_demo_dataset(paths.data_raw, overwrite=True)
    print("Demo dataset created in data/raw")
    print(stats)
