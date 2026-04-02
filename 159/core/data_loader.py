from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from core.config import DatasetConfig, STRONG_AI_LABELS, WEAK_AI_LABELS


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(output_path, index=False, encoding="utf-8")
        return output_path


class DatasetLoader:
    def __init__(self, raw_root: Path, config: DatasetConfig | None = None) -> None:
        self.raw_root = raw_root
        self.config = config or DatasetConfig()

    def load(self) -> DatasetBundle:
        records: List[dict] = []
        for label, folder in self.config.class_to_folder.items():
            class_dir = self.raw_root / folder
            if not class_dir.exists():
                continue
            records.extend(self._load_class_dir(class_dir=class_dir, label=label))

        if not records:
            raise ValueError(f"No data found in {self.raw_root}. Add txt/csv files first.")

        df = pd.DataFrame(records)
        df["text"] = df["text"].fillna("").astype(str)
        df = df[df["text"].str.strip() != ""].reset_index(drop=True)
        if df.empty:
            raise ValueError("All loaded texts are empty after cleaning.")

        df["ai_all"] = df["label"].apply(lambda x: "ai" if x != "human" else "human")
        df["ai_family"] = df["label"].apply(self._label_to_family)
        return DatasetBundle(dataframe=df)

    def build_binary_view(self, df: pd.DataFrame, scenario: str) -> pd.DataFrame:
        if scenario == "human_vs_ai_all":
            out = df[df["label"] == "human"].copy()
            ai = df[df["label"] != "human"].copy()
            out = pd.concat([out, ai], ignore_index=True)
            out["target"] = out["label"].apply(lambda x: 0 if x == "human" else 1)
            return out
        if scenario == "human_vs_ai_weak":
            out = df[df["label"].isin({"human", *WEAK_AI_LABELS})].copy()
            out["target"] = out["label"].apply(lambda x: 0 if x == "human" else 1)
            return out
        if scenario == "human_vs_ai_strong":
            out = df[df["label"].isin({"human", *STRONG_AI_LABELS})].copy()
            out["target"] = out["label"].apply(lambda x: 0 if x == "human" else 1)
            return out
        raise ValueError(f"Unknown scenario: {scenario}")

    @staticmethod
    def _label_to_family(label: str) -> str:
        if label == "human":
            return "human"
        if label in WEAK_AI_LABELS:
            return "weak"
        if label in STRONG_AI_LABELS:
            return "strong"
        return "other_ai"

    def _load_class_dir(self, class_dir: Path, label: str) -> Iterable[dict]:
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() not in self.config.supported_extensions:
                continue
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self._split_into_chunks(text)
                if not chunks:
                    chunks = [text]
                for idx, chunk in enumerate(chunks):
                    yield {
                        "text": chunk,
                        "label": label,
                        "source_file": f"{file_path.relative_to(self.raw_root.parent)}#chunk{idx}",
                    }
            elif file_path.suffix.lower() == ".csv":
                csv_df = pd.read_csv(file_path)
                if "text" not in csv_df.columns:
                    continue
                for idx, row in csv_df.iterrows():
                    text = str(row["text"])
                    for j, chunk in enumerate(self._split_into_chunks(text) or [text]):
                        yield {
                            "text": chunk,
                            "label": label,
                            "source_file": f"{file_path.name}#{idx}_chunk{j}",
                        }

    @staticmethod
    def _split_into_chunks(text: str, chunk_words: int = 220, overlap_words: int = 40) -> List[str]:
        words = text.split()
        if len(words) <= chunk_words:
            return [text.strip()] if text.strip() else []

        step = max(chunk_words - overlap_words, 1)
        chunks: List[str] = []
        for start in range(0, len(words), step):
            part = words[start : start + chunk_words]
            if len(part) < max(40, chunk_words // 4):
                break
            chunks.append(" ".join(part).strip())
        return chunks
