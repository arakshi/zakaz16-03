from __future__ import annotations

from core.data_loader import DatasetLoader


def test_split_into_chunks_long_text() -> None:
    text = " ".join(["слово"] * 1000)
    chunks = DatasetLoader._split_into_chunks(text, chunk_words=200, overlap_words=50)
    assert len(chunks) >= 4
    assert all(len(c.split()) >= 50 for c in chunks)
