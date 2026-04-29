"""FAISS vector store persistence and search."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from backend.core.config import get_settings


class FAISSVectorStore:
    """Persist embeddings and chunk identifiers using FAISS."""

    def __init__(self) -> None:
        settings = get_settings()
        self.index_path: Path = settings.vector_store_dir / "index.faiss"
        self.mapping_path: Path = settings.vector_store_dir / "mapping.json"
        self._index: faiss.Index | None = None
        self._mapping: list[int] = []
        self._load()

    def _load(self) -> None:
        """Load index and mapping if they already exist."""

        if self.index_path.exists() and self.mapping_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        """Persist the current index and chunk mapping to disk."""

        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        self.mapping_path.write_text(json.dumps(self._mapping), encoding="utf-8")

    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: list[int]) -> None:
        """Append embeddings to the FAISS index."""

        if embeddings.size == 0:
            return

        normalized = embeddings.astype("float32")
        faiss.normalize_L2(normalized)

        if self._index is None:
            self._index = faiss.IndexFlatIP(normalized.shape[1])

        self._index.add(normalized)
        self._mapping.extend(chunk_ids)
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict[str, float | int]]:
        """Search the FAISS index and return chunk identifiers."""

        if self._index is None or not self._mapping:
            return []

        normalized_query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(normalized_query)

        scores, indices = self._index.search(normalized_query, min(top_k, len(self._mapping)))
        results: list[dict[str, float | int]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index == -1:
                continue
            results.append({"chunk_id": int(self._mapping[index]), "score": float(score)})
        return results


@lru_cache
def get_vector_store() -> FAISSVectorStore:
    """Return a cached FAISS vector store instance."""

    return FAISSVectorStore()
