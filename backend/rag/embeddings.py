"""Embedding service backed by SentenceTransformers with a lightweight fallback."""

from __future__ import annotations

from functools import lru_cache
import logging

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from backend.core.config import get_settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Lazy embedding model wrapper with a lexical fallback encoder."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._fallback_vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm=None,
        )

    @property
    def model(self):
        """Load the embedding model on first use when available."""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.embedding_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers is not installed. Falling back to HashingVectorizer embeddings."
                )
                self._model = False
            except Exception as exc:  # pragma: no cover - depends on local torch/runtime state
                logger.warning(
                    "Failed to initialize sentence-transformers embeddings (%s). Falling back to HashingVectorizer.",
                    exc,
                )
                self._model = False
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text into float32 vectors."""

        model = self.model
        if model is False:
            embeddings = self._fallback_vectorizer.transform(texts).toarray()
            return embeddings.astype("float32")

        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype("float32")
        except Exception as exc:  # pragma: no cover - depends on local torch/runtime state
            logger.warning(
                "Embedding model inference failed (%s). Falling back to HashingVectorizer.",
                exc,
            )
            self._model = False
            embeddings = self._fallback_vectorizer.transform(texts).toarray()
            return embeddings.astype("float32")


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached embedding service."""

    return EmbeddingService()
