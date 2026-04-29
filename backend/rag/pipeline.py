"""RAG retrieval and grounded answer generation."""

from __future__ import annotations

from functools import lru_cache

from cachetools import TTLCache
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.db.models import DocumentChunk
from backend.schemas.rag import RetrievalSource
from backend.services.llm import llm_service
from backend.services.storage import truncate_text
from backend.rag.embeddings import get_embedding_service
from backend.rag.vector_store import get_vector_store


class RAGService:
    """Document retrieval and question-answering pipeline."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._cache: TTLCache[tuple[str, int], list[dict[str, float | int]]] = TTLCache(
            maxsize=128,
            ttl=300,
        )

    def clear_cache(self) -> None:
        """Invalidate cached retrieval results."""

        self._cache.clear()

    def retrieve(self, *, db: Session, question: str, top_k: int) -> list[RetrievalSource]:
        """Retrieve the most relevant document chunks for a question."""

        if not db.query(DocumentChunk.id).limit(1).first():
            return []

        cache_key = (question.lower().strip(), top_k)
        cached = self._cache.get(cache_key)
        if cached is None:
            embedding = get_embedding_service().encode([question])[0]
            cached = get_vector_store().search(embedding, top_k=top_k)
            self._cache[cache_key] = cached

        if not cached:
            return []

        chunk_ids = [int(item["chunk_id"]) for item in cached]
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.id.in_(chunk_ids))
            .all()
        )
        chunk_map = {chunk.id: chunk for chunk in chunks}

        ordered_sources: list[RetrievalSource] = []
        for result in cached:
            chunk = chunk_map.get(int(result["chunk_id"]))
            if not chunk:
                continue
            ordered_sources.append(
                RetrievalSource(
                    document_id=chunk.document_id,
                    document_title=chunk.document.title,
                    chunk_id=chunk.id,
                    score=float(result["score"]),
                    excerpt=truncate_text(chunk.content, limit=320),
                    metadata=chunk.metadata_json,
                )
            )
        return ordered_sources

    def answer(self, *, db: Session, question: str, top_k: int, provider: str | None = None) -> tuple[str, str, list[RetrievalSource]]:
        """Answer a question using retrieved evidence and the configured LLM."""

        sources = self.retrieve(db=db, question=question, top_k=top_k)
        context = "\n\n".join(
            f"[{index + 1}] {source.document_title}: {source.excerpt}"
            for index, source in enumerate(sources)
        )

        prompt = (
            "Use the retrieved context to answer the user's question. "
            "Be explicit when the evidence is incomplete.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context or 'No supporting documents were retrieved.'}"
        )
        system_prompt = (
            "You are an operations analyst. Answer with grounded, concise reasoning and cite retrieved evidence."
        )
        result = llm_service.generate(prompt=prompt, system_prompt=system_prompt, provider=provider)
        return result.answer, result.provider, sources


@lru_cache
def get_rag_service() -> RAGService:
    """Return a cached RAG service instance."""

    return RAGService()
