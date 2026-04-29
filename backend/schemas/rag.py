"""RAG schemas."""

from typing import Any

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)
    provider: str | None = None


class RetrievalSource(BaseModel):
    """A document chunk returned as evidence."""

    document_id: int
    document_title: str
    chunk_id: int
    score: float
    excerpt: str
    metadata: dict[str, Any]


class RAGQueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    provider: str
    grounded: bool
    sources: list[RetrievalSource]
