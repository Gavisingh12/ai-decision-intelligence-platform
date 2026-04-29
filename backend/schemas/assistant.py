"""AI assistant schemas."""

from typing import Any

from pydantic import BaseModel, Field


class AssistantQueryRequest(BaseModel):
    """AI decision assistant request."""

    question: str = Field(min_length=3)
    dataset_id: int | None = None
    forecast_run_id: int | None = None
    classification_run_id: int | None = None
    top_k: int = Field(default=4, ge=1, le=10)
    provider: str | None = None


class AssistantResponse(BaseModel):
    """Decision assistant response."""

    answer: str
    provider: str
    forecast_summary: dict[str, Any] | None
    explainability_summary: dict[str, Any] | None
    classification_summary: dict[str, Any] | None
    sources: list[dict[str, Any]]
