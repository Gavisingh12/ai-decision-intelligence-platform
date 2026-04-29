"""Data ingestion schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetRead(BaseModel):
    """Dataset metadata returned to clients."""

    id: int
    name: str
    filename: str
    time_column: str | None
    target_column: str | None
    row_count: int
    metadata_json: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetPreviewResponse(BaseModel):
    """Dataset upload result."""

    dataset: DatasetRead
    preview_rows: list[dict[str, Any]]
    warnings: list[str] = []


class DocumentRead(BaseModel):
    """Document metadata returned to clients."""

    id: int
    title: str
    filename: str
    content_type: str
    chunk_count: int
    metadata_json: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    """Document upload result."""

    document: DocumentRead
    sample_chunks: list[str]
