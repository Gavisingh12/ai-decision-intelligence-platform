"""Database models for the platform."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from backend.db.base import Base


class TimestampMixin:
    """Common timestamp fields."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class User(TimestampMixin, Base):
    """Authenticated platform user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Dataset(TimestampMixin, Base):
    """Structured dataset metadata."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    time_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    records: Mapped[list["DatasetRecord"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    forecast_runs: Mapped[list["ForecastRun"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    classification_runs: Mapped[list["ClassificationRun"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class DatasetRecord(Base):
    """Row-level structured records persisted from uploaded CSV files."""

    __tablename__ = "dataset_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    record_index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    target_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    dataset: Mapped["Dataset"] = relationship(back_populates="records")


class Document(TimestampMixin, Base):
    """Uploaded document metadata."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(100))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """Persisted document chunk linked to the FAISS index."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class ForecastRun(TimestampMixin, Base):
    """Forecast model training run metadata and artifacts."""

    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), default="xgboost")
    target_column: Mapped[str] = mapped_column(String(128))
    time_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    horizon: Mapped[int] = mapped_column(Integer)
    artifact_path: Mapped[str] = mapped_column(String(500))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(50), default="completed")

    dataset: Mapped["Dataset"] = relationship(back_populates="forecast_runs")


class ClassificationRun(TimestampMixin, Base):
    """Classification model training run metadata and artifacts."""

    __tablename__ = "classification_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), default="xgboost_classifier")
    target_column: Mapped[str] = mapped_column(String(128))
    artifact_path: Mapped[str] = mapped_column(String(500))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(50), default="completed")

    dataset: Mapped["Dataset"] = relationship(back_populates="classification_runs")


class WorkspaceTask(TimestampMixin, Base):
    """User-facing task/activity record for long or important workflow actions."""

    __tablename__ = "workspace_tasks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_view: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
