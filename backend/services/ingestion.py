"""Structured and unstructured data ingestion services."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.db.models import Dataset, DatasetRecord, Document, DocumentChunk
from backend.rag.embeddings import get_embedding_service
from backend.rag.vector_store import get_vector_store
from backend.services.persistence import hub_persistence_service
from backend.services.storage import persist_upload


def _json_safe_value(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-compatible Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


class StructuredDataService:
    """Handle CSV upload, validation, and persistence."""

    def ingest_csv(
        self,
        *,
        db: Session,
        content: bytes,
        filename: str,
        name: str | None = None,
        target_column: str | None = None,
        time_column: str | None = None,
    ) -> tuple[Dataset, list[dict[str, Any]]]:
        """Parse a CSV file, store metadata, and persist row-level records."""

        dataframe = pd.read_csv(io.BytesIO(content))
        if dataframe.empty:
            raise ValueError("Uploaded CSV file is empty.")

        resolved_target_column = (target_column or "").strip() or self._infer_target_column(dataframe)
        resolved_time_column = (time_column or "").strip() or self._infer_time_column(dataframe)

        if resolved_target_column and resolved_target_column not in dataframe.columns:
            available_columns = ", ".join(dataframe.columns.tolist())
            raise ValueError(
                f"Target column '{resolved_target_column}' was not found. Available columns: {available_columns}"
            )

        if resolved_time_column and resolved_time_column not in dataframe.columns:
            available_columns = ", ".join(dataframe.columns.tolist())
            raise ValueError(
                f"Time column '{resolved_time_column}' was not found. Leave it blank for non-time-series data, or choose one of: {available_columns}"
            )

        if resolved_time_column and resolved_time_column in dataframe.columns:
            dataframe[resolved_time_column] = pd.to_datetime(
                dataframe[resolved_time_column],
                errors="coerce",
            )

        output_path = persist_upload(content, filename, "structured")

        metadata = self._build_dataset_metadata(dataframe, resolved_time_column, resolved_target_column)
        dataset = Dataset(
            name=name or Path(filename).stem,
            filename=filename,
            file_path=str(output_path),
            time_column=resolved_time_column,
            target_column=resolved_target_column,
            row_count=int(len(dataframe)),
            metadata_json=metadata,
        )
        db.add(dataset)
        db.flush()

        records: list[DatasetRecord] = []
        for index, row in dataframe.iterrows():
            payload = {column: _json_safe_value(row[column]) for column in dataframe.columns}
            timestamp_value = None
            if resolved_time_column and isinstance(row.get(resolved_time_column), pd.Timestamp):
                timestamp_value = row[resolved_time_column].to_pydatetime()

            target_value = None
            if resolved_target_column and resolved_target_column in dataframe.columns and pd.notna(row.get(resolved_target_column)):
                try:
                    target_value = float(row[resolved_target_column])
                except (TypeError, ValueError):
                    target_value = None

            records.append(
                DatasetRecord(
                    dataset_id=dataset.id,
                    record_index=int(index),
                    timestamp=timestamp_value,
                    target_value=target_value,
                    payload=payload,
                )
            )

        db.add_all(records)
        db.commit()
        db.refresh(dataset)
        hub_persistence_service.sync_runtime_data(reason=f"dataset upload {dataset.id}")
        preview = dataframe.head(5).replace({np.nan: None}).to_dict(orient="records")
        return dataset, preview

    @staticmethod
    def _infer_time_column(dataframe: pd.DataFrame) -> str | None:
        """Infer a likely time column from common names."""

        candidates = {"date", "timestamp", "time", "ds"}
        for column in dataframe.columns:
            if column.lower() in candidates:
                return column
        return None

    @staticmethod
    def _infer_target_column(dataframe: pd.DataFrame) -> str | None:
        """Infer a likely target column from common names."""

        candidates = {"target", "label", "y", "class", "yield", "sales", "demand"}
        for column in dataframe.columns:
            if column.lower() in candidates:
                return column
        return None

    @staticmethod
    def _build_dataset_metadata(
        dataframe: pd.DataFrame,
        time_column: str | None,
        target_column: str | None,
    ) -> dict[str, Any]:
        """Create a profiling summary for a structured dataset."""

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in numeric_columns and column != time_column
        ]
        target_kind = "missing"
        recommended_workflow = "exploration"
        forecast_ready = False
        warnings: list[str] = []

        if target_column and target_column in dataframe.columns:
            target_kind = "numeric" if target_column in numeric_columns else "categorical"
            if target_kind == "numeric" and time_column:
                recommended_workflow = "forecasting"
                forecast_ready = True
            elif target_kind == "numeric":
                recommended_workflow = "regression"
                warnings.append(
                    "No time column was provided. This dataset can still be modeled numerically, but the forecasting workflow assumes sequential or time-ordered rows."
                )
            else:
                recommended_workflow = "classification"
                warnings.append(
                    f"Target column '{target_column}' is categorical. The current forecasting engine expects a numeric target, so this dataset is better suited for classification or recommendation workflows."
                )
        else:
            warnings.append(
                "No target column was selected. Upload succeeded, but model training needs an explicit target column."
            )

        sample_statistics = StructuredDataService._safe_describe(dataframe)
        return {
            "columns": dataframe.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "time_column": time_column,
            "target_column": target_column,
            "missing_values": {
                column: int(value) for column, value in dataframe.isna().sum().to_dict().items()
            },
            "sample_statistics": sample_statistics,
            "target_kind": target_kind,
            "recommended_workflow": recommended_workflow,
            "forecast_ready": forecast_ready,
            "warnings": warnings,
            "suggested_target_column": StructuredDataService._infer_target_column(dataframe),
            "suggested_time_column": StructuredDataService._infer_time_column(dataframe),
            "workflow_summary": StructuredDataService._workflow_summary(
                recommended_workflow=recommended_workflow,
                target_column=target_column,
                target_kind=target_kind,
                time_column=time_column,
            ),
        }

    @staticmethod
    def _workflow_summary(
        *,
        recommended_workflow: str,
        target_column: str | None,
        target_kind: str,
        time_column: str | None,
    ) -> dict[str, Any]:
        """Return a compact next-step summary for the frontend."""

        if recommended_workflow == "classification":
            return {
                "headline": "We detected a best-match or recommendation dataset.",
                "next_step": f"Train a recommendation model using '{target_column or 'label'}' as the answer column.",
            }
        if recommended_workflow == "forecasting":
            return {
                "headline": "We detected a future-trends dataset.",
                "next_step": f"Train a future estimate using '{target_column or 'target'}' and '{time_column or 'date'}'.",
            }
        if target_kind == "numeric":
            return {
                "headline": "We detected a numeric dataset without a clear date column.",
                "next_step": "You can still train a future estimate if the rows are already in order.",
            }
        return {
            "headline": "Your file is ready to review.",
            "next_step": "Choose a target column before training a model.",
        }

    @staticmethod
    def _safe_describe(dataframe: pd.DataFrame) -> dict[str, Any]:
        """Return a JSON-safe dataframe profile compatible across pandas versions."""

        try:
            described = dataframe.describe(include="all", datetime_is_numeric=True)
        except TypeError:
            described = dataframe.describe(include="all")

        raw = described.replace({np.nan: None}).to_dict()
        return {
            column: {metric: _json_safe_value(value) for metric, value in metrics.items()}
            for column, metrics in raw.items()
        }


class DocumentIngestionService:
    """Handle PDF/text ingestion, chunking, and vector indexing."""

    def ingest_document(
        self,
        *,
        db: Session,
        content: bytes,
        filename: str,
        content_type: str,
        title: str | None = None,
    ) -> tuple[Document, list[str]]:
        """Extract text, persist chunks, and update the FAISS index."""

        settings = get_settings()
        extracted_text = self._extract_text(content=content, filename=filename, content_type=content_type)
        if not extracted_text.strip():
            raise ValueError("The uploaded document did not contain extractable text.")

        chunks = self._chunk_text(
            extracted_text,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
        if not chunks:
            raise ValueError("No retrievable text chunks could be created from the uploaded document.")

        output_path = persist_upload(content, filename, "documents")
        document = Document(
            title=title or Path(filename).stem,
            filename=filename,
            file_path=str(output_path),
            content_type=content_type or "application/octet-stream",
            chunk_count=len(chunks),
            metadata_json={
                "characters": len(extracted_text),
                "free_mode": settings.free_mode,
            },
        )
        db.add(document)
        db.flush()

        chunk_models: list[DocumentChunk] = []
        for index, chunk in enumerate(chunks):
            chunk_models.append(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk,
                    metadata_json={"filename": filename, "chunk_index": index},
                )
            )

        db.add_all(chunk_models)
        db.flush()

        embeddings = get_embedding_service().encode(chunks)
        get_vector_store().add_embeddings(embeddings, [chunk.id for chunk in chunk_models])

        db.commit()
        db.refresh(document)
        hub_persistence_service.sync_runtime_data(reason=f"document upload {document.id}")
        return document, chunks[:3]

    @staticmethod
    def _extract_text(*, content: bytes, filename: str, content_type: str) -> str:
        """Extract text from a PDF or plain-text document."""

        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf" or content_type == "application/pdf":
            reader = PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages)

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Create overlapping text chunks for retrieval."""

        settings = get_settings()
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunks.append(normalized[start:end].strip())
            if end >= len(normalized):
                break
            start = max(end - overlap, 0)
        return chunks[: settings.max_document_chunks]


structured_data_service = StructuredDataService()
document_ingestion_service = DocumentIngestionService()
