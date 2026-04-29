"""Filesystem helpers for uploads, artifacts, and encoded assets."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from uuid import uuid4

from backend.core.config import get_settings


def ensure_app_directories() -> None:
    """Create all required runtime directories."""

    settings = get_settings()
    required = [
        settings.uploads_dir,
        settings.uploads_dir / "structured",
        settings.uploads_dir / "documents",
        settings.model_dir,
        settings.artifact_dir,
        settings.vector_store_dir,
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)


def persist_upload(content: bytes, original_filename: str, category: str) -> Path:
    """Persist an uploaded file in the configured data directory."""

    validate_upload_size(content)
    settings = get_settings()
    suffix = Path(original_filename).suffix.lower()
    output_path = settings.uploads_dir / category / f"{uuid4().hex}{suffix}"
    output_path.write_bytes(content)
    return output_path


def validate_upload_size(content: bytes) -> None:
    """Reject uploads that exceed the configured limit."""

    settings = get_settings()
    limit_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > limit_bytes:
        raise ValueError(
            f"That file is too large for this workspace. Please keep uploads under {settings.max_upload_size_mb} MB."
        )


def image_to_data_url(image_path: str | Path) -> str:
    """Convert an image file into a browser-friendly data URL."""

    path = Path(image_path)
    if not path.exists():
        return ""

    mime_type, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/png'};base64,{encoded}"


def truncate_text(text: str, limit: int = 280) -> str:
    """Truncate long text snippets for API responses."""

    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
