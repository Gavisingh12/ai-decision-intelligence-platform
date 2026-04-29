"""Workspace task schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class WorkspaceTaskRead(BaseModel):
    """Task/activity payload returned to the frontend."""

    id: int
    task_type: str
    title: str
    status: str
    detail: str | None
    target_view: str | None
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
