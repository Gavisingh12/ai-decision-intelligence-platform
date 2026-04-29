"""Workspace task logging helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.db.models import WorkspaceTask


class WorkspaceTaskService:
    """Create and update user-facing workflow task records."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def start(
        self,
        *,
        db: Session,
        task_type: str,
        title: str,
        detail: str | None = None,
        target_view: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceTask:
        """Create a new task in pending state."""

        task = WorkspaceTask(
            task_type=task_type,
            title=title,
            status="pending",
            detail=detail,
            target_view=target_view,
            metadata_json=metadata or {},
        )
        db.add(task)
        db.flush()
        return task

    @staticmethod
    def complete(
        *,
        db: Session,
        task: WorkspaceTask,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceTask:
        """Mark a task as completed."""

        task.status = "completed"
        if detail is not None:
            task.detail = detail
        if metadata:
            task.metadata_json = {**(task.metadata_json or {}), **metadata}
        db.add(task)
        return task

    @staticmethod
    def fail(
        *,
        db: Session,
        task: WorkspaceTask,
        detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceTask:
        """Mark a task as failed."""

        task.status = "failed"
        task.detail = detail
        if metadata:
            task.metadata_json = {**(task.metadata_json or {}), **metadata}
        db.add(task)
        return task

    def list_recent(self, *, db: Session) -> list[WorkspaceTask]:
        """Return recent tasks for the dashboard."""

        return (
            db.query(WorkspaceTask)
            .order_by(WorkspaceTask.updated_at.desc(), WorkspaceTask.created_at.desc())
            .limit(self.settings.recent_task_limit)
            .all()
        )


workspace_task_service = WorkspaceTaskService()
