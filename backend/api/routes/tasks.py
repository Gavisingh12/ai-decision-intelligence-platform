"""Workspace task/activity routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import User
from backend.db.session import get_db
from backend.schemas.tasks import WorkspaceTaskRead
from backend.services.workspace_tasks import workspace_task_service


router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/recent", response_model=list[WorkspaceTaskRead])
def list_recent_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[WorkspaceTaskRead]:
    """Return recent workflow tasks for the dashboard."""

    del current_user
    tasks = workspace_task_service.list_recent(db=db)
    return [WorkspaceTaskRead.model_validate(task) for task in tasks]
