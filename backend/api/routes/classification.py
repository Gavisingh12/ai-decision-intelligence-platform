"""Classification and recommendation routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import ClassificationRun, User
from backend.db.session import get_db
from backend.schemas.classification import (
    ClassificationPredictRequest,
    ClassificationPredictResponse,
    ClassificationRunRead,
    ClassificationTrainRequest,
    ClassificationTrainResponse,
)
from backend.services.classification import classification_service
from backend.services.workspace_tasks import workspace_task_service


router = APIRouter(prefix="/classification", tags=["classification"])


@router.post("/train", response_model=ClassificationTrainResponse)
def train_classification(
    payload: ClassificationTrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClassificationTrainResponse:
    """Train a classification/recommendation model."""

    del current_user
    task = workspace_task_service.start(
        db=db,
        task_type="classification_train",
        title="Training a recommendation model",
        detail=f"Preparing a model for dataset {payload.dataset_id}",
        target_view="train",
        metadata={"dataset_id": payload.dataset_id, "target_column": payload.target_column},
    )
    try:
        run, charts, insights = classification_service.train(db=db, request=payload)
    except ValueError as exc:
        workspace_task_service.fail(db=db, task=task, detail=str(exc))
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_task_service.complete(
        db=db,
        task=task,
        detail=f"Recommendation model ready for {run.target_column}.",
        metadata={"run_id": run.id},
    )
    db.commit()

    return ClassificationTrainResponse(
        run=ClassificationRunRead.model_validate(run),
        charts=charts,
        insights=insights,
    )


@router.get("/runs", response_model=list[ClassificationRunRead])
def list_classification_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ClassificationRunRead]:
    """List classification runs."""

    del current_user
    runs = db.query(ClassificationRun).order_by(ClassificationRun.created_at.desc()).all()
    return [ClassificationRunRead.model_validate(run) for run in runs]


@router.get("/runs/{run_id}")
def get_classification_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return classification run details and artifacts."""

    del current_user
    run = db.query(ClassificationRun).filter(ClassificationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Classification run not found.")
    charts, insights = classification_service.serialize_run(run)
    return {
        "run": ClassificationRunRead.model_validate(run).model_dump(),
        "charts": charts,
        "insights": insights,
    }


@router.post("/runs/{run_id}/predict", response_model=ClassificationPredictResponse)
def predict_classification(
    run_id: int,
    payload: ClassificationPredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClassificationPredictResponse:
    """Run recommendation/classification inference from manual feature values."""

    del current_user
    try:
        result = classification_service.predict_with_run(db=db, run_id=run_id, request=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ClassificationPredictResponse(**result)
