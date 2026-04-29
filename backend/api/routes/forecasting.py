"""Forecasting routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import ForecastRun, User
from backend.db.session import get_db
from backend.schemas.forecast import ForecastRunRead, ForecastTrainRequest, ForecastTrainResponse
from backend.services.forecasting import forecasting_service
from backend.services.workspace_tasks import workspace_task_service


router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/train", response_model=ForecastTrainResponse)
def train_forecast(
    payload: ForecastTrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastTrainResponse:
    """Train a time-series forecast model."""

    del current_user
    task = workspace_task_service.start(
        db=db,
        task_type="forecast_train",
        title="Training a future estimate",
        detail=f"Preparing a model for dataset {payload.dataset_id}",
        target_view="train",
        metadata={"dataset_id": payload.dataset_id, "target_column": payload.target_column},
    )
    try:
        run, charts, insights = forecasting_service.train(
            db=db,
            request=payload,
        )
    except ValueError as exc:
        workspace_task_service.fail(db=db, task=task, detail=str(exc))
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_task_service.complete(
        db=db,
        task=task,
        detail=f"Future estimate ready for {run.target_column}.",
        metadata={"run_id": run.id},
    )
    db.commit()

    return ForecastTrainResponse(
        run=ForecastRunRead.model_validate(run),
        charts=charts,
        insights=insights,
    )


@router.get("/runs", response_model=list[ForecastRunRead])
def list_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ForecastRunRead]:
    """List forecast runs."""

    del current_user
    runs = db.query(ForecastRun).order_by(ForecastRun.created_at.desc()).all()
    return [ForecastRunRead.model_validate(run) for run in runs]


@router.get("/runs/{run_id}")
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return forecast run details, charts, and explainability insights."""

    del current_user
    run = db.query(ForecastRun).filter(ForecastRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Forecast run not found.")
    charts, insights = forecasting_service.serialize_run(run)
    return {
        "run": ForecastRunRead.model_validate(run).model_dump(),
        "charts": charts,
        "insights": insights,
    }


@router.post("/runs/{run_id}/predict")
def predict_from_run(
    run_id: int,
    horizon: int = Query(default=14, ge=1, le=180),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Generate a new forecast using a saved model run."""

    del current_user
    try:
        return forecasting_service.predict_with_run(
            db=db,
            run_id=run_id,
            horizon=horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
