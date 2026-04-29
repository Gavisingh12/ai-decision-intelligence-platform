"""Dashboard metrics routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.core.config import get_settings
from backend.db.models import ClassificationRun, Dataset, Document, ForecastRun, User
from backend.db.session import get_db
from backend.schemas.metrics import DashboardMetricsResponse


router = APIRouter(prefix="/metrics", tags=["metrics"])
settings = get_settings()


@router.get("/summary", response_model=DashboardMetricsResponse)
def summary_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DashboardMetricsResponse:
    """Return dashboard-friendly totals and the latest forecast run."""

    del current_user
    latest_run = db.query(ForecastRun).order_by(ForecastRun.created_at.desc()).first()
    latest_classification = db.query(ClassificationRun).order_by(ClassificationRun.created_at.desc()).first()
    return DashboardMetricsResponse(
        totals={
            "datasets": db.query(Dataset).count(),
            "documents": db.query(Document).count(),
            "forecast_runs": db.query(ForecastRun).count(),
            "classification_runs": db.query(ClassificationRun).count(),
            "users": db.query(User).count(),
        },
        latest_forecast=latest_run.metrics_json if latest_run else None,
        latest_classification=latest_classification.metrics_json if latest_classification else None,
        hosting_profile={
            "free_mode": settings.free_mode,
            "deploy_target": settings.free_deploy_target,
        },
    )
