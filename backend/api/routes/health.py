"""Health-check routes."""

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.schemas.common import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return the application health status."""

    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.app_env)
