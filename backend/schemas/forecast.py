"""Forecasting schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ForecastTrainRequest(BaseModel):
    """Forecast training request."""

    dataset_id: int
    target_column: str
    time_column: str | None = None
    horizon: int = Field(default=14, ge=1, le=180)
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 7, 14])
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)


class ForecastRunRead(BaseModel):
    """Serialized forecast run."""

    id: int
    dataset_id: int
    model_name: str
    target_column: str
    time_column: str | None
    horizon: int
    artifact_path: str
    metrics_json: dict[str, Any]
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ForecastTrainResponse(BaseModel):
    """Training response payload."""

    run: ForecastRunRead
    charts: dict[str, str]
    insights: dict[str, Any]
