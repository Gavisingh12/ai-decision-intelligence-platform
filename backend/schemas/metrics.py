"""Dashboard metrics schemas."""

from typing import Any

from pydantic import BaseModel


class DashboardMetricsResponse(BaseModel):
    """Summary metrics used by the frontend dashboard."""

    totals: dict[str, int]
    latest_forecast: dict[str, Any] | None
    latest_classification: dict[str, Any] | None = None
    hosting_profile: dict[str, Any] | None = None
