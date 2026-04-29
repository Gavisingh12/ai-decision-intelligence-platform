"""Classification schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassificationTrainRequest(BaseModel):
    """Classification training request."""

    dataset_id: int
    target_column: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    max_classes_for_report: int = Field(default=20, ge=2, le=100)


class ClassificationRunRead(BaseModel):
    """Serialized classification run."""

    id: int
    dataset_id: int
    model_name: str
    target_column: str
    artifact_path: str
    metrics_json: dict[str, Any]
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ClassificationTrainResponse(BaseModel):
    """Classification training response payload."""

    run: ClassificationRunRead
    charts: dict[str, str]
    insights: dict[str, Any]


class ClassProbability(BaseModel):
    """Ranked probability for a predicted label."""

    label: str
    probability: float


class ClassificationPredictRequest(BaseModel):
    """Classification inference request."""

    feature_values: dict[str, Any]


class ClassificationPredictResponse(BaseModel):
    """Classification inference response."""

    predicted_label: str
    probabilities: list[ClassProbability]
