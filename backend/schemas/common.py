"""Common schemas shared across modules."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    environment: str


class MessageResponse(BaseModel):
    """Generic API response."""

    message: str
