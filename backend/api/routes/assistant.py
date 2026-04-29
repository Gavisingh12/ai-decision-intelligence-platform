"""Decision assistant routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import User
from backend.db.session import get_db
from backend.schemas.assistant import AssistantQueryRequest, AssistantResponse
from backend.services.assistant import decision_assistant_service


router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/ask", response_model=AssistantResponse)
def ask_assistant(
    payload: AssistantQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssistantResponse:
    """Ask the multi-signal decision assistant a question."""

    del current_user
    (
        answer,
        provider,
        forecast_summary,
        explainability_summary,
        classification_summary,
        sources,
    ) = decision_assistant_service.answer(
        db=db,
        question=payload.question,
        dataset_id=payload.dataset_id,
        forecast_run_id=payload.forecast_run_id,
        classification_run_id=payload.classification_run_id,
        top_k=payload.top_k,
        provider=payload.provider,
    )
    return AssistantResponse(
        answer=answer,
        provider=provider,
        forecast_summary=forecast_summary,
        explainability_summary=explainability_summary,
        classification_summary=classification_summary,
        sources=sources,
    )
