"""RAG query routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import User
from backend.db.session import get_db
from backend.rag.pipeline import get_rag_service
from backend.schemas.rag import RAGQueryRequest, RAGQueryResponse


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
def query_rag(
    payload: RAGQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RAGQueryResponse:
    """Retrieve relevant document context and answer a question."""

    del current_user
    answer, provider, sources = get_rag_service().answer(
        db=db,
        question=payload.question,
        top_k=payload.top_k,
        provider=payload.provider,
    )
    return RAGQueryResponse(answer=answer, provider=provider, grounded=bool(sources), sources=sources)
