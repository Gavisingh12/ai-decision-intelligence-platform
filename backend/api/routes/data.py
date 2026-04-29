"""Data ingestion and catalog routes."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.db.models import Dataset, Document, User
from backend.db.session import get_db
from backend.rag.pipeline import get_rag_service
from backend.schemas.data import (
    DatasetPreviewResponse,
    DatasetRead,
    DocumentRead,
    DocumentUploadResponse,
)
from backend.services.ingestion import document_ingestion_service, structured_data_service
from backend.services.workspace_tasks import workspace_task_service


router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload/csv", response_model=DatasetPreviewResponse)
def upload_csv(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    target_column: str | None = Form(None),
    time_column: str | None = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetPreviewResponse:
    """Upload a CSV file and persist its records."""

    del current_user
    content = file.file.read()
    task = workspace_task_service.start(
        db=db,
        task_type="dataset_upload",
        title="Uploading a data file",
        detail=f"Preparing {file.filename or 'dataset.csv'}",
        target_view="upload",
    )
    try:
        dataset, preview = structured_data_service.ingest_csv(
            db=db,
            content=content,
            filename=file.filename or "dataset.csv",
            name=name,
            target_column=target_column,
            time_column=time_column,
        )
    except ValueError as exc:
        workspace_task_service.fail(db=db, task=task, detail=str(exc))
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workflow = dataset.metadata_json.get("recommended_workflow", "exploration")
    summary = dataset.metadata_json.get("workflow_summary", {})
    workspace_task_service.complete(
        db=db,
        task=task,
        detail=summary.get("headline", f"{dataset.name} is ready."),
        metadata={"dataset_id": dataset.id, "workflow": workflow},
    )
    db.commit()

    return DatasetPreviewResponse(
        dataset=DatasetRead.model_validate(dataset),
        preview_rows=preview,
        warnings=dataset.metadata_json.get("warnings", []),
    )


@router.post("/upload/document", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """Upload a PDF or text document and index its chunks."""

    del current_user
    content = file.file.read()
    task = workspace_task_service.start(
        db=db,
        task_type="report_upload",
        title="Uploading a report",
        detail=f"Preparing {file.filename or 'document.txt'}",
        target_view="upload",
    )
    try:
        document, sample_chunks = document_ingestion_service.ingest_document(
            db=db,
            content=content,
            filename=file.filename or "document.txt",
            content_type=file.content_type or "text/plain",
            title=title,
        )
        get_rag_service().clear_cache()
    except ValueError as exc:
        workspace_task_service.fail(db=db, task=task, detail=str(exc))
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_task_service.complete(
        db=db,
        task=task,
        detail=f"{document.title} is ready with {document.chunk_count} report sections.",
        metadata={"document_id": document.id},
    )
    db.commit()

    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document),
        sample_chunks=sample_chunks,
    )


@router.get("/datasets", response_model=list[DatasetRead])
def list_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DatasetRead]:
    """List uploaded structured datasets."""

    del current_user
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [DatasetRead.model_validate(dataset) for dataset in datasets]


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetRead:
    """Fetch a specific dataset."""

    del current_user
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return DatasetRead.model_validate(dataset)


@router.get("/documents", response_model=list[DocumentRead])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[DocumentRead]:
    """List indexed documents."""

    del current_user
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentRead.model_validate(document) for document in documents]
