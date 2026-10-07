"""
Document management endpoints: upload, list, retrieve, delete.

Implements the "Upload PDF / Multiple PDFs / Document Library / Automatic
Text Extraction" features by delegating to :class:`DocumentService`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.deps import get_document_service
from app.core.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    PDFChatGPTError,
    UnsupportedFileTypeError,
)
from app.domain.schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/sample",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add the built-in sample research PDF to this browser session",
)
def add_sample_document(
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    try:
        document = service.import_sample()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFChatGPTError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.message
        ) from exc
    return DocumentUploadResponse(document=DocumentResponse.model_validate(document))


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
    description=(
        "Uploads a PDF, extracts its text, splits it into overlapping chunks, "
        "generates embeddings, and stores them in the FAISS vector index so "
        "the document becomes immediately searchable."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload."),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    content = await file.read()

    try:
        service.validate_upload(file.filename or "upload.pdf", len(content))
        document_id, stored_path = service.save_upload(file.filename or "upload.pdf", content)
        document = service.ingest(
            document_id, file.filename or "upload.pdf", stored_path, len(content)
        )
    except (UnsupportedFileTypeError, FileTooLargeError, EmptyDocumentError) as exc:
        logger.warning("Upload rejected: %s", exc.message)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc
    except PDFChatGPTError as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.message
        ) from exc

    return DocumentUploadResponse(document=DocumentResponse.model_validate(document))


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents = service.list_documents()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document's metadata",
)
def get_document(
    document_id: str, service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a document and remove it from the vector index",
)
def delete_document(
    document_id: str, service: DocumentService = Depends(get_document_service)
) -> Response:
    deleted = service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
