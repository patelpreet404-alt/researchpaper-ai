"""
FastAPI dependency providers.

Centralizes construction of service-layer objects so route handlers stay
declarative and testable (dependencies can be overridden in tests via
``app.dependency_overrides``).
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.infrastructure.db import get_db
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService

# The embedding client can be shared, while each visitor gets an independent
# vector store rooted in their opaque browser-session directory.
_embedding_service = EmbeddingService()


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_session_id(request: Request) -> str:
    return request.state.owner_id


@lru_cache(maxsize=2048)
def _get_vector_store_service(owner_id: str) -> VectorStoreService:
    from app.config import get_settings

    settings = get_settings()
    return VectorStoreService(
        embedding_service=_embedding_service,
        settings=settings,
        index_path=settings.vector_store_dir / owner_id,
    )


def get_document_service(
    db: Session = Depends(get_db_session), owner_id: str = Depends(get_session_id)
) -> DocumentService:
    return DocumentService(
        db=db,
        owner_id=owner_id,
        vector_store_service=_get_vector_store_service(owner_id),
    )


def get_memory_service(
    db: Session = Depends(get_db_session), owner_id: str = Depends(get_session_id)
) -> MemoryService:
    return MemoryService(db=db, owner_id=owner_id)


def get_chat_service(owner_id: str = Depends(get_session_id)) -> ChatService:
    return ChatService(
        retrieval_service=RetrievalService(vector_store_service=_get_vector_store_service(owner_id))
    )


def get_retrieval_service(owner_id: str = Depends(get_session_id)) -> RetrievalService:
    return RetrievalService(vector_store_service=_get_vector_store_service(owner_id))


def get_vector_store_service(owner_id: str = Depends(get_session_id)) -> VectorStoreService:
    return _get_vector_store_service(owner_id)
