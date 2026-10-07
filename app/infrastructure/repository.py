"""
Repository pattern implementations.

Repositories translate between ORM rows and domain dataclasses, keeping
SQLAlchemy-specific query logic out of the service layer. This is the
Clean Architecture "interface adapter" boundary between the domain and
infrastructure.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import (
    ChatMessage,
    Conversation,
    Document,
    DocumentStatus,
    MessageRole,
)
from app.infrastructure.models_db import ConversationORM, DocumentORM, MessageORM

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Persistence operations for :class:`Document` entities."""

    def __init__(self, db: Session, owner_id: str) -> None:
        self._db = db
        self._owner_id = owner_id

    def add(self, document: Document) -> Document:
        row = DocumentORM(
            id=document.id,
            owner_id=self._owner_id,
            filename=document.filename,
            stored_path=document.stored_path,
            status=document.status.value,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            size_bytes=document.size_bytes,
            error_message=document.error_message,
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._to_domain(row)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        row = self._db.scalar(
            select(DocumentORM).where(
                DocumentORM.id == document_id, DocumentORM.owner_id == self._owner_id
            )
        )
        if row is None:
            return
        row.status = status.value
        if page_count is not None:
            row.page_count = page_count
        if chunk_count is not None:
            row.chunk_count = chunk_count
        if error_message is not None:
            row.error_message = error_message
        self._db.commit()

    def get(self, document_id: str) -> Document | None:
        row = self._db.scalar(
            select(DocumentORM).where(
                DocumentORM.id == document_id, DocumentORM.owner_id == self._owner_id
            )
        )
        return self._to_domain(row) if row else None

    def list_all(self) -> list[Document]:
        rows = self._db.scalars(
            select(DocumentORM)
            .where(DocumentORM.owner_id == self._owner_id)
            .order_by(DocumentORM.uploaded_at.desc())
        ).all()
        return [self._to_domain(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        row = self._db.scalar(
            select(DocumentORM).where(
                DocumentORM.id == document_id, DocumentORM.owner_id == self._owner_id
            )
        )
        if row is None:
            return False
        self._db.delete(row)
        self._db.commit()
        return True

    @staticmethod
    def _to_domain(row: DocumentORM) -> Document:
        return Document(
            id=row.id,
            filename=row.filename,
            stored_path=row.stored_path,
            status=DocumentStatus(row.status),
            page_count=row.page_count,
            chunk_count=row.chunk_count,
            size_bytes=row.size_bytes,
            uploaded_at=row.uploaded_at,
            error_message=row.error_message,
        )


class ConversationRepository:
    """Persistence operations for conversations and their messages."""

    def __init__(self, db: Session, owner_id: str) -> None:
        self._db = db
        self._owner_id = owner_id

    def create(self, conversation: Conversation) -> Conversation:
        row = ConversationORM(id=conversation.id, owner_id=self._owner_id, title=conversation.title)
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._to_domain(row)

    def get(self, conversation_id: str) -> Conversation | None:
        row = self._db.scalar(
            select(ConversationORM).where(
                ConversationORM.id == conversation_id, ConversationORM.owner_id == self._owner_id
            )
        )
        return self._to_domain(row) if row else None

    def list_all(self) -> list[Conversation]:
        rows = self._db.scalars(
            select(ConversationORM)
            .where(ConversationORM.owner_id == self._owner_id)
            .order_by(ConversationORM.created_at.desc())
        ).all()
        return [self._to_domain(row) for row in rows]

    def add_message(self, message: ChatMessage) -> ChatMessage:
        row = MessageORM(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            sources_json=json.dumps(message.sources),
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._message_to_domain(row)

    def get_history(self, conversation_id: str, limit: int = 20) -> list[ChatMessage]:
        rows = self._db.scalars(
            select(MessageORM)
            .join(ConversationORM, MessageORM.conversation_id == ConversationORM.id)
            .where(
                MessageORM.conversation_id == conversation_id,
                ConversationORM.owner_id == self._owner_id,
            )
            .order_by(MessageORM.created_at.desc())
            .limit(limit)
        ).all()
        return [self._message_to_domain(row) for row in reversed(rows)]

    @staticmethod
    def _to_domain(row: ConversationORM) -> Conversation:
        return Conversation(
            id=row.id,
            title=row.title,
            created_at=row.created_at,
            messages=[ConversationRepository._message_to_domain(m) for m in row.messages],
        )

    @staticmethod
    def _message_to_domain(row: MessageORM) -> ChatMessage:
        try:
            sources = json.loads(row.sources_json)
        except (json.JSONDecodeError, TypeError):
            sources = []
        return ChatMessage(
            id=row.id,
            conversation_id=row.conversation_id,
            role=MessageRole(row.role),
            content=row.content,
            sources=sources,
            created_at=row.created_at,
        )
