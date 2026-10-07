"""
Conversation memory service.

Persists and retrieves chat history so that follow-up questions can be
answered with awareness of prior turns. History is stored in SQLite via
    :class:`ConversationRepository` and replayed into the Gemini chat model
as prior turns on each new request.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.domain.models import ChatMessage, Conversation, MessageRole
from app.infrastructure.repository import ConversationRepository

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12


class MemoryService:
    """Reads and writes conversation turns, providing bounded history windows."""

    def __init__(self, db: Session, owner_id: str) -> None:
        self._repo = ConversationRepository(db, owner_id)

    def get_or_create_conversation(
        self, conversation_id: str | None, title_hint: str = ""
    ) -> Conversation:
        if conversation_id:
            existing = self._repo.get(conversation_id)
            if existing:
                return existing
        default_title = title_hint or "New conversation"
        title = (title_hint[:60] + "...") if len(title_hint) > 60 else default_title
        conversation = Conversation(id=str(uuid.uuid4()), title=title or "New conversation")
        return self._repo.create(conversation)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._repo.get(conversation_id)

    def get_history(self, conversation_id: str) -> list[ChatMessage]:
        return self._repo.get_history(conversation_id, limit=MAX_HISTORY_MESSAGES)

    def append_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        sources: list[dict] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources or [],
        )
        return self._repo.add_message(message)

    def list_conversations(self) -> list[Conversation]:
        return self._repo.list_all()

    def to_prompt_history(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert stored messages into the ``role``/``content`` dict shape the
        generation provider expects for prior turns."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
