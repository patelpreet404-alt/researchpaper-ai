"""
Chat endpoints.

Provides three ways to interact with the RAG pipeline:

* ``POST /api/chat`` — synchronous structured JSON answer (demonstrates
  Gemini structured output constrained to a Pydantic schema).
* ``POST /api/chat/stream`` — Server-Sent Events streaming endpoint used by
  the web UI for a token-by-token typing effect.
* ``GET /api/conversations`` / ``GET /api/conversations/{id}`` — conversation
  history retrieval (conversation memory).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, get_memory_service
from app.core.exceptions import GeminiServiceError, MissingAPIKeyError, VectorStoreNotReadyError
from app.domain.models import MessageRole, RetrievedContext, TextChunk
from app.domain.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ConversationListResponse,
    ConversationResponse,
    StructuredAnswer,
)
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


def _context_for_request(request: ChatRequest, chat_service: ChatService) -> list[RetrievedContext]:
    if request.context_chunks is not None:
        try:
            return chat_service.retrieve_context(
                request.question, top_k=request.top_k, document_ids=request.document_ids
            )
        except VectorStoreNotReadyError:
            pass
    else:
        return chat_service.retrieve_context(
            request.question, top_k=request.top_k, document_ids=request.document_ids
        )
    return [
        RetrievedContext(
            chunk=TextChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
            ),
            score=1.0,
        )
        for chunk in request.context_chunks
        if not request.document_ids or chunk.document_id in request.document_ids
    ]


@router.post(
    "/chat",
    response_model=StructuredAnswer,
    summary="Ask a question and receive a structured JSON answer",
    description=(
        "Retrieves relevant chunks via semantic search and asks Gemini "
        "to answer using a constrained JSON schema "
        "(answer, key_points, sources, confidence)."
    ),
)
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> StructuredAnswer:
    conversation = memory_service.get_or_create_conversation(
        request.conversation_id, title_hint=request.question
    )
    history = memory_service.to_prompt_history(memory_service.get_history(conversation.id))

    try:
        contexts = _context_for_request(request, chat_service)
        answer = chat_service.generate_structured_answer(request.question, contexts, history)
    except VectorStoreNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc
    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED, detail=exc.message
        ) from exc
    except GeminiServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.message) from exc

    memory_service.append_message(conversation.id, MessageRole.USER, request.question)
    memory_service.append_message(
        conversation.id,
        MessageRole.ASSISTANT,
        answer.answer,
        sources=[s.model_dump() for s in answer.sources],
    )
    return answer


@router.post(
    "/chat/stream",
    summary="Ask a question and stream the answer via Server-Sent Events",
    description=(
        "Streams the model's answer token-by-token as it is generated, using "
        "Gemini streaming generation. The final SSE event contains "
        "the resolved source citations."
    ),
)
def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> StreamingResponse:
    conversation = memory_service.get_or_create_conversation(
        request.conversation_id, title_hint=request.question
    )
    history = memory_service.to_prompt_history(memory_service.get_history(conversation.id))

    try:
        contexts = _context_for_request(request, chat_service)
    except VectorStoreNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc

    sources = chat_service.contexts_to_sources(contexts)
    memory_service.append_message(conversation.id, MessageRole.USER, request.question)

    def event_stream():
        collected = []
        yield _sse_event("meta", {"conversation_id": conversation.id})
        try:
            for token in chat_service.stream_answer(request.question, contexts, history):
                collected.append(token)
                yield _sse_event("token", {"text": token})
        except (GeminiServiceError, MissingAPIKeyError) as exc:
            yield _sse_event("error", {"detail": exc.message})
            return

        full_answer = "".join(collected)
        memory_service.append_message(
            conversation.id,
            MessageRole.ASSISTANT,
            full_answer,
            sources=[s.model_dump() for s in sources],
        )
        yield _sse_event("sources", {"sources": [s.model_dump() for s in sources]})
        yield _sse_event("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List all conversations",
)
def list_conversations(
    memory_service: MemoryService = Depends(get_memory_service),
) -> ConversationListResponse:
    conversations = memory_service.list_conversations()
    return ConversationListResponse(
        conversations=[ConversationResponse.model_validate(c) for c in conversations],
        total=len(conversations),
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation and its full message history",
)
def get_conversation(
    conversation_id: str, memory_service: MemoryService = Depends(get_memory_service)
) -> ConversationResponse:
    conversation = memory_service.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    messages = memory_service.get_history(conversation_id)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[ChatMessageResponse.model_validate(m) for m in messages],
    )
