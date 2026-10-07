"""Gemini powered retrieval augmented generation for research papers."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from google import genai
from google.genai import errors, types

from app.config import Settings, get_settings
from app.core.constants import DEFAULT_SYSTEM_PROMPT, STRUCTURED_SYSTEM_PROMPT
from app.core.exceptions import GeminiServiceError, MissingAPIKeyError
from app.domain.models import RetrievedContext
from app.domain.schemas import SourceReference, StructuredAnswer
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class ChatService:
    """Coordinates semantic retrieval, answer generation, and citations."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._retrieval_service = retrieval_service or RetrievalService(settings=self._settings)
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise MissingAPIKeyError()
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=self._settings.gemini_request_timeout * 1000
                ),
            )
        return self._client

    def retrieve_context(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        return self._retrieval_service.retrieve(question, top_k=top_k, document_ids=document_ids)

    def _build_user_prompt(self, question: str, contexts: list[RetrievedContext]) -> str:
        context_block = self._retrieval_service.format_context(contexts)
        return (
            "Context excerpts from the user's uploaded documents:\n\n"
            f"{context_block}\n\nQuestion: {question}"
        )

    @staticmethod
    def contexts_to_sources(contexts: list[RetrievedContext]) -> list[SourceReference]:
        return [
            SourceReference(
                document_id=context.chunk.document_id,
                document_name=context.chunk.document_name,
                page_number=context.chunk.page_number,
                snippet=(
                    context.chunk.content[:280] + "..."
                    if len(context.chunk.content) > 280
                    else context.chunk.content
                ),
                relevance_score=round(context.score, 4),
            )
            for context in contexts
        ]

    def stream_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Yield an answer incrementally using Gemini's content stream."""
        if not self._settings.gemini_api_key:
            yield from self._stream_local_answer(contexts)
            return
        turns = [
            {
                "role": "model" if turn["role"] == "assistant" else "user",
                "parts": [turn["content"]],
            }
            for turn in (history or [])
        ]
        turns.append({"role": "user", "parts": [self._build_user_prompt(question, contexts)]})
        try:
            stream = self.client.models.generate_content_stream(
                model=self._settings.gemini_chat_model,
                contents=turns,
                config=types.GenerateContentConfig(
                    system_instruction=DEFAULT_SYSTEM_PROMPT,
                    max_output_tokens=self._settings.gemini_max_output_tokens,
                    temperature=0.2,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except errors.APIError as exc:
            logger.exception("Gemini streaming call failed")
            raise GeminiServiceError(f"Answer generation failed: {exc}") from exc

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Non-streaming helper used by integrations and tests."""
        return "".join(self.stream_answer(question, contexts, history))

    def generate_structured_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[dict[str, str]] | None = None,
    ) -> StructuredAnswer:
        """Generate a typed answer and replace model citations with retrieved sources."""
        if not self._settings.gemini_api_key:
            answer = self._local_answer(contexts)
            return StructuredAnswer(
                answer_found=bool(contexts),
                answer=answer,
                key_points=[context.chunk.content[:180] for context in contexts[:3]],
                sources=self.contexts_to_sources(contexts),
                confidence=0.55 if contexts else 0.0,
            )
        if not contexts:
            return StructuredAnswer(
                answer_found=False,
                answer="I couldn't find relevant information in this session's documents.",
                key_points=[],
                sources=[],
                confidence=0.0,
            )

        prior_turns = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in (history or []))
        prompt = self._build_user_prompt(question, contexts)
        if prior_turns:
            prompt = f"Conversation history:\n{prior_turns}\n\n{prompt}"
        try:
            response = self.client.models.generate_content(
                model=self._settings.gemini_chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STRUCTURED_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=StructuredAnswer,
                    max_output_tokens=self._settings.gemini_max_output_tokens,
                    temperature=0.2,
                ),
            )
        except errors.APIError as exc:
            logger.exception("Gemini structured call failed")
            raise GeminiServiceError(f"Answer generation failed: {exc}") from exc

        parsed = response.parsed
        if isinstance(parsed, dict):
            parsed = StructuredAnswer.model_validate(parsed)
        elif parsed is None and response.text:
            parsed = StructuredAnswer.model_validate_json(response.text)
        if not isinstance(parsed, StructuredAnswer):
            raise GeminiServiceError("Gemini did not return a parseable structured answer.")
        parsed.sources = self.contexts_to_sources(contexts)
        return parsed

    @staticmethod
    def _local_answer(contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "I couldn't find relevant information in this session's documents."
        excerpts = "\n\n".join(
            f"> {context.chunk.content[:650].strip()}\n\n"
            f"— *{context.chunk.document_name}, page {context.chunk.page_number}*"
            for context in contexts[:3]
        )
        return (
            "I found these relevant passages in your document. Add a server-side Gemini API key "
            "to enable generated summaries and follow-up reasoning.\n\n" + excerpts
        )

    @classmethod
    def _stream_local_answer(cls, contexts: list[RetrievedContext]) -> Iterator[str]:
        answer = cls._local_answer(contexts)
        for start in range(0, len(answer), 48):
            yield answer[start : start + 48]
