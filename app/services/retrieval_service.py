"""
Retrieval service.

Thin orchestration layer over :class:`VectorStoreService` that applies
business rules to semantic search (e.g. default top-k, score thresholds) and
formats results for prompt construction.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.domain.models import RetrievedContext
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

# Chunks scoring below this relevance threshold are dropped, since they are
# unlikely to be genuinely relevant to the question.
MIN_RELEVANCE_SCORE = 0.15


class RetrievalService:
    """Retrieves the most relevant document chunks for a given query."""

    def __init__(
        self,
        vector_store_service: VectorStoreService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vector_store = vector_store_service or VectorStoreService(settings=self._settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        """Return relevant chunks for the query, filtered by a minimum relevance score."""
        k = top_k or self._settings.retrieval_top_k
        results = self._vector_store.similarity_search(query, top_k=k, document_ids=document_ids)
        filtered = [r for r in results if r.score >= MIN_RELEVANCE_SCORE]
        logger.info(
            "Retrieval query=%r returned %d/%d chunks above threshold",
            query[:80],
            len(filtered),
            len(results),
        )
        return filtered

    @staticmethod
    def format_context(contexts: list[RetrievedContext]) -> str:
        """Render retrieved chunks into a numbered context block for the LLM prompt."""
        if not contexts:
            return "No relevant context was found in the uploaded documents."

        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            blocks.append(
                f"[Source {i}] Document: {ctx.chunk.document_name} | "
                f"Page: {ctx.chunk.page_number} | Relevance: {ctx.score:.2f}\n"
                f"{ctx.chunk.content}"
            )
        return "\n\n---\n\n".join(blocks)
