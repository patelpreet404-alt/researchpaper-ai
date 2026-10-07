"""Google Gemini embeddings for semantic research-paper retrieval."""

from __future__ import annotations

import logging

from google import genai
from google.genai import errors, types
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.core.exceptions import GeminiServiceError, MissingAPIKeyError

logger = logging.getLogger(__name__)


class EmbeddingService(Embeddings):
    """Adapt Gemini's embedding API to LangChain's FAISS interface."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding %d chunks with Gemini", len(texts))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            try:
                response = self.client.models.embed_content(
                    model=self._settings.gemini_embedding_model,
                    contents=texts[start : start + 64],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
            except errors.APIError as exc:
                raise GeminiServiceError(f"Document embedding failed: {exc}") from exc
            vectors.extend([embedding.values or [] for embedding in response.embeddings or []])
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise GeminiServiceError("Gemini returned an incomplete set of document embeddings.")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.models.embed_content(
                model=self._settings.gemini_embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
            )
        except errors.APIError as exc:
            raise GeminiServiceError(f"Query embedding failed: {exc}") from exc
        if not response.embeddings or not response.embeddings[0].values:
            raise GeminiServiceError("Gemini returned an empty query embedding.")
        return response.embeddings[0].values
