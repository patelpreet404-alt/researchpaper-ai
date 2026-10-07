"""
Vector database service.

Wraps a LangChain FAISS vector store to provide persistent, on-disk
similarity search over document chunk embeddings. FAISS (Facebook AI
Similarity Search) performs efficient approximate/exact nearest-neighbour
search over dense vectors, which is the core "retrieval" step of RAG.

The index is persisted to ``settings.vector_store_dir`` so it survives
application restarts, and is lazily loaded on first use.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from pathlib import Path

from langchain_community.docstore.document import Document as LCDocument
from langchain_community.vectorstores import FAISS

from app.config import Settings, get_settings
from app.core.constants import FAISS_INDEX_FILENAME
from app.core.exceptions import VectorStoreNotReadyError
from app.domain.models import RetrievedContext, TextChunk
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "is",
    "paper",
    "the",
    "this",
    "those",
    "what",
    "why",
    "with",
    "you",
    "your",
}


class VectorStoreService:
    """Manages a persistent FAISS index of document chunk embeddings."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
        index_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._store: FAISS | None = None
        self._lock = threading.RLock()
        self._custom_index_path = index_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self._custom_index_path or self._settings.vector_store_dir

    @property
    def _lexical_path(self) -> Path:
        return self._index_path / "chunks.json"

    def _index_exists(self) -> bool:
        return (self._index_path / FAISS_INDEX_FILENAME).exists()

    def _read_lexical_rows(self) -> list[dict]:
        return json.loads(self._lexical_path.read_text()) if self._lexical_path.exists() else []

    def _write_lexical_rows(self, rows: list[dict]) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        if not rows:
            self._lexical_path.unlink(missing_ok=True)
            return
        temporary_path = self._index_path / f"chunks-{uuid.uuid4().hex}.tmp"
        try:
            temporary_path.write_text(json.dumps(rows))
            temporary_path.replace(self._lexical_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load(self) -> FAISS | None:
        if self._store is not None:
            return self._store
        if not self._index_exists():
            return None
        logger.info("Loading FAISS index from %s", self._index_path)
        self._store = FAISS.load_local(
            str(self._index_path),
            self._embedding_service,
            allow_dangerous_deserialization=True,
        )
        return self._store

    @staticmethod
    def _chunk_to_document(chunk: TextChunk) -> LCDocument:
        return LCDocument(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """Embed and add chunks to the index, creating it if necessary."""
        if not chunks:
            return 0

        with self._lock:
            self._index_path.mkdir(parents=True, exist_ok=True)
            rows = self._read_lexical_rows()
            new_rows = [
                {
                    "content": chunk.content,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                }
                for chunk in chunks
            ]
            all_rows = [*rows, *new_rows]

            if not self._settings.gemini_api_key:
                self._write_lexical_rows(all_rows)
                logger.info(
                    "Stored %d chunks for local keyword search (no Gemini key)", len(chunks)
                )
                return len(chunks)

            existing = self._load()
            if existing is None:
                # Include papers uploaded earlier in the no-key mode when a
                # server-side key is configured later.
                all_documents = [
                    LCDocument(page_content=row["content"], metadata=row["metadata"])
                    for row in all_rows
                ]
                logger.info("Creating new FAISS index with %d chunks", len(all_documents))
                self._store = FAISS.from_documents(all_documents, self._embedding_service)
            else:
                documents = [self._chunk_to_document(chunk) for chunk in chunks]
                logger.info("Adding %d chunks to existing FAISS index", len(documents))
                existing.add_documents(documents)
                self._store = existing

            self._store.save_local(str(self._index_path))
            self._write_lexical_rows(all_rows)

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        """Perform semantic search and return the top-k most relevant chunks."""
        with self._lock:
            if not self._settings.gemini_api_key or not self._index_exists():
                return self._lexical_search(query, top_k, document_ids)

            store = self._load()
            if store is None:
                return self._lexical_search(query, top_k, document_ids)

            # Over-fetch when filtering by document id, since FAISS similarity
            # search does not natively support metadata pre-filtering.
            fetch_k = store.index.ntotal if document_ids else top_k
            results = store.similarity_search_with_relevance_scores(query, k=fetch_k)

            contexts: list[RetrievedContext] = []
            for doc, score in results:
                metadata = doc.metadata
                if document_ids and metadata.get("document_id") not in document_ids:
                    continue
                chunk = TextChunk(
                    chunk_id=metadata.get("chunk_id", ""),
                    document_id=metadata.get("document_id", ""),
                    document_name=metadata.get("document_name", "Unknown"),
                    page_number=metadata.get("page_number", 0),
                    content=doc.page_content,
                    chunk_index=metadata.get("chunk_index", 0),
                )
                contexts.append(
                    RetrievedContext(chunk=chunk, score=max(0.0, min(float(score), 1.0)))
                )
                if len(contexts) >= top_k:
                    break

            if self._lexical_path.exists():
                indexed_ids = {
                    doc.metadata.get("document_id")
                    for doc in store.docstore._dict.values()  # type: ignore[attr-defined]
                }
                contexts.extend(
                    context
                    for context in self._lexical_search(query, top_k, document_ids)
                    if context.chunk.document_id not in indexed_ids
                )
            contexts.sort(key=lambda context: context.score, reverse=True)

            logger.info("Retrieved %d chunks for query (top_k=%d)", len(contexts), top_k)
            return contexts[:top_k]

    def is_ready(self) -> bool:
        return self._index_exists() or self._lexical_path.exists()

    def _lexical_search(
        self, query: str, top_k: int, document_ids: list[str] | None
    ) -> list[RetrievedContext]:
        if not self._lexical_path.exists():
            raise VectorStoreNotReadyError()
        terms = set(re.findall(r"[a-z0-9]{2,}", query.lower())) - _STOPWORDS
        rows = self._read_lexical_rows()
        ranked: list[tuple[float, dict]] = []
        for row in rows:
            metadata = row["metadata"]
            if document_ids and metadata.get("document_id") not in document_ids:
                continue
            words = re.findall(r"[a-z0-9]{2,}", row["content"].lower())
            if not words:
                continue
            score = (
                sum(words.count(term) for term in terms) / (len(terms) * (len(words) ** 0.5))
                if terms
                else 0
            )
            if score > 0:
                ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedContext(
                chunk=TextChunk(
                    chunk_id=row["metadata"].get("chunk_id", ""),
                    document_id=row["metadata"].get("document_id", ""),
                    document_name=row["metadata"].get("document_name", "Unknown"),
                    page_number=row["metadata"].get("page_number", 0),
                    content=row["content"],
                    chunk_index=row["metadata"].get("chunk_index", 0),
                ),
                score=min(score, 1.0),
            )
            for score, row in ranked[:top_k]
        ]

    def delete_document(self, document_id: str) -> None:
        """Rebuild the index excluding all chunks belonging to a document.

        FAISS does not support efficient in-place deletion by metadata, so we
        reconstruct the index from the remaining in-memory docstore entries.
        This is acceptable for the moderate document volumes typical of a
        single-tenant / small-team deployment.
        """
        with self._lock:
            if self._lexical_path.exists():
                remaining_rows = [
                    row
                    for row in self._read_lexical_rows()
                    if row["metadata"].get("document_id") != document_id
                ]
                self._write_lexical_rows(remaining_rows)

            if not self._index_exists():
                return
            store = self._load()
            if store is None:
                return

            docstore_dict = store.docstore._dict  # type: ignore[attr-defined]
            remaining = [
                doc
                for doc in docstore_dict.values()
                if doc.metadata.get("document_id") != document_id
            ]
            if not remaining:
                self._clear_index()
                return

            self._store = FAISS.from_documents(remaining, self._embedding_service)
            self._store.save_local(str(self._index_path))
            logger.info("Rebuilt FAISS index after deleting document %s", document_id)

    def _clear_index(self) -> None:
        if self._index_path.exists():
            for filename in ("index.faiss", "index.pkl"):
                (self._index_path / filename).unlink(missing_ok=True)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store = None
        logger.info("Cleared FAISS index (no documents remaining)")
