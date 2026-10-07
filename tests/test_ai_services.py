"""Provider adapter tests using fake Gemini responses and no network access."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import get_settings
from app.domain.models import RetrievedContext, TextChunk
from app.domain.schemas import SourceReference, StructuredAnswer
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService


def test_gemini_structured_answer_uses_retrieved_citations(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    context = RetrievedContext(
        chunk=TextChunk(
            chunk_id="chunk-1",
            document_id="paper-1",
            document_name="paper.pdf",
            page_number=3,
            content="The study evaluates retrieval quality.",
            chunk_index=0,
        ),
        score=0.82,
    )
    answer = StructuredAnswer(
        answer_found=True,
        answer="The study evaluates retrieval quality.",
        sources=[
            SourceReference(
                document_id="invented",
                document_name="invented.pdf",
                page_number=999,
                snippet="",
                relevance_score=1,
            )
        ],
        confidence=0.8,
    )
    service = ChatService(settings=settings)
    service._retrieval_service = SimpleNamespace(format_context=lambda _contexts: "paper context")
    service._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: SimpleNamespace(parsed=answer))
    )

    result = service.generate_structured_answer("What was evaluated?", [context])

    assert result.answer == "The study evaluates retrieval quality."
    assert len(result.sources) == 1
    assert result.sources[0].document_id == "paper-1"
    assert result.sources[0].page_number == 3


def test_gemini_embedding_adapter_returns_document_and_query_vectors(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    class FakeModels:
        def embed_content(self, **_kwargs):
            contents = _kwargs["contents"]
            if isinstance(contents, str):
                contents = [contents]
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[float(index + 1), 0.5])
                    for index, _ in enumerate(contents)
                ]
            )

    service = EmbeddingService(settings=settings)
    service._client = SimpleNamespace(models=FakeModels())

    assert service.embed_documents(["first", "second"]) == [[1.0, 0.5], [2.0, 0.5]]
    assert service.embed_query("question") == [1.0, 0.5]
