"""Integration tests for document upload / listing / deletion endpoints.

These tests exercise the extraction and chunking pipeline end-to-end but do
NOT call the real Gemini API (embeddings are network calls), so they are
skipped automatically unless a real ``GEMINI_API_KEY`` is present. This keeps
the test suite runnable offline / in CI without incurring API costs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.exceptions import VectorStoreNotReadyError
from app.main import app
from app.services.chat_service import ChatService

client = TestClient(app)

requires_gemini_key = pytest.mark.skipif(
    os.environ.get("GEMINI_API_KEY", "").startswith("test-"),
    reason="Requires a real GEMINI_API_KEY to generate embeddings.",
)


def test_list_documents_empty_by_default() -> None:
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert "documents" in response.json()


def test_sample_search_and_history_are_isolated_by_browser_session(monkeypatch) -> None:
    # Exercise the no-key path without calling paid external services.
    monkeypatch.setattr(get_settings(), "gemini_api_key", "")
    first_browser = TestClient(app)
    second_browser = TestClient(app)

    sample = first_browser.post("/api/documents/sample")
    assert sample.status_code == 201
    document_id = sample.json()["document"]["id"]
    assert sample.json()["context_chunks"]
    assert sample.json()["context_chunks"][0]["document_id"] == document_id
    assert first_browser.get("/api/documents").json()["total"] == 1
    assert second_browser.get("/api/documents").json()["total"] == 0

    answer = first_browser.post(
        "/api/chat",
        json={"question": "What is retrieval augmented generation?", "document_ids": [document_id]},
    )
    assert answer.status_code == 200
    assert answer.json()["sources"]
    no_match = first_browser.post(
        "/api/chat", json={"question": "What is the dinosaur's favorite color?"}
    )
    assert no_match.status_code == 200
    assert no_match.json()["answer_found"] is False
    assert no_match.json()["sources"] == []
    conversation = first_browser.get("/api/conversations").json()["conversations"][0]
    assert first_browser.get("/api/conversations").json()["total"] == 2
    assert second_browser.get("/api/conversations").json()["total"] == 0
    assert second_browser.get(f"/api/documents/{document_id}").status_code == 404
    assert second_browser.get(f"/api/conversations/{conversation['id']}").status_code == 404


def test_stream_chat_uses_browser_context_when_serverless_index_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "gemini_api_key", "")

    def no_index(self, question, top_k=None, document_ids=None):
        raise VectorStoreNotReadyError()

    monkeypatch.setattr(ChatService, "retrieve_context", no_index)
    response = TestClient(app).post(
        "/api/chat/stream",
        json={
            "question": "What does this paper say about retrieval?",
            "context_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "document_name": "sample.pdf",
                    "page_number": 2,
                    "content": (
                        "Retrieval augmented generation finds relevant passages "
                        "before composing an answer."
                    ),
                    "chunk_index": 0,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "event: sources" in response.text
    assert '"page_number": 2' in response.text
    assert '"document_name": "sample.pdf"' in response.text


def test_homepage_and_workspace_are_website_routes() -> None:
    homepage = client.get("/")
    workspace = client.get("/workspace")
    assert homepage.status_code == 200
    assert "Read the paper" in homepage.text
    assert workspace.status_code == 200
    assert "Ask questions grounded" in workspace.text


def test_upload_rejects_non_pdf(tmp_path: Path) -> None:
    fake_file = tmp_path / "not-a-pdf.txt"
    fake_file.write_text("hello world")

    with fake_file.open("rb") as fh:
        response = client.post(
            "/api/documents", files={"file": ("not-a-pdf.txt", fh, "text/plain")}
        )

    assert response.status_code == 400


@requires_gemini_key
def test_upload_indexes_sample_pdf(sample_pdf_path: Path) -> None:
    with sample_pdf_path.open("rb") as fh:
        response = client.post(
            "/api/documents", files={"file": (sample_pdf_path.name, fh, "application/pdf")}
        )

    assert response.status_code == 201
    body = response.json()
    assert body["document"]["status"] == "indexed"
    assert body["document"]["chunk_count"] > 0
