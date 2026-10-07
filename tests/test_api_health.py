"""Integration test for the health-check endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "ResearchPaper AI"
    assert "gemini_configured" in body


def test_docs_endpoint_is_available() -> None:
    response = client.get("/api/docs")
    assert response.status_code == 200
