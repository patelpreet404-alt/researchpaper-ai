"""Shared pytest fixtures."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Ensure a predictable, isolated environment before any app module is imported.
_TEST_DIR = Path(tempfile.mkdtemp(prefix="researchpaper-ai-test-"))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ["DATA_DIR"] = str(_TEST_DIR / "data")
os.environ["UPLOAD_DIR"] = str(_TEST_DIR / "data" / "uploads")
os.environ["VECTOR_STORE_DIR"] = str(_TEST_DIR / "data" / "vector_store")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR / 'data' / 'test.db'}"
os.environ["LOG_DIR"] = str(_TEST_DIR / "logs")


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_dir():
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _init_test_database():
    from app.infrastructure.db import init_db

    init_db()
    yield


@pytest.fixture()
def sample_pdf_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample-rag-guide.pdf"
