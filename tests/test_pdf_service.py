"""Unit tests for PDFExtractionService."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.config import get_settings
from app.core.exceptions import UnsupportedFileTypeError
from app.services.pdf_service import PDFExtractionService


def test_extract_pages_returns_text_per_page(sample_pdf_path: Path) -> None:
    service = PDFExtractionService()
    pages = service.extract_pages(sample_pdf_path)

    assert len(pages) == 3
    assert "Retrieval-Augmented Generation" in pages[0]
    assert "Semantic Search" in pages[2]


def test_extract_pages_rejects_non_pdf(tmp_path: Path) -> None:
    fake_file = tmp_path / "not-a-pdf.txt"
    fake_file.write_text("hello world")

    service = PDFExtractionService()
    with pytest.raises(UnsupportedFileTypeError):
        service.extract_pages(fake_file)


def test_ocr_fallback_handles_image_only_pages_without_live_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scanned-paper.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.draw_rect(pymupdf.Rect(30, 30, 300, 120), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    pdf.save(pdf_path)
    pdf.close()

    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    service = PDFExtractionService(settings)
    monkeypatch.setattr(service, "_ocr_page", lambda _page, number: f"Transcribed page {number}")

    assert service.extract_pages(pdf_path) == ["Transcribed page 1"]
