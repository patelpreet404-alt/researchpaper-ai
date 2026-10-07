"""
PDF text-extraction service.

Wraps ``pypdf`` to extract per-page text from uploaded PDF files. Isolating
this behind a service means the rest of the application never needs to know
which PDF library is in use.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings, get_settings
from app.core.exceptions import EmptyDocumentError, GeminiServiceError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)


class PDFExtractionService:
    """Extracts text content from PDF files, page by page."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: genai.Client | None = None

    def extract_pages(self, file_path: Path) -> list[str]:
        """Return a list of page texts (index 0 == page 1)."""
        if file_path.suffix.lower() != ".pdf":
            raise UnsupportedFileTypeError(f"'{file_path.suffix}' is not a supported file type.")

        try:
            reader = PdfReader(str(file_path))
        except PdfReadError as exc:
            logger.exception("Failed to read PDF: %s", file_path)
            raise UnsupportedFileTypeError("The uploaded file is not a valid PDF.") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise UnsupportedFileTypeError(
                    "The PDF is password-protected and cannot be processed."
                ) from exc

        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                logger.warning("Failed to extract text from page %s of %s", index + 1, file_path)
                text = ""
            pages.append(text.strip())

        if not self._settings.gemini_api_key:
            scanned_pages = [
                index + 1
                for index, (page, text) in enumerate(zip(reader.pages, pages, strict=True))
                if len("".join(char for char in text if char.isalnum())) < 30 and bool(page.images)
            ]
            if scanned_pages:
                page_list = ", ".join(str(number) for number in scanned_pages[:8])
                raise EmptyDocumentError(
                    "Scanned pages were detected"
                    f" ({page_list}). Add a Gemini API key to OCR those pages."
                )

        # OCR only pages without a useful text layer. This keeps ordinary
        # research PDFs fast and avoids model charges for readable pages.
        if self._settings.gemini_api_key:
            pages_to_ocr = [
                index
                for index, text in enumerate(pages)
                if len("".join(char for char in text if char.isalnum())) < 30
            ]
            if len(pages_to_ocr) > self._settings.max_ocr_pages:
                raise UnsupportedFileTypeError(
                    f"This PDF has {len(pages_to_ocr)} image-only pages; "
                    f"the OCR limit is {self._settings.max_ocr_pages}."
                )
            if pages_to_ocr:
                with pymupdf.open(file_path) as rendered_pdf:
                    for index in pages_to_ocr:
                        rendered_page = rendered_pdf[index]
                        if rendered_page.get_contents():
                            pages[index] = self._ocr_page(rendered_page, index + 1)

        if not any(pages):
            raise EmptyDocumentError(
                "No readable text was found. For scanned PDFs, configure a server-side "
                "Gemini API key to enable OCR."
            )

        logger.info("Extracted %d pages from %s", len(pages), file_path.name)
        return pages

    def _ocr_page(self, page: pymupdf.Page, page_number: int) -> str:
        """Transcribe an image-only page while retaining its original page number."""
        if self._client is None:
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=self._settings.gemini_request_timeout * 1000
                ),
            )
        image = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png")
        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_chat_model,
                contents=[
                    "Transcribe all readable text on this research-paper page in reading order. "
                    "Preserve headings, formulas, table rows, and figure labels as plain text. "
                    "Do not add explanations or information that is not visible.",
                    types.Part.from_bytes(data=image, mime_type="image/png"),
                ],
                config=types.GenerateContentConfig(max_output_tokens=3000, temperature=0),
            )
        except errors.APIError as exc:
            raise GeminiServiceError(f"OCR failed on page {page_number}: {exc}") from exc
        text = (response.text or "").strip()
        if not text:
            raise EmptyDocumentError(f"OCR found no readable text on page {page_number}.")
        logger.info("OCR transcribed page %d", page_number)
        return text
