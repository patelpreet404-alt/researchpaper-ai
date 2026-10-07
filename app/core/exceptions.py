"""
Domain-specific exception hierarchy.

Keeping exceptions typed and centralized allows the API layer to translate
them into consistent, well-documented HTTP responses instead of leaking
implementation details (stack traces, third-party errors) to clients.
"""

from __future__ import annotations


class PDFChatGPTError(Exception):
    """Base class for all application-specific errors."""

    default_message = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


class DocumentNotFoundError(PDFChatGPTError):
    default_message = "The requested document was not found."


class UnsupportedFileTypeError(PDFChatGPTError):
    default_message = "Only PDF files are supported."


class FileTooLargeError(PDFChatGPTError):
    default_message = "The uploaded file exceeds the maximum allowed size."


class EmptyDocumentError(PDFChatGPTError):
    default_message = "No extractable text was found in the document."


class VectorStoreNotReadyError(PDFChatGPTError):
    default_message = "The vector store has not been built yet. Upload a document first."


class ConversationNotFoundError(PDFChatGPTError):
    default_message = "The requested conversation was not found."


class GeminiServiceError(PDFChatGPTError):
    default_message = "The AI service failed to generate a response."


class MissingAPIKeyError(PDFChatGPTError):
    default_message = (
        "GEMINI_API_KEY is not configured. Add it to your private .env file to use this feature."
    )
