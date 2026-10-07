"""
Application entry point.

Creates and configures the FastAPI application: logging, CORS, exception
handlers, routers, database initialization, and static file serving for the
single-page chat UI. Run with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.config import get_settings
from app.core.exceptions import PDFChatGPTError
from app.infrastructure.db import init_db
from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s (%s environment)",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    init_db()
    if not settings.gemini_api_key:
        logger.info(
            "GEMINI_API_KEY is not set. Local keyword search and extractive answers are enabled."
        )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        description=(
            "An enterprise-grade Retrieval-Augmented Generation (RAG) API for chatting "
            "with your PDF documents, built on FastAPI, LangChain, FAISS, and the "
            "Gemini API."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def isolate_browser_sessions(request: Request, call_next):
        session_id = request.cookies.get("researchpaper_session", "")
        try:
            session_id = str(uuid.UUID(session_id))
            is_new_session = False
        except (ValueError, AttributeError):
            session_id = str(uuid.uuid4())
            is_new_session = True
        request.state.owner_id = session_id
        response = await call_next(request)
        if is_new_session:
            response.set_cookie(
                "researchpaper_session",
                session_id,
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                secure=settings.app_env.lower() in {"prod", "production"},
                samesite="lax",
                path="/",
            )
        return response

    application.include_router(health_router, prefix="/api")
    application.include_router(documents_router)
    application.include_router(chat_router)

    _register_exception_handlers(application)
    _mount_static(application)

    return application


def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(PDFChatGPTError)
    async def handle_domain_error(request: Request, exc: PDFChatGPTError) -> JSONResponse:
        logger.warning("Domain error on %s %s: %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": exc.message, "error_type": type(exc).__name__},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "error_type": "InternalServerError",
            },
        )


def _mount_static(application: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent / "static"

    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @application.get("/workspace", include_in_schema=False)
    async def serve_workspace() -> FileResponse:
        return FileResponse(static_dir / "workspace.html")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
