"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.domain.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description=(
        "Returns basic liveness information, including whether a Gemini API key is configured."
    ),
)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        gemini_configured=bool(settings.gemini_api_key),
    )
