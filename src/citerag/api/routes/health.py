from fastapi import APIRouter

from citerag.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, object]:
    """Report whether required configuration is present."""
    configured = bool(get_settings().gemini_api_key)
    return {"status": "ready" if configured else "not_ready", "llm_configured": configured}