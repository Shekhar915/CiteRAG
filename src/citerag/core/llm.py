from langchain_google_genai import ChatGoogleGenerativeAI

from citerag.core.config import Settings, get_settings


def build_llm(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model from settings.

    Raises ValueError when no API key is configured so the failure is clear
    and happens before any network call.
    """
    settings = settings or get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )