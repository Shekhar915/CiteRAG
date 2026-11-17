from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from citerag.core.config import Settings, get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Create the Gemini embedding client; fails early if no key is set."""
    settings = settings or get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.gemini_api_key
    )