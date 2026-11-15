from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    api_key: str = ""
    log_level: str = "INFO"
    chroma_path: str = "./.chroma"
    max_upload_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 100
    llm_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()