from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    api_key: str = ""
    log_level: str = "INFO"
    chroma_path: str = "./.chroma"
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
