from fastapi import FastAPI

from citerag.api.routes import ask, documents, health
from citerag.core.config import get_settings
from citerag.core.logging import setup_logging

setup_logging(get_settings().log_level)

app = FastAPI(title="CiteRAG", version="0.1.0")
app.include_router(health.router)
app.include_router(ask.router)
app.include_router(documents.router)
