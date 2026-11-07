from fastapi import FastAPI

from citerag.api.routes import health

app = FastAPI(title="CiteRAG", version="0.1.0")
app.include_router(health.router)
