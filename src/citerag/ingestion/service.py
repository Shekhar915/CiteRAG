from pathlib import Path

from citerag.core.config import Settings, get_settings
from citerag.ingestion.chunker import Chunk, chunk_pages
from citerag.ingestion.loaders import load_document


def ingest_file(path: Path, settings: Settings | None = None) -> list[Chunk]:
    """Load a file and split it into chunks using the configured sizes."""
    settings = settings or get_settings()
    pages = load_document(path, max_bytes=settings.max_upload_mb * 1024 * 1024)
    return chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)