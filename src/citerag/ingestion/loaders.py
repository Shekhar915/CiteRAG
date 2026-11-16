from pathlib import Path

from citerag.ingestion.models import Page

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def load_text(path: Path) -> list[Page]:
    """Load a plain text or Markdown file as a single page."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Page(text=text, source=path.name)] if text else []