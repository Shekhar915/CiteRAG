from pathlib import Path

from pypdf import PdfReader

from citerag.ingestion.models import Page

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def load_text(path: Path) -> list[Page]:
    """Load a plain text or Markdown file as a single page."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Page(text=text, source=path.name)] if text else []

def load_pdf(path: Path) -> list[Page]:
    """Load a PDF, one Page per non-empty page (1-based page numbers)."""
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(text=text, source=path.name, page=number))
    return pages

def load_document(path: Path, max_bytes: int | None = None) -> list[Page]:
    """Load a supported file, enforcing the type and (optional) size limits."""
    suffix = path.suffix.lower()
    if suffix != ".pdf" and suffix not in TEXT_SUFFIXES:
        raise ValueError(f"unsupported file type: {suffix or 'none'}")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"file exceeds the {max_bytes} byte limit")
    return load_pdf(path) if suffix == ".pdf" else load_text(path)
