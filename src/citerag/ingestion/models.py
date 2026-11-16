from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """Text extracted from one page (or the whole file when pages don't apply)."""

    text: str
    source: str
    page: int | None = None