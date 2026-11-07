from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int
    start: int
    end: int


def chunk_text(text: str, source: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split text into overlapping chunks, preferring to break at whitespace."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    text = text.strip()
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space

        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, source, len(chunks), start, end))

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
