import math
from dataclasses import dataclass
from typing import Protocol

from citerag.ingestion.chunker import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]: ...

    def remove_source(self, source: str) -> int: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore:
    """Simple cosine-similarity store used for tests and local development."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._items.extend(zip(chunks, vectors, strict=True))

    def remove_source(self, source: str) -> int:
        """Delete every chunk of a document; returns how many were removed."""
        before = len(self._items)
        self._items = [(c, v) for c, v in self._items if c.source != source]
        return before - len(self._items)

    def search(
        self, vector: list[float], top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]:
        scored = [
            SearchHit(c, cosine(vector, v))
            for c, v in self._items
            if source is None or c.source == source
        ]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]