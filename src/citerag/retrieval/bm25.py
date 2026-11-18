import math
import re
from collections import Counter

from citerag.ingestion.chunker import Chunk
from citerag.retrieval.vectorstore import SearchHit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Keyword index using the Okapi BM25 ranking function."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_counts: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._total_length = 0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            counts = Counter(tokenize(chunk.text))
            self._chunks.append(chunk)
            self._term_counts.append(counts)
            self._doc_freq.update(counts.keys())
            self._total_length += sum(counts.values())

    def search(self, query: str, top_k: int = 5, source: str | None = None) -> list[SearchHit]:
        terms = tokenize(query)
        n_docs = len(self._chunks)
        if not terms or not n_docs:
            return []
        avg_length = self._total_length / n_docs
        hits: list[SearchHit] = []
        for chunk, counts in zip(self._chunks, self._term_counts, strict=True):
            if source is not None and chunk.source != source:
                continue
            length = sum(counts.values())
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = freq + self.k1 * (1 - self.b + self.b * length / avg_length)
                score += idf * freq * (self.k1 + 1) / norm
            if score > 0:
                hits.append(SearchHit(chunk, score))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
