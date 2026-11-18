import pytest

from citerag.ingestion.chunker import Chunk
from citerag.retrieval.bm25 import BM25Index
from citerag.retrieval.hybrid import hybrid_search, index_hybrid
from citerag.retrieval.vectorstore import InMemoryVectorStore


class FakeEmbedder:
    """Deterministic vectors so no network is needed.

    The semantic vector is by topic word, so a query can match a chunk by
    meaning even when it shares no keyword with it.
    """

    def _vec(self, text):
        if "car" in text or "automobile" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def chunk(text, index):
    return Chunk(text=text, source="d.txt", index=index, start=0, end=len(text))


def build():
    store, bm25 = InMemoryVectorStore(), BM25Index()
    chunks = [
        chunk("the automobile needs repair", 0),
        chunk("error code E4012 appears on startup", 1),
        chunk("lunch menu for friday", 2),
    ]
    index_hybrid(chunks, FakeEmbedder(), store, bm25)
    return store, bm25


def test_hybrid_finds_exact_identifier_via_keywords():
    store, bm25 = build()
    hits = hybrid_search("E4012", FakeEmbedder(), store, bm25, top_k=1)
    assert hits[0].chunk.index == 1


def test_hybrid_finds_semantic_match_via_vectors():
    store, bm25 = build()
    hits = hybrid_search("my car broke", FakeEmbedder(), store, bm25, top_k=1)
    assert hits[0].chunk.index == 0


def test_empty_query_is_rejected():
    store, bm25 = build()
    with pytest.raises(ValueError):
        hybrid_search("  ", FakeEmbedder(), store, bm25)


def test_index_hybrid_handles_empty_input():
    assert index_hybrid([], FakeEmbedder(), InMemoryVectorStore(), BM25Index()) == 0
