import pytest

from citerag.ingestion.chunker import Chunk
from citerag.retrieval import service
from citerag.retrieval.vectorstore import InMemoryVectorStore


class FakeEmbedder:
    """Maps a few known words to fixed vectors, so no network is needed."""

    VECTORS = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}

    def embed_documents(self, texts):
        return [self.VECTORS[t] for t in texts]

    def embed_query(self, text):
        return self.VECTORS[text]


def chunk(text, index):
    return Chunk(text=text, source="d.txt", index=index, start=0, end=len(text))


def test_index_and_search_roundtrip():
    store = InMemoryVectorStore()
    count = service.index_chunks([chunk("cat", 0), chunk("dog", 1)], FakeEmbedder(), store)
    hits = service.search("dog", FakeEmbedder(), store, top_k=1)
    assert count == 2
    assert hits[0].chunk.text == "dog"


def test_indexing_is_batched(monkeypatch):
    monkeypatch.setattr(service, "BATCH_SIZE", 2)
    calls = []

    class CountingEmbedder(FakeEmbedder):
        def embed_documents(self, texts):
            calls.append(len(texts))
            return super().embed_documents(texts)

    chunks = [chunk("cat", i) for i in range(5)]
    service.index_chunks(chunks, CountingEmbedder(), InMemoryVectorStore())
    assert calls == [2, 2, 1]


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        service.search("  ", FakeEmbedder(), InMemoryVectorStore())

def test_search_passes_source_filter():
    store = InMemoryVectorStore()
    a = Chunk(text="cat", source="one.txt", index=0, start=0, end=3)
    b = Chunk(text="cat", source="two.txt", index=0, start=0, end=3)
    store.add([a, b], [[1.0, 0.0], [1.0, 0.0]])
    hits = service.search("cat", FakeEmbedder(), store, source="two.txt")
    assert [h.chunk.source for h in hits] == ["two.txt"]