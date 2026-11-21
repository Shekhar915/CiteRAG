import pytest

from citerag.ingestion.chunker import Chunk
from citerag.retrieval.vectorstore import InMemoryVectorStore, cosine


def make_chunk(text, index=0):
    return Chunk(text=text, source="d.txt", index=index, start=0, end=len(text))


def test_cosine_of_identical_and_orthogonal_vectors():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_search_returns_best_match_first():
    store = InMemoryVectorStore()
    store.add([make_chunk("cats", 0), make_chunk("dogs", 1)], [[1, 0], [0, 1]])
    hits = store.search([0.9, 0.1], top_k=2)
    assert hits[0].chunk.text == "cats"
    assert hits[0].score > hits[1].score


def test_top_k_limits_results():
    store = InMemoryVectorStore()
    store.add([make_chunk(str(i), i) for i in range(5)], [[1, i] for i in range(5)])
    assert len(store.search([1, 0], top_k=2)) == 2


def test_add_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        InMemoryVectorStore().add([make_chunk("a")], [])

def test_remove_source_deletes_only_that_document():
    store = InMemoryVectorStore()
    a = Chunk(text="a", source="one.txt", index=0, start=0, end=1)
    b = Chunk(text="b", source="two.txt", index=0, start=0, end=1)
    store.add([a, b], [[1, 0], [1, 0]])
    assert store.remove_source("one.txt") == 1
    assert store.remove_source("missing.txt") == 0
    assert [h.chunk.source for h in store.search([1, 0])] == ["two.txt"]


def test_search_can_filter_by_source():
    store = InMemoryVectorStore()
    a = Chunk(text="a", source="one.txt", index=0, start=0, end=1)
    b = Chunk(text="b", source="two.txt", index=0, start=0, end=1)
    store.add([a, b], [[1, 0], [1, 0]])
    hits = store.search([1, 0], top_k=5, source="two.txt")
    assert [h.chunk.source for h in hits] == ["two.txt"]