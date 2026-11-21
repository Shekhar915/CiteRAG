from citerag.ingestion.chunker import Chunk
from citerag.retrieval.bm25 import BM25Index, tokenize


def make_chunk(text, index=0, source="d.txt"):
    return Chunk(text=text, source=source, index=index, start=0, end=len(text))


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Error-404: Not Found!") == ["error", "404", "not", "found"]


def test_exact_term_ranks_matching_chunk_first():
    index = BM25Index()
    index.add([make_chunk("the invoice total is due", 0), make_chunk("weather is sunny", 1)])
    hits = index.search("invoice", top_k=2)
    assert len(hits) == 1
    assert hits[0].chunk.index == 0


def test_rarer_term_scores_higher_than_common_term():
    index = BM25Index()
    index.add(
        [
            make_chunk("policy policy policy rollout", 0),
            make_chunk("policy review", 1),
            make_chunk("policy summary", 2),
            make_chunk("zebra crossing", 3),
        ]
    )
    rare = index.search("zebra", top_k=1)[0].score
    common = index.search("policy", top_k=1)[0].score
    assert rare > common


def test_empty_index_and_empty_query_return_nothing():
    assert BM25Index().search("anything") == []
    index = BM25Index()
    index.add([make_chunk("hello world")])
    assert index.search("   ") == []


def test_remove_source_rebuilds_index():
    index = BM25Index()
    index.add([make_chunk("alpha", 0, "one.txt"), make_chunk("beta", 0, "two.txt")])
    assert index.remove_source("one.txt") == 1
    assert index.search("alpha") == []
    assert index.search("beta")[0].chunk.source == "two.txt"
    assert index.remove_source("one.txt") == 0


def test_source_filter():
    index = BM25Index()
    index.add([make_chunk("alpha", 0, "one.txt"), make_chunk("alpha", 0, "two.txt")])
    hits = index.search("alpha", source="two.txt")
    assert [h.chunk.source for h in hits] == ["two.txt"]
