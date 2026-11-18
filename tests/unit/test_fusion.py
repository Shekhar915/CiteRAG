from citerag.ingestion.chunker import Chunk
from citerag.retrieval.fusion import reciprocal_rank_fusion
from citerag.retrieval.vectorstore import SearchHit


def hit(index, score=1.0):
    return SearchHit(Chunk(text=str(index), source="d.txt", index=index, start=0, end=1), score)


def test_chunk_in_both_lists_beats_chunk_in_one():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]
    fused = reciprocal_rank_fusion([vector, keyword], top_k=4)
    assert fused[0].chunk.index == 3


def test_agreement_on_top_rank_wins():
    fused = reciprocal_rank_fusion([[hit(1), hit(2)], [hit(1), hit(3)]], top_k=3)
    assert fused[0].chunk.index == 1


def test_duplicates_are_merged_and_top_k_applies():
    fused = reciprocal_rank_fusion([[hit(1), hit(2)], [hit(1), hit(2)]], top_k=1)
    assert len(fused) == 1


def test_empty_rankings_give_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []
