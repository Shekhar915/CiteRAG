from citerag.retrieval.vectorstore import SearchHit


def _key(hit: SearchHit) -> tuple[str, int]:
    return (hit.chunk.source, hit.chunk.index)


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]], k: int = 60, top_k: int = 5
) -> list[SearchHit]:
    """Merge several ranked lists with Reciprocal Rank Fusion.

    Each list contributes 1 / (k + rank) per chunk, so chunks that rank well in
    more than one list rise to the top without needing comparable raw scores.
    """
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = _key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, hit)
    fused = [SearchHit(chunks[key].chunk, score) for key, score in scores.items()]
    return sorted(fused, key=lambda h: h.score, reverse=True)[:top_k]
