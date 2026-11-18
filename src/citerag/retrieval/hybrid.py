from citerag.ingestion.chunker import Chunk
from citerag.retrieval.bm25 import BM25Index
from citerag.retrieval.embeddings import Embedder
from citerag.retrieval.fusion import reciprocal_rank_fusion
from citerag.retrieval.vectorstore import SearchHit, VectorStore


def hybrid_search(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    bm25: BM25Index,
    top_k: int = 5,
    candidates: int = 20,
    source: str | None = None,
) -> list[SearchHit]:
    """Combine vector and BM25 results with Reciprocal Rank Fusion."""
    if not query.strip():
        raise ValueError("query must not be empty")
    vector_hits = store.search(embedder.embed_query(query), top_k=candidates, source=source)
    keyword_hits = bm25.search(query, top_k=candidates, source=source)
    return reciprocal_rank_fusion([vector_hits, keyword_hits], top_k=top_k)


def index_hybrid(
    chunks: list[Chunk], embedder: Embedder, store: VectorStore, bm25: BM25Index
) -> int:
    """Add chunks to both the vector store and the keyword index."""
    if chunks:
        store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
        bm25.add(chunks)
    return len(chunks)
