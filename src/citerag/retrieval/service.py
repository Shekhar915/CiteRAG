from citerag.ingestion.chunker import Chunk
from citerag.retrieval.embeddings import Embedder
from citerag.retrieval.vectorstore import SearchHit, VectorStore

BATCH_SIZE = 64


def index_chunks(chunks: list[Chunk], embedder: Embedder, store: VectorStore) -> int:
    """Embed chunks in batches and add them to the store; returns the count."""
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embedder.embed_documents([c.text for c in batch])
        store.add(batch, vectors)
    return len(chunks)


def search(query: str, embedder: Embedder, store: VectorStore, top_k: int = 5) -> list[SearchHit]:
    """Embed the query and return the closest chunks."""
    if not query.strip():
        raise ValueError("query must not be empty")
    return store.search(embedder.embed_query(query), top_k=top_k)