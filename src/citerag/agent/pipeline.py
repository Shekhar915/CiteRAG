from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citerag.agent.answering import answer_question
from citerag.core.config import Settings, get_settings
from citerag.ingestion.service import ingest_file
from citerag.retrieval.bm25 import BM25Index
from citerag.retrieval.embeddings import Embedder, build_embedder
from citerag.retrieval.hybrid import hybrid_search, index_hybrid
from citerag.retrieval.vectorstore import InMemoryVectorStore, VectorStore
from citerag.schemas.qa import AskResponse


@dataclass
class QAPipeline:
    """Ingests documents and answers questions with citations."""

    embedder: Embedder
    store: VectorStore
    bm25: BM25Index
    llm: Any
    settings: Settings

    def ingest(self, path: Path) -> int:
        chunks = ingest_file(path, self.settings)
        return index_hybrid(chunks, self.embedder, self.store, self.bm25)

    def ask(self, question: str, top_k: int = 5) -> AskResponse:
        hits = hybrid_search(question, self.embedder, self.store, self.bm25, top_k=top_k)
        return answer_question(question, hits, self.llm)


def build_pipeline(settings: Settings | None = None) -> QAPipeline:
    """Create the default pipeline; raises ValueError if no API key is configured."""
    from citerag.core.llm import build_llm

    settings = settings or get_settings()
    return QAPipeline(
        embedder=build_embedder(settings),
        store=InMemoryVectorStore(),
        bm25=BM25Index(),
        llm=build_llm(settings),
        settings=settings,
    )
