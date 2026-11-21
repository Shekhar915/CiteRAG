import hashlib
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    sha256: str
    chunks: int


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class QAPipeline:
    """Ingests documents and answers questions with citations."""

    embedder: Embedder
    store: VectorStore
    bm25: BM25Index
    llm: Any
    settings: Settings
    documents: dict[str, DocumentInfo] = field(default_factory=dict)

    def is_up_to_date(self, path: Path) -> bool:
        """True if a file with this name and identical content is already indexed."""
        known = self.documents.get(path.name)
        return known is not None and known.sha256 == file_hash(path)

    def ingest(self, path: Path) -> int:
        """Index a file; returns the number of chunks added (0 if unchanged).

        Re-ingesting a changed file replaces its old chunks, so outdated text
        can never be retrieved or cited.
        """
        if self.is_up_to_date(path):
            return 0
        chunks = ingest_file(path, self.settings)
        self.delete_document(path.name)
        added = index_hybrid(chunks, self.embedder, self.store, self.bm25)
        self.documents[path.name] = DocumentInfo(path.name, file_hash(path), added)
        return added

    def delete_document(self, name: str) -> bool:
        """Remove a document everywhere; returns True if it was indexed."""
        self.store.remove_source(name)
        self.bm25.remove_source(name)
        return self.documents.pop(name, None) is not None

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(self.documents.values(), key=lambda d: d.name)

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
