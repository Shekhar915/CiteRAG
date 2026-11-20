from types import SimpleNamespace

from citerag.agent.pipeline import QAPipeline
from citerag.core.config import Settings
from citerag.retrieval.bm25 import BM25Index
from citerag.retrieval.vectorstore import InMemoryVectorStore


class FakeEmbedder:
    def _vec(self, text):
        return [1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="Refunds take 14 days [1].")


def make_pipeline():
    return QAPipeline(
        embedder=FakeEmbedder(),
        store=InMemoryVectorStore(),
        bm25=BM25Index(),
        llm=FakeLLM(),
        settings=Settings(_env_file=None),
    )


def test_ingest_then_ask_returns_cited_answer(tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Refund requests are processed within 14 days.", encoding="utf-8")
    pipeline = make_pipeline()

    assert pipeline.ingest(doc) == 1
    response = pipeline.ask("How long do refunds take?")

    assert response.grounded is True
    assert response.citations[0].source == "policy.txt"


def test_ask_with_nothing_indexed_is_ungrounded():
    response = make_pipeline().ask("anything at all")
    assert response.grounded is False
    assert response.citations == []
