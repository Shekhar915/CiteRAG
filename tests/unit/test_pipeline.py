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


def test_reingesting_unchanged_file_is_skipped(tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Refund requests are processed within 14 days.", encoding="utf-8")
    pipeline = make_pipeline()

    assert pipeline.ingest(doc) == 1
    assert pipeline.is_up_to_date(doc) is True
    assert pipeline.ingest(doc) == 0
    assert len(pipeline.list_documents()) == 1


def test_changed_file_replaces_old_chunks(tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Refund requests take 14 days.", encoding="utf-8")
    pipeline = make_pipeline()
    pipeline.ingest(doc)

    doc.write_text("Refund requests take 30 days now.", encoding="utf-8")
    assert pipeline.is_up_to_date(doc) is False
    assert pipeline.ingest(doc) == 1

    hits = pipeline.bm25.search("refund", top_k=10)
    assert len(hits) == 1
    assert "30 days" in hits[0].chunk.text
    assert pipeline.list_documents()[0].chunks == 1


def test_delete_document_removes_it_from_search(tmp_path):
    doc = tmp_path / "policy.txt"
    doc.write_text("Refund requests take 14 days.", encoding="utf-8")
    pipeline = make_pipeline()
    pipeline.ingest(doc)

    assert pipeline.delete_document("policy.txt") is True
    assert pipeline.delete_document("policy.txt") is False
    assert pipeline.list_documents() == []
    assert pipeline.ask("How long do refunds take?").grounded is False


def test_ask_with_nothing_indexed_is_ungrounded():
    response = make_pipeline().ask("anything at all")
    assert response.grounded is False
    assert response.citations == []
