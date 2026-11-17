import pytest

from citerag.core.config import Settings
from citerag.retrieval import embeddings


def test_build_embedder_requires_api_key():
    with pytest.raises(ValueError):
        embeddings.build_embedder(Settings(_env_file=None, gemini_api_key=""))


def test_build_embedder_uses_configured_model(monkeypatch):
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return "embedder"

    monkeypatch.setattr(embeddings, "GoogleGenerativeAIEmbeddings", fake)
    settings = Settings(_env_file=None, gemini_api_key="k", embedding_model="emb-x")

    assert embeddings.build_embedder(settings) == "embedder"
    assert captured["model"] == "emb-x"