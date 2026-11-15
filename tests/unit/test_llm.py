import pytest

from citerag.core import llm
from citerag.core.config import Settings


def test_build_llm_requires_api_key():
    with pytest.raises(ValueError):
        llm.build_llm(Settings(_env_file=None, gemini_api_key=""))


def test_build_llm_passes_settings(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return "model"

    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", fake_model)
    settings = Settings(_env_file=None, gemini_api_key="k", gemini_model="m", llm_timeout_seconds=5)

    assert llm.build_llm(settings) == "model"
    assert captured["model"] == "m"
    assert captured["timeout"] == 5
    assert captured["temperature"] == 0