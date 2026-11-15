from fastapi.testclient import TestClient

from citerag.api.main import app
from citerag.api.routes import health
from citerag.core.config import Settings

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_missing_key(monkeypatch):
    monkeypatch.setattr(health, "get_settings", lambda: Settings(_env_file=None, gemini_api_key=""))
    body = client.get("/ready").json()
    assert body == {"status": "not_ready", "llm_configured": False}


def test_ready_reports_configured(monkeypatch):
    monkeypatch.setattr(health, "get_settings", lambda: Settings(_env_file=None, gemini_api_key="k"))
    assert client.get("/ready").json()["status"] == "ready"