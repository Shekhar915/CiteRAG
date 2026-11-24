from fastapi.testclient import TestClient

from citerag.api.deps import get_pipeline
from citerag.api.main import app
from citerag.schemas.qa import AskResponse, Citation


class FakePipeline:
    def __init__(self):
        self.last = None

    def ask_agentic(self, question, top_k=5):
        self.last = ("agentic", question, top_k)
        return self.ask(question, top_k)

    def ask(self, question, top_k=5):
        if self.last is None or self.last[0] != "agentic":
            self.last = (question, top_k)
        return AskResponse(
            answer="Refunds take 14 days [1].",
            citations=[Citation(source="policy.txt", chunk_index=0, snippet="Refund requests")],
        )


def client_with(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_ask_returns_answer_and_citations():
    pipeline = FakePipeline()
    response = client_with(pipeline).post("/v1/ask", json={"question": "How long do refunds take?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"][0]["source"] == "policy.txt"
    assert pipeline.last == ("How long do refunds take?", 5)


def test_ask_passes_top_k():
    pipeline = FakePipeline()
    client_with(pipeline).post("/v1/ask", json={"question": "valid question", "top_k": 3})
    assert pipeline.last[1] == 3


def test_agentic_flag_uses_the_agent():
    pipeline = FakePipeline()
    response = client_with(pipeline).post(
        "/v1/ask", json={"question": "valid question", "agentic": True}
    )
    assert response.status_code == 200
    assert pipeline.last == ("agentic", "valid question", 5)


def test_ask_validates_input():
    client = client_with(FakePipeline())
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/ask", json={}).status_code == 422


def test_ask_returns_503_when_not_configured(monkeypatch):
    from citerag.api import deps

    deps._default_pipeline.cache_clear()
    monkeypatch.setattr(
        deps, "build_pipeline", lambda: (_ for _ in ()).throw(ValueError("GEMINI_API_KEY is not set"))
    )
    app.dependency_overrides.clear()

    response = TestClient(app).post("/v1/ask", json={"question": "valid question"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
    deps._default_pipeline.cache_clear()
