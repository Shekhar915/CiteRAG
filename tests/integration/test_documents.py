from types import SimpleNamespace

from fastapi.testclient import TestClient

from citerag.agent.pipeline import DocumentInfo
from citerag.api.deps import get_pipeline
from citerag.api.main import app


class FakePipeline:
    def __init__(self):
        self.docs = {}

    def is_up_to_date(self, path):
        info = self.docs.get(path.name)
        return info is not None and info.sha256 == path.read_text(encoding="utf-8")

    def ingest(self, path):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".exe":
            raise ValueError("unsupported file type: .exe")
        self.docs[path.name] = DocumentInfo(path.name, text, 2)
        return 2

    def list_documents(self):
        return sorted(self.docs.values(), key=lambda d: d.name)

    def delete_document(self, name):
        return self.docs.pop(name, None) is not None


def client_with(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def upload(client, name, content=b"hello"):
    return client.post("/v1/documents", files={"file": (name, content, "text/plain")})


def test_upload_indexes_and_lists_document():
    client = client_with(FakePipeline())
    response = upload(client, "policy.txt")
    assert response.status_code == 200
    assert response.json() == {"name": "policy.txt", "chunks_added": 2, "unchanged": False}

    listed = client.get("/v1/documents").json()
    assert [d["name"] for d in listed] == ["policy.txt"]


def test_reuploading_same_content_is_reported_unchanged():
    client = client_with(FakePipeline())
    upload(client, "a.txt", b"same")
    body = upload(client, "a.txt", b"same").json()
    assert body["unchanged"] is True
    assert body["chunks_added"] == 0


def test_unsupported_type_returns_422():
    response = upload(client_with(FakePipeline()), "tool.exe")
    assert response.status_code == 422
    assert "unsupported" in response.json()["detail"]


def test_oversized_upload_returns_413(monkeypatch):
    from citerag.api.routes import documents

    monkeypatch.setattr(
        documents, "get_settings", lambda: SimpleNamespace(max_upload_mb=0)
    )
    response = upload(client_with(FakePipeline()), "big.txt", b"x")
    assert response.status_code == 413


def test_upload_filename_cannot_escape_temp_folder():
    pipeline = FakePipeline()
    response = upload(client_with(pipeline), "../../evil.txt")
    assert response.json()["name"] == "evil.txt"
    assert "evil.txt" in pipeline.docs


def test_delete_removes_document_and_404s_when_missing():
    client = client_with(FakePipeline())
    upload(client, "a.txt")
    assert client.delete("/v1/documents/a.txt").status_code == 204
    assert client.get("/v1/documents").json() == []
    assert client.delete("/v1/documents/a.txt").status_code == 404
