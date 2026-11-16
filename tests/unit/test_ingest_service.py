from citerag.core.config import Settings
from citerag.ingestion.service import ingest_file


def test_ingest_file_chunks_with_configured_size(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text(" ".join(f"word{i}" for i in range(200)), encoding="utf-8")
    settings = Settings(_env_file=None, chunk_size=150, chunk_overlap=20)

    chunks = ingest_file(f, settings)

    assert len(chunks) > 1
    assert all(c.source == "doc.txt" for c in chunks)
    assert all(len(c.text) <= 150 for c in chunks)