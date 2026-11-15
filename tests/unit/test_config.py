from citerag.core.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.max_upload_mb == 10


def test_chunk_settings_are_consistent():
    settings = Settings(_env_file=None)
    assert 0 <= settings.chunk_overlap < settings.chunk_size


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-test"
    assert settings.chunk_size == 500