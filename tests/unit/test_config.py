from citerag.core.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.max_upload_mb == 10
