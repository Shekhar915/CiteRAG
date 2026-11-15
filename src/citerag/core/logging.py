import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once; an invalid level falls back to INFO."""
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        resolved = logging.INFO
    logging.basicConfig(level=resolved, format=_FORMAT, force=True)