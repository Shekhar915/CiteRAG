import logging

from citerag.core.logging import setup_logging


def test_setup_logging_sets_level():
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_invalid_level_falls_back_to_info():
    setup_logging("not-a-level")
    assert logging.getLogger().level == logging.INFO