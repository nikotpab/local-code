from __future__ import annotations

import logging

from local_code.logging_setup import ROOT_LOGGER_NAME, configure_logging, resolve_level


def test_resolve_level_precedence():
    assert resolve_level(debug=True, verbose=True) == logging.DEBUG
    assert resolve_level(debug=True, verbose=False) == logging.DEBUG
    assert resolve_level(debug=False, verbose=True) == logging.INFO
    assert resolve_level(debug=False, verbose=False) == logging.WARNING


def test_configure_logging_is_idempotent():
    first = configure_logging()
    count_after_first = len(first.handlers)
    second = configure_logging()
    assert second is first
    assert len(second.handlers) == count_after_first


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    logger = configure_logging(debug=True, log_file=log_file)
    logger.debug("hello-from-test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    assert "hello-from-test" in log_file.read_text()


def test_configure_logging_does_not_touch_root_logger():
    configure_logging()
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    assert logger.propagate is False
