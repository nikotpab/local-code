from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

# The package-root logger. Everything under local_code.* propagates here, so
# configuring this one handler set covers the whole app without touching the
# global root logger (which third-party libs also use).
ROOT_LOGGER_NAME = "local_code"


def resolve_level(debug: bool, verbose: bool) -> int:
    """Map the two verbosity flags to a logging level.

    --debug wins over --verbose; with neither, stay quiet (WARNING).
    """
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.INFO
    return logging.WARNING


def configure_logging(
    debug: bool = False,
    verbose: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the local_code logger hierarchy.

    Console output goes to stderr so stdout stays clean for piping and
    --json. A --log-file always captures DEBUG regardless of console level,
    so a failure can be diagnosed after the fact even without --debug.

    Idempotent: repeated calls clear previously installed handlers first.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()  # defaults to stderr
    console_handler.setLevel(resolve_level(debug, verbose))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
