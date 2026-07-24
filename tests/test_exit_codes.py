from __future__ import annotations

from local_code.backends import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)
from local_code.cli import (
    EXIT_CONNECTION,
    EXIT_ERROR,
    EXIT_MODEL_NOT_FOUND,
    exit_code_for,
)


def test_exit_code_connection():
    assert exit_code_for(OllamaConnectionError("down")) == EXIT_CONNECTION


def test_exit_code_model_not_found():
    assert exit_code_for(ModelNotFoundError("nope")) == EXIT_MODEL_NOT_FOUND


def test_exit_code_generic_backend():
    assert exit_code_for(OllamaError("boom")) == EXIT_ERROR


def test_report_backend_error_returns_code_and_prints_hint():
    import io

    from rich.console import Console

    from local_code.cli import report_backend_error

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    code = report_backend_error(console, OllamaConnectionError("refused"))
    assert code == EXIT_CONNECTION
    assert "hint" in buf.getvalue()
