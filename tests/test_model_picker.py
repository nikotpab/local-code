from __future__ import annotations

import io

from rich.console import Console

from local_code.cli import (
    _is_current_model,
    handle_command,
    render_models_table,
    resolve_model_choice,
)


def test_handle_command_models():
    assert handle_command("/models") == ("models", None)


def test_handle_command_model_still_parses_arg():
    assert handle_command("/model llama3.2") == ("model", "llama3.2")
    assert handle_command("/model") == ("model", None)


def test_resolve_by_index():
    names = ["a:1", "b:2", "c:3"]
    assert resolve_model_choice("2", names) == "b:2"
    assert resolve_model_choice(" 1 ", names) == "a:1"


def test_resolve_index_out_of_range():
    assert resolve_model_choice("9", ["a:1"]) is None
    assert resolve_model_choice("0", ["a:1"]) is None


def test_resolve_blank_is_none():
    assert resolve_model_choice("", ["a:1"]) is None
    assert resolve_model_choice("   ", ["a:1"]) is None


def test_resolve_name_passthrough():
    # A non-numeric choice is trusted verbatim even if not in the list.
    assert resolve_model_choice("qwen2.5-coder:7b", ["a:1"]) == "qwen2.5-coder:7b"


def test_is_current_model_matches_with_or_without_tag():
    assert _is_current_model("qwen2.5-coder:7b", "qwen2.5-coder") is True
    assert _is_current_model("qwen2.5-coder:7b", "qwen2.5-coder:7b") is True
    assert _is_current_model("llama3.2:latest", "qwen2.5-coder") is False


def test_render_models_table_marks_current():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    render_models_table(console, ["qwen2.5-coder:7b", "llama3.2:latest"], "llama3.2", "/models")
    out = buf.getvalue()
    assert "qwen2.5-coder:7b" in out
    assert "llama3.2:latest" in out
    assert "current" in out
