from __future__ import annotations

from local_code.cli import handle_command, parse_args, style_preview_lines


def test_parse_args_defaults():
    args = parse_args([])
    assert args.model is None
    assert args.yolo is False
    assert args.system is None
    assert args.prompt == []


def test_parse_args_flags_and_prompt():
    args = parse_args(["--model", "llama3.1", "--yolo", "--system", "sos rust dev", "arregla", "el", "bug"])
    assert args.model == "llama3.1"
    assert args.yolo is True
    assert args.system == "sos rust dev"
    assert args.prompt == ["arregla", "el", "bug"]


def test_handle_command_chat():
    assert handle_command("hola mundo") == ("chat", "hola mundo")


def test_handle_command_clear_exit():
    assert handle_command("/clear") == ("clear", None)
    assert handle_command("/exit") == ("exit", None)


def test_handle_command_model():
    assert handle_command("/model llama3.1") == ("model", "llama3.1")
    assert handle_command("/model") == ("model", None)


def test_handle_command_unknown():
    assert handle_command("/wat") == ("unknown", "/wat")


def test_style_preview_lines_diff_colors():
    preview = "--- a/f\n+++ b/f\n-old\n+new\ncontext"
    styled = style_preview_lines(preview)
    assert styled == [
        ("--- a/f", ""),
        ("+++ b/f", ""),
        ("-old", "red"),
        ("+new", "green"),
        ("context", ""),
    ]
