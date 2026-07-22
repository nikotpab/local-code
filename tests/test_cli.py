from __future__ import annotations

from local_code.cli import MarkdownStreamer, handle_command, history_summary, parse_args, style_preview_lines


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


def test_parse_args_resume_absent():
    assert parse_args([]).resume is None


def test_parse_args_resume_bare():
    assert parse_args(["--resume"]).resume == "latest"


def test_parse_args_resume_with_id():
    assert parse_args(["--resume", "20260722-120000"]).resume == "20260722-120000"


def test_handle_command_new_commands():
    assert handle_command("/help") == ("help", None)
    assert handle_command("/tools") == ("tools", None)
    assert handle_command("/history") == ("history", None)
    assert handle_command("/sessions") == ("sessions", None)


def test_history_summary_empty():
    assert history_summary("s1", "qwen", []) == "session s1 · model qwen · empty"


def test_history_summary_counts_roles():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = history_summary("s1", "qwen", history)
    assert out == "session s1 · model qwen · 3 messages (assistant: 1, user: 2)"


def test_markdown_streamer_lifecycle():
    import io

    from rich.console import Console

    streamer = MarkdownStreamer(Console(file=io.StringIO(), force_terminal=False))
    streamer.token("ignored before start")
    streamer.start()
    streamer.token("# hola\n")
    streamer.token("mundo")
    assert streamer.buffer == "# hola\nmundo"
    streamer.end()
    streamer.start()
    assert streamer.buffer == ""
    streamer.end()


def test_smoke_help_exits_zero():
    import subprocess

    result = subprocess.run(
        [".venv/bin/local-code", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--resume" in result.stdout


def test_smoke_resume_unknown_session_exits_one(tmp_path):
    import subprocess

    result = subprocess.run(
        [".venv/bin/local-code", "--resume", "nonexistent-id"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 1
