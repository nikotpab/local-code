from __future__ import annotations

import io

from rich.console import Console

from local_code import ui


def sio_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=80), buf


def test_tool_icon_by_category():
    assert ui.tool_icon("read_file") == "▸"
    assert ui.tool_icon("grep") == "⌕"
    assert ui.tool_icon("write_file") == "✎"
    assert ui.tool_icon("bash") == "»"


def test_tool_icon_mcp_and_default():
    assert ui.tool_icon("github__list_issues") == "◇"
    assert ui.tool_icon("totally_unknown") == "·"


def test_short_args_uses_path_and_command():
    assert ui.short_args("read_file", {"path": "src/main.py"}) == "src/main.py"
    assert ui.short_args("bash", {"command": "npm test"}) == "npm test"


def test_short_args_truncates():
    out = ui.short_args("read_file", {"path": "x" * 200})
    assert out.endswith("…")
    assert len(out) <= ui.MAX_ARG_CHARS


def test_shorten_path():
    assert ui.shorten_path("/Users/niko/proyecto", "/Users/niko") == "~/proyecto"
    assert ui.shorten_path("/Users/niko", "/Users/niko") == "~"
    assert ui.shorten_path("/etc/hosts", "/Users/niko") == "/etc/hosts"


def test_format_context():
    assert ui.format_context(12345, 32768).startswith("12k")
    assert "/" in ui.format_context(12345, 32768)
    assert ui.format_context(500, 0) == "500"


def test_status_line_has_facts():
    line = ui.status_line("qwen", "~/proyecto", 12000, 32000, plan_mode=True)
    assert "qwen" in line
    assert "~/proyecto" in line
    assert "plan on" in line
    assert "12k" in line


def test_render_banner_contains_facts():
    console, buf = sio_console()
    ui.render_banner(console, "qwen", "ollama", True, "~/proyecto", False)
    out = buf.getvalue()
    assert "local-code" in out
    assert "qwen" in out
    assert "ollama" in out
    assert "~/proyecto" in out


def test_render_tool_start_and_end():
    console, buf = sio_console()
    ui.render_tool_start(console, "read_file", {"path": "a.py"})
    ui.render_tool_end(console, "read_file", "42 chars")
    out = buf.getvalue()
    assert "read_file" in out
    assert "a.py" in out
    assert "42 chars" in out


def test_render_tool_end_marks_errors():
    console, buf = sio_console()
    ui.render_tool_end(console, "bash", "Error: boom")
    assert "Error: boom" in buf.getvalue()


def test_response_view_lifecycle():
    console, _ = sio_console()
    view = ui.ResponseView(console)
    view.token("ignored before start")  # no-op-ish: starts a live lazily
    view.end()
    view.start()
    assert view._status is not None
    view.token("# hola\n")
    assert view._status is None
    assert view._live is not None
    view.token("mundo")
    assert view.buffer == "# hola\nmundo"
    view.end()
    assert view._status is None
    assert view._live is None
    view.end()  # safe to call again
