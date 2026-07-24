from __future__ import annotations

import json

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

ACCENT = "cyan"
DIM = "dim"

TOOL_ICONS = {
    "read_file": "▸",
    "list_dir": "▸",
    "glob": "⌕",
    "grep": "⌕",
    "write_file": "✎",
    "edit_file": "✎",
    "multi_edit": "✎",
    "bash": "»",
    "web_fetch": "↗",
    "set_todos": "☰",
    "spawn_agent": "⇢",
}

MAX_ARG_CHARS = 80
MAX_RESULT_CHARS = 80


def tool_icon(name: str) -> str:
    if name in TOOL_ICONS:
        return TOOL_ICONS[name]
    if "__" in name:  # MCP tools are namespaced as server__tool
        return "◇"
    return "·"


def short_args(name: str, arguments: dict) -> str:
    if not isinstance(arguments, dict):
        return ""
    if name == "bash":
        value = arguments.get("command", "")
    elif "path" in arguments:
        value = arguments.get("path", "")
    elif "pattern" in arguments:
        value = arguments.get("pattern", "")
    elif "url" in arguments:
        value = arguments.get("url", "")
    else:
        value = json.dumps(arguments, ensure_ascii=False)
    value = str(value)
    if len(value) > MAX_ARG_CHARS:
        value = value[: MAX_ARG_CHARS - 1] + "…"
    return value


def shorten_path(path: str, home: str) -> str:
    home = home.rstrip("/")
    if home and (path == home or path.startswith(home + "/")):
        return "~" + path[len(home) :]
    return path


def _k(n: int) -> str:
    return f"{round(n / 1000)}k" if n >= 1000 else str(n)


def format_context(used: int, window: int) -> str:
    if window <= 0:
        return _k(used)
    return f"{_k(used)}/{_k(window)}"


def status_line(
    model: str, cwd: str, used_tokens: int, window: int, plan_mode: bool
) -> str:
    plan = "on" if plan_mode else "off"
    ctx = format_context(used_tokens, window)
    return f" {model} · {cwd} · {ctx} · plan {plan} "


def render_banner(
    console: Console,
    model: str,
    backend: str,
    native: bool,
    cwd: str,
    plan_mode: bool,
) -> None:
    mode = "native tools" if native else "ReAct fallback"
    body = Text()
    body.append(f"{model}", style=f"bold {ACCENT}")
    body.append(f" · {backend} · {mode}\n", style=DIM)
    body.append(f"{cwd} · plan: {'on' if plan_mode else 'off'}", style=DIM)
    console.print(
        Panel(body, title="local-code", subtitle="/help", border_style=ACCENT)
    )


def render_tool_start(console: Console, name: str, arguments: dict) -> None:
    icon = tool_icon(name)
    args = short_args(name, arguments)
    line = Text()
    line.append(f"{icon} ", style=ACCENT)
    line.append(name, style=f"bold {ACCENT}")
    if args:
        line.append(f"  {args}", style=DIM)
    console.print(line)


def render_tool_end(console: Console, name: str, result: str) -> None:
    summary = (result or "").splitlines()[0] if result else ""
    if len(summary) > MAX_RESULT_CHARS:
        summary = summary[: MAX_RESULT_CHARS - 1] + "…"
    lowered = summary.lower()
    if lowered.startswith("error") or "declined" in lowered:
        style = "red dim"
    elif lowered.startswith("blocked"):
        style = "yellow dim"
    else:
        style = DIM
    console.print(Text(f"  ↳ {summary}", style=style))


class ResponseView:
    """Owns the 'thinking' spinner and the markdown stream, handing off between
    them (rich allows only one live display per console at a time)."""

    def __init__(self, console: Console):
        self.console = console
        self.buffer = ""
        self._status = None
        self._live: Live | None = None

    def _stop_all(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def start(self) -> None:
        self._stop_all()
        self.buffer = ""
        self._status = self.console.status("pensando…", spinner="dots")
        self._status.start()

    def token(self, token: str) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self._live is None:
            self._live = Live(
                Markdown(""), console=self.console, refresh_per_second=10
            )
            self._live.start()
        self.buffer += token
        self._live.update(Markdown(self.buffer))

    def end(self) -> None:
        self._stop_all()
