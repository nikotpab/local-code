from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from rich.text import Text
from rich.panel import Panel

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Markdown, Static, TextArea

from local_code.cli import CONFIRM_CHOICES, TODO_ICONS, style_preview_lines
from local_code.ui import shorten_path


class HeaderBar(Static):
    """Top bar displaying environment, model, mode, and token context info."""

    def __init__(
        self,
        model: str = "",
        backend: str = "",
        mode: str = "",
        context_usage: str = "",
        plan_mode: bool = False,
        cwd: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend
        self.mode = mode
        self.context_usage = context_usage
        self.plan_mode = plan_mode
        self.cwd = cwd or shorten_path(str(Path.cwd()), str(Path.home()))
        self.update_content()

    def update_content(self) -> None:
        plan_str = " · [bold yellow][plan][/bold yellow]" if self.plan_mode else ""
        content = (
            f"[bold cyan]local-code[/bold cyan] · "
            f"model: [bold white]{self.model}[/bold white] · "
            f"backend: {self.backend} ({self.mode}) · "
            f"ctx: {self.context_usage}"
            f"{plan_str} · "
            f"[dim]{self.cwd}[/dim]"
        )
        self.update(content)

    def set_state(
        self,
        model: str | None = None,
        backend: str | None = None,
        mode: str | None = None,
        context_usage: str | None = None,
        plan_mode: bool | None = None,
        cwd: str | None = None,
    ) -> None:
        if model is not None:
            self.model = model
        if backend is not None:
            self.backend = backend
        if mode is not None:
            self.mode = mode
        if context_usage is not None:
            self.context_usage = context_usage
        if plan_mode is not None:
            self.plan_mode = plan_mode
        if cwd is not None:
            self.cwd = cwd
        self.update_content()


class ConversationPane(VerticalScroll):
    """Scrollable conversation pane displaying chat messages, markdown, and tool activity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_assistant_md: Markdown | None = None
        self._current_assistant_content: str = ""
        self._thinking_widget: Static | None = None

    def add_user_message(self, text: str) -> None:
        self.show_thinking(False)
        widget = Static(f"[bold cyan]USER>[/bold cyan] {text}")
        self.mount(widget)
        self.scroll_end(animate=False)

    def show_thinking(self, show: bool = True) -> None:
        if show:
            if self._thinking_widget is None:
                self._thinking_widget = Static("[dim italic]Thinking…[/dim italic]")
                self.mount(self._thinking_widget)
                self.scroll_end(animate=False)
        else:
            if self._thinking_widget is not None:
                self._thinking_widget.remove()
                self._thinking_widget = None

    def start_assistant_message(self) -> None:
        self.show_thinking(False)
        if self._current_assistant_md is None:
            self._current_assistant_content = ""
            self._current_assistant_md = Markdown("")
            self.mount(self._current_assistant_md)
            self.scroll_end(animate=False)

    def append_assistant_token(self, token: str) -> None:
        self.show_thinking(False)
        if self._current_assistant_md is None:
            self.start_assistant_message()
        self._current_assistant_content += token
        if self._current_assistant_md is not None:
            self._current_assistant_md.update(self._current_assistant_content)
            self.scroll_end(animate=False)

    def end_assistant_message(self) -> None:
        self.show_thinking(False)
        self._current_assistant_md = None

    def add_tool_start(self, name: str, arguments: dict) -> None:
        self.show_thinking(False)
        arg_str = json.dumps(arguments, ensure_ascii=False)
        if len(arg_str) > 100:
            arg_str = arg_str[:100] + "…"
        widget = Static(f"[dim yellow]⚡ {name}({arg_str})[/dim yellow]")
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_tool_end(self, name: str, result: str) -> None:
        self.show_thinking(False)
        if result.startswith("Error:"):
            styled = f"[bold red]└─ {result}[/bold red]"
        else:
            first_line = result.splitlines()[0] if result else "ok"
            if len(first_line) > 100:
                first_line = first_line[:100] + "…"
            styled = f"[dim]  └─ {first_line}[/dim]"
        self.mount(Static(styled))
        self.scroll_end(animate=False)

    def add_notify(self, msg: str) -> None:
        self.show_thinking(False)
        if msg.startswith("Error:"):
            widget = Static(f"[red]{msg}[/red]")
        else:
            widget = Static(f"[dim]{msg}[/dim]")
        self.mount(widget)
        self.scroll_end(animate=False)

    def clear_conversation(self) -> None:
        self.show_thinking(False)
        self._current_assistant_md = None
        self._current_assistant_content = ""
        for child in list(self.children):
            child.remove()


class ActivityPane(Container):
    """Side panel displaying set_todos checklist, recent code diff, and MCP status."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._todos_widget = Static("[dim]📋 Todos: (empty)[/dim]")
        self._diff_widget = Static("[dim]📝 Diff: (none)[/dim]")
        self._mcp_widget = Static("[dim]🔌 MCP: 0 servers[/dim]")

    def compose(self) -> ComposeResult:
        yield Container(
            Static("[bold yellow]Activity Panel[/bold yellow] [dim](Ctrl+B toggle)[/dim]\n"),
            self._todos_widget,
            Static("\n"),
            self._diff_widget,
            Static("\n"),
            self._mcp_widget,
            id="activity_container",
        )

    def update_todos(self, todos: list) -> None:
        if not todos:
            self._todos_widget.update("[dim]📋 Todos: (empty)[/dim]")
            return
        lines = ["[bold yellow]📋 Todos:[/bold yellow]"]
        for t in todos:
            status = t.get("status", "pending")
            icon = TODO_ICONS.get(status, "☐")
            text = t.get("text", "")
            if status == "done":
                lines.append(f"  [green]{icon} {text}[/green]")
            elif status == "in_progress":
                lines.append(f"  [cyan]{icon} {text}[/cyan]")
            else:
                lines.append(f"  [dim]{icon} {text}[/dim]")
        self._todos_widget.update("\n".join(lines))

    def update_diff(self, preview_text: str) -> None:
        if not preview_text.strip():
            self._diff_widget.update("[dim]📝 Diff: (none)[/dim]")
            return
        lines = ["[bold yellow]📝 Latest Diff / Preview:[/bold yellow]"]
        styled_lines = style_preview_lines(preview_text)
        for line, style in styled_lines[:40]:  # limit lines for pane
            if style == "green":
                lines.append(f"[green]{line}[/green]")
            elif style == "red":
                lines.append(f"[red]{line}[/red]")
            else:
                lines.append(f"[dim]{line}[/dim]")
        if len(styled_lines) > 40:
            lines.append("[dim]… [truncated][/dim]")
        self._diff_widget.update("\n".join(lines))

    def update_mcp(self, summaries: list[dict]) -> None:
        if not summaries:
            self._mcp_widget.update("[dim]🔌 MCP: 0 servers[/dim]")
            return
        parts = [f"{s['name']}: {s['tool_count']} tools" for s in summaries]
        self._mcp_widget.update(f"[bold yellow]🔌 MCP:[/bold yellow] " + ", ".join(parts))


class ConfirmationModal(ModalScreen[str]):
    """Modal screen for tool execution confirmation."""

    BINDINGS = [
        Binding("y", "choose('yes')", "Yes"),
        Binding("n", "choose('no')", "No"),
        Binding("a", "choose('always')", "Always"),
        Binding("escape", "choose('no')", "Cancel"),
    ]

    def __init__(
        self,
        name: str,
        preview: str,
        callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_name = name
        self.preview = preview
        self.callback = callback

    def compose(self) -> ComposeResult:
        styled_text = Text()
        for line, style in style_preview_lines(self.preview):
            styled_text.append(line + "\n", style=style)

        yield Container(
            Label(f"[bold yellow]Run tool '{self.tool_name}'?[/bold yellow]", id="modal_title"),
            Static(styled_text, id="modal_preview"),
            Horizontal(
                Button("Yes (y)", variant="success", id="btn_yes"),
                Button("No (n)", variant="error", id="btn_no"),
                Button("Always (a)", variant="primary", id="btn_always"),
                id="modal_buttons",
            ),
            id="modal_dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn_yes":
            self.action_choose("yes")
        elif button_id == "btn_always":
            self.action_choose("always")
        else:
            self.action_choose("no")

    def action_choose(self, choice: str) -> None:
        if self.callback:
            self.callback(choice)
        self.dismiss(choice)
