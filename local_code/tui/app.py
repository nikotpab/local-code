from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, TextArea, Static, Label
from textual.worker import Worker

from local_code import tools
from local_code.agent import Agent, AgentConfig
from local_code.backends import OllamaError
from local_code.checkpoints import CheckpointStore
from local_code.cli import (
    AppContext,
    build_agent,
    handle_command,
    help_text,
    history_summary,
    make_spawn_factory,
    parse_args,
    print_mcp_table,
    print_sessions_table,
    print_tools_table,
    save_session,
    setup_app_context,
    _last_assistant_message,
)
from local_code.compaction import estimate_tokens
from local_code.custom_commands import load_custom_command
from local_code.mentions import expand_file_mentions
from local_code.session import Session
from local_code.session_store import SessionStore
from local_code.tui.bridge import ThreadSafeConfirmationBridge
from local_code.tui.widgets import (
    ActivityPane,
    ConfirmationModal,
    ConversationPane,
    HeaderBar,
)


TUI_CSS = """
Screen {
    layout: vertical;
    background: $surface;
}

HeaderBar {
    height: 1;
    background: $panel;
    padding: 0 1;
}

#main_split {
    layout: horizontal;
    height: 1fr;
}

#conv_container {
    width: 1fr;
    height: 1fr;
    border: solid $accent;
    margin: 0;
}

ConversationPane {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
}

#activity_container {
    width: 38;
    height: 1fr;
    border: solid $primary;
    padding: 0 1;
}

.hidden {
    display: none;
}

#input_container {
    height: 4;
    padding: 0 1;
    margin-bottom: 0;
}

#input_box {
    height: 1fr;
    border: solid $accent;
}

#modal_dialog {
    width: 70;
    height: auto;
    max-height: 80%;
    border: thick $warning;
    background: $surface;
    padding: 1 2;
    align: center middle;
}

#modal_title {
    margin-bottom: 1;
}

#modal_preview {
    height: auto;
    max-height: 15;
    overflow-y: scroll;
    border: solid $accent;
    margin-bottom: 1;
    padding: 0 1;
}

#modal_buttons {
    height: 3;
    align: center middle;
}

#modal_buttons Button {
    margin: 0 1;
}
"""


class ChatInputArea(TextArea):
    """Multi-line capable text area that submits on Enter and inserts newline on Shift+Enter / Alt+Enter."""

    def __init__(self, submit_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.submit_callback = submit_callback

    def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text and self.submit_callback:
                self.clear()
                self.submit_callback(text)
            return
        elif event.key in ("shift+enter", "alt+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        super()._on_key(event)


class LocalCodeApp(App):
    """Full-screen split-pane Textual TUI for local-code."""

    CSS = TUI_CSS
    TITLE = "local-code"

    BINDINGS = [
        Binding("ctrl+b", "toggle_activity", "Toggle Side Pane"),
        Binding("ctrl+c", "cancel_turn", "Cancel / Quit"),
        Binding("ctrl+q", "quit_app", "Quit"),
    ]

    def __init__(self, app_ctx: AppContext, **kwargs):
        super().__init__(**kwargs)
        self.app_ctx = app_ctx
        self.agent: Agent | None = None
        self.confirmation_bridge = ThreadSafeConfirmationBridge(self, self._show_confirmation_modal)
        self._is_running: bool = False
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header_bar")
        yield Horizontal(
            Container(ConversationPane(id="conv_pane"), id="conv_container"),
            ActivityPane(id="activity_pane"),
            id="main_split",
        )
        yield Container(ChatInputArea(submit_callback=self.on_user_submit, id="input_box"), id="input_container")
        yield Footer()

    def on_mount(self) -> None:
        self._init_agent()
        self._update_header()
        self._update_activity_mcp()

        conv_pane = self.query_one(ConversationPane)
        conv_pane.add_notify(
            f"local-code ready. Model: [bold]{self.app_ctx.model}[/bold] · Type /help for commands."
        )

    def _init_agent(self) -> None:
        """Construct the Agent instance using the shared build_agent helper."""
        def confirm_wrapper(name: str, preview: str) -> str:
            # Called synchronously from agent worker thread
            return self.confirmation_bridge.confirm(name, preview)

        def on_token_ui(token: str) -> None:
            self.call_from_thread(self._on_token, token)

        def on_stream_start_ui() -> None:
            self.call_from_thread(self._on_stream_start)

        def on_stream_end_ui() -> None:
            self.call_from_thread(self._on_stream_end)

        def notify_ui(msg: str) -> None:
            self.call_from_thread(self._notify, msg)

        def on_todos_ui(todos: list) -> None:
            self.call_from_thread(self._on_todos, todos)

        def on_tool_start_ui(name: str, arguments: dict) -> None:
            self.call_from_thread(self._on_tool_start, name, arguments)

        def on_tool_end_ui(name: str, result: str) -> None:
            self.call_from_thread(self._on_tool_end, name, result)

        self.agent = build_agent(
            client=self.app_ctx.client,
            session=self.app_ctx.session,
            cfg=self.app_ctx.cfg,
            model=self.app_ctx.model,
            yolo=self.app_ctx.args.yolo,
            detector=self.app_ctx.detector,
            console=None,
            streamer=None,
            checkpoint_store=self.app_ctx.checkpoints,
            plan_mode=self.app_ctx.plan_mode,
            spawn_factory=self.app_ctx.spawn_factory,
            confirm=confirm_wrapper,
            on_token=on_token_ui,
            notify=notify_ui,
            on_stream_start=on_stream_start_ui,
            on_stream_end=on_stream_end_ui,
            on_todos=on_todos_ui,
            on_tool_start=on_tool_start_ui,
            on_tool_end=on_tool_end_ui,
        )

    def _update_header(self) -> None:
        if not self.agent:
            return
        native = getattr(self.agent, "use_native", False)
        mode = "native" if native else "ReAct"
        backend_name = getattr(self.app_ctx.client, "name", "backend")
        window = self.agent.compactor.context_window if self.agent.compactor is not None else 0
        est_tokens = estimate_tokens(self.app_ctx.session.messages)
        window_str = f"{window // 1000}k" if window else "?"
        est_str = f"{est_tokens // 1000}k" if est_tokens >= 1000 else str(est_tokens)
        ctx_usage = f"{est_str}/{window_str}"

        header = self.query_one(HeaderBar)
        header.set_state(
            model=self.agent.config.model,
            backend=backend_name,
            mode=mode,
            context_usage=ctx_usage,
            plan_mode=self.agent.config.plan_mode,
        )

    def _update_activity_mcp(self) -> None:
        summaries = self.app_ctx.mcp_manager.server_summaries()
        activity = self.query_one(ActivityPane)
        activity.update_mcp(summaries)

    # -----------------------------------------------------------------------
    # UI Thread Callback Targets
    # -----------------------------------------------------------------------
    def _on_token(self, token: str) -> None:
        conv_pane = self.query_one(ConversationPane)
        conv_pane.append_assistant_token(token)

    def _on_stream_start(self) -> None:
        conv_pane = self.query_one(ConversationPane)
        conv_pane.start_assistant_message()

    def _on_stream_end(self) -> None:
        conv_pane = self.query_one(ConversationPane)
        conv_pane.end_assistant_message()
        self._update_header()

    def _notify(self, msg: str) -> None:
        conv_pane = self.query_one(ConversationPane)
        conv_pane.add_notify(msg)

    def _on_todos(self, todos: list) -> None:
        activity = self.query_one(ActivityPane)
        activity.update_todos(todos)

    def _on_tool_start(self, name: str, arguments: dict) -> None:
        conv_pane = self.query_one(ConversationPane)
        conv_pane.add_tool_start(name, arguments)
        # Surface the diff/preview of file edits in the activity pane.
        tool = tools.get_tool(name)
        if tool is not None and hasattr(tool, "preview"):
            try:
                self.query_one(ActivityPane).update_diff(tool.preview(arguments))
            except Exception:
                pass

    def _on_tool_end(self, name: str, result: str) -> None:
        self.query_one(ConversationPane).add_tool_end(name, result)

    def _show_confirmation_modal(
        self, name: str, preview: str, callback: Callable[[str], None]
    ) -> None:
        # Also update diff preview in ActivityPane if preview is non-empty
        activity = self.query_one(ActivityPane)
        activity.update_diff(preview)
        modal = ConfirmationModal(name, preview, callback)
        self.push_screen(modal)

    # -----------------------------------------------------------------------
    # User Input & Turn Dispatching
    # -----------------------------------------------------------------------
    def on_user_submit(self, text: str) -> None:
        if self._is_running:
            conv_pane = self.query_one(ConversationPane)
            conv_pane.add_notify("Error: agent turn already in progress. Please wait.")
            return

        action, arg = handle_command(text)
        if action != "chat":
            self.execute_command(action, arg, text)
            return

        self.run_turn_worker(text)

    def execute_command(self, action: str, arg: str | None, raw_line: str) -> None:
        conv_pane = self.query_one(ConversationPane)
        if action == "exit":
            self.action_quit_app()
            return
        if action == "clear":
            self.app_ctx.session.clear()
            self.app_ctx.session_id = self.app_ctx.store.new_id()
            conv_pane.clear_conversation()
            conv_pane.add_notify(f"History cleared (new session {self.app_ctx.session_id})")
            self._update_header()
            return
        if action == "help":
            conv_pane.add_notify(help_text())
            return
        if action == "tools":
            for tool_mod in tools.ALL_TOOLS:
                conf = " (requires confirm)" if tool_mod.REQUIRES_CONFIRMATION else ""
                conv_pane.add_notify(f"• {tool_mod.NAME}: {tool_mod.DESCRIPTION}{conf}")
            return
        if action == "mcp":
            summaries = self.app_ctx.mcp_manager.server_summaries()
            if not summaries:
                conv_pane.add_notify("No MCP servers connected.")
            else:
                for s in summaries:
                    conv_pane.add_notify(f"• MCP Server '{s['name']}': {s['tool_count']} tools")
            return
        if action == "history":
            summary = history_summary(self.app_ctx.session_id, self.app_ctx.model, self.app_ctx.session.history)
            conv_pane.add_notify(summary)
            return
        if action == "sessions":
            sessions = self.app_ctx.store.list_sessions(10)
            if not sessions:
                conv_pane.add_notify("No saved sessions.")
            else:
                conv_pane.add_notify("Saved Sessions:")
                for s in sessions:
                    conv_pane.add_notify(f"• {s['id']} [{s['updated_at']}] model={s['model']}: {s['first_message']}")
            return
        if action == "undo":
            msg = self.app_ctx.checkpoints.undo_last()
            conv_pane.add_notify(msg)
            return
        if action == "plan":
            if self.agent:
                self.agent.config.plan_mode = not self.agent.config.plan_mode
                status = "on" if self.agent.config.plan_mode else "off"
                conv_pane.add_notify(f"Plan mode {status}")
                self._update_header()
            return
        if action == "approve":
            last_plan = _last_assistant_message(self.app_ctx.session)
            if last_plan is None:
                conv_pane.add_notify("No plan to approve — run a plan-mode turn first.")
                return
            if self.agent:
                self.agent.config.plan_mode = False
                self._update_header()
            conv_pane.add_notify("Plan mode off — executing plan…")
            self.run_turn_worker(f"Execute this plan:\n\n{last_plan}")
            return
        if action == "model":
            if not arg:
                conv_pane.add_notify("Usage: /model <name>")
                return
            try:
                self.app_ctx.model = arg
                self._init_agent()
                self._update_header()
                conv_pane.add_notify(f"Switched model to {arg}")
            except OllamaError as e:
                conv_pane.add_notify(f"Error switching model: {e}")
            return
        if action == "custom":
            parts = raw_line.split(maxsplit=1)
            name = parts[0][1:]
            cmd_args = parts[1] if len(parts) > 1 else ""
            custom_text = load_custom_command(name, cmd_args)
            if custom_text is None:
                conv_pane.add_notify(f"Unknown command: {raw_line}")
                return
            self.run_turn_worker(custom_text)
            return

    # -----------------------------------------------------------------------
    # Worker Thread Turn Execution (@work(thread=True))
    # -----------------------------------------------------------------------
    def run_turn_worker(self, prompt: str) -> None:
        self._is_running = True
        conv_pane = self.query_one(ConversationPane)
        conv_pane.add_user_message(prompt)
        conv_pane.show_thinking(True)

        def worker_task() -> None:
            try:
                expanded, warnings = expand_file_mentions(prompt)
                for w in warnings:
                    self.call_from_thread(self._notify, f"Warning: {w}")

                if self.agent:
                    self.agent.run_turn(expanded)

                save_session(
                    self.app_ctx.store,
                    self.app_ctx.session_id,
                    self.app_ctx.model,
                    self.app_ctx.session,
                    Console(quiet=True),
                )
            except OllamaError as e:
                self.call_from_thread(self._notify, f"Error: {e}")
            except Exception as e:
                self.call_from_thread(self._notify, f"Error during turn: {type(e).__name__}: {e}")
            finally:
                self.call_from_thread(self._on_turn_finished)

        self._current_worker = self.run_worker(worker_task, thread=True)

    def _on_turn_finished(self) -> None:
        self._is_running = False
        self._current_worker = None
        conv_pane = self.query_one(ConversationPane)
        conv_pane.show_thinking(False)
        self._update_header()

    # -----------------------------------------------------------------------
    # Actions & Key Bindings
    # -----------------------------------------------------------------------
    def action_toggle_activity(self) -> None:
        activity = self.query_one(ActivityPane)
        activity.toggle_class("hidden")

    def action_cancel_turn(self) -> None:
        if self._is_running and self._current_worker:
            self._current_worker.cancel()
            conv_pane = self.query_one(ConversationPane)
            conv_pane.add_notify("Turn cancelled.")
            self._on_turn_finished()
        else:
            self.action_quit_app()

    def action_quit_app(self) -> None:
        if self.app_ctx.mcp_manager:
            self.app_ctx.mcp_manager.shutdown()
        self.exit(0)


def run_tui(app_ctx: AppContext) -> int:
    """Launch the Textual TUI application."""
    app = LocalCodeApp(app_ctx)
    return app.run() or 0
