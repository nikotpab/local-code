from __future__ import annotations

import argparse
import os
from collections import Counter

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from local_code import tools
from local_code.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentConfig
from local_code.capabilities import CapabilityDetector
from local_code.client import OllamaClient, OllamaError
from local_code.compaction import Compactor, detect_context_window
from local_code.config import Config, load_config
from local_code.custom_commands import list_custom_commands, load_custom_command
from local_code.mentions import expand_file_mentions
from local_code.permissions import PermissionStore
from local_code.project_context import load_project_context
from local_code.session import Session
from local_code.session_store import SessionNotFoundError, SessionStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="local-code",
        description="Agentic coding CLI for any local model served by Ollama",
    )
    parser.add_argument("--model", help="Model name (default: from ~/.local-code/config.yaml)")
    parser.add_argument("--yolo", action="store_true", help="Skip all tool confirmations")
    parser.add_argument("--system", help="Override the system prompt")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="ID",
        help="Resume the latest session, or a specific session id",
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt; omit for interactive REPL")
    return parser.parse_args(argv)


def handle_command(line: str) -> tuple[str, str | None]:
    stripped = line.strip()
    if not stripped.startswith("/"):
        return ("chat", stripped)
    parts = stripped.split(maxsplit=1)
    cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
    if cmd == "/clear":
        return ("clear", None)
    if cmd == "/exit":
        return ("exit", None)
    if cmd == "/model":
        return ("model", arg)
    if cmd == "/help":
        return ("help", None)
    if cmd == "/tools":
        return ("tools", None)
    if cmd == "/history":
        return ("history", None)
    if cmd == "/sessions":
        return ("sessions", None)
    return ("custom", stripped)


def style_preview_lines(preview: str) -> list[tuple[str, str]]:
    styled = []
    for line in preview.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            styled.append((line, "green"))
        elif line.startswith("-") and not line.startswith("---"):
            styled.append((line, "red"))
        else:
            styled.append((line, ""))
    return styled


CONFIRM_CHOICES = {"y": "yes", "n": "no", "a": "always"}
TODO_ICONS = {"pending": "☐", "in_progress": "◐", "done": "☑"}


def make_todo_renderer(console: Console):
    def render(todos: list) -> None:
        lines = "\n".join(
            f"{TODO_ICONS.get(t.get('status'), '?')} {t.get('text', '')}" for t in todos
        )
        console.print(
            Panel(Text(lines or "(vacío)"), title="todos", border_style="cyan")
        )

    return render


class MarkdownStreamer:
    def __init__(self, console: Console):
        self.console = console
        self.buffer = ""
        self._live: Live | None = None

    def start(self) -> None:
        self.buffer = ""
        self._live = Live(
            Markdown(""), console=self.console, refresh_per_second=10
        )
        self._live.start()

    def token(self, token: str) -> None:
        if self._live is None:
            return
        self.buffer += token
        self._live.update(Markdown(self.buffer))

    def end(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


def history_summary(session_id: str, model: str, history: list[dict]) -> str:
    if not history:
        return f"session {session_id} · model {model} · empty"
    counts = Counter(m.get("role", "?") for m in history)
    parts = ", ".join(f"{role}: {n}" for role, n in sorted(counts.items()))
    return f"session {session_id} · model {model} · {len(history)} messages ({parts})"


def make_confirmer(console: Console):
    def confirm(name: str, preview: str) -> str:
        text = Text()
        for line, style in style_preview_lines(preview):
            text.append(line + "\n", style=style)
        console.print(Panel(text, title=f"[bold]{name}[/bold]", border_style="yellow"))
        choice = Prompt.ask(
            "Run this? [y=yes / n=no / a=always]", choices=["y", "n", "a"], default="n"
        )
        return CONFIRM_CHOICES[choice]

    return confirm


def build_agent(
    client: OllamaClient,
    session: Session,
    cfg: Config,
    model: str,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console,
    streamer: MarkdownStreamer,
) -> Agent:
    native = detector.supports_tools(model)
    mode = "native tool calling" if native else "ReAct fallback (no native tool support)"
    console.print(f"[dim]model: {model} · {mode}[/dim]")
    agent_cfg = AgentConfig(
        model=model,
        max_iterations=cfg.max_iterations,
        bash_timeout=cfg.bash_timeout_seconds,
        yolo=yolo,
    )
    try:
        window = int(cfg.context_window) if cfg.context_window else 0
    except (TypeError, ValueError):
        window = 0
    window = window or detect_context_window(client, model)
    compactor = Compactor(
        client, model, window,
        notify=lambda message: console.print(f"[dim]{message}[/dim]"),
    )
    permission_store = PermissionStore()
    return Agent(
        client,
        session,
        agent_cfg,
        use_native=native,
        confirm=make_confirmer(console),
        on_token=streamer.token,
        notify=lambda message: console.print(f"\n[dim]{message}[/dim]"),
        on_stream_start=streamer.start,
        on_stream_end=streamer.end,
        compactor=compactor,
        permission_store=permission_store,
        on_todos=make_todo_renderer(console),
    )


def save_session(
    store: SessionStore,
    session_id: str,
    model: str,
    session: Session,
    console: Console,
) -> None:
    try:
        store.save(session_id, model, session.system_prompt, session.history)
    except OSError as e:
        console.print(f"[dim yellow]session save failed: {e}[/dim yellow]")


def help_text() -> str:
    base = (
        "Comandos: /help · /tools · /history · /sessions · /clear · /model <name> · /exit\n"
        "Flags de arranque: --model NAME · --yolo · --system TEXT · --resume [ID]\n"
        "Confirmaciones: y = sí · n = no · a = siempre (se guarda en ~/.local-code/permissions.yaml)\n"
        "Menciones: @ruta/archivo inyecta el contenido del archivo en tu mensaje."
    )
    customs = list_custom_commands()
    if customs:
        base += "\nCustom: " + " · ".join(f"/{c}" for c in customs)
    return base


def print_tools_table(console: Console) -> None:
    table = Table(title="tools")
    table.add_column("name")
    table.add_column("description")
    table.add_column("confirma")
    for mod in tools.ALL_TOOLS:
        table.add_row(mod.NAME, mod.DESCRIPTION, "Sí" if mod.REQUIRES_CONFIRMATION else "No")
    console.print(table)


def print_sessions_table(store: SessionStore, console: Console) -> None:
    sessions = store.list_sessions(10)
    if not sessions:
        console.print("[dim]no hay sesiones guardadas[/dim]")
        return
    table = Table(title="sessions")
    table.add_column("id")
    table.add_column("updated")
    table.add_column("model")
    table.add_column("first message")
    for s in sessions:
        table.add_row(s["id"], s["updated_at"], s["model"], s["first_message"])
    console.print(table)


def repl(
    client: OllamaClient,
    session: Session,
    cfg: Config,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console,
    agent: Agent,
    store: SessionStore,
    session_id: str,
    streamer: MarkdownStreamer,
) -> int:
    console.print("[bold]local-code[/bold] — /help para comandos")

    def run_chat(text: str) -> None:
        expanded, warnings = expand_file_mentions(text)
        for w in warnings:
            console.print(f"[dim yellow]{w}[/dim yellow]")
        try:
            agent.run_turn(expanded)
            print()
        except KeyboardInterrupt:
            streamer.end()
            console.print("\n[dim]turn interrupted[/dim]")
        except OllamaError as e:
            console.print(f"\n[red]{e}[/red]")
        save_session(store, session_id, agent.config.model, session, console)

    while True:
        try:
            line = console.input("\n[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        action, arg = handle_command(line)
        if action == "exit":
            return 0
        if action == "clear":
            session.clear()
            session_id = store.new_id()
            console.print(f"[dim]history cleared (new session {session_id})[/dim]")
            continue
        if action == "model":
            if not arg:
                console.print("[red]usage: /model <name>[/red]")
                continue
            try:
                agent = build_agent(
                    client, session, cfg, arg, yolo, detector, console, streamer
                )
            except OllamaError as e:
                console.print(f"[red]{e}[/red]")
            continue
        if action == "help":
            console.print(help_text())
            continue
        if action == "tools":
            print_tools_table(console)
            continue
        if action == "history":
            console.print(
                f"[dim]{history_summary(session_id, agent.config.model, session.history)}[/dim]"
            )
            continue
        if action == "sessions":
            print_sessions_table(store, console)
            continue
        if action == "custom":
            parts = arg.split(maxsplit=1)
            name = parts[0][1:]
            cmd_args = parts[1] if len(parts) > 1 else ""
            text = load_custom_command(name, cmd_args)
            if text is None:
                console.print(f"[red]unknown command: {arg}[/red]")
                continue
            run_chat(text)
            continue
        if not arg:
            continue
        run_chat(arg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    console = Console()
    client = OllamaClient(host=cfg.ollama_host)
    detector = CapabilityDetector(client)
    model = args.model or cfg.default_model

    if args.system:
        base_system = args.system
    elif cfg.system_prompt:
        base_system = cfg.system_prompt
    else:
        base_system = DEFAULT_SYSTEM_PROMPT.format(cwd=os.getcwd())

    ctx = load_project_context()
    if ctx is not None:
        ctx_name, ctx_content = ctx
        base_system = f"{base_system}\n\n# Project context ({ctx_name})\n\n{ctx_content}"
        console.print(f"[dim]project context: {ctx_name}[/dim]")

    store = SessionStore()
    if args.resume is not None:
        resume_id = store.latest_id() if args.resume == "latest" else args.resume
        if resume_id is None:
            console.print("[red]No hay sesiones guardadas para retomar.[/red]")
            return 1
        try:
            data = store.load(resume_id)
        except SessionNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        session = Session(system_prompt=data.get("system_prompt") or base_system)
        session.history.extend(data.get("history", []))
        session_id = resume_id
        console.print(
            f"[dim]resumed session {resume_id} "
            f"({len(session.history)} messages, saved model: {data.get('model', '?')})[/dim]"
        )
    else:
        session = Session(system_prompt=base_system)
        session_id = store.new_id()

    streamer = MarkdownStreamer(console)
    try:
        agent = build_agent(
            client, session, cfg, model, args.yolo, detector, console, streamer
        )
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    if args.prompt:
        text, warnings = expand_file_mentions(" ".join(args.prompt))
        for w in warnings:
            console.print(f"[dim yellow]{w}[/dim yellow]")
        try:
            agent.run_turn(text)
        except OllamaError as e:
            console.print(f"\n[red]{e}[/red]")
            return 1
        finally:
            save_session(store, session_id, model, session, console)
        print()
        return 0

    return repl(
        client, session, cfg, args.yolo, detector, console, agent, store, session_id, streamer
    )
