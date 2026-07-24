from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from local_code import tools
from local_code.agent import DEFAULT_SYSTEM_PROMPT, PLAN_MODE_INSTRUCTION, Agent, AgentConfig
from local_code.backends import OllamaError, select_backend
from local_code.capabilities import CapabilityDetector
from local_code.checkpoints import CheckpointStore
from local_code import ui
from local_code.compaction import Compactor, detect_context_window, estimate_tokens
from local_code.config import Config, load_config, load_mcp_server_configs
from local_code.custom_commands import list_custom_commands, load_custom_command
from local_code.environment import environment_block
from local_code.hooks import HookRunner
from local_code.mcp import MCPManager
from local_code.mentions import expand_file_mentions
from local_code.permissions import PermissionStore
from local_code.project_context import load_project_context
from local_code.session import Session
from local_code.session_store import SessionNotFoundError, SessionStore
from local_code.tools.context import ToolContext

SUBAGENT_MAX_ITERATIONS = 15
SUBAGENT_REPORT_MAX_CHARS = 10_000

SUBAGENT_SYSTEM_PROMPT = (
    "You are a read-only research subagent. "
    "Your sole purpose is to investigate the task given to you and produce a "
    "detailed, accurate report. "
    "Use only read-only tools: read_file, list_dir, glob, grep, set_todos. "
    "Do NOT write files, run shell commands, or spawn further subagents. "
    "When you have gathered enough information, output your complete findings."
)


@dataclass
class AppContext:
    args: argparse.Namespace
    cfg: Config
    console: Console
    client: object
    detector: CapabilityDetector
    model: str
    plan_mode: bool
    mcp_manager: MCPManager
    store: SessionStore
    checkpoints: CheckpointStore
    session: Session
    session_id: str
    spawn_factory: object


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="local-code",
        description="Agentic coding CLI for any local model served by Ollama",
    )
    parser.add_argument("--model", help="Model name (default: from ~/.local-code/config.yaml)")
    parser.add_argument("--yolo", action="store_true", help="Skip all tool confirmations")
    parser.add_argument("--plan", action="store_true", help="Start in plan mode (read-only investigation)")
    parser.add_argument("--system", help="Override the system prompt")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default=None,
        help="Force a backend instead of inferring it from the host URL",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="ID",
        help="Resume the latest session, or a specific session id",
    )
    parser.add_argument("--no-tui", action="store_true", help="Force line-based REPL interface instead of full-screen TUI")
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
    if cmd == "/undo":
        return ("undo", None)
    if cmd == "/mcp":
        return ("mcp", None)
    if cmd == "/plan":
        return ("plan", None)
    if cmd == "/approve":
        return ("approve", None)
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
    client,
    session: Session,
    cfg: Config,
    model: str,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console | None = None,
    streamer: ui.ResponseView | None = None,
    checkpoint_store: CheckpointStore | None = None,
    plan_mode: bool = False,
    spawn_factory=None,
    confirm: Callable[[str, str], str] | None = None,
    on_token: Callable[[str], None] | None = None,
    notify: Callable[[str], None] | None = None,
    on_stream_start: Callable[[], None] | None = None,
    on_stream_end: Callable[[], None] | None = None,
    on_todos: Callable[[list], None] | None = None,
) -> Agent:
    native = detector.supports_tools(model)
    mode = "native tool calling" if native else "ReAct fallback (no native tool support)"
    plan_tag = " · plan mode" if plan_mode else ""
    if console:
        console.print(f"[dim]model: {model} · {mode}{plan_tag}[/dim]")
    agent_cfg = AgentConfig(
        model=model,
        max_iterations=cfg.max_iterations,
        bash_timeout=cfg.bash_timeout_seconds,
        yolo=yolo,
        plan_mode=plan_mode,
    )
    try:
        window = int(cfg.context_window) if cfg.context_window else 0
    except (TypeError, ValueError):
        window = 0
    window = window or detect_context_window(client, model)

    eff_notify = notify or (lambda message: console.print(f"\n[dim]{message}[/dim]") if console else (lambda msg: None))
    compactor = Compactor(
        client, model, window,
        notify=eff_notify,
    )
    permission_store = PermissionStore()
    hook_runner = HookRunner()

    eff_confirm = confirm or (make_confirmer(console) if console else (lambda n, p: "yes"))
    eff_on_token = on_token or (streamer.token if streamer else (lambda t: None))
    eff_on_stream_start = on_stream_start or (streamer.start if streamer else (lambda: None))
    eff_on_stream_end = on_stream_end or (streamer.end if streamer else (lambda: None))
    eff_on_todos = on_todos or (make_todo_renderer(console) if console else None)

    on_tool_start = (lambda n, a: ui.render_tool_start(console, n, a)) if console else None
    on_tool_end = (lambda n, r: ui.render_tool_end(console, n, r)) if console else None

    agent = Agent(
        client,
        session,
        agent_cfg,
        use_native=native,
        confirm=eff_confirm,
        on_token=eff_on_token,
        notify=eff_notify,
        on_stream_start=eff_on_stream_start,
        on_stream_end=eff_on_stream_end,
        compactor=compactor,
        permission_store=permission_store,
        on_todos=eff_on_todos,
        hook_runner=hook_runner,
        checkpoint_store=checkpoint_store,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )
    if spawn_factory is not None:
        agent.context.spawn = spawn_factory
    return agent


def setup_app_context(args: argparse.Namespace, console: Console | None = None) -> AppContext:
    cfg = load_config()
    client = select_backend(
        cfg.ollama_host,
        override=args.backend or cfg.backend,
        api_key=cfg.api_key,
    )
    detector = CapabilityDetector(client)
    model = args.model or cfg.default_model
    plan_mode: bool = getattr(args, "plan", False)

    if args.system:
        base_system = args.system
    elif cfg.system_prompt:
        base_system = cfg.system_prompt
    else:
        base_system = DEFAULT_SYSTEM_PROMPT

    base_system = f"{base_system}\n\n{environment_block()}"

    ctx = load_project_context()
    if ctx is not None:
        ctx_name, ctx_content = ctx
        base_system = f"{base_system}\n\n# Project context ({ctx_name})\n\n{ctx_content}"
        if console:
            console.print(f"[dim]project context: {ctx_name}[/dim]")

    mcp_manager = MCPManager(notify=lambda msg: console.print(msg) if console else None)
    mcp_configs = load_mcp_server_configs()
    if mcp_configs:
        mcp_manager.start(mcp_configs)
        adapters = mcp_manager.tool_adapters
        if adapters:
            from local_code import tools as _tools
            _tools.register_mcp_tools(adapters)

    store = SessionStore()
    checkpoints = CheckpointStore()
    if args.resume is not None:
        resume_id = store.latest_id() if args.resume == "latest" else args.resume
        if resume_id is None:
            raise RuntimeError("No hay sesiones guardadas para retomar.")
        data = store.load(resume_id)
        session = Session(system_prompt=data.get("system_prompt") or base_system)
        session.history.extend(data.get("history", []))
        session_id = resume_id
        if console:
            console.print(
                f"[dim]resumed session {resume_id} "
                f"({len(session.history)} messages, saved model: {data.get('model', '?')})[/dim]"
            )
    else:
        session = Session(system_prompt=base_system)
        session_id = store.new_id()

    spawn_factory = make_spawn_factory(client, cfg, model, console)

    return AppContext(
        args=args,
        cfg=cfg,
        console=console or Console(),
        client=client,
        detector=detector,
        model=model,
        plan_mode=plan_mode,
        mcp_manager=mcp_manager,
        store=store,
        checkpoints=checkpoints,
        session=session,
        session_id=session_id,
        spawn_factory=spawn_factory,
    )


def make_spawn_factory(
    parent_client,
    parent_cfg: Config,
    parent_model: str,
    console: Console,
):
    """Return a spawn(task, model) -> str factory for subagents.

    The subagent:
    - uses a fresh read-only Session
    - is always in plan_mode (read-only gate)
    - has spawn=None on its ToolContext (recursion guard)
    - is bounded to SUBAGENT_MAX_ITERATIONS
    - streams are silent (no on_token/on_stream_start/on_stream_end)
    - any failure is caught and returned as "Error: ..."
    """

    def spawn(task: str, model: str | None) -> str:
        effective_model = model or parent_model
        try:
            # Reuse the parent's backend/host; a different model name just
            # changes what we pass to chat().
            sub_client = parent_client

            sub_session = Session(system_prompt=SUBAGENT_SYSTEM_PROMPT)
            sub_cfg = AgentConfig(
                model=effective_model,
                max_iterations=SUBAGENT_MAX_ITERATIONS,
                bash_timeout=parent_cfg.bash_timeout_seconds,
                yolo=False,
                plan_mode=True,  # enforces read-only gate
            )
            # Detect native tool support for the subagent's model.
            # We check against parent_client which may not support the model —
            # fall back gracefully.
            try:
                from local_code.capabilities import CapabilityDetector
                detector = CapabilityDetector(sub_client)
                use_native = detector.supports_tools(effective_model)
            except Exception:
                use_native = False

            console.print(f"\n[dim][subagent: {effective_model}] starting…[/dim]")

            sub_agent = Agent(
                sub_client,
                sub_session,
                sub_cfg,
                use_native=use_native,
                # Subagents confirm nothing (plan_mode gate handles it).
                confirm=lambda name, preview: "yes",
                on_token=lambda token: None,
                notify=lambda msg: console.print(f"[dim][subagent] {msg}[/dim]"),
                on_stream_start=lambda: None,
                on_stream_end=lambda: None,
            )
            # Explicit recursion guard: subagent cannot spawn further subagents.
            sub_agent.context.spawn = None

            result = sub_agent.run_turn(task)
            if len(result) > SUBAGENT_REPORT_MAX_CHARS:
                result = result[:SUBAGENT_REPORT_MAX_CHARS] + "\n…[truncated]"
            console.print(f"[dim][subagent: {effective_model}] done[/dim]")
            return f"[Subagent report from {effective_model}]\n\n{result}"
        except Exception as exc:
            return f"Error: subagent failed: {type(exc).__name__}: {exc}"

    return spawn


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
        "Comandos: /help · /tools · /mcp · /plan · /approve · "
        "/history · /sessions · /undo · /clear · /model <name> · /exit\n"
        "Flags de arranque: --model NAME · --yolo · --plan · --system TEXT · --resume [ID]\n"
        "Confirmaciones: y = sí · n = no · a = siempre (se guarda en ~/.local-code/permissions.yaml)\n"
        "Menciones: @ruta/archivo inyecta el contenido del archivo en tu mensaje.\n"
        "/plan  — toggle plan mode (read-only investigation + numbered plan output)\n"
        "/approve — execute the last plan: turns off plan mode and re-runs the last assistant message"
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


def print_mcp_table(console: Console, mcp_manager: MCPManager) -> None:
    summaries = mcp_manager.server_summaries()
    if not summaries:
        console.print("[dim]no MCP servers connected[/dim]")
        return
    table = Table(title="MCP servers")
    table.add_column("server")
    table.add_column("tools")
    for s in summaries:
        table.add_row(s["name"], str(s["tool_count"]))
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


def _last_assistant_message(session: Session) -> str | None:
    """Return the content of the most recent assistant message, or None."""
    for msg in reversed(session.history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                return content
    return None


def repl(
    client,
    session: Session,
    cfg: Config,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console,
    agent: Agent,
    store: SessionStore,
    session_id: str,
    streamer: ui.ResponseView,
    checkpoints: CheckpointStore,
    mcp_manager: MCPManager,
    plan_mode: bool = False,
    spawn_factory=None,
) -> int:
    native = getattr(agent, "use_native", False)
    ui.render_banner(
        console,
        agent.config.model,
        getattr(client, "name", "?"),
        native,
        ui.shorten_path(str(Path.cwd()), str(Path.home())),
        agent.config.plan_mode,
    )

    def run_chat(text: str) -> None:
        nonlocal session_id
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
        window = agent.compactor.context_window if agent.compactor is not None else 0
        console.print(
            ui.status_line(
                agent.config.model,
                ui.shorten_path(str(Path.cwd()), str(Path.home())),
                estimate_tokens(session.messages),
                window,
                agent.config.plan_mode,
            ),
            style="dim",
        )
        plan_indicator = "[plan] " if agent.config.plan_mode else ""
        try:
            line = console.input(f"[bold cyan]{plan_indicator}> [/bold cyan]")
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
                    client, session, cfg, arg, yolo, detector, console, streamer, checkpoints,
                    plan_mode=agent.config.plan_mode, spawn_factory=spawn_factory,
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
        if action == "mcp":
            print_mcp_table(console, mcp_manager)
            continue
        if action == "plan":
            # Toggle plan mode on/off.
            new_plan_mode = not agent.config.plan_mode
            agent.config.plan_mode = new_plan_mode
            status = "on" if new_plan_mode else "off"
            console.print(f"[dim]plan mode {status}[/dim]")
            continue
        if action == "approve":
            # Turn off plan mode and feed the last assistant message back as
            # "Execute this plan:\n\n{plan}" so the agent carries it out.
            last_plan = _last_assistant_message(session)
            if last_plan is None:
                console.print("[dim]no plan to approve — run a plan-mode turn first[/dim]")
                continue
            agent.config.plan_mode = False
            console.print("[dim]plan mode off — executing plan…[/dim]")
            run_chat(f"Execute this plan:\n\n{last_plan}")
            continue
        if action == "history":
            console.print(
                f"[dim]{history_summary(session_id, agent.config.model, session.history)}[/dim]"
            )
            continue
        if action == "sessions":
            print_sessions_table(store, console)
            continue
        if action == "undo":
            console.print(checkpoints.undo_last())
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
    console = Console()
    try:
        app_ctx = setup_app_context(args, console)
    except (ValueError, RuntimeError, SessionNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        return 1

    use_tui = not bool(args.prompt) and sys.stdout.isatty() and not getattr(args, "no_tui", False)

    if use_tui:
        from local_code.tui import run_tui
        return run_tui(app_ctx)

    streamer = ui.ResponseView(console)
    try:
        agent = build_agent(
            app_ctx.client,
            app_ctx.session,
            app_ctx.cfg,
            app_ctx.model,
            app_ctx.args.yolo,
            app_ctx.detector,
            console,
            streamer,
            app_ctx.checkpoints,
            plan_mode=app_ctx.plan_mode,
            spawn_factory=app_ctx.spawn_factory,
        )
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        app_ctx.mcp_manager.shutdown()
        return 1

    if app_ctx.args.prompt:
        text, warnings = expand_file_mentions(" ".join(app_ctx.args.prompt))
        for w in warnings:
            console.print(f"[dim yellow]{w}[/dim yellow]")
        try:
            agent.run_turn(text)
        except OllamaError as e:
            console.print(f"\n[red]{e}[/red]")
            return 1
        finally:
            save_session(app_ctx.store, app_ctx.session_id, app_ctx.model, app_ctx.session, console)
            app_ctx.mcp_manager.shutdown()
        print()
        return 0

    try:
        return repl(
            app_ctx.client,
            app_ctx.session,
            app_ctx.cfg,
            app_ctx.args.yolo,
            app_ctx.detector,
            console,
            agent,
            app_ctx.store,
            app_ctx.session_id,
            streamer,
            app_ctx.checkpoints,
            app_ctx.mcp_manager,
            plan_mode=app_ctx.plan_mode,
            spawn_factory=app_ctx.spawn_factory,
        )
    finally:
        app_ctx.mcp_manager.shutdown()

