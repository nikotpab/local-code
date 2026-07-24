from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from local_code import __version__, tools, ui
from local_code.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentConfig
from local_code.backends import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
    select_backend,
)
from local_code.capabilities import CapabilityDetector
from local_code.checkpoints import CheckpointStore
from local_code.compaction import Compactor, detect_context_window, estimate_tokens
from local_code.config import (
    CONFIG_PATH,
    Config,
    load_config,
    load_mcp_server_configs,
    validate_config,
)
from local_code.custom_commands import list_custom_commands, load_custom_command
from local_code.environment import environment_block
from local_code.hooks import HookRunner
from local_code.logging_setup import configure_logging
from local_code.mcp import MCPManager
from local_code.mentions import expand_file_mentions
from local_code.permissions import PermissionStore
from local_code.project_context import load_project_context
from local_code.session import Session
from local_code.session_store import SessionNotFoundError, SessionStore

SUBAGENT_MAX_ITERATIONS = 15
SUBAGENT_REPORT_MAX_CHARS = 10_000

logger = logging.getLogger(__name__)

# Exit codes. 0/1/2 follow convention (ok / generic error / argparse usage);
# 130 is the shell's SIGINT convention. Backend failures get distinct codes so
# scripts can branch on *why* a run failed, not just that it did.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130
EXIT_CONNECTION = 3
EXIT_MODEL_NOT_FOUND = 4


def exit_code_for(exc: Exception) -> int:
    """Map a backend exception to a process exit code."""
    if isinstance(exc, OllamaConnectionError):
        return EXIT_CONNECTION
    if isinstance(exc, ModelNotFoundError):
        return EXIT_MODEL_NOT_FOUND
    return EXIT_ERROR


def report_backend_error(console: Console, exc: Exception) -> int:
    """Log, print a user-facing message with a hint, and return an exit code."""
    logger.error("%s: %s", type(exc).__name__, exc, exc_info=True)
    console.print(f"\n[red]{exc}[/red]")
    if isinstance(exc, OllamaConnectionError):
        console.print("[dim]hint: is the model server running? (e.g. `ollama serve`)[/dim]")
    elif isinstance(exc, ModelNotFoundError):
        console.print("[dim]hint: pull or pick another model (e.g. `ollama pull <name>`) or use --model[/dim]")
    return exit_code_for(exc)

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
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the version and exit",
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="One-shot only: emit a JSON result on stdout (diagnostics go to stderr)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log INFO-level diagnostics to stderr")
    parser.add_argument("--debug", action="store_true", help="Log DEBUG-level diagnostics to stderr")
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Also write DEBUG-level logs to this file",
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt; omit for interactive REPL")
    return parser.parse_args(argv)


def read_piped_stdin() -> str | None:
    """Return piped stdin content, or None when stdin is an interactive tty.

    Empty/whitespace-only input counts as none so an accidental empty pipe
    doesn't send a blank prompt.
    """
    if sys.stdin.isatty():
        return None
    data = sys.stdin.read()
    return data if data.strip() else None


def build_oneshot_text(prompt_args: list[str], stdin_text: str | None) -> str | None:
    """Combine positional prompt args with piped stdin into one prompt.

    - both present  -> "<prompt>\\n\\n<stdin>" (stdin as trailing context)
    - only one      -> that one
    - neither       -> None (fall through to REPL/TUI)
    """
    joined = " ".join(prompt_args).strip()
    if joined and stdin_text:
        return f"{joined}\n\n{stdin_text}"
    return joined or stdin_text or None


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
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_end: Callable[[str, str], None] | None = None,
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

    def _default_notify(message: str) -> None:
        if console is not None:
            console.print(f"\n[dim]{message}[/dim]")

    eff_notify: Callable[[str], None] = notify or _default_notify
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

    eff_on_tool_start = on_tool_start or (
        (lambda n, a: ui.render_tool_start(console, n, a)) if console else None
    )
    eff_on_tool_end = on_tool_end or (
        (lambda n, r: ui.render_tool_end(console, n, r)) if console else None
    )

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
        on_tool_start=eff_on_tool_start,
        on_tool_end=eff_on_tool_end,
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
    console: Console | None,
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

    def say(message: str) -> None:
        if console is not None:
            console.print(message)

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

            say(f"\n[dim][subagent: {effective_model}] starting…[/dim]")

            sub_agent = Agent(
                sub_client,
                sub_session,
                sub_cfg,
                use_native=use_native,
                # Subagents confirm nothing (plan_mode gate handles it).
                confirm=lambda name, preview: "yes",
                on_token=lambda token: None,
                notify=lambda msg: say(f"[dim][subagent] {msg}[/dim]"),
                on_stream_start=lambda: None,
                on_stream_end=lambda: None,
            )
            # Explicit recursion guard: subagent cannot spawn further subagents.
            sub_agent.context.spawn = None

            result = sub_agent.run_turn(task)
            if len(result) > SUBAGENT_REPORT_MAX_CHARS:
                result = result[:SUBAGENT_REPORT_MAX_CHARS] + "\n…[truncated]"
            say(f"[dim][subagent: {effective_model}] done[/dim]")
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
        if action == "custom" and arg:
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


CONFIG_USAGE = "usage: local-code config {show|path|validate}"

# Top-level flags and subcommands, single source of truth for shell completion.
COMPLETION_FLAGS = [
    "--help", "--version", "--model", "--yolo", "--plan", "--system",
    "--backend", "--resume", "--no-tui", "--json", "-v", "--verbose",
    "--debug", "--log-file",
]
COMPLETION_SUBCOMMANDS = ["config", "completion"]
COMPLETION_SHELLS = ("bash", "zsh", "fish")
COMPLETION_USAGE = "usage: local-code completion {bash|zsh|fish}"


def _completion_script(shell: str) -> str:
    words = " ".join(COMPLETION_FLAGS + COMPLETION_SUBCOMMANDS)
    if shell == "bash":
        return (
            "# local-code bash completion — add to ~/.bashrc:\n"
            "#   source <(local-code completion bash)\n"
            "_local_code_complete() {\n"
            '    local cur="${COMP_WORDS[COMP_CWORD]}"\n'
            f'    COMPREPLY=( $(compgen -W "{words}" -- "$cur") )\n'
            "    return 0\n"
            "}\n"
            "complete -o default -F _local_code_complete local-code\n"
        )
    if shell == "zsh":
        return (
            "# local-code zsh completion — add to ~/.zshrc:\n"
            "#   source <(local-code completion zsh)\n"
            "_local_code_complete() {\n"
            f'    local -a opts; opts=({words})\n'
            "    compadd -- $opts\n"
            "    _files\n"
            "}\n"
            "compdef _local_code_complete local-code\n"
        )
    # fish
    lines = ["# local-code fish completion — save to:",
             "#   ~/.config/fish/completions/local-code.fish"]
    for flag in COMPLETION_FLAGS:
        lines.append(f"complete -c local-code -a '{flag}'")
    for sub in COMPLETION_SUBCOMMANDS:
        lines.append(f"complete -c local-code -f -a '{sub}'")
    return "\n".join(lines) + "\n"


def run_completion_command(sub_argv: list[str], console: Console) -> int:
    """Print a shell completion script to stdout."""
    if not sub_argv:
        console.print(f"[red]{COMPLETION_USAGE}[/red]")
        return 2
    shell = sub_argv[0]
    if shell not in COMPLETION_SHELLS:
        console.print(f"[red]unsupported shell: {shell}[/red]")
        console.print(f"[dim]{COMPLETION_USAGE}[/dim]")
        return 2
    # Print raw (no rich markup) so the script is emitted verbatim.
    print(_completion_script(shell), end="")
    return EXIT_OK


def run_config_command(sub_argv: list[str], console: Console) -> int:
    """Handle the `config` subcommand (show / path / validate)."""
    from dataclasses import asdict

    import yaml

    action = sub_argv[0] if sub_argv else "show"

    if action == "path":
        console.print(str(CONFIG_PATH))
        return EXIT_OK

    if action == "show":
        data = asdict(load_config())
        if data.get("api_key"):
            data["api_key"] = "***redacted***"
        console.print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip())
        exists = CONFIG_PATH.exists()
        console.print(
            f"[dim]# source: {CONFIG_PATH}"
            f"{'' if exists else ' (not found — showing defaults)'}[/dim]"
        )
        return EXIT_OK

    if action == "validate":
        problems = validate_config()
        if not problems:
            console.print(f"[green]config OK[/green] [dim]({CONFIG_PATH})[/dim]")
            return EXIT_OK
        console.print(f"[red]config has {len(problems)} problem(s):[/red]")
        for p in problems:
            console.print(f"  [red]- {p}[/red]")
        return EXIT_ERROR

    console.print(f"[red]unknown config action: {action}[/red]")
    console.print(f"[dim]{CONFIG_USAGE}[/dim]")
    return 2


def run_oneshot_json(app_ctx: AppContext, text: str) -> int:
    """Run a single turn and emit one JSON object on stdout.

    Human-facing diagnostics (mode line, tool activity, warnings) go to a
    stderr console so stdout carries only the JSON payload — safe to pipe
    into `jq` or another program.
    """
    err_console = Console(stderr=True)

    def emit(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False))

    try:
        agent = build_agent(
            app_ctx.client,
            app_ctx.session,
            app_ctx.cfg,
            app_ctx.model,
            app_ctx.args.yolo,
            app_ctx.detector,
            err_console,
            streamer=None,
            checkpoint_store=app_ctx.checkpoints,
            plan_mode=app_ctx.plan_mode,
            spawn_factory=app_ctx.spawn_factory,
        )
    except OllamaError as e:
        logger.error("%s: %s", type(e).__name__, e, exc_info=True)
        emit({"ok": False, "model": app_ctx.model, "error": str(e), "error_type": type(e).__name__})
        app_ctx.mcp_manager.shutdown()
        return exit_code_for(e)

    expanded, warnings = expand_file_mentions(text)
    for w in warnings:
        err_console.print(f"[dim yellow]{w}[/dim yellow]")

    payload: dict
    try:
        result = agent.run_turn(expanded)
        payload = {
            "ok": True,
            "model": app_ctx.model,
            "session_id": app_ctx.session_id,
            "response": result,
        }
        code = EXIT_OK
    except KeyboardInterrupt:
        payload = {"ok": False, "model": app_ctx.model, "error": "interrupted"}
        code = EXIT_INTERRUPTED
    except OllamaError as e:
        logger.error("%s: %s", type(e).__name__, e, exc_info=True)
        payload = {"ok": False, "model": app_ctx.model, "error": str(e), "error_type": type(e).__name__}
        code = exit_code_for(e)
    finally:
        save_session(app_ctx.store, app_ctx.session_id, app_ctx.model, app_ctx.session, err_console)
        app_ctx.mcp_manager.shutdown()

    emit(payload)
    return code


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "config":
        return run_config_command(raw[1:], Console())
    if raw and raw[0] == "completion":
        return run_completion_command(raw[1:], Console())

    args = parse_args(argv)
    logger = configure_logging(
        debug=getattr(args, "debug", False),
        verbose=getattr(args, "verbose", False),
        log_file=getattr(args, "log_file", None),
    )
    logger.debug("local-code %s starting (argv=%r)", __version__, argv)
    console = Console()

    stdin_text = read_piped_stdin()
    oneshot_text = build_oneshot_text(args.prompt, stdin_text)

    try:
        app_ctx = setup_app_context(args, console)
    except (ValueError, RuntimeError, SessionNotFoundError) as e:
        logger.error("startup failed: %s", e, exc_info=True)
        console.print(f"[red]{e}[/red]")
        return EXIT_ERROR

    if oneshot_text is not None and getattr(args, "json", False):
        return run_oneshot_json(app_ctx, oneshot_text)

    if getattr(args, "json", False):
        console.print("[red]--json requires a one-shot prompt (positional or piped stdin)[/red]")
        app_ctx.mcp_manager.shutdown()
        return EXIT_ERROR

    use_tui = oneshot_text is None and sys.stdout.isatty() and not getattr(args, "no_tui", False)

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
        code = report_backend_error(console, e)
        app_ctx.mcp_manager.shutdown()
        return code

    if oneshot_text is not None:
        text, warnings = expand_file_mentions(oneshot_text)
        for w in warnings:
            console.print(f"[dim yellow]{w}[/dim yellow]")
        code = EXIT_OK
        try:
            agent.run_turn(text)
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted[/dim]")
            code = EXIT_INTERRUPTED
        except OllamaError as e:
            code = report_backend_error(console, e)
        finally:
            save_session(app_ctx.store, app_ctx.session_id, app_ctx.model, app_ctx.session, console)
            app_ctx.mcp_manager.shutdown()
        if code == EXIT_OK:
            print()
        return code

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

