from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from local_code.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentConfig
from local_code.capabilities import CapabilityDetector
from local_code.client import OllamaClient, OllamaError
from local_code.config import Config, load_config
from local_code.session import Session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="local-code",
        description="Agentic coding CLI for any local model served by Ollama",
    )
    parser.add_argument("--model", help="Model name (default: from ~/.local-code/config.yaml)")
    parser.add_argument("--yolo", action="store_true", help="Skip all tool confirmations")
    parser.add_argument("--system", help="Override the system prompt")
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
    return ("unknown", stripped)


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


def make_confirmer(console: Console):
    def confirm(name: str, preview: str) -> bool:
        text = Text()
        for line, style in style_preview_lines(preview):
            text.append(line + "\n", style=style)
        console.print(Panel(text, title=f"[bold]{name}[/bold]", border_style="yellow"))
        return Confirm.ask("Run this?", default=False)

    return confirm


def build_agent(
    client: OllamaClient,
    session: Session,
    cfg: Config,
    model: str,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console,
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
    return Agent(
        client,
        session,
        agent_cfg,
        use_native=native,
        confirm=make_confirmer(console),
        on_token=lambda token: (sys.stdout.write(token), sys.stdout.flush()),
        notify=lambda message: console.print(f"\n[dim]{message}[/dim]"),
    )


def repl(
    client: OllamaClient,
    session: Session,
    cfg: Config,
    yolo: bool,
    detector: CapabilityDetector,
    console: Console,
    agent: Agent,
) -> int:
    console.print("[bold]local-code[/bold] — /clear · /model <name> · /exit")
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
            console.print("[dim]history cleared[/dim]")
            continue
        if action == "model":
            if not arg:
                console.print("[red]usage: /model <name>[/red]")
                continue
            try:
                agent = build_agent(client, session, cfg, arg, yolo, detector, console)
            except OllamaError as e:
                console.print(f"[red]{e}[/red]")
            continue
        if action == "unknown":
            console.print(f"[red]unknown command: {arg}[/red]")
            continue
        if not arg:
            continue
        try:
            agent.run_turn(arg)
            print()
        except KeyboardInterrupt:
            console.print("\n[dim]turn interrupted[/dim]")
        except OllamaError as e:
            console.print(f"\n[red]{e}[/red]")


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

    session = Session(system_prompt=base_system)
    try:
        agent = build_agent(client, session, cfg, model, args.yolo, detector, console)
    except OllamaError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    if args.prompt:
        try:
            agent.run_turn(" ".join(args.prompt))
        except OllamaError as e:
            console.print(f"\n[red]{e}[/red]")
            return 1
        print()
        return 0

    return repl(client, session, cfg, args.yolo, detector, console, agent)
