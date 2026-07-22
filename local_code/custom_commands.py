from __future__ import annotations

import re
from pathlib import Path

COMMANDS_DIR = Path.home() / ".local-code" / "commands"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def list_custom_commands(dir: Path | None = None) -> list[str]:
    base = dir or COMMANDS_DIR
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.md"))


def load_custom_command(name: str, args: str, dir: Path | None = None) -> str | None:
    if not NAME_RE.fullmatch(name):
        return None
    base = dir or COMMANDS_DIR
    path = base / f"{name}.md"
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return content.replace("$ARGUMENTS", args)
