from __future__ import annotations

from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "list_dir"
DESCRIPTION = "List the entries of a directory. Directories are suffixed with '/'."
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory path (default: current directory)"}
    },
    "required": [],
}
REQUIRES_CONFIRMATION = False


def run(arguments: dict, context: ToolContext) -> str:
    base = Path(arguments.get("path", "."))
    entries = sorted(base.iterdir(), key=lambda p: p.name)
    lines = [e.name + ("/" if e.is_dir() else "") for e in entries]
    return "\n".join(lines) if lines else "(empty directory)"
