from __future__ import annotations

import re
from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "grep"
DESCRIPTION = "Search file contents recursively with a Python regex. Returns 'path:line_number: line' matches."
PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Python regular expression"},
        "path": {"type": "string", "description": "Directory to search (default: current directory)"},
    },
    "required": ["pattern"],
}
REQUIRES_CONFIRMATION = False
MAX_RESULTS = 200
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__"}


def run(arguments: dict, context: ToolContext) -> str:
    rx = re.compile(arguments["pattern"])
    root = Path(arguments.get("path", "."))
    results: list[str] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                results.append(f"{f}:{i}: {line}")
                if len(results) >= MAX_RESULTS:
                    results.append("...[truncated]")
                    return "\n".join(results)
    return "\n".join(results) if results else "(no matches)"
