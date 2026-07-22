from __future__ import annotations

from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "write_file"
DESCRIPTION = "Write content to a file, creating parent directories. Overwrites existing files."
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Full file content"},
    },
    "required": ["path", "content"],
}
REQUIRES_CONFIRMATION = True


def run(arguments: dict, context: ToolContext) -> str:
    p = Path(arguments["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(arguments["content"])
    return f"Wrote {len(arguments['content'])} chars to {arguments['path']}"


def preview(arguments: dict) -> str:
    head = arguments["content"][:500]
    return f"write_file → {arguments['path']}\n---\n{head}"
