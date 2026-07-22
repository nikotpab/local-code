from __future__ import annotations

from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "read_file"
DESCRIPTION = "Read a text file and return its full content."
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path, relative to the working directory"}
    },
    "required": ["path"],
}
REQUIRES_CONFIRMATION = False
MAX_CHARS = 100_000


def run(arguments: dict, context: ToolContext) -> str:
    content = Path(arguments["path"]).read_text()
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "\n...[truncated]"
    return content
