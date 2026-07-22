from __future__ import annotations

from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "glob"
DESCRIPTION = "Find files matching a glob pattern relative to the working directory. Supports ** recursion."
PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern, e.g. 'src/**/*.py'"}
    },
    "required": ["pattern"],
}
REQUIRES_CONFIRMATION = False
MAX_RESULTS = 200


def run(arguments: dict, context: ToolContext) -> str:
    matches = sorted(str(p) for p in Path(".").glob(arguments["pattern"]))
    if not matches:
        return "(no matches)"
    out = matches[:MAX_RESULTS]
    if len(matches) > MAX_RESULTS:
        out.append("...[truncated]")
    return "\n".join(out)
