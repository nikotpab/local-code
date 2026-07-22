from __future__ import annotations

import difflib
from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "edit_file"
DESCRIPTION = (
    "Replace one unique occurrence of old_string with new_string in a file. "
    "old_string must appear exactly once; include surrounding lines to disambiguate."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to edit"},
        "old_string": {"type": "string", "description": "Exact text to replace (must be unique in the file)"},
        "new_string": {"type": "string", "description": "Replacement text"},
    },
    "required": ["path", "old_string", "new_string"],
}
REQUIRES_CONFIRMATION = True


def run(arguments: dict, context: ToolContext) -> str:
    path = arguments["path"]
    p = Path(path)
    content = p.read_text()
    old = arguments["old_string"]
    n = content.count(old)
    if n == 0:
        return f"Error: old_string not found in {path}"
    if n > 1:
        return f"Error: old_string appears {n} times in {path}; provide a larger unique string"
    p.write_text(content.replace(old, arguments["new_string"], 1))
    return f"Edited {path}"


def preview(arguments: dict) -> str:
    path = arguments["path"]
    try:
        content = Path(path).read_text()
    except OSError as e:
        return f"edit_file → {path} (cannot read: {e})"
    new_content = content.replace(arguments["old_string"], arguments["new_string"], 1)
    diff = difflib.unified_diff(
        content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff) or "(no changes)"
