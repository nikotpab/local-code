from __future__ import annotations

import difflib
from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "multi_edit"
DESCRIPTION = (
    "Apply several old_string→new_string replacements to one file in a single "
    "atomic operation. Each old_string must be unique in the file at the time "
    "it is applied; if any edit fails, nothing is written."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to edit"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["old_string", "new_string"],
            },
        },
    },
    "required": ["path", "edits"],
}
REQUIRES_CONFIRMATION = True


def _apply(content: str, edits: list, path: str) -> tuple[str | None, str]:
    for i, edit in enumerate(edits):
        old = edit.get("old_string", "")
        n = content.count(old) if old else 0
        if n == 0:
            return None, f"edit {i}: old_string not found in {path}"
        if n > 1:
            return None, f"edit {i}: old_string appears {n} times in {path}"
        content = content.replace(old, edit.get("new_string", ""), 1)
    return content, ""


def run(arguments: dict, context: ToolContext) -> str:
    path = arguments["path"]
    edits = arguments["edits"]
    if not isinstance(edits, list) or not edits:
        return "Error: edits must be a non-empty list"
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    new_content, error = _apply(content, edits, path)
    if new_content is None:
        return f"Error: {error}"
    p.write_text(new_content, encoding="utf-8")
    return f"Applied {len(edits)} edits to {path}"


def preview(arguments: dict) -> str:
    path = arguments["path"]
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        return f"multi_edit → {path} (error: cannot read: {e})"
    edits = arguments.get("edits") or []
    new_content, error = _apply(content, edits, path)
    if new_content is None:
        return f"multi_edit → {path} (error: {error})"
    diff = difflib.unified_diff(
        content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff) or "(no changes)"
