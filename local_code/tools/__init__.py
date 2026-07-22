from __future__ import annotations

import json

from local_code.tools import (
    bash,
    edit_file,
    glob,
    grep,
    list_dir,
    multi_edit,
    read_file,
    set_todos,
    web_fetch,
    write_file,
)
from local_code.tools.context import ToolContext

__all__ = ["ALL_TOOLS", "ToolContext", "execute", "get_preview", "get_tool", "requires_confirmation", "tool_schemas"]

ALL_TOOLS = [
    bash,
    edit_file,
    glob,
    grep,
    list_dir,
    multi_edit,
    read_file,
    set_todos,
    web_fetch,
    write_file,
]
_BY_NAME = {t.NAME: t for t in ALL_TOOLS}


def get_tool(name: str):
    return _BY_NAME.get(name)


def requires_confirmation(name: str) -> bool:
    tool = _BY_NAME.get(name)
    return tool is not None and tool.REQUIRES_CONFIRMATION


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.NAME,
                "description": t.DESCRIPTION,
                "parameters": t.PARAMETERS,
            },
        }
        for t in ALL_TOOLS
    ]


def get_preview(name: str, arguments: dict) -> str:
    tool = _BY_NAME.get(name)
    if tool is not None and hasattr(tool, "preview"):
        return tool.preview(arguments)
    return f"{name}({json.dumps(arguments, ensure_ascii=False)})"


def execute(name: str, arguments: dict, context: ToolContext) -> str:
    tool = _BY_NAME.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return tool.run(arguments, context)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
