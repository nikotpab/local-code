from __future__ import annotations

import json
import logging
import warnings

from local_code.tools import (
    bash,
    edit_file,
    glob,
    grep,
    list_dir,
    multi_edit,
    read_file,
    set_todos,
    spawn_agent,
    web_fetch,
    write_file,
)
from local_code.tools.context import ToolContext

__all__ = [
    "ALL_TOOLS",
    "ToolContext",
    "execute",
    "get_preview",
    "get_tool",
    "register_mcp_tools",
    "requires_confirmation",
    "tool_schemas",
]

logger = logging.getLogger(__name__)

ALL_TOOLS = [
    bash,
    edit_file,
    glob,
    grep,
    list_dir,
    multi_edit,
    read_file,
    set_todos,
    spawn_agent,
    web_fetch,
    write_file,
]

# Mutable registry — local tools first, MCP tools appended after startup.
_BY_NAME: dict[str, object] = {t.NAME: t for t in ALL_TOOLS}


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------

def register_mcp_tools(adapters) -> None:
    """Register a list of MCPToolAdapter objects into the live registry.

    Local tools always win: if an adapter's NAME clashes with an existing
    entry, it is skipped with a warning.  This should never happen in practice
    because MCP tools are namespaced as ``{server}__{tool}``, but a misbehaving
    server could in theory return a name that collides.
    """
    for adapter in adapters:
        name = adapter.NAME
        if name in _BY_NAME:
            logger.warning(
                "mcp: tool name '%s' clashes with an existing tool; skipping MCP adapter",
                name,
            )
            continue
        _BY_NAME[name] = adapter
        ALL_TOOLS.append(adapter)


# ---------------------------------------------------------------------------
# Public API (unchanged contract)
# ---------------------------------------------------------------------------

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
