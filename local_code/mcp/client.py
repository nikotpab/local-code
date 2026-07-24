from __future__ import annotations

"""MCP client — speaks the Model Context Protocol lifecycle over a StdioTransport.

Responsibilities:
- initialize handshake (send initialize request, receive result, send initialized notification)
- tools/list  → returns list of raw tool dicts from the server
- tools/call  → sends call, parses content blocks, handles isError
"""

from dataclasses import dataclass
from typing import Any

from local_code.mcp.transport import StdioTransport, TransportError

PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "local-code"
CLIENT_VERSION = "1.0.0"


@dataclass
class MCPToolInfo:
    """Raw tool descriptor as returned by the server."""

    name: str
    description: str
    input_schema: dict  # JSON-Schema object for the tool's arguments


class MCPClient:
    """High-level MCP client.  Operates on an already-open StdioTransport."""

    def __init__(self, transport: StdioTransport, timeout: float = 30.0) -> None:
        self._transport = transport
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake.

        Returns the server's capabilities dict.
        Raises TransportError on any failure.
        """
        result = self._transport.request(
            "initialize",
            params={
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
            timeout=self._timeout,
        )
        # After a successful result we must send the notification.
        self._transport.send_notification("notifications/initialized")
        return result

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def list_tools(self) -> list[MCPToolInfo]:
        """Call tools/list and return a list of MCPToolInfo.

        Raises TransportError on any failure.
        """
        result = self._transport.request("tools/list", timeout=self._timeout)
        tools_raw = result.get("tools", [])
        out: list[MCPToolInfo] = []
        for t in tools_raw:
            if not isinstance(t, dict):
                continue
            name = t.get("name", "")
            description = t.get("description", "")
            # inputSchema may be absent for argument-free tools; default to empty object schema.
            input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            out.append(MCPToolInfo(name=name, description=description, input_schema=input_schema))
        return out

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call tools/call with *name* and *arguments*.

        Returns a string result.  If isError is true in the response, the result
        is prefixed with "Error: ".  Raises TransportError on transport failure.
        """
        result = self._transport.request(
            "tools/call",
            params={"name": name, "arguments": arguments},
            timeout=self._timeout,
        )
        return _parse_call_result(result)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_call_result(result: dict) -> str:
    """Turn a tools/call result into a plain string.

    Concatenates all text content blocks.  Non-text blocks are skipped.
    If isError is true, wraps the content in an "Error: ..." string.
    """
    is_error: bool = bool(result.get("isError"))
    content_blocks = result.get("content") or []
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    text = "\n".join(parts)
    if is_error:
        return f"Error: {text}" if text else "Error: tool call failed"
    return text
