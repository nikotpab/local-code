from __future__ import annotations

"""MCP server manager.

Reads a list of server configs, spawns each subprocess, performs the
initialize handshake, discovers tools, and exposes them as adapter objects
that satisfy the local tool contract:

    NAME, DESCRIPTION, PARAMETERS, REQUIRES_CONFIRMATION, run(arguments, context) -> str

Failures at any stage (spawn, handshake, list) are caught and logged as dim
warnings; the CLI never crashes because of a broken MCP server.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from local_code.mcp.client import MCPClient, MCPToolInfo
from local_code.mcp.transport import StdioTransport, TransportError

if TYPE_CHECKING:
    from local_code.tools.context import ToolContext

SEPARATOR = "__"  # {server_name}__{tool_name}


# ---------------------------------------------------------------------------
# Tool adapter — satisfies the local tool contract
# ---------------------------------------------------------------------------

class MCPToolAdapter:
    """Wraps a single MCP tool so it looks like a local tool module."""

    def __init__(
        self,
        server_name: str,
        tool_info: MCPToolInfo,
        client: MCPClient,
        requires_confirmation: bool,
    ) -> None:
        self.NAME: str = f"{server_name}{SEPARATOR}{tool_info.name}"
        self.DESCRIPTION: str = (
            f"[MCP/{server_name}] {tool_info.description}"
            if tool_info.description
            else f"[MCP/{server_name}] {tool_info.name}"
        )
        self.PARAMETERS: dict = tool_info.input_schema
        self.REQUIRES_CONFIRMATION: bool = requires_confirmation
        self._server_name = server_name
        self._tool_name = tool_info.name
        self._client = client

    def run(self, arguments: dict, context: "ToolContext") -> str:  # noqa: ARG002
        try:
            return self._client.call_tool(self._tool_name, arguments)
        except TransportError as exc:
            return f"Error: MCP server '{self._server_name}' unreachable: {exc}"
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Per-server state
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    name: str
    client: MCPClient
    transport: StdioTransport
    adapters: list[MCPToolAdapter] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MCPManager:
    """Owns all MCP server subprocesses for the lifetime of the CLI."""

    def __init__(self, notify=None) -> None:
        # notify is called with a string to show dim warnings/info to the user.
        self._notify = notify or (lambda msg: None)
        self._servers: list[ServerState] = []

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self, configs: list[dict]) -> None:
        """Spawn each server described in *configs* and perform the handshake.

        Each config dict must have:
            name    str      server name (used as namespace prefix)
            command str      executable
            args    list     command-line arguments
            env     dict     extra environment variables (merged with os.environ)
            trust   bool     if True, tools don't require confirmation

        Failures are swallowed and logged as warnings.
        """
        for cfg in configs:
            self._start_one(cfg)

    def _start_one(self, cfg: dict) -> None:
        name: str = cfg.get("name", "")
        command: str = cfg.get("command", "")
        args: list[str] = cfg.get("args") or []
        env_extra: dict = cfg.get("env") or {}
        trusted: bool = bool(cfg.get("trust", False))

        if not name or not command:
            self._notify(f"[dim]mcp: skipping server with missing name or command[/dim]")
            return

        # Build environment
        env = {**os.environ, **{k: str(v) for k, v in env_extra.items()}}

        # Spawn
        try:
            proc = subprocess.Popen(
                [command] + list(args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except (OSError, FileNotFoundError) as exc:
            self._notify(f"[dim]mcp: failed to spawn '{name}': {exc}[/dim]")
            return

        transport = StdioTransport(proc)
        client = MCPClient(transport)

        # Handshake
        try:
            client.initialize()
        except TransportError as exc:
            self._notify(f"[dim]mcp: '{name}' initialize failed: {exc}[/dim]")
            transport.close()
            return

        # Discover tools
        try:
            tool_infos = client.list_tools()
        except TransportError as exc:
            self._notify(f"[dim]mcp: '{name}' tools/list failed: {exc}[/dim]")
            transport.close()
            return

        adapters: list[MCPToolAdapter] = []
        for info in tool_infos:
            adapter = MCPToolAdapter(
                server_name=name,
                tool_info=info,
                client=client,
                requires_confirmation=not trusted,
            )
            adapters.append(adapter)

        state = ServerState(name=name, client=client, transport=transport, adapters=adapters)
        self._servers.append(state)
        self._notify(f"[dim]mcp: '{name}' connected ({len(adapters)} tools)[/dim]")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def tool_adapters(self) -> list[MCPToolAdapter]:
        """Flat list of all MCPToolAdapter objects across all connected servers."""
        out: list[MCPToolAdapter] = []
        for s in self._servers:
            out.extend(s.adapters)
        return out

    def server_summaries(self) -> list[dict]:
        """List of {name, tool_count} dicts for the /mcp command."""
        return [{"name": s.name, "tool_count": len(s.adapters)} for s in self._servers]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate all server subprocesses."""
        for state in self._servers:
            try:
                state.transport.close()
            except Exception:
                pass
        self._servers.clear()
