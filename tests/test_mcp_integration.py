from __future__ import annotations

"""Integration tests: real MCPManager + real StdioTransport + fake MCP server subprocess.

The fake server is ``tests/fake_mcp_server.py``.  Each test spawns it in a
different mode to exercise discovery, routing, graceful degradation, and error
handling end-to-end.
"""

import sys
from pathlib import Path

import pytest

from local_code.mcp.manager import MCPManager
from local_code.tools.context import ToolContext

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


def _cfg(mode: str, trusted: bool = False) -> dict:
    return {
        "name": f"fake_{mode}",
        "command": sys.executable,
        "args": [FAKE_SERVER, mode],
        "env": {},
        "trust": trusted,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_normal_server_discovered(self):
        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            summaries = manager.server_summaries()
            assert len(summaries) == 1
            assert summaries[0]["name"] == "fake_normal"
            assert summaries[0]["tool_count"] == 1
        finally:
            manager.shutdown()

    def test_adapter_name_is_namespaced(self):
        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            adapters = manager.tool_adapters
            assert len(adapters) == 1
            assert adapters[0].NAME == "fake_normal__echo"
        finally:
            manager.shutdown()

    def test_adapter_parameters_from_input_schema(self):
        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            adapter = manager.tool_adapters[0]
            assert adapter.PARAMETERS["type"] == "object"
            assert "message" in adapter.PARAMETERS["properties"]
        finally:
            manager.shutdown()

    def test_adapter_requires_confirmation_by_default(self):
        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            assert manager.tool_adapters[0].REQUIRES_CONFIRMATION is True
        finally:
            manager.shutdown()

    def test_trusted_server_no_confirmation(self):
        manager = MCPManager()
        manager.start([_cfg("normal", trusted=True)])
        try:
            assert manager.tool_adapters[0].REQUIRES_CONFIRMATION is False
        finally:
            manager.shutdown()


# ---------------------------------------------------------------------------
# Routing (tools/call)
# ---------------------------------------------------------------------------

class TestRouting:
    def test_call_returns_text(self):
        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            adapter = manager.tool_adapters[0]
            result = adapter.run({"message": "hello"}, ToolContext())
            assert result == "echo: hello"
        finally:
            manager.shutdown()

    def test_call_multi_blocks_concatenated(self):
        manager = MCPManager()
        manager.start([_cfg("multi_blocks")])
        try:
            adapter = manager.tool_adapters[0]
            result = adapter.run({"message": "x"}, ToolContext())
            assert "block one" in result
            assert "block two" in result
        finally:
            manager.shutdown()

    def test_call_is_error_returns_error_string(self):
        manager = MCPManager()
        manager.start([_cfg("is_error")])
        try:
            adapter = manager.tool_adapters[0]
            result = adapter.run({"message": "x"}, ToolContext())
            assert result.startswith("Error:")
            assert "intentional failure" in result
        finally:
            manager.shutdown()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestDegradation:
    def test_server_dies_early(self):
        """A server that exits immediately should be skipped silently."""
        warnings = []
        manager = MCPManager(notify=lambda msg: warnings.append(msg))
        manager.start([_cfg("die_early")])
        try:
            assert manager.server_summaries() == []
        finally:
            manager.shutdown()

    def test_server_times_out_on_init(self):
        """A server that hangs on initialize should be skipped after timeout."""
        manager = MCPManager()
        # Use a very short timeout to keep the test fast.
        # We bypass the normal start() to inject a short timeout.
        import subprocess
        from local_code.mcp.client import MCPClient
        from local_code.mcp.transport import StdioTransport, TransportError

        proc = subprocess.Popen(
            [sys.executable, FAKE_SERVER, "timeout"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        transport = StdioTransport(proc)
        client = MCPClient(transport, timeout=1.0)
        try:
            with pytest.raises(TransportError, match="timed out"):
                client.initialize()
        finally:
            transport.close()

    def test_server_returns_error_on_init(self):
        """A server that returns a JSON-RPC error on initialize should be skipped."""
        warnings = []
        manager = MCPManager(notify=lambda msg: warnings.append(msg))
        manager.start([_cfg("error_init")])
        try:
            assert manager.server_summaries() == []
            assert any("error_init" in w for w in warnings)
        finally:
            manager.shutdown()

    def test_bad_json_server_skipped(self):
        """A server that sends malformed JSON should be skipped."""
        warnings = []
        manager = MCPManager(notify=lambda msg: warnings.append(msg))
        manager.start([_cfg("bad_json")])
        try:
            assert manager.server_summaries() == []
        finally:
            manager.shutdown()

    def test_multiple_servers_partial_failure(self):
        """One broken server should not prevent the working one from connecting."""
        manager = MCPManager()
        manager.start([_cfg("die_early"), _cfg("normal")])
        try:
            summaries = manager.server_summaries()
            names = [s["name"] for s in summaries]
            assert "fake_normal" in names
            assert "fake_die_early" not in names
        finally:
            manager.shutdown()

    def test_shutdown_terminates_subprocesses(self):
        """After shutdown(), server_summaries() should be empty."""
        manager = MCPManager()
        manager.start([_cfg("normal")])
        assert len(manager.server_summaries()) == 1
        manager.shutdown()
        assert manager.server_summaries() == []


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_adapter_visible_via_registry_execute(self):
        """After register_mcp_tools(), tools.execute() can route to the MCP tool."""
        import local_code.tools as reg

        manager = MCPManager()
        manager.start([_cfg("normal")])
        try:
            adapters = manager.tool_adapters
            # Only register if not already in registry (pytest might re-run)
            names = {t.NAME for t in reg.ALL_TOOLS}
            new_adapters = [a for a in adapters if a.NAME not in names]
            reg.register_mcp_tools(new_adapters)

            if new_adapters:
                name = new_adapters[0].NAME
                result = reg.execute(name, {"message": "world"}, ToolContext())
                assert result == "echo: world"
        finally:
            manager.shutdown()
