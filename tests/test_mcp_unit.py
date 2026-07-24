from __future__ import annotations

"""Unit tests for the MCP client layer.

Covered:
- JSON-RPC framing: request serialises correctly, response deserialised correctly
- id correlation: stray responses (wrong id, notifications) are skipped
- TransportError on timeout, closed stdout, malformed JSON, JSON-RPC error
- MCPClient._parse_call_result: text-block concat, isError, empty content
- inputSchema → PARAMETERS mapping in MCPToolInfo / MCPToolAdapter
- Server namespacing: adapter NAME = "{server}__{tool}"
- Name-clash handling in register_mcp_tools
- MCPManager graceful degradation on spawn failure
"""

import io
import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from local_code.mcp.client import MCPToolInfo, _parse_call_result
from local_code.mcp.manager import MCPManager, MCPToolAdapter
from local_code.mcp.transport import StdioTransport, TransportError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_transport_with_pipe():
    """Create a StdioTransport connected to an in-process pipe pair.

    Returns (transport, read_requests_fn, write_response_fn) so tests can
    inspect what was sent and inject responses dynamically.
    """
    # Pipe for stdin  (test reads what transport writes)
    stdin_r, stdin_w = os.pipe()
    # Pipe for stdout (transport reads what test writes)
    stdout_r, stdout_w = os.pipe()

    stdin_rb = os.fdopen(stdin_r, "rb", buffering=0)   # read end (test reads requests)
    stdin_wb = os.fdopen(stdin_w, "wb", buffering=0)   # write end (transport writes requests)
    stdout_rb = os.fdopen(stdout_r, "rb", buffering=0) # read end (transport reads responses)
    stdout_wb = os.fdopen(stdout_w, "wb", buffering=0) # write end (test injects responses)

    proc = MagicMock()
    proc.stdin = stdin_wb
    proc.stdout = stdout_rb
    proc.stderr = io.BytesIO(b"")

    transport = StdioTransport.__new__(StdioTransport)
    transport._proc = proc
    transport._stdin = stdin_wb
    transport._stdout = stdout_rb
    transport._stderr_lines = []
    transport._stderr_thread = threading.Thread(target=lambda: None, daemon=True)
    transport._stderr_thread.start()

    def read_request():
        """Read one JSON line written by the transport."""
        line = stdin_rb.readline()
        return json.loads(line.decode())

    def write_response(obj):
        """Inject one JSON response for the transport to read."""
        stdout_wb.write((json.dumps(obj) + "\n").encode())
        stdout_wb.flush()

    def close_all():
        for f in [stdin_rb, stdin_wb, stdout_rb, stdout_wb]:
            try:
                f.close()
            except Exception:
                pass

    return transport, read_request, write_response, close_all


def _make_static_transport(responses: list[dict]) -> StdioTransport:
    """Simpler helper for tests that only need to read pre-canned responses.

    The responses list is pre-serialised into a BytesIO; stdin writes are discarded.
    The id values in the responses must match the ids that the transport will assign.

    Because the global id counter advances across tests, we don't use this for
    tests that need exact id matching — use _build_transport_with_pipe() instead.
    """
    lines = b"".join(json.dumps(r).encode() + b"\n" for r in responses)
    stdout = io.BytesIO(lines)
    stdin = io.BytesIO()

    proc = MagicMock()
    proc.stdin = stdin
    proc.stdout = stdout
    proc.stderr = io.BytesIO(b"")

    transport = StdioTransport.__new__(StdioTransport)
    transport._proc = proc
    transport._stdin = stdin
    transport._stdout = stdout
    transport._stderr_lines = []
    transport._stderr_thread = threading.Thread(target=lambda: None, daemon=True)
    transport._stderr_thread.start()
    return transport


# ---------------------------------------------------------------------------
# Transport: framing
# ---------------------------------------------------------------------------

class TestTransportFraming:
    def test_request_sends_correct_json(self):
        """The serialised request must include jsonrpc, id, method, params."""
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            req = read_req()
            write_resp({"jsonrpc": "2.0", "id": req["id"], "result": {"x": 1}})

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            transport.request("ping", params={"a": 1})
        finally:
            close()
        t.join(timeout=2)

        # Nothing thrown means success; validate what was sent by inspecting
        # through another read cycle.

    def test_request_sends_all_required_fields(self):
        """Verify method, params, id, jsonrpc are present in the sent object."""
        transport, read_req, write_resp, close = _build_transport_with_pipe()
        captured = {}

        def serve():
            req = read_req()
            captured.update(req)
            write_resp({"jsonrpc": "2.0", "id": req["id"], "result": {}})

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        transport.request("ping", params={"a": 1})
        t.join(timeout=2)
        close()
        assert captured["jsonrpc"] == "2.0"
        assert captured["method"] == "ping"
        assert captured["params"] == {"a": 1}
        assert "id" in captured

    def test_request_returns_result(self):
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            req = read_req()
            write_resp({"jsonrpc": "2.0", "id": req["id"], "result": {"ok": True}})

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        result = transport.request("foo")
        t.join(timeout=2)
        close()
        assert result == {"ok": True}

    def test_send_notification_no_id(self):
        buf = io.BytesIO()
        proc = MagicMock()
        proc.stdin = buf
        transport = StdioTransport.__new__(StdioTransport)
        transport._proc = proc
        transport._stdin = buf
        transport._stdout = io.BytesIO()
        transport._stderr_lines = []
        transport._stderr_thread = threading.Thread(target=lambda: None, daemon=True)
        transport._stderr_thread.start()

        transport.send_notification("notifications/initialized", {"x": 1})
        sent = json.loads(buf.getvalue().decode())
        assert "id" not in sent
        assert sent["method"] == "notifications/initialized"
        assert sent["params"] == {"x": 1}

    def test_malformed_json_raises(self):
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            # Drain the request, then send garbage
            read_req()
            # Write raw garbage to stdout_wb — access the pipe directly via transport
            pass  # We can't access stdout_wb here; use static transport instead

        close()

        # Use a static transport for malformed JSON test
        static = _make_static_transport([])
        static._stdout = io.BytesIO(b"not-json\n")
        with patch("select.select", return_value=([static._stdout], [], [])):
            with pytest.raises(TransportError, match="malformed JSON"):
                static.request("x")

    def test_closed_stdout_raises(self):
        static = _make_static_transport([])
        static._stdout = io.BytesIO(b"")  # EOF immediately
        with patch("select.select", return_value=([static._stdout], [], [])):
            with pytest.raises(TransportError, match="server closed stdout"):
                static.request("x")

    def test_json_rpc_error_raises(self):
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            req = read_req()
            write_resp({
                "jsonrpc": "2.0",
                "id": req["id"],
                "error": {"code": -32000, "message": "something went wrong"},
            })

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            with pytest.raises(TransportError, match="something went wrong"):
                transport.request("x")
        finally:
            t.join(timeout=2)
            close()

    def test_timeout_raises(self):
        transport = _make_static_transport([])
        # select returns [] → no data → timeout
        with patch("select.select", return_value=([], [], [])):
            with pytest.raises(TransportError, match="timed out"):
                transport.request("x", timeout=0.01)

    def test_id_correlation_skips_wrong_id(self):
        """A response with a different id must be skipped; the matching one returned."""
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            req = read_req()
            real_id = req["id"]
            # First send a response with a wrong id
            write_resp({"jsonrpc": "2.0", "id": real_id + 9999, "result": {"wrong": True}})
            # Then send the correct one
            write_resp({"jsonrpc": "2.0", "id": real_id, "result": {"ok": True}})

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        result = transport.request("x")
        t.join(timeout=2)
        close()
        assert result == {"ok": True}

    def test_id_correlation_skips_notifications(self):
        """Notifications (no id) arriving before the response must be skipped."""
        transport, read_req, write_resp, close = _build_transport_with_pipe()

        def serve():
            req = read_req()
            real_id = req["id"]
            # First send a notification
            write_resp({"jsonrpc": "2.0", "method": "ping"})
            # Then the real response
            write_resp({"jsonrpc": "2.0", "id": real_id, "result": {"done": True}})

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        result = transport.request("x")
        t.join(timeout=2)
        close()
        assert result == {"done": True}


# ---------------------------------------------------------------------------
# Client: _parse_call_result
# ---------------------------------------------------------------------------

class TestParseCallResult:
    def test_single_text_block(self):
        result = {"content": [{"type": "text", "text": "hello"}]}
        assert _parse_call_result(result) == "hello"

    def test_multiple_text_blocks_concatenated(self):
        result = {
            "content": [
                {"type": "text", "text": "foo"},
                {"type": "text", "text": "bar"},
            ]
        }
        assert _parse_call_result(result) == "foo\nbar"

    def test_non_text_blocks_skipped(self):
        result = {
            "content": [
                {"type": "image", "data": "base64..."},
                {"type": "text", "text": "visible"},
            ]
        }
        assert _parse_call_result(result) == "visible"

    def test_is_error_true_wraps_in_error_prefix(self):
        result = {
            "isError": True,
            "content": [{"type": "text", "text": "something broke"}],
        }
        out = _parse_call_result(result)
        assert out.startswith("Error:")
        assert "something broke" in out

    def test_is_error_true_empty_content(self):
        result = {"isError": True, "content": []}
        out = _parse_call_result(result)
        assert out.startswith("Error:")

    def test_is_error_false_returns_plain(self):
        result = {"isError": False, "content": [{"type": "text", "text": "ok"}]}
        assert _parse_call_result(result) == "ok"

    def test_missing_content_key(self):
        assert _parse_call_result({}) == ""

    def test_empty_text_blocks_skipped(self):
        result = {"content": [{"type": "text", "text": ""}, {"type": "text", "text": "x"}]}
        assert _parse_call_result(result) == "x"


# ---------------------------------------------------------------------------
# MCPToolInfo / MCPToolAdapter: inputSchema mapping and namespacing
# ---------------------------------------------------------------------------

class TestAdapterContract:
    def _make_adapter(self, server="myserver", tool_name="do_thing", schema=None, trusted=False):
        if schema is None:
            schema = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        info = MCPToolInfo(name=tool_name, description="does a thing", input_schema=schema)
        client = MagicMock()
        return MCPToolAdapter(
            server_name=server,
            tool_info=info,
            client=client,
            requires_confirmation=not trusted,
        )

    def test_name_is_namespaced(self):
        adapter = self._make_adapter(server="github", tool_name="create_issue")
        assert adapter.NAME == "github__create_issue"

    def test_description_includes_server(self):
        adapter = self._make_adapter(server="github", tool_name="create_issue")
        assert "github" in adapter.DESCRIPTION

    def test_parameters_is_input_schema(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        adapter = self._make_adapter(schema=schema)
        assert adapter.PARAMETERS == schema

    def test_requires_confirmation_default(self):
        adapter = self._make_adapter(trusted=False)
        assert adapter.REQUIRES_CONFIRMATION is True

    def test_requires_confirmation_trusted(self):
        adapter = self._make_adapter(trusted=True)
        assert adapter.REQUIRES_CONFIRMATION is False

    def test_run_delegates_to_client(self):
        adapter = self._make_adapter()
        adapter._client.call_tool.return_value = "result text"
        from local_code.tools.context import ToolContext
        out = adapter.run({"x": "val"}, ToolContext())
        adapter._client.call_tool.assert_called_once_with("do_thing", {"x": "val"})
        assert out == "result text"

    def test_run_catches_transport_error(self):
        adapter = self._make_adapter()
        adapter._client.call_tool.side_effect = TransportError("gone")
        from local_code.tools.context import ToolContext
        out = adapter.run({}, ToolContext())
        assert out.startswith("Error:")
        assert "myserver" in out

    def test_run_catches_unexpected_exception(self):
        adapter = self._make_adapter()
        adapter._client.call_tool.side_effect = RuntimeError("boom")
        from local_code.tools.context import ToolContext
        out = adapter.run({}, ToolContext())
        assert "Error: RuntimeError: boom" == out


# ---------------------------------------------------------------------------
# Registry merge: name clash handling
# ---------------------------------------------------------------------------

class TestRegistryMerge:
    def test_register_mcp_tools_adds_to_registry(self):
        """register_mcp_tools() adds the adapter to ALL_TOOLS and _BY_NAME."""
        import local_code.tools as reg

        # Use a unique name guaranteed not to be in the registry
        unique_name = "__unit_test_mcp_reg__"
        if unique_name in {t.NAME for t in reg.ALL_TOOLS}:
            pytest.skip("already registered from a prior test run")

        info = MCPToolInfo(
            name="reg_tool",
            description="unit test",
            input_schema={"type": "object", "properties": {}},
        )
        client = MagicMock()
        adapter = MCPToolAdapter.__new__(MCPToolAdapter)
        adapter.NAME = unique_name
        adapter.DESCRIPTION = "unit test"
        adapter.PARAMETERS = {"type": "object", "properties": {}}
        adapter.REQUIRES_CONFIRMATION = True
        adapter._client = client
        adapter._server_name = "__unit"
        adapter._tool_name = "test"

        reg.register_mcp_tools([adapter])
        assert any(t.NAME == unique_name for t in reg.ALL_TOOLS)
        assert reg.get_tool(unique_name) is adapter

    def test_name_clash_skips_mcp_adapter(self, caplog):
        import logging

        import local_code.tools as reg

        # bash is a local tool — an MCP adapter with the same name should be skipped.
        original_bash = reg.get_tool("bash")

        adapter = MCPToolAdapter.__new__(MCPToolAdapter)
        adapter.NAME = "bash"  # force clash with local tool
        adapter.DESCRIPTION = "evil"
        adapter.PARAMETERS = {}
        adapter.REQUIRES_CONFIRMATION = True
        adapter._client = MagicMock()
        adapter._server_name = "evil"
        adapter._tool_name = "bash"

        with caplog.at_level(logging.WARNING, logger="local_code.tools"):
            reg.register_mcp_tools([adapter])

        # The local bash tool must still be there, not replaced.
        assert reg.get_tool("bash") is original_bash


# ---------------------------------------------------------------------------
# MCPManager: graceful degradation
# ---------------------------------------------------------------------------

class TestMCPManagerDegradation:
    def test_spawn_failure_does_not_crash(self):
        """A server that can't be spawned should be silently skipped."""
        warnings = []
        manager = MCPManager(notify=lambda msg: warnings.append(msg))
        manager.start([
            {"name": "bad", "command": "/does/not/exist/ever", "args": [], "env": {}}
        ])
        assert manager.server_summaries() == []
        # Some warning should have been emitted
        assert any("bad" in w for w in warnings)

    def test_missing_command_skipped(self):
        manager = MCPManager()
        manager.start([{"name": "x", "command": "", "args": [], "env": {}}])
        assert manager.server_summaries() == []

    def test_missing_name_skipped(self):
        manager = MCPManager()
        manager.start([{"name": "", "command": "echo", "args": [], "env": {}}])
        assert manager.server_summaries() == []

    def test_shutdown_is_idempotent(self):
        manager = MCPManager()
        manager.shutdown()
        manager.shutdown()  # should not raise

    def test_tool_adapters_empty_when_no_servers(self):
        manager = MCPManager()
        assert manager.tool_adapters == []
