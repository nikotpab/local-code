from __future__ import annotations

"""Stdio JSON-RPC 2.0 transport for MCP servers.

One JSON object per line (newline-delimited), UTF-8.
Reads stderr in a background thread so it never blocks the main path.
"""

import json
import subprocess
import threading
from typing import IO

_ID_COUNTER_LOCK = threading.Lock()
_ID_COUNTER = 0


def _next_id() -> int:
    global _ID_COUNTER
    with _ID_COUNTER_LOCK:
        _ID_COUNTER += 1
        return _ID_COUNTER


class TransportError(Exception):
    """Raised when the transport layer fails (process gone, malformed JSON, etc.)."""


class StdioTransport:
    """Wraps a subprocess and speaks newline-delimited JSON-RPC 2.0 over its stdio.

    stderr is drained in a background daemon thread and stored so callers
    can surface diagnostics without blocking.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._stdin: IO[bytes] = proc.stdin  # type: ignore[assignment]
        self._stdout: IO[bytes] = proc.stdout  # type: ignore[assignment]
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="mcp-stderr"
        )
        self._stderr_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_stderr(self) -> None:
        try:
            for raw in self._proc.stderr:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").rstrip()
                self._stderr_lines.append(line)
        except Exception:
            pass

    def _write(self, obj: dict) -> None:
        """Serialise *obj* and send it as a single newline-terminated line."""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        try:
            self._stdin.write(line.encode("utf-8"))
            self._stdin.flush()
        except OSError as exc:
            raise TransportError(f"write failed: {exc}") from exc

    def _read_line(self) -> dict:
        """Read exactly one newline-terminated JSON object from stdout."""
        try:
            raw = self._stdout.readline()
        except OSError as exc:
            raise TransportError(f"read failed: {exc}") from exc
        if not raw:
            raise TransportError("server closed stdout")
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TransportError(f"malformed JSON from server: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def request(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and return the *result* field of the response.

        Raises ``TransportError`` on timeout, transport failure, or JSON-RPC error.
        """
        req_id = _next_id()
        msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

        # Read responses until we find the one matching our id.
        # A well-behaved server sends exactly one response per request, but we
        # skip stray notifications (messages without an "id") gracefully.
        import select as _select

        deadline_remaining = timeout
        import time
        t0 = time.monotonic()

        while True:
            elapsed = time.monotonic() - t0
            remaining = timeout - elapsed
            if remaining <= 0:
                raise TransportError(f"request '{method}' timed out after {timeout}s")

            # Wait for data to be available on stdout with a timeout.
            try:
                ready, _, _ = _select.select([self._stdout], [], [], remaining)
            except (OSError, ValueError):
                raise TransportError("stdout not selectable; process may have died")
            if not ready:
                raise TransportError(f"request '{method}' timed out after {timeout}s")

            resp = self._read_line()

            # Skip notifications (no id) and responses to other ids.
            resp_id = resp.get("id")
            if resp_id is None:
                continue
            if resp_id != req_id:
                continue

            if "error" in resp:
                err = resp["error"]
                msg_text = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise TransportError(f"JSON-RPC error from server: {msg_text}")

            return resp.get("result", {})

    @property
    def stderr_output(self) -> list[str]:
        return list(self._stderr_lines)

    def close(self) -> None:
        """Close the transport and terminate the subprocess."""
        try:
            self._stdin.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
