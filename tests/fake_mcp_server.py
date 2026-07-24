#!/usr/bin/env python3
"""Fake MCP server for integration tests.

Reads JSON-RPC requests from stdin line-by-line and responds on stdout.
Behaviour is selected by the first command-line argument:

  normal       — responds correctly to initialize, tools/list, tools/call
  timeout      — never responds (hangs)
  error_init   — returns a JSON-RPC error on initialize
  bad_json     — sends malformed JSON on first message
  die_early    — exits immediately (simulates crash)
  is_error     — responds to tools/call with isError=true
  multi_blocks — responds to tools/call with multiple text content blocks
  trusted      — same as normal but the manager test uses trust=true
"""
from __future__ import annotations

import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "Echoes the input",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    }
]


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def send_error(req_id, message: str) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": message}})


def send_result(req_id, result: dict) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if mode == "die_early":
        sys.exit(1)

    if mode == "timeout":
        time.sleep(9999)
        return

    if mode == "bad_json":
        sys.stdout.write("this is not json at all\n")
        sys.stdout.flush()
        time.sleep(9999)
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")

        # Notifications have no id — ignore silently
        if req_id is None:
            continue

        if mode == "error_init" and method == "initialize":
            send_error(req_id, "server refused to initialize")
            continue

        if method == "initialize":
            send_result(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
            })

        elif method == "tools/list":
            send_result(req_id, {"tools": TOOLS})

        elif method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if mode == "is_error":
                send_result(req_id, {
                    "isError": True,
                    "content": [{"type": "text", "text": "intentional failure"}],
                })
            elif mode == "multi_blocks":
                send_result(req_id, {
                    "content": [
                        {"type": "text", "text": "block one"},
                        {"type": "text", "text": "block two"},
                    ]
                })
            else:
                # normal/trusted: echo the message argument
                msg = args.get("message", "(no message)")
                send_result(req_id, {
                    "content": [{"type": "text", "text": f"echo: {msg}"}]
                })

        else:
            send_error(req_id, f"unknown method: {method}")


if __name__ == "__main__":
    main()
