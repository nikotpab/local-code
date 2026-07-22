from __future__ import annotations

import subprocess

from local_code.tools.context import ToolContext

NAME = "bash"
DESCRIPTION = "Run a shell command and return its exit code, stdout and stderr."
PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"}
    },
    "required": ["command"],
}
REQUIRES_CONFIRMATION = True
MAX_OUTPUT_CHARS = 10_000


def run(arguments: dict, context: ToolContext) -> str:
    try:
        proc = subprocess.run(
            arguments["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=context.bash_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {context.bash_timeout}s"
    out = proc.stdout[:MAX_OUTPUT_CHARS]
    err = proc.stderr[:MAX_OUTPUT_CHARS]
    return f"exit code: {proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"


def preview(arguments: dict) -> str:
    return f"$ {arguments['command']}"
