"""spawn_agent tool — delegates a focused sub-task to a fresh read-only agent.

The subagent investigates with read-only tools and returns a text report.
It cannot spawn further subagents (recursion guard: ToolContext.spawn is None).
Write-capable subagents are outside the scope of v1.
"""

from __future__ import annotations

from local_code.tools.context import ToolContext

NAME = "spawn_agent"
DESCRIPTION = (
    "Delegate a focused investigation task to a fresh read-only subagent. "
    "The subagent uses read-only tools (read_file, list_dir, glob, grep) to "
    "research the task and returns a detailed report. It cannot modify files, "
    "run commands, or spawn further subagents. Use this to research a question "
    "in depth without cluttering the main conversation."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The investigation task for the subagent to perform.",
        },
        "model": {
            "type": "string",
            "description": (
                "Optional model name for the subagent. "
                "Defaults to the parent model when omitted."
            ),
        },
    },
    "required": ["task"],
}
REQUIRES_CONFIRMATION = False


def run(arguments: dict, context: ToolContext) -> str:
    task: str = arguments.get("task", "")
    model: str | None = arguments.get("model") or None

    if not task.strip():
        return "Error: spawn_agent requires a non-empty 'task' argument."

    if context.spawn is None:
        return "Error: subagents are not available in this context."

    return context.spawn(task, model)
