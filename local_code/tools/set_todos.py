from __future__ import annotations

from local_code.tools.context import ToolContext

NAME = "set_todos"
DESCRIPTION = (
    "Set or update the visible task checklist for multi-step work. "
    "Replaces the whole list; include every item with its current status."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done"],
                    },
                },
                "required": ["text", "status"],
            },
        }
    },
    "required": ["todos"],
}
REQUIRES_CONFIRMATION = False
VALID_STATUSES = {"pending", "in_progress", "done"}


def run(arguments: dict, context: ToolContext) -> str:
    todos = arguments.get("todos")
    if not isinstance(todos, list):
        return "Error: todos must be a list"
    for i, item in enumerate(todos):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or item.get("status") not in VALID_STATUSES
        ):
            return (
                f"Error: item {i} must be an object with text (string) and "
                "status (pending|in_progress|done)"
            )
    if context.on_todos is not None:
        context.on_todos(todos)
    return f"Todos updated ({len(todos)} items)"
