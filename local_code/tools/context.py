from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ToolContext:
    bash_timeout: int = 120
    on_todos: Callable[[list], None] | None = None
    spawn: Callable[[str, str | None], str] | None = None
