from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolContext:
    bash_timeout: int = 120
    on_todos: Callable[[list], None] | None = None
