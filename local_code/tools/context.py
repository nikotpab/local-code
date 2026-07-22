from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolContext:
    bash_timeout: int = 120
