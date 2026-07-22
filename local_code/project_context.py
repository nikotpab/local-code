from __future__ import annotations

from pathlib import Path

CONTEXT_FILENAMES = ("LOCALCODE.md", "AGENTS.md")
MAX_CONTEXT_CHARS = 20_000


def load_project_context(cwd: Path | None = None) -> tuple[str, str] | None:
    base = cwd or Path.cwd()
    for name in CONTEXT_FILENAMES:
        path = base / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if len(content) > MAX_CONTEXT_CHARS:
            content = content[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
        return name, content
    return None
