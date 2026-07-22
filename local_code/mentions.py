from __future__ import annotations

import re
from pathlib import Path

MAX_FILE_CHARS = 50_000
MENTION_RE = re.compile(r"@([^\s@]+)")
_TRAILING_PUNCT = ",.;:!?)]}"


def expand_file_mentions(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for raw in MENTION_RE.findall(text):
        token = raw.rstrip(_TRAILING_PUNCT)
        if not token or token in seen:
            continue
        seen.add(token)
        path = Path(token)
        if not path.is_file():
            warnings.append(f"@{token} no encontrado")
            continue
        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            warnings.append(f"@{token} no se pudo leer")
            continue
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n...[truncated]"
        blocks.append(f"```{token}\n{content}\n```")
    if not blocks:
        return text, warnings
    return text + "\n\n" + "\n\n".join(blocks), warnings
