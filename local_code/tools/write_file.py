from __future__ import annotations

from pathlib import Path

from local_code.tools.context import ToolContext

NAME = "write_file"
DESCRIPTION = "Write content to a file, creating parent directories. Overwrites existing files."
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Full file content"},
    },
    "required": ["path", "content"],
}
REQUIRES_CONFIRMATION = True


def run(arguments: dict, context: ToolContext) -> str:
    p = Path(arguments["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(arguments["content"])
    return f"Wrote {len(arguments['content'])} chars to {arguments['path']}"


def suspicious_new_home_dir(path: str, home: Path | None = None) -> str | None:
    """Return the offending directory if writing *path* would create a brand-new
    top-level folder under the home directory, else None.

    Catches the model inventing a localized path (e.g. ~/Escritorio) that does
    not exist, instead of the real folder (~/Desktop).
    """
    home = home or Path.home()
    try:
        target = Path(path).resolve()
        home = home.resolve()
    except OSError:
        return None
    try:
        rel = target.relative_to(home)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        # Writing directly into home (~/file.txt) — no new folder created.
        return None
    first = home / parts[0]
    if first.is_dir():
        return None
    return str(first)


def preview(arguments: dict, home: Path | None = None) -> str:
    head = arguments["content"][:500]
    text = f"write_file → {arguments['path']}\n---\n{head}"
    flagged = suspicious_new_home_dir(arguments["path"], home=home)
    if flagged is not None:
        text += (
            f"\n⚠ Esto crearía una carpeta nueva que no existe: {flagged}. "
            "¿Es la ruta correcta?"
        )
    return text
