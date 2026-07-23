from __future__ import annotations

import itertools
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

CHECKPOINTS_DIR = Path.home() / ".local-code" / "checkpoints"
MUTATING_TOOLS = {"write_file", "edit_file", "multi_edit"}
MAX_CHECKPOINTS = 200


@dataclass
class Checkpoint:
    id: str
    path: str
    backup: str | None
    tool: str
    created_at: str


class CheckpointStore:
    def __init__(self, dir: Path | None = None):
        self.dir = dir or CHECKPOINTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index: list[Checkpoint] = self._load_index()
        self._seq = itertools.count()

    @property
    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load_index(self) -> list[Checkpoint]:
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if isinstance(item, dict):
                try:
                    out.append(Checkpoint(**item))
                except TypeError:
                    continue
        return out

    def _save_index(self) -> None:
        tmp = self._index_path.parent / (self._index_path.name + ".tmp")
        tmp.write_text(
            json.dumps([asdict(c) for c in self._index], indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._index_path)

    def _trim(self) -> None:
        while len(self._index) > MAX_CHECKPOINTS:
            old = self._index.pop(0)
            if old.backup:
                Path(old.backup).unlink(missing_ok=True)

    def snapshot(self, tool_name: str, arguments: dict) -> Checkpoint | None:
        if tool_name not in MUTATING_TOOLS:
            return None
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        target = Path(raw_path).resolve()
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S") + "-"
        backup: str | None = None
        try:
            if target.is_file():
                # Reserve the backup file atomically so a second snapshot in the
                # same wall-clock second can never overwrite this one's copy —
                # the OS guarantees the name is unique. The id is that name's
                # stem so `{id}.bak` still locates the backup.
                fd, backup_path = tempfile.mkstemp(
                    prefix=prefix, suffix=".bak", dir=self.dir
                )
                os.close(fd)
                shutil.copy2(target, backup_path)
                backup = backup_path
                cp_id = Path(backup_path).stem
            else:
                cp_id = f"{prefix}{secrets.token_hex(4)}-{next(self._seq)}"
            checkpoint = Checkpoint(
                id=cp_id,
                path=str(target),
                backup=backup,
                tool=tool_name,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            self._index.append(checkpoint)
            self._trim()
            self._save_index()
        except OSError:
            return None
        return checkpoint

    def undo_last(self) -> str:
        if not self._index:
            return "No hay nada para deshacer."
        checkpoint = self._index[-1]
        target = Path(checkpoint.path)
        try:
            if checkpoint.backup is None:
                target.unlink(missing_ok=True)
                message = f"Deshecho: {checkpoint.path} eliminado (era un archivo nuevo)."
            else:
                shutil.copy2(checkpoint.backup, target)
                Path(checkpoint.backup).unlink(missing_ok=True)
                message = f"Deshecho: {checkpoint.path} restaurado."
        except OSError as e:
            return f"No se pudo deshacer {checkpoint.path}: {e}"
        self._index.pop()
        try:
            self._save_index()
        except OSError:
            pass
        return message

    def list_recent(self, limit: int = 10) -> list[Checkpoint]:
        return list(reversed(self._index))[:limit]
