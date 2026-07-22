from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path.home() / ".local-code" / "sessions"


class SessionNotFoundError(Exception):
    pass


class SessionStore:
    def __init__(self, dir: Path | None = None):
        self.dir = dir or SESSIONS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(
        self,
        session_id: str,
        model: str,
        system_prompt: str | None,
        history: list[dict],
    ) -> None:
        path = self._path(session_id)
        now = datetime.now().isoformat(timespec="seconds")
        created_at = now
        cwd = str(Path.cwd())
        if path.exists():
            try:
                old = json.loads(path.read_text())
                created_at = old.get("created_at", now)
                cwd = old.get("cwd", cwd)
            except (OSError, json.JSONDecodeError):
                pass
        data = {
            "id": session_id,
            "created_at": created_at,
            "updated_at": now,
            "model": model,
            "cwd": cwd,
            "system_prompt": system_prompt,
            "history": history,
        }
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        os.replace(tmp, path)

    def load(self, session_id: str) -> dict:
        path = self._path(session_id)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError as e:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found in {self.dir}"
            ) from e
        except (OSError, json.JSONDecodeError) as e:
            raise SessionNotFoundError(
                f"Session '{session_id}' is unreadable: {e}"
            ) from e

    def latest_id(self) -> str | None:
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return files[-1].stem if files else None

    def list_sessions(self, limit: int = 10) -> list[dict]:
        files = sorted(
            self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        out: list[dict] = []
        for f in files:
            if len(out) >= limit:
                break
            try:
                data = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            first = ""
            for m in data.get("history", []):
                if m.get("role") == "user":
                    first = str(m.get("content", ""))[:60]
                    break
            out.append(
                {
                    "id": data.get("id", f.stem),
                    "updated_at": data.get("updated_at", ""),
                    "model": data.get("model", ""),
                    "first_message": first,
                }
            )
        return out
