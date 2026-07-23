from __future__ import annotations

import json
from pathlib import Path

from local_code.backends.base import OllamaError

CAPABILITIES_CACHE = Path.home() / ".local-code" / "capabilities.json"

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "probe",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class CapabilityDetector:
    def __init__(self, client, cache_path: Path | None = None):
        self._client = client
        self._cache_path = cache_path or CAPABILITIES_CACHE
        self._cache: dict[str, bool] = {}

    def _backend_name(self) -> str:
        return getattr(self._client, "name", "ollama")

    def _key(self, model: str) -> str:
        return f"{self._backend_name()}:{model}"

    def _read_disk(self) -> dict:
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_disk(self, key: str, value: bool) -> None:
        data = self._read_disk()
        data[key] = value
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def _probe(self, model: str) -> bool:
        try:
            stream = self._client.chat(
                model, [{"role": "user", "content": "hi"}], tools=[PROBE_TOOL]
            )
            for _ in stream:
                break
            close = getattr(stream, "close", None)
            if close is not None:
                close()
            return True
        except Exception:
            return False

    def _detect(self, model: str) -> bool:
        if self._backend_name() == "ollama":
            try:
                info = self._client.show(model)
            except OllamaError:
                return False
            return "tools" in info.get("capabilities", [])
        return self._probe(model)

    def supports_tools(self, model: str) -> bool:
        key = self._key(model)
        if key in self._cache:
            return self._cache[key]
        # The disk cache exists to spare backends with no /api/show (the
        # probe path) a real chat round-trip on every run. ollama's show()
        # is already a cheap, local, authoritative call, so it is always
        # re-verified rather than trusted from disk; its result is still
        # written to disk below for completeness.
        if self._backend_name() != "ollama":
            disk = self._read_disk()
            if isinstance(disk.get(key), bool):
                self._cache[key] = disk[key]
                return disk[key]
        result = self._detect(model)
        self._cache[key] = result
        self._write_disk(key, result)
        return result
