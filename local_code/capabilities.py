from __future__ import annotations


class CapabilityDetector:
    def __init__(self, client):
        self._client = client
        self._cache: dict[str, bool] = {}

    def supports_tools(self, model: str) -> bool:
        if model not in self._cache:
            info = self._client.show(model)
            self._cache[model] = "tools" in info.get("capabilities", [])
        return self._cache[model]
