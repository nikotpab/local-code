from __future__ import annotations


class Session:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt
        self.history: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        base = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )
        return base + self.history

    def add(self, message: dict) -> None:
        self.history.append(message)

    def clear(self) -> None:
        self.history.clear()
