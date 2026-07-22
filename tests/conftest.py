from __future__ import annotations


class FakeClient:
    """Scripted Ollama client. Each chat() call pops the next chunk list."""

    def __init__(self, scripted: list[list[dict]]):
        self.scripted = list(scripted)
        self.calls: list[tuple[str, list[dict], list[dict] | None]] = []

    def chat(self, model, messages, tools=None):
        self.calls.append((model, [dict(m) for m in messages], tools))
        return iter(self.scripted.pop(0))


def text_chunks(text: str) -> list[dict]:
    return [
        {"message": {"role": "assistant", "content": text}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]


def tool_call_chunks(name: str, arguments: dict, content: str = "") -> list[dict]:
    return [
        {
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            },
            "done": False,
        },
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
