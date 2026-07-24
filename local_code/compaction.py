from __future__ import annotations

import json
from collections.abc import Callable

DEFAULT_CONTEXT_WINDOW = 8192
THRESHOLD = 0.7
KEEP_RECENT = 6

SUMMARY_PROMPT = (
    "Resumí la siguiente conversación (JSON) en texto plano, máximo 300 "
    "palabras, preservando: decisiones tomadas, archivos creados o "
    "modificados, comandos ejecutados y tareas pendientes. Respondé SOLO "
    "con el resumen."
)


def estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False)) // 4


def detect_context_window(client, model: str) -> int:
    try:
        info = client.show(model).get("model_info", {})
    except Exception:
        return DEFAULT_CONTEXT_WINDOW
    for key, value in info.items():
        if key.endswith(".context_length"):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return DEFAULT_CONTEXT_WINDOW


class Compactor:
    def __init__(
        self,
        client,
        model: str,
        context_window: int,
        notify: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.model = model
        self.context_window = context_window
        self.notify = notify or (lambda message: None)

    def _summarize(self, old: list[dict]) -> str:
        prompt = SUMMARY_PROMPT + "\n\n" + json.dumps(old, ensure_ascii=False)
        content = ""
        for chunk in self.client.chat(
            self.model, [{"role": "user", "content": prompt}], tools=None
        ):
            content += chunk.get("message", {}).get("content", "")
        return content.strip()

    def maybe_compact(self, session) -> bool:
        before = estimate_tokens(session.messages)
        if before <= int(self.context_window * THRESHOLD):
            return False
        if len(session.history) <= KEEP_RECENT:
            return False
        recent = session.history[-KEEP_RECENT:]
        # A tool result is meaningless without the assistant tool_calls message
        # that produced it, so never start the kept slice on one.
        while recent and recent[0].get("role") == "tool":
            recent = recent[1:]
        old = session.history[: len(session.history) - len(recent)]
        try:
            summary = self._summarize(old)
        except Exception:
            summary = ""
        if summary:
            session.history[:] = [
                {
                    "role": "user",
                    "content": f"[Resumen de conversación previa]\n{summary}",
                }
            ] + recent
            verb = "compacted"
        else:
            session.history[:] = recent
            verb = "truncated"
        after = estimate_tokens(session.messages)
        self.notify(f"context {verb} (~{before}→{after} tokens)")
        return True
