from __future__ import annotations

import json
import os
from typing import Iterator

import requests

from local_code.backends.base import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)

API_KEY_ENV = "LOCAL_CODE_API_KEY"


class OpenAICompatClient:
    name = "openai"

    def __init__(
        self,
        host: str = "http://localhost:1234/v1",
        timeout: float = 300.0,
        api_key: str | None = None,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict:
        key = os.environ.get(API_KEY_ENV) or self.api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    @staticmethod
    def _to_openai_messages(messages: list[dict]) -> list[dict]:
        """Translate our Ollama-shaped history back into OpenAI wire format.

        The agent stores assistant tool calls with dict arguments and no id,
        and tool results keyed by tool_name. Strict OpenAI-compatible servers
        (e.g. vLLM) reject that, so on the way out we serialize arguments to a
        JSON string, add the required id/type, and pair each tool result with
        its call id in order.
        """
        out: list[dict] = []
        pending_ids: list[str] = []
        counter = 0
        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                calls = []
                for call in msg["tool_calls"]:
                    fn = call.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, (dict, list)):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = "{}"
                    call_id = f"call_{counter}"
                    counter += 1
                    pending_ids.append(call_id)
                    calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn.get("name", ""), "arguments": args},
                        }
                    )
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": calls,
                    }
                )
            elif role == "tool":
                if pending_ids:
                    call_id = pending_ids.pop(0)
                else:
                    call_id = f"call_{counter}"
                    counter += 1
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": msg.get("content", ""),
                    }
                )
            else:
                out.append(msg)
        return out

    def chat(
        self, model: str, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[dict]:
        payload: dict = {
            "model": model,
            "messages": self._to_openai_messages(messages),
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools
        try:
            resp = requests.post(
                f"{self.host}/chat/completions",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot reach the model server at {self.host}. Is it running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaConnectionError(
                f"Timed out talking to the model server at {self.host}."
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Request to the model server failed: {e}") from e
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model}' not found at {self.host}."
            )
        if resp.status_code != 200:
            raise OllamaError(f"Model server error {resp.status_code}: {resp.text}")
        yield from self._stream(resp)

    def _stream(self, resp) -> Iterator[dict]:
        pending: dict[int, dict] = {}
        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError as e:
                    raise OllamaError(f"Invalid JSON in stream: {e}") from e
                if "error" in event:
                    raise OllamaError(f"Model server stream error: {event['error']}")
                choices = event.get("choices") or [{}]
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield {
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                    }
                for call in delta.get("tool_calls") or []:
                    slot = pending.setdefault(
                        call.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason") and pending:
                    yield self._flush(pending)
                    pending = {}
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Stream interrupted: {e}") from e
        finally:
            resp.close()
        if pending:
            yield self._flush(pending)
        yield {"message": {"role": "assistant", "content": ""}, "done": True}

    @staticmethod
    def _flush(pending: dict[int, dict]) -> dict:
        calls = []
        for _, slot in sorted(pending.items()):
            try:
                arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                {"function": {"name": slot["name"], "arguments": arguments}}
            )
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": calls},
            "done": False,
        }

    def show(self, model: str) -> dict:
        return {}
