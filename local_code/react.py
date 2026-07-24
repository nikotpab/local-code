from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


class ToolCallParseError(Exception):
    pass


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolCallParseError(
                f"Invalid JSON inside <tool_call>: {e}. Content was: {raw[:200]}"
            ) from e
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            raise ToolCallParseError(
                f'<tool_call> must contain an object with a string "name". Got: {raw[:200]}'
            )
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ToolCallParseError(
                f'"arguments" must be a JSON object. Got: {raw[:200]}'
            )
        calls.append(ParsedToolCall(name=data["name"], arguments=arguments))
    return calls


def _object_to_call(obj: object) -> ParsedToolCall | None:
    """Turn a decoded JSON value into a ParsedToolCall if it looks like a tool
    call (`{"name": str, "arguments": dict?}`), else None."""
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    arguments = obj.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return ParsedToolCall(name=obj["name"], arguments=arguments)


def find_tool_calls_in_text(
    text: str, is_tool: Callable[[str], bool]
) -> list[ParsedToolCall]:
    """Best-effort recovery of tool calls a model leaked into plain text.

    Some local models emit a tool call as text — either wrapped in
    ``<tool_call>`` tags or as a bare ``{"name": ..., "arguments": ...}`` JSON
    object — instead of the structured tool_calls array, even in native mode.
    This harvests those, but only when the name is a real registered tool
    (`is_tool(name)`), so ordinary prose that merely mentions JSON is never
    mistaken for a call.
    """
    # 1) Tagged blocks first (the format we ask ReAct models to use).
    tagged: list[ParsedToolCall] = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            call = _object_to_call(json.loads(raw))
        except json.JSONDecodeError:
            call = None
        if call is not None and is_tool(call.name):
            tagged.append(call)
    if tagged:
        return tagged

    # 2) Bare JSON objects anywhere in the text.
    decoder = json.JSONDecoder()
    calls: list[ParsedToolCall] = []
    idx = 0
    while True:
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        call = _object_to_call(obj)
        if call is not None and is_tool(call.name):
            calls.append(call)
        idx = max(end, brace + 1)
    return calls


def format_observation(result: str) -> str:
    return f"Observation: {result}"


def build_system_prompt(base: str, schemas: list[dict]) -> str:
    lines = []
    for s in schemas:
        fn = s["function"]
        lines.append(
            f"- {fn['name']}: {fn['description']} Parameters (JSON schema): {json.dumps(fn['parameters'])}"
        )
    tool_block = "\n".join(lines)
    return f"""{base}

You have access to the following tools:

{tool_block}

To call a tool, output exactly this format (valid JSON inside the tags):
<tool_call>{{"name": "<tool_name>", "arguments": {{}}}}</tool_call>

You may emit multiple <tool_call> blocks in one response; they run in order.
After your tool calls you will receive one line per call starting with
"Observation:" containing the result. Never write "Observation:" lines
yourself. When you have enough information, answer the user directly WITHOUT
any <tool_call> block. Only use the tools listed above. Arguments must be
valid JSON."""
