from __future__ import annotations

import json
import re
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
