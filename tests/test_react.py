from __future__ import annotations

import pytest

from local_code.react import (
    ParsedToolCall,
    ToolCallParseError,
    build_system_prompt,
    format_observation,
    parse_tool_calls,
)


def test_parse_single_call():
    text = 'Voy a leer el archivo.\n<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [ParsedToolCall(name="read_file", arguments={"path": "x.py"})]


def test_parse_multiple_calls_in_order():
    text = (
        '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>\n'
        'y después\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["glob", "read_file"]


def test_parse_no_calls_returns_empty():
    assert parse_tool_calls("Listo, el bug era un off-by-one.") == []


def test_parse_missing_arguments_defaults_empty():
    calls = parse_tool_calls('<tool_call>{"name": "list_dir"}</tool_call>')
    assert calls == [ParsedToolCall(name="list_dir", arguments={})]


def test_parse_invalid_json_raises():
    with pytest.raises(ToolCallParseError, match="Invalid JSON"):
        parse_tool_calls('<tool_call>{"name": broken}</tool_call>')


def test_parse_missing_name_raises():
    with pytest.raises(ToolCallParseError):
        parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')


def test_parse_non_dict_arguments_raises():
    with pytest.raises(ToolCallParseError):
        parse_tool_calls('<tool_call>{"name": "x", "arguments": [1]}</tool_call>')


def test_parse_multiline_json():
    text = '<tool_call>\n{"name": "read_file",\n "arguments": {"path": "x"}}\n</tool_call>'
    assert parse_tool_calls(text)[0].name == "read_file"


def test_format_observation():
    assert format_observation("contenido") == "Observation: contenido"


def test_build_system_prompt_lists_tools_and_format():
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    prompt = build_system_prompt("Sos un agente.", schemas)
    assert prompt.startswith("Sos un agente.")
    assert "read_file: Read a file." in prompt
    assert "<tool_call>" in prompt and "</tool_call>" in prompt
    assert "Observation:" in prompt


def _is(name):
    return name in {"write_file", "read_file", "bash"}


def test_find_tool_calls_bare_json():
    from local_code.react import find_tool_calls_in_text

    text = '{"name": "write_file", "arguments": {"path": "x", "content": "y"}}'
    calls = find_tool_calls_in_text(text, _is)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "x", "content": "y"}


def test_find_tool_calls_with_surrounding_prose():
    from local_code.react import find_tool_calls_in_text

    text = 'Voy a hacerlo: {"name": "read_file", "arguments": {"path": "a.py"}} listo.'
    calls = find_tool_calls_in_text(text, _is)
    assert [c.name for c in calls] == ["read_file"]


def test_find_tool_calls_ignores_unknown_names():
    from local_code.react import find_tool_calls_in_text

    text = '{"name": "not_a_tool", "arguments": {}}'
    assert find_tool_calls_in_text(text, _is) == []


def test_find_tool_calls_prefers_tagged():
    from local_code.react import find_tool_calls_in_text

    text = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    calls = find_tool_calls_in_text(text, _is)
    assert calls[0].name == "bash"


def test_find_tool_calls_empty_on_prose():
    from local_code.react import find_tool_calls_in_text

    assert find_tool_calls_in_text("solo texto sin json", _is) == []
