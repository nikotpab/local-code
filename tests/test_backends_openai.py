from __future__ import annotations

import json

import pytest
import requests

import local_code.backends.openai_compat as oc
from local_code.backends.base import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)
from local_code.backends.openai_compat import OpenAICompatClient


class FakeSSEResponse:
    def __init__(self, status_code=200, events=None, text=""):
        self.status_code = status_code
        self._events = events or []
        self.text = text

    def iter_lines(self):
        for e in self._events:
            yield e if isinstance(e, bytes) else e.encode()

    def close(self):
        pass


def data(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def delta(content=None, tool_calls=None, finish_reason=None) -> dict:
    d = {}
    if content is not None:
        d["content"] = content
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return {"choices": [{"delta": d, "finish_reason": finish_reason}]}


def install_post(monkeypatch, response=None, exc=None):
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(oc.requests, "post", fake_post)
    return calls


def collect(client, **kwargs):
    return list(client.chat("m", [{"role": "user", "content": "hi"}], **kwargs))


def test_text_stream(monkeypatch):
    events = [data(delta(content="ho")), data(delta(content="la")), "data: [DONE]"]
    calls = install_post(monkeypatch, FakeSSEResponse(events=events))
    chunks = collect(OpenAICompatClient("http://x/v1"))
    text = "".join(c["message"].get("content", "") for c in chunks)
    assert text == "hola"
    assert chunks[-1]["done"] is True
    assert calls[0]["url"] == "http://x/v1/chat/completions"
    assert calls[0]["json"]["stream"] is True
    assert "tools" not in calls[0]["json"]


def test_tools_included_when_given(monkeypatch):
    calls = install_post(monkeypatch, FakeSSEResponse(events=["data: [DONE]"]))
    schemas = [{"type": "function", "function": {"name": "t"}}]
    collect(OpenAICompatClient("http://x/v1"), tools=schemas)
    assert calls[0]["json"]["tools"] == schemas


def test_tool_call_arguments_assembled_across_chunks(monkeypatch):
    events = [
        data(delta(tool_calls=[{"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}])),
        data(delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "x.py"}'}}])),
        data(delta(finish_reason="tool_calls")),
        "data: [DONE]",
    ]
    install_post(monkeypatch, FakeSSEResponse(events=events))
    chunks = collect(OpenAICompatClient("http://x/v1"))
    calls = [tc for c in chunks for tc in c["message"].get("tool_calls", [])]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == {"path": "x.py"}


def test_invalid_tool_arguments_become_empty_dict(monkeypatch):
    events = [
        data(delta(tool_calls=[{"index": 0, "function": {"name": "t", "arguments": "{not json"}}])),
        data(delta(finish_reason="tool_calls")),
        "data: [DONE]",
    ]
    install_post(monkeypatch, FakeSSEResponse(events=events))
    chunks = collect(OpenAICompatClient("http://x/v1"))
    calls = [tc for c in chunks for tc in c["message"].get("tool_calls", [])]
    assert calls[0]["function"]["arguments"] == {}


def test_auth_header_present_only_with_key(monkeypatch):
    calls = install_post(monkeypatch, FakeSSEResponse(events=["data: [DONE]"]))
    collect(OpenAICompatClient("http://x/v1", api_key="secret"))
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"

    calls2 = install_post(monkeypatch, FakeSSEResponse(events=["data: [DONE]"]))
    collect(OpenAICompatClient("http://x/v1"))
    assert "Authorization" not in calls2[0].get("headers", {})


def test_env_key_overrides_argument(monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_API_KEY", "from-env")
    calls = install_post(monkeypatch, FakeSSEResponse(events=["data: [DONE]"]))
    collect(OpenAICompatClient("http://x/v1", api_key="from-arg"))
    assert calls[0]["headers"]["Authorization"] == "Bearer from-env"


def test_model_not_found(monkeypatch):
    install_post(monkeypatch, FakeSSEResponse(status_code=404))
    with pytest.raises(ModelNotFoundError, match="m"):
        collect(OpenAICompatClient("http://x/v1"))


def test_connection_error(monkeypatch):
    install_post(monkeypatch, exc=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(OllamaConnectionError, match="http://x/v1"):
        collect(OpenAICompatClient("http://x/v1"))


def test_other_http_error(monkeypatch):
    install_post(monkeypatch, FakeSSEResponse(status_code=500, text="boom"))
    with pytest.raises(OllamaError, match="500"):
        collect(OpenAICompatClient("http://x/v1"))


def test_midstream_failure(monkeypatch):
    class Exploding(FakeSSEResponse):
        def iter_lines(self):
            yield data(delta(content="a")).encode()
            raise requests.exceptions.ChunkedEncodingError("dropped")

    install_post(monkeypatch, Exploding())
    with pytest.raises(OllamaError):
        collect(OpenAICompatClient("http://x/v1"))


def test_show_returns_empty(monkeypatch):
    assert OpenAICompatClient("http://x/v1").show("m") == {}


def test_name_attribute():
    assert OpenAICompatClient.name == "openai"


def test_outbound_plain_messages_pass_through():
    msgs = [
        {"role": "system", "content": "sos util"},
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "hey"},
    ]
    assert OpenAICompatClient._to_openai_messages(msgs) == msgs


def test_outbound_tool_calls_get_id_type_and_string_arguments():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "x.py"}}}
            ],
        }
    ]
    out = OpenAICompatClient._to_openai_messages(msgs)
    call = out[0]["tool_calls"][0]
    assert call["id"] == "call_0"
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert call["function"]["arguments"] == '{"path": "x.py"}'


def test_outbound_tool_result_gets_matching_call_id():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}],
        },
        {"role": "tool", "tool_name": "read_file", "content": "contenido"},
    ]
    out = OpenAICompatClient._to_openai_messages(msgs)
    assert out[1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "contenido",
    }
    assert "tool_name" not in out[1]


def test_outbound_multiple_calls_pair_with_results_in_order():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "a"}}},
                {"function": {"name": "read_file", "arguments": {"path": "b"}}},
            ],
        },
        {"role": "tool", "tool_name": "read_file", "content": "A"},
        {"role": "tool", "tool_name": "read_file", "content": "B"},
    ]
    out = OpenAICompatClient._to_openai_messages(msgs)
    assert [c["id"] for c in out[0]["tool_calls"]] == ["call_0", "call_1"]
    assert out[1]["tool_call_id"] == "call_0" and out[1]["content"] == "A"
    assert out[2]["tool_call_id"] == "call_1" and out[2]["content"] == "B"


def test_outbound_ids_unique_across_multiple_rounds():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]},
        {"role": "tool", "tool_name": "t", "content": "1"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]},
        {"role": "tool", "tool_name": "t", "content": "2"},
    ]
    out = OpenAICompatClient._to_openai_messages(msgs)
    assert out[0]["tool_calls"][0]["id"] == "call_0"
    assert out[1]["tool_call_id"] == "call_0"
    assert out[2]["tool_calls"][0]["id"] == "call_1"
    assert out[3]["tool_call_id"] == "call_1"


def test_chat_sends_translated_messages(monkeypatch):
    calls = install_post(monkeypatch, FakeSSEResponse(events=["data: [DONE]"]))
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": {"k": 1}}}]},
        {"role": "tool", "tool_name": "t", "content": "r"},
    ]
    list(OpenAICompatClient("http://x/v1").chat("m", msgs))
    sent = calls[0]["json"]["messages"]
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == '{"k": 1}'
    assert sent[1]["tool_call_id"] == "call_0"
