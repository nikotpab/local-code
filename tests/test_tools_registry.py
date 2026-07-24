from __future__ import annotations

from local_code import tools
from local_code.tools import ToolContext

CTX = ToolContext()

LOCAL_TOOL_NAMES = {
    "bash",
    "edit_file",
    "glob",
    "grep",
    "list_dir",
    "multi_edit",
    "read_file",
    "set_todos",
    "web_fetch",
    "write_file",
}


def test_all_tools_registered():
    names = {t.NAME for t in tools.ALL_TOOLS}
    # All 10 built-in tools must be present; MCP adapters may also be registered.
    assert LOCAL_TOOL_NAMES.issubset(names)


def test_get_tool():
    assert tools.get_tool("read_file").NAME == "read_file"
    assert tools.get_tool("nope") is None


def test_requires_confirmation():
    assert tools.requires_confirmation("bash") is True
    assert tools.requires_confirmation("read_file") is False
    assert tools.requires_confirmation("nope") is False
    assert tools.requires_confirmation("multi_edit") is True
    assert tools.requires_confirmation("web_fetch") is True
    assert tools.requires_confirmation("set_todos") is False


def test_tool_schemas_ollama_format():
    schemas = tools.tool_schemas()
    # Must have at least the 10 built-in tools; MCP adapters may add more.
    assert len(schemas) >= 10
    local_schemas = [s for s in schemas if s["function"]["name"] in LOCAL_TOOL_NAMES]
    assert len(local_schemas) == 10
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"


def test_execute_success(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("contenido")
    assert tools.execute("read_file", {"path": str(f)}, CTX) == "contenido"


def test_execute_unknown_tool():
    assert tools.execute("nope", {}, CTX) == "Error: unknown tool 'nope'"


def test_execute_captures_exceptions(tmp_path):
    out = tools.execute("read_file", {"path": str(tmp_path / "nope.txt")}, CTX)
    assert out.startswith("Error: FileNotFoundError:")


def test_execute_captures_missing_argument():
    out = tools.execute("read_file", {}, CTX)
    assert out.startswith("Error: KeyError:")


def test_get_preview_uses_tool_preview():
    assert tools.get_preview("bash", {"command": "ls"}) == "$ ls"


def test_get_preview_fallback_for_tools_without_preview():
    assert tools.get_preview("read_file", {"path": "x"}) == 'read_file({"path": "x"})'


def test_get_preview_unknown_tool():
    assert tools.get_preview("nope", {"a": 1}) == 'nope({"a": 1})'
