from __future__ import annotations

from local_code.tools import multi_edit
from local_code.tools.context import ToolContext

CTX = ToolContext()


def test_applies_edits_in_order(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\nz = 3\n")
    out = multi_edit.run(
        {
            "path": str(f),
            "edits": [
                {"old_string": "x = 1", "new_string": "x = 10"},
                {"old_string": "z = 3", "new_string": "z = 30"},
            ],
        },
        CTX,
    )
    assert out == f"Applied 2 edits to {f}"
    assert f.read_text() == "x = 10\ny = 2\nz = 30\n"


def test_later_edit_sees_earlier_result(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("aaa\n")
    out = multi_edit.run(
        {
            "path": str(f),
            "edits": [
                {"old_string": "aaa", "new_string": "bbb"},
                {"old_string": "bbb", "new_string": "ccc"},
            ],
        },
        CTX,
    )
    assert out.startswith("Applied 2 edits")
    assert f.read_text() == "ccc\n"


def test_atomic_on_failure(tmp_path):
    f = tmp_path / "a.py"
    original = "x = 1\ny = 2\n"
    f.write_text(original)
    out = multi_edit.run(
        {
            "path": str(f),
            "edits": [
                {"old_string": "x = 1", "new_string": "x = 10"},
                {"old_string": "no existe", "new_string": "q"},
            ],
        },
        CTX,
    )
    assert out == f"Error: edit 1: old_string not found in {f}"
    assert f.read_text() == original


def test_ambiguous(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("dup\ndup\n")
    out = multi_edit.run(
        {"path": str(f), "edits": [{"old_string": "dup", "new_string": "q"}]}, CTX
    )
    assert out == f"Error: edit 0: old_string appears 2 times in {f}"
    assert f.read_text() == "dup\ndup\n"


def test_empty_edits(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    out = multi_edit.run({"path": str(f), "edits": []}, CTX)
    assert out == "Error: edits must be a non-empty list"


def test_preview_unified_diff(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    p = multi_edit.preview(
        {"path": str(f), "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]}
    )
    assert f"--- a/{f}" in p
    assert "-x = 1" in p
    assert "+x = 2" in p


def test_preview_with_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    p = multi_edit.preview(
        {"path": str(f), "edits": [{"old_string": "nope", "new_string": "q"}]}
    )
    assert p.startswith(f"multi_edit → {f} (error:")


def test_contract():
    assert multi_edit.NAME == "multi_edit"
    assert multi_edit.REQUIRES_CONFIRMATION is True
