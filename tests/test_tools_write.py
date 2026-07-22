from __future__ import annotations

from local_code.tools import edit_file, write_file
from local_code.tools.context import ToolContext

CTX = ToolContext()


def test_confirmation_flags():
    assert write_file.REQUIRES_CONFIRMATION is True
    assert edit_file.REQUIRES_CONFIRMATION is True


def test_write_file_creates_dirs(tmp_path):
    target = tmp_path / "deep" / "dir" / "f.txt"
    out = write_file.run({"path": str(target), "content": "hola"}, CTX)
    assert target.read_text() == "hola"
    assert out == f"Wrote 4 chars to {target}"


def test_write_file_preview_truncates():
    p = write_file.preview({"path": "x.txt", "content": "a" * 600})
    assert p.startswith("write_file → x.txt\n---\n")
    assert len(p) < 600


def test_edit_file_replaces_unique(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    out = edit_file.run(
        {"path": str(f), "old_string": "y = 2", "new_string": "y = 99"}, CTX
    )
    assert out == f"Edited {f}"
    assert f.read_text() == "x = 1\ny = 99\n"


def test_edit_file_not_found(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    out = edit_file.run(
        {"path": str(f), "old_string": "zzz", "new_string": "q"}, CTX
    )
    assert out == f"Error: old_string not found in {f}"
    assert f.read_text() == "x = 1\n"


def test_edit_file_ambiguous(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("dup\ndup\n")
    out = edit_file.run({"path": str(f), "old_string": "dup", "new_string": "q"}, CTX)
    assert out == f"Error: old_string appears 2 times in {f}; provide a larger unique string"
    assert f.read_text() == "dup\ndup\n"


def test_edit_file_preview_is_unified_diff(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    p = edit_file.preview(
        {"path": str(f), "old_string": "y = 2", "new_string": "y = 99"}
    )
    assert f"--- a/{f}" in p
    assert f"+++ b/{f}" in p
    assert "-y = 2" in p
    assert "+y = 99" in p


def test_edit_file_preview_unreadable(tmp_path):
    p = edit_file.preview(
        {"path": str(tmp_path / "nope.py"), "old_string": "a", "new_string": "b"}
    )
    assert "cannot read" in p
