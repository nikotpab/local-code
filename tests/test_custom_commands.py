from __future__ import annotations

from local_code.custom_commands import list_custom_commands, load_custom_command


def test_list_empty_when_dir_missing(tmp_path):
    assert list_custom_commands(dir=tmp_path / "nope") == []


def test_list_sorted_names(tmp_path):
    (tmp_path / "b.md").write_text("B")
    (tmp_path / "a.md").write_text("A")
    (tmp_path / "no-md.txt").write_text("x")
    assert list_custom_commands(dir=tmp_path) == ["a", "b"]


def test_load_replaces_arguments(tmp_path):
    (tmp_path / "review.md").write_text("Revisá esto: $ARGUMENTS. Fin $ARGUMENTS")
    out = load_custom_command("review", "src/main.py", dir=tmp_path)
    assert out == "Revisá esto: src/main.py. Fin src/main.py"


def test_load_without_args(tmp_path):
    (tmp_path / "plan.md").write_text("Armá un plan. $ARGUMENTS")
    assert load_custom_command("plan", "", dir=tmp_path) == "Armá un plan. "


def test_load_missing_returns_none(tmp_path):
    assert load_custom_command("nope", "", dir=tmp_path) is None


def test_load_rejects_traversal_names(tmp_path):
    (tmp_path / "ok.md").write_text("x")
    assert load_custom_command("../ok", "", dir=tmp_path) is None
    assert load_custom_command("a/b", "", dir=tmp_path) is None
    assert load_custom_command("", "", dir=tmp_path) is None
