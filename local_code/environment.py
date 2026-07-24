from __future__ import annotations

import getpass
import os
import platform
from datetime import datetime
from pathlib import Path

MAX_HOME_ENTRIES = 40


def git_repo_root(start: Path) -> Path | None:
    try:
        current = start.resolve()
    except OSError:
        return None
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    return None


def home_directories(home: Path) -> list[str]:
    try:
        entries = sorted(
            p.name
            for p in home.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    except OSError:
        return []
    if len(entries) > MAX_HOME_ENTRIES:
        return entries[:MAX_HOME_ENTRIES] + ["…"]
    return entries


def _os_label() -> str:
    try:
        system = platform.system()
        if system == "Darwin":
            version = platform.mac_ver()[0] or platform.release()
            return f"macOS {version} (Darwin)"
        if system == "Linux":
            return f"Linux {platform.release()}"
        if system == "Windows":
            return f"Windows {platform.release()}"
        return platform.platform()
    except Exception:
        return "unknown"


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")


def environment_block(cwd: Path | None = None, home: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    home = home or Path.home()

    repo = git_repo_root(cwd)
    git_line = str(repo) if repo is not None else "not a git repository"
    try:
        today = datetime.now().strftime("%Y-%m-%d")
    except Exception:
        today = "unknown"

    lines = [
        "# Environment",
        "",
        f"- Operating system: {_os_label()}",
        f"- Home directory: {home}",
        f"- Working directory: {cwd}",
        f"- User: {_user()}",
        f"- Shell: {os.environ.get('SHELL', 'unknown')}",
        f"- Today: {today}",
        f"- Git repository: {git_line}",
    ]

    folders = home_directories(home)
    if folders:
        lines.append("")
        lines.append(
            "Your home directory contains these folders (use the exact names "
            "shown; do not translate or guess folder names):"
        )
        lines.append(", ".join(folders))

    return "\n".join(lines)
