# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--version` flag.
- Real logging: `--verbose`/`-v`, `--debug`, and `--log-file PATH` (console
  logs go to stderr; a log file always captures DEBUG).
- Distinct process exit codes: `3` connection error, `4` model-not-found,
  `130` interrupted, `1` generic, `0` success.
- Piped stdin support: `cat file | local-code "explain this"` (stdin is
  appended to any positional prompt, or used as the prompt on its own).
- `--json` one-shot mode: emits a single JSON result on stdout while
  diagnostics go to stderr — safe to pipe into `jq`.
- `local-code config {show|path|validate}` subcommand (`show` redacts the API
  key).
- `local-code completion {bash|zsh|fish}` shell-completion generator.
- Linting (`ruff`) and type-checking (`mypy`) configuration, plus a GitHub
  Actions CI matrix across Python 3.10–3.13.
- LICENSE (MIT), CONTRIBUTING guide, and this changelog.

### Fixed
- Module docstrings in the MCP and `spawn_agent` modules now precede
  `from __future__ import annotations`, so they are real docstrings.

## [0.1.0]

- Initial release: agentic coding CLI for local models (Ollama and
  OpenAI-compatible backends), native tool-calling and ReAct fallback, TUI,
  MCP support, sessions, checkpoints/undo, hooks, permissions, plan mode,
  subagents, context compaction, and custom commands.
