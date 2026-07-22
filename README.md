# local-code

Agentic coding CLI — Claude Code style — for **any** local model served by
[Ollama](https://ollama.com). Models with native tool-calling support use it
directly; models without it work through a ReAct-style prompt fallback, so
even older models can drive the agent. Responses render as markdown live in
the terminal.

## Install

```bash
pipx install -e .        # or: pip install -e .
```

Requires Python ≥ 3.10 and a running Ollama (`ollama serve`).

## Usage

```bash
local-code                             # interactive REPL with the default model
local-code --model qwen2.5-coder       # pick a model
local-code --yolo                      # skip all confirmations (careful)
local-code --system "sos experto en Rust"
local-code --resume                    # resume the latest session
local-code --resume 20260722-153045    # resume a specific session
local-code "arregla el bug en x.py"    # one-shot, no REPL
```

REPL commands: `/help` · `/tools` (tool table) · `/history` (current session
summary) · `/sessions` (saved sessions) · `/clear` (reset history, starts a
new session) · `/model <name>` (hot-switch model, re-detects tool support) ·
`/exit`.

## Project context

If a `LOCALCODE.md` (or, as fallback, `AGENTS.md`) file exists in the
directory where you launch `local-code`, its content is appended to the
system prompt automatically — use it for project conventions, build
commands, and style rules. Content is capped at 20k characters.

## Sessions

Every conversation autosaves to `~/.local-code/sessions/<id>.json` after
each turn. Resume the latest with `local-code --resume`, or a specific one
with `local-code --resume <id>` (list them with `/sessions`). `/clear`
starts a fresh session id.

## @file mentions

Mention files in your prompt with `@path/to/file` and their content is
attached to the message automatically:

```bash
local-code "explicá qué hace @local_code/agent.py"
```

Missing files produce a warning and the message is sent anyway.

## Tools available to the model

| Tool | Asks confirmation |
|---|---|
| `read_file`, `list_dir`, `glob`, `grep` | No |
| `write_file`, `edit_file` (shows diff) | Yes |
| `bash` (shows command) | Yes |

## Configuration

`~/.local-code/config.yaml` (all keys optional):

```yaml
default_model: qwen2.5-coder
max_iterations: 25
bash_timeout_seconds: 120
system_prompt: null
ollama_host: http://localhost:11434
```

## How "any model" works

At startup `local-code` asks Ollama (`/api/show`) whether the model supports
native tool calling.

- **Native mode** — tool schemas are sent with each request; the model returns
  structured `tool_calls`.
- **ReAct fallback** — the system prompt teaches the model to emit
  `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks, which
  local-code parses and executes, feeding results back as `Observation:` lines.
  Malformed calls get error feedback with up to 2 retries before the turn
  aborts.

## Security warning

The `bash` tool runs commands with **no sandbox** — same power and same risk
as typing them into your own shell. Confirmations are the only guardrail;
`--yolo` removes even that. Don't point this at repos you don't trust, and
read every command before approving it.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
