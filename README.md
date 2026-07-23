# local-code

Agentic coding CLI for **any** local model — served by
[Ollama](https://ollama.com) or by any OpenAI-compatible server (LM Studio,
llama.cpp, vLLM). Models with native tool-calling support use it directly;
models without it work through a ReAct-style prompt fallback, so even older
models can drive the agent. Responses render as markdown live in the
terminal, and the agent keeps a visible todo checklist while it works
through multi-step tasks.

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
local-code --backend openai            # force the OpenAI-compatible client
local-code --resume                    # resume the latest session
local-code --resume 20260722-153045-a1b2  # resume a specific session
local-code "arregla el bug en x.py"    # one-shot, no REPL
```

REPL commands: `/help` · `/tools` (tool table) · `/history` (current session
summary) · `/sessions` (saved sessions) · `/undo` (revert the last file
change) · `/clear` (reset history, starts a new session) · `/model <name>`
(hot-switch model, re-detects tool support) · `/exit`.

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
| `read_file`, `list_dir`, `glob`, `grep`, `set_todos` | No |
| `write_file`, `edit_file` (shows diff), `multi_edit` (shows diff) | Yes |
| `bash` (shows command), `web_fetch` (shows URL) | Yes |

## Context compaction

Local models have small context windows. When the conversation grows past
~70% of the model's window (auto-detected from Ollama, or set
`context_window` in the config), older messages are summarized by the same
model into a single note and recent messages are kept. If summarization
fails, older messages are simply dropped. You'll see a dim
`context compacted (...)` notice when it happens.

## Permissions

Confirmation prompts accept `y` (yes, once), `n` (no) and `a` (always).
Choosing `a` stores the tool — or, for `bash`, the exact command as a
prefix — in `~/.local-code/permissions.yaml`, and it won't ask again.
Edit that file by hand to shorten prefixes (e.g. leave just `npm test`)
or revoke grants.

A stored bash prefix only covers commands that continue with plain
arguments: approving `npm test` also allows `npm test -- --watch`, but not
`npm testify`, and not anything that chains, pipes, redirects or substitutes
(`npm test; rm -rf /` still asks). It is still a text match, not a parser —
grant prefixes you would be comfortable running unattended.

## Custom commands

Drop markdown files in `~/.local-code/commands/`. A file named `review.md`
becomes `/review` in the REPL; `$ARGUMENTS` inside the file is replaced by
whatever you type after the command:

```markdown
<!-- ~/.local-code/commands/review.md -->
Revisá $ARGUMENTS buscando bugs y problemas de estilo.
```

`/help` lists the custom commands it finds.

## Backends

By default the host URL decides: one ending in `/v1` uses the
OpenAI-compatible client, anything else uses Ollama. Force it with
`--backend ollama|openai` or `backend:` in the config.

```yaml
ollama_host: http://localhost:1234/v1   # LM Studio, llama.cpp, vLLM…
backend: openai                          # optional, inferred from the URL
api_key: null                            # or set LOCAL_CODE_API_KEY
```

Servers without an `/api/show` endpoint can't advertise tool support, so
local-code probes once with a dummy tool and caches the answer in
`~/.local-code/capabilities.json`.

## Undo

Before every `write_file`, `edit_file` or `multi_edit`, the previous
contents are copied to `~/.local-code/checkpoints/`. `/undo` restores the
most recent one (or deletes the file, if the agent had just created it).
Repeat it to walk further back.

## Hooks

Put executable scripts at `~/.local-code/hooks/pre_tool` and
`~/.local-code/hooks/post_tool`. They receive a JSON payload on stdin
(`tool`, `arguments`, `cwd`, plus `result` for post) and have 10 seconds to
answer.

A non-zero exit from `pre_tool` **blocks** the tool, and its stderr is shown
to the model so it can adjust — use it for rules the model must not talk its
way around:

```sh
#!/bin/sh
# ~/.local-code/hooks/pre_tool — never touch production config
grep -q '"path": *"[^"]*prod' && { echo "prod files are off limits" >&2; exit 1; }
exit 0
```

`post_tool` only observes; its stdout is printed as a note. A hook that
times out or can't run blocks the tool — a broken guard is treated as a
closed one.

## Configuration

`~/.local-code/config.yaml` (all keys optional):

```yaml
default_model: qwen2.5-coder
max_iterations: 25
bash_timeout_seconds: 120
system_prompt: null
ollama_host: http://localhost:11434
context_window: null   # tokens; null = auto-detect from the model
backend: null          # ollama | openai; null = infer from the host URL
api_key: null          # for OpenAI-compatible servers; or LOCAL_CODE_API_KEY
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
