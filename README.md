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
local-code --no-tui                 # force line-based REPL instead of TUI
local-code "arregla el bug en x.py"    # one-shot, no REPL
```

REPL commands: `/help` · `/tools` (tool table) · `/history` (current session
summary) · `/sessions` (saved sessions) · `/undo` (revert the last file
change) · `/clear` (reset history, starts a new session) · `/model <name>`
(hot-switch model, re-detects tool support) · `/plan` (toggle plan mode) ·
`/approve` (execute the current plan) · `/mcp` (MCP server status) · `/exit`.

## TUI (Full-Screen Interface)

When launched interactively on a terminal, `local-code` opens a full-screen
split-pane TUI built on `textual`.

- **Header bar**: Model name, backend info, native/ReAct mode, token context usage, plan mode indicator, and working directory.
- **Conversation Pane**: Left pane with live streaming Markdown assistant responses, user messages, and tool activity logs.
- **Activity Pane**: Right side panel (toggle with `Ctrl+B`) showing live `set_todos` checklist, syntax-highlighted code diffs from file edits, and MCP server status.
- **Input Area**: Multi-line prompt input box supporting all slash commands.
- **Confirmation Modals**: Dialog popups for tool confirmations (`y`=yes, `n`=no, `a`=always).

### TUI Key Bindings

- **Enter**: Submit prompt or slash command
- **Shift+Enter**: Insert newline in input
- **Ctrl+B**: Toggle side activity pane
- **Ctrl+C**: Cancel current turn (or exit if idle)
- **Ctrl+Q**: Exit application

Use `--no-tui` to force the classic line-based REPL interface.

## Plan mode

Start with `--plan` or toggle it with `/plan` in the REPL. In plan mode the
model is restricted to **read-only tools** (read_file, list_dir, glob, grep,
set_todos) — side-effecting tools (write_file, edit_file, bash, …) are blocked
even if `--yolo` is set. The system prompt instructs the model to investigate
and then output a **numbered step-by-step plan**. The prompt prefix shows
`[plan]` while active.

```bash
local-code --plan "refactor the auth module"  # produces a plan, then stops
```

Inside the REPL:

```
> /plan         # turn plan mode on
[plan] > investigate src/auth
        …model reads files and outputs a plan…
[plan] > /approve   # turn plan mode off and execute the last plan
```

`/approve` turns off plan mode and feeds the last assistant message back as
`"Execute this plan:\n\n{plan}"`, so the model carries out every step with
full tools.  If there is no prior assistant message, it prints a hint.

## Subagents

The `spawn_agent` tool lets the model delegate a focused investigation task to
a **fresh read-only subagent**. The subagent uses only read-only tools, runs for
up to 15 iterations, and returns a text report — it cannot modify files, run
commands, or spawn further subagents (hard recursion guard).

```json
{
  "name": "spawn_agent",
  "arguments": {
    "task": "list all places where Config is read and summarise the defaults",
    "model": "llama3.2"   // optional — defaults to the parent model
  }
}
```

The report is prefixed with `[Subagent report from <model>]` and truncated at
10 000 characters so it doesn't flood the parent context.

**Write-capable subagents are out of scope for v1** — a subagent that can
modify files without the user watching is a different risk model that needs
explicit confirmation wiring.  Add a note in a future feature request if you
need that.

## Project context

If a `LOCALCODE.md` (or, as fallback, `AGENTS.md`) file exists in the
directory where you launch `local-code`, its content is appended to the
system prompt automatically — use it for project conventions, build
commands, and style rules. Content is capped at 20k characters.

## Environment awareness

The model is given a `# Environment` block with real machine facts — OS, home
directory, working directory, user, shell, date, git repository — plus the
**actual** folder names in your home directory. This stops the model from
inventing localized paths (e.g. writing to `~/Escritorio` instead of the real
`~/Desktop`). Folder names come from what exists on disk, so it is correct on
any OS and locale.

As a safety net, if a `write_file` would create a brand-new top-level folder
under your home directory, the confirmation prompt flags it (`⚠ this would
create a new folder that doesn't exist`) so you can catch a wrong path before
approving. Under `--yolo` there is no confirmation, so this check is skipped.

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

## MCP (Model Context Protocol) servers

local-code can expose tools from external MCP servers — GitHub, Postgres,
filesystem, Brave Search, and anything else that speaks the stdio MCP protocol
— alongside its built-in tools, with no extra code.

### Config: `~/.local-code/mcp.json`

Create this file (standard MCP client config format):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    }
  }
}
```

On startup local-code spawns each listed server, performs the MCP
`initialize` handshake, and calls `tools/list` to discover available tools.
A missing or malformed `mcp.json` is silently ignored — the CLI works
normally with zero MCP servers.

### Tool namespacing

Every MCP tool is exposed to the model as `{server}__{tool}` (double
underscore), so `filesystem`'s `read_file` becomes `filesystem__read_file`.
This prevents any collision with local built-in tools or between servers.
The tool description shown to the model notes the originating server.

### Confirmation

MCP tools require confirmation by default (they call external processes that
may have side effects). Add `"trust": true` to a server entry to skip
confirmation for that server:

```json
"filesystem": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  "trust": true
}
```

Trust is per-server; other servers still ask.

### `/mcp` command

In the REPL, `/mcp` lists the connected servers and how many tools each
exposes:

```
┌──────────────┬───────┐
│ server       │ tools │
├──────────────┼───────┤
│ filesystem   │ 7     │
│ github       │ 23    │
└──────────────┴───────┘
```

### Graceful degradation

A server that fails to start, times out during the handshake, or dies
mid-session never crashes the CLI. A dim warning is shown and the server's
tools are simply absent. A `tools/call` to a dead server returns an
`"Error: …"` string to the model instead of raising.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
