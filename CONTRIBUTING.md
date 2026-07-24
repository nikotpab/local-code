# Contributing

Thanks for your interest in improving local-code.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Before opening a pull request

Run the full local gate — the same checks CI enforces:

```bash
.venv/bin/ruff check .      # lint
.venv/bin/mypy              # type-check
.venv/bin/pytest -q         # tests
```

All three must pass. CI runs them on Python 3.10 through 3.13.

## Conventions

- **Tests first.** New behavior comes with tests under `tests/`; the suite is
  the contract. Mirror the style of the nearest existing test module.
- **Keep stdout clean.** Human-facing diagnostics go to stderr (or a Rich
  console pointed at stderr); stdout is reserved for machine-readable output
  such as `--json`.
- **Fail gracefully.** External failures (backend down, MCP server crash, bad
  config) should degrade to a clear message and a meaningful exit code, never
  a traceback.
- **Line length** is 100 (see `pyproject.toml`). Long prompt/description
  strings are exempt (`E501` is ignored).

## Commit messages

Use short, conventional-style prefixes (`feat:`, `fix:`, `style:`,
`docs:`, `test:`) matching the existing history.
