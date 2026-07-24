#!/usr/bin/env bash
#
# local-code installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/nikotpab/local-code/main/install.sh | bash
#
# Installs the `local-code` CLI with pipx (preferred, isolated) or falls back
# to `pip install --user`. Overridable via environment variables:
#
#   LOCAL_CODE_REPO   git repo URL      (default: https://github.com/nikotpab/local-code)
#   LOCAL_CODE_REF    branch/tag/commit (default: main)
#   LOCAL_CODE_PYPI   PyPI package name; if set, installs from PyPI instead of git
#
set -euo pipefail

REPO="${LOCAL_CODE_REPO:-https://github.com/nikotpab/local-code}"
REF="${LOCAL_CODE_REF:-main}"
PYPI="${LOCAL_CODE_PYPI:-}"
MIN_PY_MINOR=10  # require Python 3.10+

# --- pretty output -----------------------------------------------------------
if [ -t 1 ]; then
  BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m'); RED=$(printf '\033[31m')
  GREEN=$(printf '\033[32m'); YELLOW=$(printf '\033[33m'); RESET=$(printf '\033[0m')
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi
info()  { printf '%s==>%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()  { printf '%s==>%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()   { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# --- locate a suitable python ------------------------------------------------
find_python() {
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, ${MIN_PY_MINOR}) else 1)" 2>/dev/null; then
        echo "$cand"; return 0
      fi
    fi
  done
  return 1
}

OS="$(uname -s)"
case "$OS" in
  Darwin) info "detected macOS" ;;
  Linux)  info "detected Linux" ;;
  *)      warn "unrecognized OS '$OS' — attempting anyway" ;;
esac

PY="$(find_python)" || die "Python 3.${MIN_PY_MINOR}+ not found. Install it first (e.g. 'brew install python' or your distro's package)."
PY_VER="$("$PY" -c 'import platform; print(platform.python_version())')"
info "using $PY ($PY_VER)"

# --- resolve install source -------------------------------------------------
if [ -n "$PYPI" ]; then
  SOURCE="$PYPI"
  info "installing from PyPI: $SOURCE"
else
  SOURCE="git+${REPO}.git@${REF}"
  info "installing from git: $SOURCE"
fi

# --- prefer pipx -------------------------------------------------------------
ensure_pipx() {
  if command -v pipx >/dev/null 2>&1; then return 0; fi
  if "$PY" -m pipx --version >/dev/null 2>&1; then return 0; fi
  warn "pipx not found — installing it with pip (user site)"
  "$PY" -m pip install --user -q pipx || return 1
  "$PY" -m pipx ensurepath >/dev/null 2>&1 || true
  return 0
}

run_pipx() {
  if command -v pipx >/dev/null 2>&1; then pipx "$@"; else "$PY" -m pipx "$@"; fi
}

if ensure_pipx; then
  info "installing with pipx"
  # --force makes re-runs upgrade cleanly.
  run_pipx install --force "$SOURCE"
else
  warn "falling back to 'pip install --user'"
  "$PY" -m pip install --user --upgrade "$SOURCE"
fi

# --- verify ------------------------------------------------------------------
if command -v local-code >/dev/null 2>&1; then
  info "${BOLD}installed:${RESET} $(command -v local-code)"
  local-code --version || true
else
  warn "installed, but 'local-code' is not on your PATH yet."
  warn "open a new shell, or add pipx's bin dir:  ${DIM}\$($PY -m pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo ~/.local/bin)${RESET}"
fi

info "done. run: ${BOLD}local-code${RESET}"
printf '%srequires a running model server (e.g. \`ollama serve\`).%s\n' "$DIM" "$RESET"
