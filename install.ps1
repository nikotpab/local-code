<#
.SYNOPSIS
  local-code installer for Windows (PowerShell).

.DESCRIPTION
  Installs the `local-code` CLI with pipx (preferred, isolated) or falls back
  to `pip install --user`.

  Run:
    irm https://raw.githubusercontent.com/nikotpab/local-code/main/install.ps1 | iex

  Override via environment variables before running:
    $env:LOCAL_CODE_REPO  git repo URL      (default: https://github.com/nikotpab/local-code)
    $env:LOCAL_CODE_REF   branch/tag/commit (default: main)
    $env:LOCAL_CODE_PYPI  PyPI package name; if set, installs from PyPI instead of git
#>

$ErrorActionPreference = 'Stop'

$Repo    = if ($env:LOCAL_CODE_REPO) { $env:LOCAL_CODE_REPO } else { 'https://github.com/nikotpab/local-code' }
$Ref     = if ($env:LOCAL_CODE_REF)  { $env:LOCAL_CODE_REF }  else { 'main' }
$Pypi    = $env:LOCAL_CODE_PYPI
$MinorPy = 10  # require Python 3.10+

function Info($m) { Write-Host "==> $m" -ForegroundColor Green }
function Warn($m) { Write-Host "==> $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

# Python is stored as an executable name plus a fixed argument list, so we
# never slice arrays (which is error-prone in PowerShell for single elements).
$script:PyExe  = $null
$script:PyArgs = @()

function Find-Python {
  $checks = @(
    @{ exe = 'py';      args = @('-3') },
    @{ exe = 'python';  args = @() },
    @{ exe = 'python3'; args = @() }
  )
  foreach ($c in $checks) {
    if (Get-Command $c.exe -ErrorAction SilentlyContinue) {
      $code = "import sys; sys.exit(0 if sys.version_info[:2] >= (3, $MinorPy) else 1)"
      & $c.exe @($c.args) '-c' $code 2>$null
      if ($LASTEXITCODE -eq 0) {
        $script:PyExe  = $c.exe
        $script:PyArgs = $c.args
        return $true
      }
    }
  }
  return $false
}

function Invoke-Py {
  & $script:PyExe @($script:PyArgs) @args
}

if (-not (Find-Python)) {
  Die "Python 3.$MinorPy+ not found. Install it from https://www.python.org/downloads/ or 'winget install Python.Python.3.12'."
}

$PyVer = Invoke-Py '-c' 'import platform; print(platform.python_version())'
Info "using $script:PyExe $($script:PyArgs -join ' ') ($PyVer)"

# --- resolve install source --------------------------------------------------
if ($Pypi) {
  $Source = $Pypi
  Info "installing from PyPI: $Source"
} else {
  $Source = "git+$Repo.git@$Ref"
  Info "installing from git: $Source"
}

# --- prefer pipx -------------------------------------------------------------
$havePipx = $false
if (Get-Command pipx -ErrorAction SilentlyContinue) {
  $havePipx = $true
} else {
  Invoke-Py '-m' 'pipx' '--version' 2>$null
  if ($LASTEXITCODE -eq 0) { $havePipx = $true }
}

if (-not $havePipx) {
  Warn "pipx not found — installing it with pip (user site)"
  Invoke-Py '-m' 'pip' 'install' '--user' '-q' 'pipx'
  try { Invoke-Py '-m' 'pipx' 'ensurepath' } catch {}
  $havePipx = $true
}

if ($havePipx) {
  Info "installing with pipx"
  if (Get-Command pipx -ErrorAction SilentlyContinue) {
    pipx install --force $Source
  } else {
    Invoke-Py '-m' 'pipx' 'install' '--force' $Source
  }
} else {
  Warn "falling back to 'pip install --user'"
  Invoke-Py '-m' 'pip' 'install' '--user' '--upgrade' $Source
}

# --- verify ------------------------------------------------------------------
if (Get-Command local-code -ErrorAction SilentlyContinue) {
  Info "installed: $((Get-Command local-code).Source)"
  local-code --version
} else {
  Warn "installed, but 'local-code' is not on PATH yet. Open a new terminal (pipx ensurepath updates it)."
}

Info "done. run: local-code"
Write-Host "requires a running model server (e.g. 'ollama serve')." -ForegroundColor DarkGray
