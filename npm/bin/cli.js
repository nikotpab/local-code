#!/usr/bin/env node
//
// Node launcher for local-code (a Python CLI).
//
// `local-code` (Python) is not a Node program; this wrapper exists so it can be
// installed and run through the npm ecosystem (`npm i -g local-code`,
// `npx local-code`). On first run it bootstraps an isolated virtualenv under
// ~/.local-code/node-venv and pip-installs the Python package into it; every
// run afterwards just execs the venv's `local-code` with your arguments.
//
// Environment overrides (match install.sh / install.ps1):
//   LOCAL_CODE_REPO       git repo URL      (default: https://github.com/nikotpab/local-code)
//   LOCAL_CODE_REF        branch/tag/commit (default: main)
//   LOCAL_CODE_PYPI       PyPI package name; if set, installs from PyPI instead of git
//   LOCAL_CODE_REINSTALL  if set, recreate the venv before running

'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const REPO = process.env.LOCAL_CODE_REPO || 'https://github.com/nikotpab/local-code';
const REF = process.env.LOCAL_CODE_REF || 'main';
const PYPI = process.env.LOCAL_CODE_PYPI || '';
const MIN_MINOR = 10; // Python 3.10+

const isWin = process.platform === 'win32';
const VENV = path.join(os.homedir(), '.local-code', 'node-venv');
const binDir = path.join(VENV, isWin ? 'Scripts' : 'bin');
const exe = path.join(binDir, isWin ? 'local-code.exe' : 'local-code');
const venvPython = path.join(binDir, isWin ? 'python.exe' : 'python');

function findPython() {
  const candidates = isWin
    ? [['py', '-3'], ['python'], ['python3']]
    : [['python3'], ['python']];
  const probe = `import sys; sys.exit(0 if sys.version_info[:2] >= (3, ${MIN_MINOR}) else 1)`;
  for (const cand of candidates) {
    const [cmd, ...pre] = cand;
    const r = spawnSync(cmd, [...pre, '-c', probe], { stdio: 'ignore' });
    if (r.status === 0) return cand;
  }
  return null;
}

function run(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: 'inherit' });
  if (r.error) {
    console.error(`local-code: failed to run ${cmd}: ${r.error.message}`);
    process.exit(1);
  }
  return r.status === null ? 1 : r.status;
}

function bootstrap() {
  const py = findPython();
  if (!py) {
    console.error(`local-code: Python 3.${MIN_MINOR}+ not found. Install it first (https://www.python.org/downloads/).`);
    process.exit(1);
  }
  const [pcmd, ...ppre] = py;
  console.error('local-code: first-time setup — creating an isolated environment…');
  fs.mkdirSync(path.dirname(VENV), { recursive: true });
  if (run(pcmd, [...ppre, '-m', 'venv', VENV]) !== 0) process.exit(1);

  const source = PYPI ? PYPI : `git+${REPO}.git@${REF}`;
  run(venvPython, ['-m', 'pip', 'install', '--quiet', '--upgrade', 'pip']);
  if (run(venvPython, ['-m', 'pip', 'install', '--upgrade', source]) !== 0) {
    console.error('local-code: installation failed.');
    process.exit(1);
  }
}

function main() {
  if (process.env.LOCAL_CODE_REINSTALL && fs.existsSync(VENV)) {
    fs.rmSync(VENV, { recursive: true, force: true });
  }
  if (!fs.existsSync(exe)) bootstrap();
  process.exit(run(exe, process.argv.slice(2)));
}

main();
