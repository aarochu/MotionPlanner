#!/usr/bin/env bash
# Checks that a checkout works end to end with uv. Read-only: nothing here moves an arm.
#   scripts/validate.sh          offline checks: exact install from uv.lock, compile, unit tests
#   scripts/validate.sh --live   also renders the dashboard against armfarm16 (needs env.sh, machine online)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== $(uv --version)"
echo "== install exactly what uv.lock pins (fails if pyproject.toml and uv.lock disagree)"
uv sync --locked

echo "== Python $(uv run python -c 'import platform; print(platform.python_version())')"
echo "== compile all modules"
uv run python -m py_compile *.py tests/*.py

echo "== unit tests"
uv run python -m unittest discover -s tests

if [[ "${1:-}" == "--live" ]]; then
  [[ -f env.sh ]] || { echo "env.sh missing: copy env.sh.example and fill in your Viam API key"; exit 1; }
  # shellcheck disable=SC1091
  source ./env.sh
  echo "== live smoke test: render the dashboard for arm and arm2 (read-only, no buttons pressed)"
  uv run python scripts/smoke_app.py
fi
echo "== all checks passed"
