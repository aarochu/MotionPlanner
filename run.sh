#!/usr/bin/env bash
# Start the FastAPI backend and serve the single-page frontend.
# Requires: uv (https://github.com/astral-sh/uv)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"

# Source env if present (VIAM_API_KEY_ID, VIAM_API_KEY, VIAM_MACHINE_ADDRESS)
if [[ -f "$ROOT/.env" ]]; then
  set -a; . "$ROOT/.env"; set +a
elif [[ -f "$ROOT/env.sh" ]]; then
  set -a; . "$ROOT/env.sh"; set +a
fi

export POSES_FILE="${POSES_FILE:-$ROOT/poses.json}"

echo "==> installing backend deps (uv sync)"
uv sync --directory "$BACKEND"

echo "==> starting backend on http://127.0.0.1:8000"
uv run --directory "$BACKEND" uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

sleep 2

echo "==> frontend at http://127.0.0.1:8000"
if command -v open >/dev/null 2>&1; then
  open http://127.0.0.1:8000
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://127.0.0.1:8000
fi

wait "$BACKEND_PID"
