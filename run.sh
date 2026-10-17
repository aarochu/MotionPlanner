#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ./env.sh
case "$VIAM_API_KEY" in PASTE_*) echo "Edit env.sh and paste your API key first."; exit 1;; esac
exec uv run streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
