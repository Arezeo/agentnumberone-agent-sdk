#!/usr/bin/env bash
# One-time setup for a machine that will run the GGUF policy (e.g. the mac mini).
set -euo pipefail
command -v brew >/dev/null || { echo "needs Homebrew"; exit 1; }
command -v llama-server >/dev/null || brew install llama.cpp
command -v uv >/dev/null || { echo "needs uv (brew install uv)"; exit 1; }
uv python install 3.12
uv venv --python 3.12 .venv-control
. .venv-control/bin/activate
uv pip install -r "$(dirname "$0")/requirements.txt"
echo "ready: .venv-control/bin/python scripts/remote-control/agent.py ..."
