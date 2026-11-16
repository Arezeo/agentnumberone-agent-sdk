#!/usr/bin/env bash
# Serve a GameScaling GGUF for the SDK harness.
#   ./serve.sh /path/to/GameScaling-9B-S3.Q4_K_M.gguf [/path/to/mmproj-f16.gguf]
set -euo pipefail
MODEL="${1:?usage: serve.sh MODEL.gguf [MMPROJ.gguf]}"
MMPROJ="${2:-}"
PORT="${ANO_PORT:-8901}"
CTX="${ANO_CTX:-4096}"
ARGS=(-m "$MODEL" -c "$CTX" --host 127.0.0.1 --port "$PORT" -t "${ANO_THREADS:-6}")
[[ -n "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")
echo "llama-server ${ARGS[*]}" >&2
exec llama-server "${ARGS[@]}"
