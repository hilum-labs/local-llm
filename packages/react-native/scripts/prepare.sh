#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PKG_DIR/../../vendor/hilum-local-llm-engine"

# Only create symlink in monorepo dev (not in published npm)
if [ -d "$ENGINE_DIR" ] && [ ! -d "$PKG_DIR/cpp/src" ]; then
  rm -rf "$PKG_DIR/cpp"
  ln -sf "$ENGINE_DIR" "$PKG_DIR/cpp"
  echo "Symlinked cpp/ -> vendor/hilum-local-llm-engine"
fi
