#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PKG_DIR/../../vendor/hilum-local-llm-engine"

if [ ! -d "$ENGINE_DIR/src" ]; then
  echo "ERROR: Engine source not found at $ENGINE_DIR"
  exit 1
fi

# Remove symlink or stale copy
rm -rf "$PKG_DIR/cpp"
mkdir -p "$PKG_DIR/cpp"

echo "Copying engine source (~20 MB)..."

# Core inference engine
cp -r "$ENGINE_DIR/src"     "$PKG_DIR/cpp/src"
cp -r "$ENGINE_DIR/include" "$PKG_DIR/cpp/include"

# Tensor library + Metal shaders
cp -r "$ENGINE_DIR/ggml"    "$PKG_DIR/cpp/ggml"

# Grammar support, chat templates, json-schema-to-grammar
cp -r "$ENGINE_DIR/common"  "$PKG_DIR/cpp/common"

# Vision / multimodal
cp -r "$ENGINE_DIR/tools/mtmd" "$PKG_DIR/cpp/mtmd"

# Third-party headers (nlohmann/json, stb_image)
mkdir -p "$PKG_DIR/cpp/vendor"
cp -r "$ENGINE_DIR/vendor/nlohmann" "$PKG_DIR/cpp/vendor/nlohmann"
cp -r "$ENGINE_DIR/vendor/stb"      "$PKG_DIR/cpp/vendor/stb"

# Top-level CMakeLists.txt (needed by ggml's internal references)
cp "$ENGINE_DIR/CMakeLists.txt" "$PKG_DIR/cpp/CMakeLists.txt"

# Strip what iOS doesn't need from the copy
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-cuda"
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-vulkan"
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-sycl"
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-cann"
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-hip"
rm -rf "$PKG_DIR/cpp/ggml/src/ggml-kompute"

FINAL_SIZE=$(du -sh "$PKG_DIR/cpp" | cut -f1)
FILE_COUNT=$(find "$PKG_DIR/cpp" -type f | wc -l | tr -d ' ')
echo "Packed cpp/ -- ${FINAL_SIZE}, ${FILE_COUNT} files"
