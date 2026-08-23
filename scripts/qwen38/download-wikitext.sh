#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${QWEN38_CACHE_ROOT:-/mnt/S/.cache/llama.cpp/qwen38}"
DEST="${CACHE_ROOT}/wikitext-2-raw"
ARCHIVE="${CACHE_ROOT}/wikitext-2-raw-v1.zip"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

mkdir -p "$CACHE_ROOT"
if [[ ! -f "${DEST}/wiki.test.raw" ]]; then
    curl --fail --location --retry 5 --continue-at - --output "$ARCHIVE" "$URL"
    unzip -o "$ARCHIVE" -d "$CACHE_ROOT"
fi
test -s "${DEST}/wiki.test.raw"

