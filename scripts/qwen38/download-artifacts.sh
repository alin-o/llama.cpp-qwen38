#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e}"
MODEL_REVISION="27af057ecb382ddfea5d12837360a8980560e3ed"
BASE_URL="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/${MODEL_REVISION}"

mkdir -p "${MODEL_DIR}/MTP"

download() {
    local relative_path="$1"
    local destination="${MODEL_DIR}/${relative_path}"
    curl --fail --location --retry 5 --continue-at - --output "$destination" "${BASE_URL}/${relative_path}"
}

download Qwen3.8-27B-UD-Q4_K_XL.gguf
download MTP/mtp-Qwen3.8-27B-Q4_0.gguf
download mmproj-BF16.gguf

(cd "$MODEL_DIR" && sha256sum -c "${ROOT}/docker/artifacts.sha256")
