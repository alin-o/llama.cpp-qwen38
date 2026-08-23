#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8882/v1}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/qwen38/results/codex-smoke}"
FIXTURE_DIR="$(mktemp -d)"
trap 'rm -rf "$FIXTURE_DIR"' EXIT

cp -a "${ROOT}/qwen38/codex-fixture/." "$FIXTURE_DIR/"
git -C "$FIXTURE_DIR" init -q
mkdir -p "$RESULTS_DIR"

codex exec \
    --ephemeral \
    --full-auto \
    --cd "$FIXTURE_DIR" \
    --model qwen38 \
    --config 'model_provider="qwen38_local"' \
    --config 'model_reasoning_effort="high"' \
    --config 'model_providers.qwen38_local.name="Qwen3.8 llama.cpp"' \
    --config "model_providers.qwen38_local.base_url=\"${BASE_URL}\"" \
    --config 'model_providers.qwen38_local.wire_api="responses"' \
    --config 'model_providers.qwen38_local.requires_openai_auth=false' \
    --output-last-message "${RESULTS_DIR}/last-message.txt" \
    'Implement add(a, b) in math_utils.py. Run test_math_utils.py and stop only after it passes. Do not change the test file.' \
    2>&1 | tee "${RESULTS_DIR}/codex.log"

python3 "${FIXTURE_DIR}/test_math_utils.py"
cmp "${ROOT}/qwen38/codex-fixture/test_math_utils.py" "${FIXTURE_DIR}/test_math_utils.py"
cp "$FIXTURE_DIR/math_utils.py" "${RESULTS_DIR}/math_utils.py"
cp "${ROOT}/qwen38/codex-fixture/PASS" "${RESULTS_DIR}/status.txt"
