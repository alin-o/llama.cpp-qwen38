#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# /models must be the unsloth store for these runs (compose default is the
# cache root); this bench references /models/<file> paths directly.
export MODELS_DIR="${MODELS_DIR:-/mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e}"
COMPOSE=(docker compose -f "${ROOT}/docker-compose.yml")
MODEL=/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
DATA=/models/wikitext-2-raw/wiki.test.raw
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/docker/results/$(date -u +%Y%m%dT%H%M%SZ)-ppl}"
mkdir -p "$RESULTS_DIR"

run_one() {
    local engine="$1"
    local profile="$2"
    local cache_k="$3"
    local cache_v="$4"
    local service
    if [[ "$engine" == upstream ]]; then
        service=llama-server
    fi
    "${COMPOSE[@]}" --profile "$engine" run --rm --no-deps \
        --entrypoint /usr/local/bin/llama-perplexity "$service" \
        --model "$MODEL" \
        --file "$DATA" \
        --ctx-size 4096 \
        --batch-size 512 \
        --ubatch-size 512 \
        --n-gpu-layers 99 \
        --flash-attn on \
        --cache-type-k "$cache_k" \
        --cache-type-v "$cache_v" \
        2>&1 | tee "${RESULTS_DIR}/${profile}.log"
}

run_one fork fork-turbo3 turbo3 turbo3
run_one upstream upstream-q8q5 q8_0 q5_0
run_one upstream upstream-iq4nl iq4_nl iq4_nl
python3 "${ROOT}/scripts/qwen38/evaluate-perplexity.py" --results-dir "$RESULTS_DIR" --output "${RESULTS_DIR}/quality.json"
