#!/usr/bin/env bash
set -euo pipefail

SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL="${MODEL_GGUF:-${MODEL_DIR}/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MTP_MODEL="${MTP_GGUF:-${MODEL_DIR}/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
MMPROJ="${MMPROJ_GGUF:-${MODEL_DIR}/mmproj-BF16.gguf}"
SPEC="${SPEC:-off}"

for artifact in "$MODEL" "$MTP_MODEL" "$MMPROJ"; do
    if [[ ! -f "$artifact" ]]; then
        echo "error: missing artifact: $artifact" >&2
        exit 1
    fi
done

ARGS=(
    --model "$MODEL"
    --mmproj "$MMPROJ"
    --no-mmproj-offload
    --ctx-size "${CTX:-102400}"
    --n-gpu-layers "${NGL:-99}"
    --cache-type-k turbo3
    --cache-type-v turbo3
    --flash-attn on
    --parallel 1
    --kv-unified
    --jinja
    --reasoning-format deepseek
    --metrics
    --slots
    --slot-save-path "${SLOT_SAVE_PATH:-/state}/"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8080}"
    --threads "${THREADS:-4}"
    --threads-http "${THREADS_HTTP:-4}"
)

if [[ "$SPEC" == "nextn" ]]; then
    ARGS+=(
        --model-draft "$MODEL"
        --spec-type nextn
        --draft-max "${DRAFT_MAX:-4}"
        --cache-type-k-draft f16
        --cache-type-v-draft f16
    )
elif [[ "$SPEC" != "off" ]]; then
    echo "error: SPEC must be off or nextn" >&2
    exit 2
fi

echo "qwen38: engine=fork profile=turbo3 spec=${SPEC} ctx=${CTX:-102400}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
