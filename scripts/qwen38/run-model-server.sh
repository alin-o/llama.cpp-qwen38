#!/usr/bin/env bash
# Generic llama-server launcher for models outside the qwen38 evaluation
# stack (e.g. manually downloaded GGUFs in the cache root). The qwen38
# entrypoint assumes Qwen3.8 weights, its chat template and an mmproj;
# this one assumes nothing beyond a GGUF.
#
# Required: MODEL_GGUF (in-container path, e.g. /qwen38-cache/<file>.gguf).
# Optional: CTX NP CACHE_PROFILE CACHE_REUSE THREADS THREADS_HTTP HOST PORT
#           SLOT_PROMPT_CACHE_THRESHOLD SLOT_SAVE_PATH
#           MMPROJ_GGUF   attach a vision tower when the file exists
#           SPEC=mtp      load a separate MTP head from MTP_GGUF
#                         (no embedded-head detection here; pass the file)
set -euo pipefail

SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server}"
MODEL="${MODEL_GGUF:?MODEL_GGUF is required (in-container path to the GGUF)}"
[[ -f "$MODEL" ]] || { echo "error: missing model: $MODEL" >&2; exit 1; }

PROFILE="${CACHE_PROFILE:-turbo3}"
case "$PROFILE" in
    q8q5)
        CACHE_K=q8_0
        CACHE_V=q5_0
        ;;
    iq4nl)
        CACHE_K=iq4_nl
        CACHE_V=iq4_nl
        ;;
    turbo3)
        CACHE_K=turbo3
        CACHE_V=turbo3
        ;;
    q8turbo3)
        CACHE_K=q8_0
        CACHE_V=turbo3
        ;;
    turbo4)
        CACHE_K=turbo4
        CACHE_V=turbo4
        ;;
    q8turbo4)
        CACHE_K=q8_0
        CACHE_V=turbo4
        ;;
    *)
        echo "error: CACHE_PROFILE must be q8q5, iq4nl, turbo3, q8turbo3, turbo4, or q8turbo4" >&2
        exit 2
        ;;
esac

ARGS=(
    --model "$MODEL"
    --ctx-size "${CTX:-32768}"
    -ngl 999
    --cache-type-k "$CACHE_K"
    --cache-type-v "$CACHE_V"
    --cache-reuse "${CACHE_REUSE:-256}"
    --flash-attn on
    -b 2048
    -ub 2048
    --parallel "${NP:-1}"
    --kv-unified
    --jinja
    --metrics
    --slots
    --slot-save-path "${SLOT_SAVE_PATH:-/state}/"
    --slot-prompt-cache-threshold "${SLOT_PROMPT_CACHE_THRESHOLD:-0.5}"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8080}"
    --threads "${THREADS:-4}"
    --threads-http "${THREADS_HTTP:-4}"
)

if [[ -n "${MMPROJ_GGUF:-}" ]]; then
    [[ -f "$MMPROJ_GGUF" ]] || { echo "error: missing mmproj: $MMPROJ_GGUF" >&2; exit 1; }
    ARGS+=(--mmproj "$MMPROJ_GGUF" --no-mmproj-offload)
fi

if [[ "${SPEC:-off}" == "mtp" ]]; then
    [[ -n "${MTP_GGUF:-}" && -f "${MTP_GGUF}" ]] \
        || { echo "error: SPEC=mtp needs MTP_GGUF (separate head file)" >&2; exit 1; }
    ARGS+=(
        --spec-type draft-mtp
        --spec-draft-model "$MTP_GGUF"
        --spec-draft-n-max "${DRAFT_MAX:-4}"
        --spec-draft-type-k "${DRAFT_CACHE_K:-q8_0}"
        --spec-draft-type-v "${DRAFT_CACHE_V:-q8_0}"
    )
elif [[ "${SPEC:-off}" != "off" ]]; then
    echo "error: SPEC must be off or mtp" >&2
    exit 2
fi

echo "model: engine=generic profile=${PROFILE} spec=${SPEC:-off} ctx=${CTX:-32768} np=${NP:-1} model=${MODEL}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
