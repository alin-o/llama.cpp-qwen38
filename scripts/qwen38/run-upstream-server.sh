#!/usr/bin/env bash
set -euo pipefail

SERVER="${LLAMA_SERVER:-/usr/local/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-/models}"
CACHE_ROOT="${CACHE_ROOT:-/qwen38-cache}"
MODEL="${MODEL_GGUF:-${MODEL_DIR}/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MTP_MODEL="${MTP_GGUF:-${MODEL_DIR}/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
DRAFT_MODEL="${DRAFT_GGUF:-${MODEL_DIR}/DSpark/Qwen3.8-27B-DSpark-Q8_0.gguf}"
MMPROJ="${MMPROJ_GGUF:-${MODEL_DIR}/mmproj-BF16.gguf}"
PROFILE="${CACHE_PROFILE:-q8q5}"

# Optional post-template prompt capture. These files contain raw system
# instructions, tools, and conversation content and must remain local.
PROMPT_LOG_DIR="${PROMPT_LOG_DIR:-}"

# A main GGUF either embeds its MTP head (blk.64.* tensors, e.g. the unsloth
# Q4_K_XL) or not (e.g. the ggml-org Q4_K_M trunk). Detect by scanning the file.
has_embedded_mtp_head() {
    LC_ALL=C grep -aq 'blk\.64\.' "$1" 2>/dev/null
}
# No explicit MTP_GGUF and the model carries no embedded head: prefer the
# separate cache-root head file if it is mounted.
if [[ -z "${MTP_GGUF:-}" ]] && ! has_embedded_mtp_head "$MODEL"; then
    if [[ -f "${CACHE_ROOT}/mtp-Qwen3.8-27B-Q4_0.gguf" ]]; then
        MTP_MODEL="${CACHE_ROOT}/mtp-Qwen3.8-27B-Q4_0.gguf"
    fi
fi
SPEC="${SPEC:-off}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/qwen38/Qwen3.8-developer.jinja}"
KV_PAGED_ENABLED=0
if [[ "${KV_PAGED:-0}" == "1" || "${KV_PAGED:-0}" == "on" ]]; then
    KV_PAGED_ENABLED=1
fi
CACHE_REUSE_EFFECTIVE="${CACHE_REUSE:-256}"
if [[ "$KV_PAGED_ENABLED" == "1" ]]; then
    CACHE_REUSE_EFFECTIVE=0
fi

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

for artifact in "$MODEL" "$MMPROJ"; do
    if [[ ! -f "$artifact" ]]; then
        echo "error: missing artifact: $artifact" >&2
        exit 1
    fi
done
if [[ "$SPEC" == "mtp" ]]; then
    # a separate head file is only required when the trunk does not embed one
    if ! has_embedded_mtp_head "$MODEL" && [[ ! -f "$MTP_MODEL" ]]; then
        echo "error: missing MTP draft: $MTP_MODEL" >&2
        exit 1
    fi
elif [[ "$SPEC" == "dspark" ]]; then
    if [[ ! -f "$DRAFT_MODEL" ]]; then
        echo "error: missing DSpark draft: $DRAFT_MODEL" >&2
        exit 1
    fi
fi

ARGS=(
    --model "$MODEL"
    --mmproj "$MMPROJ"
    --no-mmproj-offload
    --ctx-size "${CTX:-102400}"
    --n-gpu-layers "${NGL:-99}"
    --cache-type-k "$CACHE_K"
    --cache-type-v "$CACHE_V"
    --cache-reuse "$CACHE_REUSE_EFFECTIVE"
    --flash-attn on
    -b 2048
    -ub 2048
    -ngl 999
    --parallel "${NP:-1}"
    --jinja
    --chat-template-file "$CHAT_TEMPLATE_FILE"
    --reasoning-format deepseek
    --metrics
    --slots
    --slot-save-path "${SLOT_SAVE_PATH:-/state}/"
    --slot-prompt-cache-threshold "${SLOT_PROMPT_CACHE_THRESHOLD:-0.5}"
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8080}"
    --threads "${THREADS:-4}"
    --threads-http "${THREADS_HTTP:-4}"
)
#--cont-batching
#--cache-idle-slots

# KV cache backend: unified buffer by default, paged block pool when KV_PAGED.
if [[ "$KV_PAGED_ENABLED" == "1" ]]; then
    KV_MODE="paged"
    ARGS+=(--kv-paged)
    [[ -n "${KV_BLOCK_SIZE:-}" ]]      && ARGS+=(--kv-block-size "$KV_BLOCK_SIZE")
    [[ -n "${N_GPU_BLOCKS:-}" ]]       && ARGS+=(--n-gpu-blocks "$N_GPU_BLOCKS")
    [[ -n "${N_CPU_BLOCKS:-}" ]]       && ARGS+=(--n-cpu-blocks "$N_CPU_BLOCKS")
    [[ -n "${KV_PAGED_WATERMARK:-}" ]] && ARGS+=(--kv-paged-watermark "$KV_PAGED_WATERMARK")
else
    KV_MODE="unified"
    ARGS+=(--kv-unified)
fi

[[ -n "${FIT:-}" ]] && ARGS+=(--fit "$FIT")
[[ -n "$PROMPT_LOG_DIR" ]] && ARGS+=(--log-prompts-dir "$PROMPT_LOG_DIR")

if [[ "$SPEC" == "mtp" ]]; then
    ARGS+=(
        --spec-type draft-mtp
        --spec-draft-n-max "${DRAFT_MAX:-4}"
        --spec-draft-type-k "${DRAFT_CACHE_K:-q8_0}"
        --spec-draft-type-v "${DRAFT_CACHE_V:-q8_0}"
    )
    if ! has_embedded_mtp_head "$MODEL"; then
        # trunk GGUF has no embedded head: load the separate head file
        ARGS+=(--spec-draft-model "$MTP_MODEL")
    fi
elif [[ "$SPEC" == "dspark" ]]; then
    ARGS+=(
        --spec-draft-model "$DRAFT_MODEL"
        --spec-type draft-dspark
        --spec-draft-n-max "${DRAFT_MAX:-7}"
        --spec-draft-type-k "${DRAFT_CACHE_K:-q8_0}"
        --spec-draft-type-v "${DRAFT_CACHE_V:-q8_0}"
    )
elif [[ "$SPEC" != "off" ]]; then
    echo "error: SPEC must be off, mtp, or dspark" >&2
    exit 2
fi

echo "qwen38: engine=upstream profile=${PROFILE} spec=${SPEC} ctx=${CTX:-102400} np=${NP:-1} kv=${KV_MODE} cache_reuse=${CACHE_REUSE_EFFECTIVE} prompt_log=${PROMPT_LOG_DIR:-off}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
