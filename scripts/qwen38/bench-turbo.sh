#!/usr/bin/env bash
set -euo pipefail

# Bench turbo3 vs turbo4 KV cache in llama-bench (pp + tg), upstream container.
# Per profile runs: pp512 (compute-bound), tg256, tg256 at DEPTH (KV-bandwidth-bound).
# Usage: CACHE_PROFILE=turbo3,turbo4,t3k-t4v,t4k-t3v DEPTH=8192 ./bench-turbo.sh

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# /models must be the unsloth store for these runs (compose default is the
# cache root); this bench references /models/<file> paths directly.
export MODELS_DIR="${MODELS_DIR:-/mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e}"
COMPOSE=(docker compose -f "${ROOT}/docker-compose.yml")
MODEL=/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
PROFILES="${CACHE_PROFILE:-turbo4,turbo3,t3k-t4v,t4k-t3v}"
DEPTH="${DEPTH:-16384}"
PP="${PP:-512}"
TG="${TG:-256}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/docker/results/$(date -u +%Y%m%dT%H%M%SZ)-bench-turbo}"
mkdir -p "$RESULTS_DIR"

for profile in ${PROFILES//,/ }; do
    case "$profile" in
        turbo3)  CACHE_K=turbo3; CACHE_V=turbo3 ;;
        turbo4)  CACHE_K=turbo4; CACHE_V=turbo4 ;;
        t3k-t4v) CACHE_K=turbo3; CACHE_V=turbo4 ;;
        t4k-t3v) CACHE_K=turbo4; CACHE_V=turbo3 ;;
        q8turbo3)  CACHE_K=q8_0; CACHE_V=turbo3 ;;
        q8turbo4)  CACHE_K=q8_0; CACHE_V=turbo4 ;;
        q8q5)  CACHE_K=q8_0; CACHE_V=q5_0 ;;
        iq4nl) CACHE_K=iq4_nl; CACHE_V=iq4_nl ;;
        *) echo "error: unknown CACHE_PROFILE: $profile" >&2; exit 2 ;;
    esac

    echo "== profile=${profile} k=${CACHE_K} v=${CACHE_V} depth=${DEPTH}" >&2
    "${COMPOSE[@]}" --profile upstream run --rm --no-deps \
        --entrypoint /usr/local/bin/llama-bench "llama-server" \
        -m "$MODEL" \
        --n-gpu-layers 99 \
        --flash-attn on \
        --cache-type-k "$CACHE_K" \
        --cache-type-v "$CACHE_V" \
        -p "${PP}" \
        -n "${TG}" \
        -d "${DEPTH}" \
        2>&1 | tee "${RESULTS_DIR}/${profile}.log"
    
    sleep 15
done

echo >&2
echo "results in ${RESULTS_DIR}" >&2
